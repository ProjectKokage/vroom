"""Deterministic curation of sampled VRMA clips."""

from __future__ import annotations

import hashlib
import math
import struct
from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import Any

from .errors import GlbError, VrmaError
from .glb import AccessorReader, Glb
from .motion import Bone, MotionClip, Quat, Skeleton, Vec3
from .vrm import HUMANOID_PARENTS
from .vrma import VrmaFile, build_vrma, read_vrma, validate_vrma

CURATION_METADATA_KEY = "vroomCuration"
CURATION_METADATA_VERSION = 3
IDENTITY_QUATERNION: Quat = (0.0, 0.0, 0.0, 1.0)
MAX_ANIMATION_NAME_BYTES = 256
MAX_CURATION_DURATION_SECONDS = 3_600.0
MAX_CURATION_OUTPUT_FRAMES = 18_000
MAX_CURATION_OUTPUT_BYTES = 16 * 1024 * 1024
MAX_CURATION_SOURCE_BYTES = 64 * 1024 * 1024
MAX_CURATION_SOURCE_COMPONENTS = 1_000_000
MAX_CURATION_SOURCE_FRAMES = 18_000
MAX_CURATION_SOURCE_NODES = 512
MAX_CURATION_SOURCE_TRACKS = 256
MAX_CURATED_ROOT_DISPLACEMENT_METERS = 0.5
MAX_ROTATION_GAIN = 4.0
ROOT_RETURN_MINIMUM_EXTRA_METERS = 0.05
ROOT_RETURN_MAXIMUM_EXTRA_RATIO = 0.25
ROTATION_PROJECTIONS = ("none", "twist-x", "twist-y", "twist-z")
HIPS_TRANSLATION_MODES = ("strip", "relative-return")
MIN_RETURN_TAIL_INTERVALS = 4
_ACCESSOR_TYPE_WIDTHS = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
    "MAT2": 4,
    "MAT3": 9,
    "MAT4": 16,
}

type Matrix4 = tuple[
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
]
type VrmaSource = str | Path | bytes | Glb


def _require_rotation_gain(value: float, *, label: str = "rotation gain") -> None:
    if not math.isfinite(value) or value < 0 or value > MAX_ROTATION_GAIN:
        raise VrmaError(f"{label} must be a finite number from 0 to {MAX_ROTATION_GAIN:g}")


def _require_rotation_projection(
    value: str,
    *,
    label: str = "rotation projection",
) -> None:
    if not isinstance(value, str) or value not in ROTATION_PROJECTIONS:
        raise VrmaError(f"{label} must be none, twist-x, twist-y, or twist-z")


@dataclass(frozen=True, slots=True)
class _ReturnedTracks:
    times: tuple[float, ...]
    hips_translations: tuple[Vec3, ...]
    rotations: dict[str, tuple[Quat, ...]]
    start_hold_duration: float
    return_intervals: int


@dataclass(frozen=True, slots=True)
class BoneCurationPolicy:
    """Optional gain and projection overrides for one selected rotation bone."""

    bone: str
    rotation_gain: float | None = None
    rotation_projection: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.bone, str) or not self.bone:
            raise VrmaError("bone policy names must be non-empty strings")
        if self.rotation_gain is not None:
            _require_rotation_gain(self.rotation_gain, label=f"{self.bone} rotation gain")
        if self.rotation_projection is not None:
            _require_rotation_projection(
                self.rotation_projection,
                label=f"{self.bone} rotation projection",
            )
        if self.rotation_gain is None and self.rotation_projection is None:
            raise VrmaError(f"{self.bone} bone policy must override rotation gain or projection")


@dataclass(frozen=True, slots=True)
class HipsTranslationPolicy:
    """Closed policy for stripping or safely returning first-relative root motion."""

    mode: str = "strip"
    return_duration: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.mode, str) or self.mode not in HIPS_TRANSLATION_MODES:
            raise VrmaError("hips translation mode must be strip or relative-return")
        if not math.isfinite(self.return_duration) or self.return_duration < 0:
            raise VrmaError("hips return duration must be a non-negative finite number")
        if self.mode == "strip" and self.return_duration != 0:
            raise VrmaError("hips return duration requires relative-return translation mode")
        if self.mode == "relative-return" and self.return_duration == 0:
            raise VrmaError("relative-return translation mode requires a positive return duration")


@dataclass(frozen=True, slots=True)
class CurationOptions:
    """Closed, reusable parameters for a deterministic curation transform."""

    bones: tuple[str, ...]
    duration: float
    fade_in: float = 0.0
    fade_out: float = 0.0
    rotation_gain: float = 1.0
    rotation_projection: str = "none"
    start_frame: int = 0
    end_frame: int | None = None
    name: str | None = None
    bone_policies: tuple[BoneCurationPolicy, ...] = ()
    hips_translation_policy: HipsTranslationPolicy = field(default_factory=HipsTranslationPolicy)

    def __post_init__(self) -> None:
        if not self.bones:
            raise VrmaError("at least one humanoid rotation bone is required")
        if len(self.bones) > len(HUMANOID_PARENTS):
            raise VrmaError("too many humanoid rotation bones were requested")
        if any(not isinstance(bone, str) or not bone for bone in self.bones):
            raise VrmaError("bone names must be non-empty strings")
        if len(set(self.bones)) != len(self.bones):
            raise VrmaError("humanoid rotation bones must not be repeated")
        if not math.isfinite(self.duration) or self.duration <= 0:
            raise VrmaError("curated duration must be a positive finite number")
        if self.duration > MAX_CURATION_DURATION_SECONDS:
            raise VrmaError(
                f"curated duration must not exceed {MAX_CURATION_DURATION_SECONDS:g} seconds"
            )
        for label, value in (("fade-in", self.fade_in), ("fade-out", self.fade_out)):
            if not math.isfinite(value) or value < 0:
                raise VrmaError(f"{label} duration must be a non-negative finite number")
            if value > self.duration:
                raise VrmaError(f"{label} duration must not exceed the curated duration")
        if self.fade_in + self.fade_out > self.duration:
            raise VrmaError("fade-in and fade-out durations must not overlap")
        _require_rotation_gain(self.rotation_gain)
        _require_rotation_projection(self.rotation_projection)
        if (
            not isinstance(self.start_frame, int)
            or isinstance(self.start_frame, bool)
            or self.start_frame < 0
        ):
            raise VrmaError("start frame must be a non-negative integer")
        if self.end_frame is not None and (
            not isinstance(self.end_frame, int)
            or isinstance(self.end_frame, bool)
            or self.end_frame < self.start_frame
        ):
            raise VrmaError("end frame must be an integer at or after the start frame")
        if self.name is not None:
            _validate_animation_name(self.name)
        if any(not isinstance(policy, BoneCurationPolicy) for policy in self.bone_policies):
            raise VrmaError("bone policies must contain BoneCurationPolicy values")
        policy_bones = tuple(policy.bone for policy in self.bone_policies)
        if len(set(policy_bones)) != len(policy_bones):
            raise VrmaError("bone policies must not repeat a humanoid bone")
        unselected_policy_bones = set(policy_bones) - set(self.bones)
        if unselected_policy_bones:
            raise VrmaError(
                "bone policies require selected bones: "
                f"{', '.join(sorted(unselected_policy_bones))}"
            )
        if not isinstance(self.hips_translation_policy, HipsTranslationPolicy):
            raise VrmaError("hips translation policy must be a HipsTranslationPolicy")
        if (
            self.duration + self.hips_translation_policy.return_duration
            > MAX_CURATION_DURATION_SECONDS
        ):
            raise VrmaError(
                "main and hips return durations must not exceed "
                f"{MAX_CURATION_DURATION_SECONDS:g} seconds"
            )


def _effective_bone_policies(
    options: CurationOptions,
    ordered_bones: tuple[str, ...],
) -> dict[str, BoneCurationPolicy]:
    overrides = {policy.bone: policy for policy in options.bone_policies}
    return {
        bone: BoneCurationPolicy(
            bone=bone,
            rotation_gain=(
                options.rotation_gain
                if overrides.get(bone) is None or overrides[bone].rotation_gain is None
                else overrides[bone].rotation_gain
            ),
            rotation_projection=(
                options.rotation_projection
                if overrides.get(bone) is None or overrides[bone].rotation_projection is None
                else overrides[bone].rotation_projection
            ),
        )
        for bone in ordered_bones
    }


def curate_vrma(
    source: VrmaSource,
    *,
    bones: Sequence[str],
    duration: float,
    fade_in: float = 0.0,
    fade_out: float = 0.0,
    rotation_gain: float = 1.0,
    rotation_projection: str = "none",
    start_frame: int = 0,
    end_frame: int | None = None,
    name: str | None = None,
    bone_policies: Sequence[BoneCurationPolicy] = (),
    hips_translation_policy: HipsTranslationPolicy | None = None,
) -> Glb:
    """Create a bounded first-frame-relative VRMA from sampled LINEAR input.

    The default strips hips translation and retains the legacy rotation fade
    behavior. ``relative-return`` translation mode preserves the cropped main
    root trajectory, adds a rest hold when needed, and appends a sampled
    rest-to-rest return tail for both translation and selected rotations.
    """

    options = CurationOptions(
        bones=tuple(bones),
        duration=float(duration),
        fade_in=float(fade_in),
        fade_out=float(fade_out),
        rotation_gain=float(rotation_gain),
        rotation_projection=rotation_projection,
        start_frame=start_frame,
        end_frame=end_frame,
        name=name,
        bone_policies=tuple(bone_policies),
        hips_translation_policy=(
            HipsTranslationPolicy() if hips_translation_policy is None else hips_translation_policy
        ),
    )
    source_bytes, source_glb = _load_source(source)
    validate_vrma(source_glb).require_valid()
    source_vrma = read_vrma(source_glb)
    _require_linear_animation(source_vrma)

    missing_bones = set(options.bones) - source_vrma.humanoid_bones.keys()
    if missing_bones:
        raise VrmaError(f"input VRMA has no humanoid bones: {', '.join(sorted(missing_bones))}")

    skeleton = _skeleton_from_vrma(source_vrma)
    ordered_bones = tuple(bone.name for bone in skeleton.bones if bone.name in set(options.bones))
    source_times, source_tracks = _selected_rotation_tracks(source_vrma, ordered_bones)
    if options.start_frame >= len(source_times):
        raise VrmaError(
            f"start frame {options.start_frame} is outside the input's {len(source_times)} frames"
        )
    last_frame = len(source_times) - 1 if options.end_frame is None else options.end_frame
    if last_frame >= len(source_times):
        raise VrmaError(f"end frame {last_frame} is outside the input's {len(source_times)} frames")
    frame_count = last_frame - options.start_frame + 1
    if frame_count < 2:
        raise VrmaError("curation requires at least two selected frames")
    retained_times = source_times[options.start_frame : last_frame + 1]
    source_duration = retained_times[-1] - retained_times[0]
    output_times = tuple(
        _float32_at_or_below(
            0.0
            if frame == 0
            else (
                options.duration
                if frame == frame_count - 1
                else (source_time - retained_times[0]) / source_duration * options.duration
            )
        )
        for frame, source_time in enumerate(retained_times)
    )

    effective_policies = _effective_bone_policies(options, ordered_bones)
    closes_with_return_tail = options.hips_translation_policy.mode == "relative-return"
    rotations = {
        bone: _curate_rotation_track(
            values[options.start_frame : last_frame + 1],
            label=bone,
            sample_times=output_times,
            fade_in=options.fade_in,
            fade_out=options.fade_out,
            rotation_gain=effective_policies[bone].rotation_gain,
            rotation_projection=effective_policies[bone].rotation_projection,
            force_identity_end=not closes_with_return_tail,
        )
        for bone, values in source_tracks.items()
    }
    fps = (frame_count - 1) / options.duration
    if not math.isfinite(fps) or fps <= 0:
        raise VrmaError("curated duration is too small for the selected frame range")

    hips_translations: tuple[Vec3, ...] = ()
    start_hold_duration = 0.0
    return_intervals = 0
    if closes_with_return_tail:
        source_hips_translations = _hips_translation_track(source_vrma, source_times)
        main_hips_translations = _first_relative_hips_track(
            source_hips_translations[options.start_frame : last_frame + 1],
            rest_translation=skeleton.bones[0].translation,
        )
        returned = _append_rest_return(
            sample_times=output_times,
            hips_translations=main_hips_translations,
            rotations=rotations,
            return_duration=options.hips_translation_policy.return_duration,
            nominal_fps=fps,
        )
        output_times = returned.times
        hips_translations = returned.hips_translations
        rotations = returned.rotations
        start_hold_duration = returned.start_hold_duration
        return_intervals = returned.return_intervals

    if len(output_times) > MAX_CURATION_OUTPUT_FRAMES:
        raise VrmaError(
            f"curated output has {len(output_times)} frames; limit is {MAX_CURATION_OUTPUT_FRAMES}"
        )
    output_duration = (
        options.duration + start_hold_duration + options.hips_translation_policy.return_duration
    )
    if output_duration > MAX_CURATION_DURATION_SECONDS:
        raise VrmaError(
            f"curated output duration must not exceed {MAX_CURATION_DURATION_SECONDS:g} seconds"
        )

    output_name = _curated_name(options.name, source_vrma.animation.get("name"))
    output = build_vrma(
        MotionClip(
            fps=fps,
            hips_translations=hips_translations,
            rotations=rotations,
            name=output_name,
            sample_times=output_times,
        ),
        skeleton,
    )
    policy_metadata = [
        {
            "bone": bone,
            "rotationGain": effective_policies[bone].rotation_gain,
            "rotationProjection": effective_policies[bone].rotation_projection,
        }
        for bone in ordered_bones
    ]
    maximum_root_displacement, maximum_root_velocity = _root_motion_metrics(output)
    transform_metadata: dict[str, Any] = {
        "bones": list(ordered_bones),
        "startFrame": options.start_frame,
        "endFrame": last_frame,
        "durationSeconds": output_duration,
        "mainDurationSeconds": options.duration,
        "fadeInSeconds": 0.0 if options.fade_in == 0 else options.fade_in,
        "fadeOutSeconds": 0.0 if options.fade_out == 0 else options.fade_out,
        "rotationGain": (0.0 if options.rotation_gain == 0 else options.rotation_gain),
        "rotationProjection": options.rotation_projection,
        "rotationBasis": "inverse(first) * frame",
        "quaternionPath": "shortest",
        "endpoints": (
            "identity" if not closes_with_return_tail else "rest with zero start and end velocity"
        ),
        "hipsTranslationReturn": (
            {"mode": "none"}
            if not closes_with_return_tail
            else {
                "mode": "relative-return",
                "basis": "rest + (frame - first)",
                "mainSamples": "preserved",
                "startHoldSeconds": start_hold_duration,
                "returnDurationSeconds": options.hips_translation_policy.return_duration,
                "returnIntervals": return_intervals,
                "returnInterpolation": "sampled cubic Hermite",
                "endpointVelocity": "zero",
                "maximumExtraDisplacementRatio": ROOT_RETURN_MAXIMUM_EXTRA_RATIO,
                "minimumExtraDisplacementMeters": ROOT_RETURN_MINIMUM_EXTRA_METERS,
                "absoluteMaximumDisplacementMeters": (MAX_CURATED_ROOT_DISPLACEMENT_METERS),
            }
        ),
    }
    encoded_duration = read_vrma(output).duration
    output.document["asset"]["extras"] = {
        CURATION_METADATA_KEY: {
            "version": CURATION_METADATA_VERSION,
            "source": {
                "sha256": hashlib.sha256(source_bytes).hexdigest(),
                "byteLength": len(source_bytes),
            },
            "perBonePolicies": policy_metadata,
            "hipsTranslationPolicy": ("relative-return" if closes_with_return_tail else "none"),
            "mainDurationMicroseconds": _duration_microseconds(options.duration),
            "returnDurationMicroseconds": _duration_microseconds(
                options.hips_translation_policy.return_duration
            ),
            "durationMicroseconds": _duration_microseconds(encoded_duration),
            "maximumRootDisplacementMeters": maximum_root_displacement,
            "maximumRootVelocityMetersPerSecond": maximum_root_velocity,
            "transform": transform_metadata,
        }
    }
    output_report = validate_vrma(output)
    output_report.require_valid()
    if any("quaternion sign discontinuity" in warning for warning in output_report.warnings):
        raise VrmaError("curated output has a quaternion sign discontinuity after float32 encoding")
    if closes_with_return_tail:
        _require_serialized_rest_endpoints(output)
    output_byte_length = len(output.to_bytes())
    if output_byte_length > MAX_CURATION_OUTPUT_BYTES:
        raise VrmaError(
            f"curated output has {output_byte_length} bytes; "
            f"limit is {MAX_CURATION_OUTPUT_BYTES} bytes"
        )
    return output


def _float32_at_or_below(value: float) -> float:
    """Return the nearest nonnegative float32 that does not exceed ``value``."""

    encoded = struct.pack("<f", value)
    quantized = struct.unpack("<f", encoded)[0]
    if quantized <= value:
        return quantized
    bits = struct.unpack("<I", encoded)[0]
    return struct.unpack("<f", struct.pack("<I", bits - 1))[0]


def _float32_at_or_above(value: float) -> float:
    """Return the nearest nonnegative float32 that is not below ``value``."""

    if value == 0:
        return 0.0
    try:
        encoded = struct.pack("<f", value)
    except OverflowError as exc:
        raise VrmaError("curated root-motion bound exceeds finite float32") from exc
    quantized = struct.unpack("<f", encoded)[0]
    if quantized >= value:
        return quantized
    bits = struct.unpack("<I", encoded)[0]
    return struct.unpack("<f", struct.pack("<I", bits + 1))[0]


def _duration_microseconds(value: float) -> int:
    return math.floor(value * 1_000_000 + 0.5)


def write_curated_vrma(
    source: VrmaSource,
    output: str | Path,
    *,
    bones: Sequence[str],
    duration: float,
    fade_in: float = 0.0,
    fade_out: float = 0.0,
    rotation_gain: float = 1.0,
    rotation_projection: str = "none",
    start_frame: int = 0,
    end_frame: int | None = None,
    name: str | None = None,
    bone_policies: Sequence[BoneCurationPolicy] = (),
    hips_translation_policy: HipsTranslationPolicy | None = None,
) -> Path:
    """Curate and validate a VRMA before writing it to ``output``."""

    target = Path(output)
    if target.suffix.lower() != ".vrma":
        raise VrmaError("VRM Animation output must use the .vrma extension")
    curated = curate_vrma(
        source,
        bones=bones,
        duration=duration,
        fade_in=fade_in,
        fade_out=fade_out,
        rotation_gain=rotation_gain,
        rotation_projection=rotation_projection,
        start_frame=start_frame,
        end_frame=end_frame,
        name=name,
        bone_policies=bone_policies,
        hips_translation_policy=hips_translation_policy,
    )
    return curated.write(target)


def _load_source(source: VrmaSource) -> tuple[bytes, Glb]:
    if isinstance(source, Glb):
        _require_source_byte_bound(len(source.binary))
        _require_source_structure_bounds(source)
        payload = source.to_bytes()
        _require_source_byte_bound(len(payload))
        return payload, source
    if isinstance(source, bytes):
        _require_source_byte_bound(len(source))
        source_glb = Glb.from_bytes(source)
        _require_source_structure_bounds(source_glb)
        return source, source_glb
    path = Path(source)
    try:
        _require_source_byte_bound(path.stat().st_size)
        with path.open("rb") as source_file:
            payload = source_file.read(MAX_CURATION_SOURCE_BYTES + 1)
    except OSError as exc:
        raise GlbError(f"cannot read {path}: {exc}") from exc
    _require_source_byte_bound(len(payload))
    source_glb = Glb.from_bytes(payload)
    _require_source_structure_bounds(source_glb)
    return payload, source_glb


def _require_source_byte_bound(byte_length: int) -> None:
    if byte_length > MAX_CURATION_SOURCE_BYTES:
        raise VrmaError(
            f"curation source has {byte_length} bytes; limit is {MAX_CURATION_SOURCE_BYTES} bytes"
        )


def _require_source_structure_bounds(source: Glb) -> None:
    document = source.document
    nodes = document.get("nodes")
    if isinstance(nodes, list) and len(nodes) > MAX_CURATION_SOURCE_NODES:
        raise VrmaError(
            f"curation source has {len(nodes)} nodes; limit is {MAX_CURATION_SOURCE_NODES}"
        )

    animations = document.get("animations")
    if not isinstance(animations, list) or not animations:
        return
    animation = animations[0]
    if not isinstance(animation, dict):
        return
    for label in ("channels", "samplers"):
        values = animation.get(label)
        if isinstance(values, list) and len(values) > MAX_CURATION_SOURCE_TRACKS:
            raise VrmaError(
                f"curation source has {len(values)} animation {label}; "
                f"limit is {MAX_CURATION_SOURCE_TRACKS}"
            )

    accessors = document.get("accessors")
    if not isinstance(accessors, list):
        return
    max_accessors = MAX_CURATION_SOURCE_TRACKS * 2
    if len(accessors) > max_accessors:
        raise VrmaError(f"curation source has {len(accessors)} accessors; limit is {max_accessors}")
    component_count = 0
    for accessor_index, accessor in enumerate(accessors):
        if not isinstance(accessor, dict):
            continue
        count = accessor.get("count")
        if (
            isinstance(count, int)
            and not isinstance(count, bool)
            and count > MAX_CURATION_SOURCE_FRAMES
        ):
            raise VrmaError(
                f"curation source accessor {accessor_index} has {count} samples; "
                f"limit is {MAX_CURATION_SOURCE_FRAMES}"
            )
        width = _ACCESSOR_TYPE_WIDTHS.get(accessor.get("type"))
        if (
            width is not None
            and isinstance(count, int)
            and not isinstance(count, bool)
            and count >= 0
        ):
            component_count += count * width
            if component_count > MAX_CURATION_SOURCE_COMPONENTS:
                raise VrmaError(
                    f"curation source accessors contain {component_count} numeric components; "
                    f"limit is {MAX_CURATION_SOURCE_COMPONENTS}"
                )


def _require_linear_animation(vrma: VrmaFile) -> None:
    samplers = vrma.animation["samplers"]
    channels = vrma.animation["channels"]
    for channel_index, channel in enumerate(channels):
        sampler = samplers[channel["sampler"]]
        interpolation = sampler.get("interpolation", "LINEAR")
        if interpolation != "LINEAR":
            raise VrmaError(
                f"animation channel {channel_index} uses {interpolation}; "
                "curation requires sampled LINEAR tracks"
            )


def _selected_rotation_tracks(
    vrma: VrmaFile,
    bones: tuple[str, ...],
) -> tuple[tuple[float, ...], dict[str, tuple[Quat, ...]]]:
    selected = set(bones)
    bone_by_node = {node: bone for bone, node in vrma.humanoid_bones.items()}
    reader = AccessorReader(vrma.glb)
    reference_times: tuple[float, ...] | None = None
    tracks: dict[str, tuple[Quat, ...]] = {}
    samplers = vrma.animation["samplers"]
    for channel in vrma.animation["channels"]:
        target = channel["target"]
        bone = bone_by_node.get(target["node"])
        if bone not in selected or target["path"] != "rotation":
            continue
        sampler = samplers[channel["sampler"]]
        times = tuple(float(value[0]) for value in reader.read(sampler["input"]))
        values = tuple(
            _normalize_quaternion(value, f"{bone} rotation at frame {frame}")
            for frame, value in enumerate(reader.read(sampler["output"]))
        )
        if reference_times is None:
            reference_times = times
        elif times != reference_times:
            raise VrmaError("selected rotation tracks do not share the same sample times")
        tracks[bone] = values

    missing_tracks = selected - tracks.keys()
    if missing_tracks:
        raise VrmaError(
            f"input VRMA has no rotation tracks for: {', '.join(sorted(missing_tracks))}"
        )
    if reference_times is None:
        raise VrmaError("input VRMA has no selected rotation samples")
    return reference_times, {bone: tracks[bone] for bone in bones}


def _hips_translation_track(
    vrma: VrmaFile,
    reference_times: tuple[float, ...],
) -> tuple[Vec3, ...]:
    hips_node = vrma.humanoid_bones.get("hips")
    reader = AccessorReader(vrma.glb)
    samplers = vrma.animation["samplers"]
    track: tuple[Vec3, ...] | None = None
    for channel in vrma.animation["channels"]:
        target = channel["target"]
        if target["node"] != hips_node or target["path"] != "translation":
            continue
        if track is not None:
            raise VrmaError("input VRMA has more than one hips translation track")
        sampler = samplers[channel["sampler"]]
        times = tuple(float(value[0]) for value in reader.read(sampler["input"]))
        if times != reference_times:
            raise VrmaError(
                "hips translation and selected rotation tracks do not share sample times"
            )
        track = tuple(
            _vec3(value, f"hips translation at frame {frame}")
            for frame, value in enumerate(reader.read(sampler["output"]))
        )
    if track is None:
        raise VrmaError("relative-return translation mode requires a hips translation track")
    return track


def _first_relative_hips_track(
    values: tuple[Vec3, ...],
    *,
    rest_translation: Vec3,
) -> tuple[Vec3, ...]:
    first = values[0]
    return tuple(
        tuple(rest_translation[axis] + value[axis] - first[axis] for axis in range(3))
        for value in values
    )


def _root_motion_metrics(output: Glb) -> tuple[float, float]:
    vrma = read_vrma(output)
    reader = AccessorReader(output)
    samplers = vrma.animation["samplers"]
    for channel in vrma.animation["channels"]:
        if channel["target"]["path"] != "translation":
            continue
        sampler = samplers[channel["sampler"]]
        times = tuple(float(value[0]) for value in reader.read(sampler["input"]))
        positions = tuple(
            _vec3(value, f"encoded hips translation at frame {frame}")
            for frame, value in enumerate(reader.read(sampler["output"]))
        )
        rest = _vec3(
            output.document["nodes"][channel["target"]["node"]].get(
                "translation",
                (0.0, 0.0, 0.0),
            ),
            "encoded hips rest translation",
        )
        maximum_displacement = max(_vector_distance(position, rest) for position in positions)
        if maximum_displacement > MAX_CURATED_ROOT_DISPLACEMENT_METERS:
            raise VrmaError(
                "encoded hips translation exceeds the absolute "
                f"{MAX_CURATED_ROOT_DISPLACEMENT_METERS:g} meter curation cap"
            )
        maximum_velocity = max(
            (
                _vector_distance(positions[index], positions[index - 1])
                / (times[index] - times[index - 1])
                for index in range(1, len(times))
            ),
            default=0.0,
        )
        return (
            _float32_at_or_above(maximum_displacement),
            _float32_at_or_above(maximum_velocity),
        )
    return (0.0, 0.0)


def _curate_rotation_track(
    values: tuple[Quat, ...],
    *,
    label: str,
    sample_times: tuple[float, ...],
    fade_in: float,
    fade_out: float,
    rotation_gain: float,
    rotation_projection: str,
    force_identity_end: bool,
) -> tuple[Quat, ...]:
    first_inverse = _quaternion_conjugate(values[0])
    relative: list[Quat] = []
    previous = IDENTITY_QUATERNION
    for value in values:
        current = _normalize_quaternion(
            _quaternion_multiply(first_inverse, value),
            "relative rotation",
        )
        if _quaternion_dot(previous, current) < 0:
            current = _quaternion_negate(current)
        relative.append(current)
        previous = current
    relative[0] = IDENTITY_QUATERNION

    if rotation_projection != "none":
        axis = rotation_projection[-1]
        projected: list[Quat] = []
        previous = IDENTITY_QUATERNION
        for value in relative:
            current = _quaternion_twist(value, axis)
            if _quaternion_dot(previous, current) < 0:
                current = _quaternion_negate(current)
            projected.append(current)
            previous = current
        relative = projected
        relative[0] = IDENTITY_QUATERNION

    amplified = (
        relative
        if rotation_gain == 1.0
        else [_quaternion_slerp(IDENTITY_QUATERNION, value, rotation_gain) for value in relative]
    )
    amplified[0] = IDENTITY_QUATERNION

    last_index = len(amplified) - 1
    duration = sample_times[-1]
    faded: list[Quat] = []
    for frame, value in enumerate(amplified):
        if frame == 0 or (frame == last_index and force_identity_end):
            faded.append(IDENTITY_QUATERNION)
            continue
        time = sample_times[frame]
        fade_in_weight = 1.0 if fade_in == 0 else _smoothstep(min(1.0, time / fade_in))
        fade_out_weight = (
            1.0 if fade_out == 0 else _smoothstep(min(1.0, (duration - time) / fade_out))
        )
        faded.append(
            _quaternion_slerp(
                IDENTITY_QUATERNION,
                value,
                min(fade_in_weight, fade_out_weight),
            )
        )

    continuous: list[Quat] = [IDENTITY_QUATERNION]
    retained_end = -1 if force_identity_end else None
    for value in faded[1:retained_end]:
        if _quaternion_dot(continuous[-1], value) < 0:
            value = _quaternion_negate(value)
        continuous.append(value)
    if force_identity_end:
        if _quaternion_dot(continuous[-1], IDENTITY_QUATERNION) < 0:
            raise VrmaError(
                f"{label} rotation cannot reach the exact identity endpoint "
                "on a shortest quaternion path; increase fade-out or crop the source"
            )
        continuous.append(IDENTITY_QUATERNION)
    return tuple(continuous)


def _append_rest_return(
    *,
    sample_times: tuple[float, ...],
    hips_translations: tuple[Vec3, ...],
    rotations: dict[str, tuple[Quat, ...]],
    return_duration: float,
    nominal_fps: float,
) -> _ReturnedTracks:
    return_intervals = max(
        MIN_RETURN_TAIL_INTERVALS,
        math.ceil(return_duration * nominal_fps),
    )
    needs_start_hold = _needs_start_hold(hips_translations, rotations)
    start_hold_duration = sample_times[1] - sample_times[0] if needs_start_hold else 0.0
    output_frame_count = len(sample_times) + return_intervals + (1 if needs_start_hold else 0)
    if output_frame_count > MAX_CURATION_OUTPUT_FRAMES:
        raise VrmaError(
            f"curated output has {output_frame_count} frames; limit is {MAX_CURATION_OUTPUT_FRAMES}"
        )

    shifted_main_times = tuple(
        _float32_at_or_below(time + start_hold_duration) for time in sample_times
    )
    tail_start = shifted_main_times[-1]
    tail_times = tuple(
        _float32_at_or_below(tail_start + return_duration * interval / return_intervals)
        for interval in range(1, return_intervals + 1)
    )
    times = ((0.0,) if needs_start_hold else ()) + shifted_main_times + tail_times
    if any(later <= earlier for earlier, later in pairwise(times)):
        raise VrmaError(
            "curated return samples collapse after float32 time encoding; "
            "increase the return duration or reduce the selected frame count"
        )

    hips_tail = _translation_return_tail(
        hips_translations,
        sample_times=sample_times,
        return_duration=return_duration,
        return_intervals=return_intervals,
    )
    returned_hips = (
        ((hips_translations[0],) if needs_start_hold else ()) + hips_translations + hips_tail
    )
    returned_rotations = {
        bone: (
            ((IDENTITY_QUATERNION,) if needs_start_hold else ())
            + values
            + _rotation_return_tail(
                values,
                label=bone,
                sample_times=sample_times,
                return_duration=return_duration,
                return_intervals=return_intervals,
            )
        )
        for bone, values in rotations.items()
    }
    return _ReturnedTracks(
        times=times,
        hips_translations=returned_hips,
        rotations=returned_rotations,
        start_hold_duration=start_hold_duration,
        return_intervals=return_intervals,
    )


def _needs_start_hold(
    hips_translations: tuple[Vec3, ...],
    rotations: dict[str, tuple[Quat, ...]],
) -> bool:
    if _vector_distance(hips_translations[0], hips_translations[1]) > 1e-12:
        return True
    return any(
        _quaternion_distance(IDENTITY_QUATERNION, values[1]) > 1e-12
        for values in rotations.values()
    )


def _translation_return_tail(
    values: tuple[Vec3, ...],
    *,
    sample_times: tuple[float, ...],
    return_duration: float,
    return_intervals: int,
) -> tuple[Vec3, ...]:
    rest = values[0]
    last = values[-1]
    source_interval = sample_times[-1] - sample_times[-2]
    incoming_velocity = tuple(
        (last[axis] - values[-2][axis]) / source_interval for axis in range(3)
    )
    return_interval = return_duration / return_intervals
    curve_duration = return_interval * (return_intervals - 2)
    main_bound = max(_vector_distance(value, rest) for value in values)
    if main_bound > MAX_CURATED_ROOT_DISPLACEMENT_METERS:
        raise VrmaError(
            "retained main hips translation exceeds the absolute "
            f"{MAX_CURATED_ROOT_DISPLACEMENT_METERS:g} meter curation cap"
        )
    continuation_bound = min(
        MAX_CURATED_ROOT_DISPLACEMENT_METERS,
        main_bound
        + max(
            main_bound * ROOT_RETURN_MAXIMUM_EXTRA_RATIO,
            ROOT_RETURN_MINIMUM_EXTRA_METERS,
        ),
    )
    bound_tolerance = max(1e-9, continuation_bound * 1e-7)

    first_return = tuple(
        last[axis] + incoming_velocity[axis] * return_interval for axis in range(3)
    )
    tail: list[Vec3] = [first_return]
    for interval in range(2, return_intervals):
        amount = (interval - 1) / (return_intervals - 2)
        position_basis, velocity_basis, rest_basis = _hermite_bases(amount)
        value = tuple(
            (
                position_basis * first_return[axis]
                + velocity_basis * curve_duration * incoming_velocity[axis]
                + rest_basis * rest[axis]
            )
            for axis in range(3)
        )
        if interval == return_intervals - 1:
            value = rest
        if _vector_distance(value, rest) > continuation_bound + bound_tolerance:
            raise VrmaError(
                "hips return tail exceeds the bounded translation continuation; "
                "crop at a slower endpoint or shorten the return duration"
            )
        tail.append(value)
    if _vector_distance(first_return, rest) > continuation_bound + bound_tolerance:
        raise VrmaError(
            "hips return tail exceeds the bounded translation continuation; "
            "crop at a slower endpoint or shorten the return duration"
        )
    tail.append(rest)
    return tuple(tail)


def _rotation_return_tail(
    values: tuple[Quat, ...],
    *,
    label: str,
    sample_times: tuple[float, ...],
    return_duration: float,
    return_intervals: int,
) -> tuple[Quat, ...]:
    last = values[-1]
    if _quaternion_distance(last, IDENTITY_QUATERNION) <= 1e-12:
        return (IDENTITY_QUATERNION,) * return_intervals
    if _quaternion_dot(last, IDENTITY_QUATERNION) < 0:
        raise VrmaError(
            f"{label} rotation cannot reach the exact identity return endpoint "
            "on a shortest quaternion path; crop the source or reduce its gain"
        )

    source_interval = sample_times[-1] - sample_times[-2]
    incoming_velocity = tuple(
        (last[axis] - values[-2][axis]) / source_interval for axis in range(4)
    )
    return_interval = return_duration / return_intervals
    first_return = _normalize_quaternion(
        tuple(last[axis] + incoming_velocity[axis] * return_interval for axis in range(4)),
        f"{label} first return rotation",
    )
    if _quaternion_dot(last, first_return) < 0:
        first_return = _quaternion_negate(first_return)
    hermite_velocity = tuple(
        (first_return[axis] - last[axis]) / return_interval for axis in range(4)
    )
    curve_duration = return_interval * (return_intervals - 2)
    tail: list[Quat] = [first_return]
    previous = first_return
    for interval in range(2, return_intervals):
        amount = (interval - 1) / (return_intervals - 2)
        position_basis, velocity_basis, rest_basis = _hermite_bases(amount)
        raw = tuple(
            (
                position_basis * first_return[axis]
                + velocity_basis * curve_duration * hermite_velocity[axis]
                + rest_basis * IDENTITY_QUATERNION[axis]
            )
            for axis in range(4)
        )
        value = (
            IDENTITY_QUATERNION
            if interval == return_intervals - 1
            else _normalize_quaternion(raw, f"{label} return rotation")
        )
        if _quaternion_dot(previous, value) < 0:
            raise VrmaError(
                f"{label} return tail leaves the shortest quaternion path; "
                "crop at a slower endpoint or shorten the return duration"
            )
        tail.append(value)
        previous = value
    tail.append(IDENTITY_QUATERNION)
    return tuple(tail)


def _hermite_bases(amount: float) -> tuple[float, float, float]:
    squared = amount * amount
    cubed = squared * amount
    return (
        2.0 * cubed - 3.0 * squared + 1.0,
        cubed - 2.0 * squared + amount,
        -2.0 * cubed + 3.0 * squared,
    )


def _vector_distance(left: Sequence[float], right: Sequence[float]) -> float:
    return math.sqrt(sum((a - b) * (a - b) for a, b in zip(left, right, strict=True)))


def _quaternion_distance(left: Quat, right: Quat) -> float:
    dot = abs(_quaternion_dot(left, right))
    return 2.0 * math.acos(min(1.0, max(-1.0, dot)))


def _normalize_quaternion(value: Sequence[int | float], label: str) -> Quat:
    if len(value) != 4:
        raise VrmaError(f"{label} must contain four values")
    result = tuple(float(component) for component in value)
    if any(not math.isfinite(component) for component in result):
        raise VrmaError(f"{label} must contain finite values")
    norm = math.sqrt(sum(component * component for component in result))
    if norm < 1e-12:
        raise VrmaError(f"{label} has zero length")
    return tuple(component / norm for component in result)


def _quaternion_conjugate(value: Quat) -> Quat:
    return (-value[0], -value[1], -value[2], value[3])


def _quaternion_multiply(left: Quat, right: Quat) -> Quat:
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    return (
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    )


def _quaternion_dot(left: Quat, right: Quat) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def _quaternion_negate(value: Quat) -> Quat:
    return tuple(-component for component in value)


def _quaternion_twist(value: Quat, axis: str) -> Quat:
    """Keep only a quaternion's twist around one local coordinate axis."""

    axis_index = {"x": 0, "y": 1, "z": 2}[axis]
    components = [0.0, 0.0, 0.0, value[3]]
    components[axis_index] = value[axis_index]
    norm = math.hypot(components[axis_index], components[3])
    if norm < 1e-12:
        return IDENTITY_QUATERNION
    return tuple(component / norm for component in components)


def _quaternion_slerp(start: Quat, end: Quat, amount: float) -> Quat:
    dot = _quaternion_dot(start, end)
    if dot < 0:
        end = _quaternion_negate(end)
        dot = -dot
    dot = min(1.0, max(-1.0, dot))
    if dot > 0.9995:
        return _normalize_quaternion(
            tuple(a + amount * (b - a) for a, b in zip(start, end, strict=True)),
            "interpolated rotation",
        )
    angle = math.acos(dot)
    scale = math.sin(angle)
    start_weight = math.sin((1.0 - amount) * angle) / scale
    end_weight = math.sin(amount * angle) / scale
    return _normalize_quaternion(
        tuple(start_weight * a + end_weight * b for a, b in zip(start, end, strict=True)),
        "interpolated rotation",
    )


def _smoothstep(value: float) -> float:
    return value * value * (3.0 - 2.0 * value)


def _skeleton_from_vrma(vrma: VrmaFile) -> Skeleton:
    nodes = vrma.glb.document["nodes"]
    parents: list[int | None] = [None] * len(nodes)
    for parent_index, node in enumerate(nodes):
        for child_index in node.get("children", []):
            parents[child_index] = parent_index

    local_matrices = tuple(
        _node_local_matrix(node, node_index) for node_index, node in enumerate(nodes)
    )
    world_matrices: list[Matrix4 | None] = [None] * len(nodes)

    def world_matrix(node_index: int) -> Matrix4:
        cached = world_matrices[node_index]
        if cached is not None:
            return cached
        parent_index = parents[node_index]
        local = local_matrices[node_index]
        result = (
            local if parent_index is None else _matrix_multiply(world_matrix(parent_index), local)
        )
        world_matrices[node_index] = result
        return result

    positions = {
        bone: _matrix_translation(world_matrix(node_index))
        for bone, node_index in vrma.humanoid_bones.items()
    }
    humanoid_order = {name: index for index, name in enumerate(HUMANOID_PARENTS)}
    ordered_names = sorted(
        vrma.humanoid_bones,
        key=lambda name: (_humanoid_depth(name), humanoid_order[name]),
    )
    output_indices: dict[str, int] = {}
    bones: list[Bone] = []
    source_positions: list[Vec3] = []
    mapped_names = set(ordered_names)
    for name in ordered_names:
        parent_name = HUMANOID_PARENTS[name]
        while parent_name is not None and parent_name not in mapped_names:
            parent_name = HUMANOID_PARENTS[parent_name]
        parent_index = None if parent_name is None else output_indices[parent_name]
        position = positions[name]
        translation = (
            position
            if parent_name is None
            else tuple(position[axis] - positions[parent_name][axis] for axis in range(3))
        )
        _finite_vector(translation, f"{name} rest translation")
        output_indices[name] = len(bones)
        bones.append(Bone(name=name, parent=parent_index, translation=translation))
        source_positions.append(position)
    return Skeleton(
        bones=tuple(bones),
        source_positions=tuple(source_positions),
        source_floor_y=min(position[1] for position in source_positions),
    )


def _humanoid_depth(name: str) -> int:
    depth = 0
    parent = HUMANOID_PARENTS[name]
    while parent is not None:
        depth += 1
        parent = HUMANOID_PARENTS[parent]
    return depth


def _node_local_matrix(node: dict[str, Any], node_index: int) -> Matrix4:
    if "matrix" in node:
        if any(key in node for key in ("translation", "rotation", "scale")):
            raise VrmaError(f"node {node_index} mixes matrix and TRS rest transforms")
        raw_matrix = _finite_sequence(node["matrix"], 16, f"node {node_index} matrix")
        return tuple(tuple(raw_matrix[column * 4 + row] for column in range(4)) for row in range(4))

    translation = _finite_sequence(
        node.get("translation", [0.0, 0.0, 0.0]),
        3,
        f"node {node_index} translation",
    )
    scale = _finite_sequence(
        node.get("scale", [1.0, 1.0, 1.0]),
        3,
        f"node {node_index} scale",
    )
    rotation = _normalize_quaternion(
        _finite_sequence(
            node.get("rotation", [0.0, 0.0, 0.0, 1.0]),
            4,
            f"node {node_index} rotation",
        ),
        f"node {node_index} rotation",
    )
    x, y, z, w = rotation
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    xw, yw, zw = x * w, y * w, z * w
    sx, sy, sz = scale
    tx, ty, tz = translation
    return (
        (
            (1.0 - 2.0 * (yy + zz)) * sx,
            2.0 * (xy - zw) * sy,
            2.0 * (xz + yw) * sz,
            tx,
        ),
        (
            2.0 * (xy + zw) * sx,
            (1.0 - 2.0 * (xx + zz)) * sy,
            2.0 * (yz - xw) * sz,
            ty,
        ),
        (
            2.0 * (xz - yw) * sx,
            2.0 * (yz + xw) * sy,
            (1.0 - 2.0 * (xx + yy)) * sz,
            tz,
        ),
        (0.0, 0.0, 0.0, 1.0),
    )


def _matrix_multiply(left: Matrix4, right: Matrix4) -> Matrix4:
    return tuple(
        tuple(
            sum(left[row][axis] * right[axis][column] for axis in range(4)) for column in range(4)
        )
        for row in range(4)
    )


def _matrix_translation(value: Matrix4) -> Vec3:
    result = (value[0][3], value[1][3], value[2][3])
    _finite_vector(result, "world-space rest translation")
    return result


def _finite_sequence(value: Any, width: int, label: str) -> tuple[float, ...]:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != width
        or any(
            not isinstance(component, (int, float))
            or isinstance(component, bool)
            or not math.isfinite(float(component))
            for component in value
        )
    ):
        raise VrmaError(f"{label} must contain {width} finite numbers")
    return tuple(float(component) for component in value)


def _vec3(value: Any, label: str) -> Vec3:
    result = _finite_sequence(value, 3, label)
    return (result[0], result[1], result[2])


def _finite_vector(value: Sequence[float], label: str) -> None:
    if len(value) != 3 or any(not math.isfinite(component) for component in value):
        raise VrmaError(f"{label} must contain three finite numbers")


def _require_serialized_rest_endpoints(output: Glb) -> None:
    vrma = read_vrma(output)
    reader = AccessorReader(output)
    samplers = vrma.animation["samplers"]
    has_hips_translation = False
    for channel in vrma.animation["channels"]:
        target = channel["target"]
        path = target["path"]
        values = reader.read(samplers[channel["sampler"]]["output"])
        if len(values) < 4:
            raise VrmaError("curated return track needs at least four encoded samples")
        endpoint_values = (values[0], values[1], values[-2], values[-1])
        if path == "translation":
            has_hips_translation = True
            rest = _vec3(
                output.document["nodes"][target["node"]].get(
                    "translation",
                    (0.0, 0.0, 0.0),
                ),
                "encoded hips rest translation",
            )
            if any(_vector_distance(value, rest) > 1e-6 for value in endpoint_values):
                raise VrmaError(
                    "curated hips translation did not encode exact rest holds at both endpoints"
                )
        elif path == "rotation" and any(
            _vector_distance(value, IDENTITY_QUATERNION) > 1e-6 for value in endpoint_values
        ):
            raise VrmaError(
                "curated rotation did not encode exact identity holds at both endpoints"
            )
    if not has_hips_translation:
        raise VrmaError("curated relative-return output has no hips translation track")


def _curated_name(requested: str | None, source_name: Any) -> str:
    if requested is not None:
        return requested
    if isinstance(source_name, str):
        try:
            _validate_animation_name(source_name)
        except VrmaError:
            pass
        else:
            return source_name
    return "Curated motion"


def _validate_animation_name(name: str) -> None:
    if not name or len(name.encode("utf-8")) > MAX_ANIMATION_NAME_BYTES:
        raise VrmaError(f"animation name must contain 1 to {MAX_ANIMATION_NAME_BYTES} UTF-8 bytes")

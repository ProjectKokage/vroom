"""Deterministic curation of sampled VRMA clips."""

from __future__ import annotations

import hashlib
import math
import struct
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import GlbError, VrmaError
from .glb import AccessorReader, Glb
from .motion import Bone, MotionClip, Quat, Skeleton, Vec3
from .vrm import HUMANOID_PARENTS
from .vrma import VrmaFile, build_vrma, read_vrma, validate_vrma

CURATION_METADATA_KEY = "vroomCuration"
CURATION_METADATA_VERSION = 2
IDENTITY_QUATERNION: Quat = (0.0, 0.0, 0.0, 1.0)
MAX_ANIMATION_NAME_BYTES = 256
MAX_CURATION_DURATION_SECONDS = 3_600.0
MAX_CURATION_OUTPUT_BYTES = 16 * 1024 * 1024
MAX_CURATION_SOURCE_BYTES = 64 * 1024 * 1024
MAX_CURATION_SOURCE_COMPONENTS = 1_000_000
MAX_CURATION_SOURCE_FRAMES = 18_000
MAX_CURATION_SOURCE_NODES = 512
MAX_CURATION_SOURCE_TRACKS = 256
MAX_ROTATION_GAIN = 4.0
ROTATION_PROJECTIONS = ("none", "twist-x", "twist-y", "twist-z")
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
        if (
            not math.isfinite(self.rotation_gain)
            or self.rotation_gain < 0
            or self.rotation_gain > MAX_ROTATION_GAIN
        ):
            raise VrmaError(
                f"rotation gain must be a finite number from 0 to {MAX_ROTATION_GAIN:g}"
            )
        if (
            not isinstance(self.rotation_projection, str)
            or self.rotation_projection not in ROTATION_PROJECTIONS
        ):
            raise VrmaError("rotation projection must be none, twist-x, twist-y, or twist-z")
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
) -> Glb:
    """Create a rotation-only, first-frame-relative VRMA from sampled LINEAR input."""

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

    rotations = {
        bone: _curate_rotation_track(
            values[options.start_frame : last_frame + 1],
            label=bone,
            sample_times=output_times,
            fade_in=options.fade_in,
            fade_out=options.fade_out,
            rotation_gain=options.rotation_gain,
            rotation_projection=options.rotation_projection,
        )
        for bone, values in source_tracks.items()
    }
    fps = (frame_count - 1) / options.duration
    if not math.isfinite(fps) or fps <= 0:
        raise VrmaError("curated duration is too small for the selected frame range")
    output_name = _curated_name(options.name, source_vrma.animation.get("name"))
    output = build_vrma(
        MotionClip(
            fps=fps,
            hips_translations=(),
            rotations=rotations,
            name=output_name,
            sample_times=output_times,
        ),
        skeleton,
    )
    transform_metadata: dict[str, Any] = {
        "bones": list(ordered_bones),
        "startFrame": options.start_frame,
        "endFrame": last_frame,
        "durationSeconds": options.duration,
        "fadeInSeconds": 0.0 if options.fade_in == 0 else options.fade_in,
        "fadeOutSeconds": 0.0 if options.fade_out == 0 else options.fade_out,
        "rotationGain": (0.0 if options.rotation_gain == 0 else options.rotation_gain),
        "rotationProjection": options.rotation_projection,
        "rotationBasis": "inverse(first) * frame",
        "quaternionPath": "shortest",
        "endpoints": "identity",
    }
    output.document["asset"]["extras"] = {
        CURATION_METADATA_KEY: {
            "version": CURATION_METADATA_VERSION,
            "source": {
                "sha256": hashlib.sha256(source_bytes).hexdigest(),
                "byteLength": len(source_bytes),
            },
            "transform": transform_metadata,
        }
    }
    output_report = validate_vrma(output)
    output_report.require_valid()
    if any("quaternion sign discontinuity" in warning for warning in output_report.warnings):
        raise VrmaError("curated output has a quaternion sign discontinuity after float32 encoding")
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


def _curate_rotation_track(
    values: tuple[Quat, ...],
    *,
    label: str,
    sample_times: tuple[float, ...],
    fade_in: float,
    fade_out: float,
    rotation_gain: float,
    rotation_projection: str,
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
        if frame in {0, last_index}:
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
    for value in faded[1:-1]:
        if _quaternion_dot(continuous[-1], value) < 0:
            value = _quaternion_negate(value)
        continuous.append(value)
    if _quaternion_dot(continuous[-1], IDENTITY_QUATERNION) < 0:
        raise VrmaError(
            f"{label} rotation cannot reach the exact identity endpoint "
            "on a shortest quaternion path; increase fade-out or crop the source"
        )
    continuous.append(IDENTITY_QUATERNION)
    return tuple(continuous)


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


def _finite_vector(value: Sequence[float], label: str) -> None:
    if len(value) != 3 or any(not math.isfinite(component) for component in value):
        raise VrmaError(f"{label} must contain three finite numbers")


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

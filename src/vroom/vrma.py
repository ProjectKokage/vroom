"""VRM Animation 1.0 serializer, reader, and structural validator."""

from __future__ import annotations

import math
import statistics
import struct
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import Any

from .errors import GlbError, VrmaError
from .glb import AccessorReader, Glb
from .motion import MotionClip, Skeleton
from .vrm import (
    HUMANOID_PARENTS,
    REQUIRED_HUMANOID_BONES,
    REQUIRED_OPTIONAL_PARENTS,
)

VRMA_EXTENSION = "VRMC_vrm_animation"


class _BufferBuilder:
    def __init__(self) -> None:
        self.data = bytearray()
        self.views: list[dict[str, Any]] = []
        self.accessors: list[dict[str, Any]] = []

    def add_floats(self, values: tuple[tuple[float, ...], ...], accessor_type: str) -> int:
        widths = {"SCALAR": 1, "VEC3": 3, "VEC4": 4}
        width = widths[accessor_type]
        if not values or any(len(value) != width for value in values):
            raise VrmaError(f"{accessor_type} accessor values have an invalid shape")
        while len(self.data) % 4:
            self.data.append(0)
        offset = len(self.data)
        stored_values: list[tuple[float, ...]] = []
        for value in values:
            try:
                packed = struct.pack(f"<{width}f", *(float(component) for component in value))
            except (OverflowError, struct.error) as exc:
                raise VrmaError(
                    f"{accessor_type} accessor contains a value outside the float32 range"
                ) from exc
            self.data.extend(packed)
            stored_values.append(struct.unpack(f"<{width}f", packed))
        byte_length = len(self.data) - offset
        view_index = len(self.views)
        self.views.append({"buffer": 0, "byteOffset": offset, "byteLength": byte_length})

        columns = tuple(zip(*stored_values, strict=True))
        accessor: dict[str, Any] = {
            "bufferView": view_index,
            "componentType": 5126,
            "count": len(values),
            "type": accessor_type,
            "min": [min(column) for column in columns],
            "max": [max(column) for column in columns],
        }
        accessor_index = len(self.accessors)
        self.accessors.append(accessor)
        return accessor_index


def build_vrma(clip: MotionClip, skeleton: Skeleton) -> Glb:
    """Build a standalone VRMA 1.0 GLB from a sampled motion clip."""

    bone_names = skeleton.bone_names
    if bone_names[0] != "hips" or skeleton.bones[0].parent is not None:
        raise VrmaError("the skeleton must start with a root hips bone")
    missing = REQUIRED_HUMANOID_BONES - set(bone_names)
    if missing:
        raise VrmaError(f"skeleton is missing required VRM bones: {', '.join(sorted(missing))}")
    unknown_tracks = set(clip.rotations) - set(bone_names)
    if unknown_tracks:
        raise VrmaError(f"motion contains unknown bones: {', '.join(sorted(unknown_tracks))}")

    active_indices = {
        index
        for index, bone in enumerate(skeleton.bones)
        if bone.name in REQUIRED_HUMANOID_BONES or bone.name in clip.rotations
    }
    # Retain every ancestor needed to keep the exported hierarchy connected.
    pending = list(active_indices)
    while pending:
        parent = skeleton.bones[pending.pop()].parent
        if parent is not None and parent not in active_indices:
            active_indices.add(parent)
            pending.append(parent)
    ordered_indices = sorted(active_indices)
    node_indices = {bone_index: node_index for node_index, bone_index in enumerate(ordered_indices)}

    nodes: list[dict[str, Any]] = []
    for bone_index in ordered_indices:
        bone = skeleton.bones[bone_index]
        children = [
            node_indices[child_index]
            for child_index in ordered_indices
            if skeleton.bones[child_index].parent == bone_index
        ]
        node: dict[str, Any] = {
            "name": f"VRMA_{bone.name}",
            "translation": list(bone.translation),
        }
        if children:
            node["children"] = children
        nodes.append(node)

    builder = _BufferBuilder()
    times = (
        tuple((value,) for value in clip.sample_times)
        if clip.sample_times is not None
        else tuple((frame / clip.fps,) for frame in range(clip.frame_count))
    )
    time_accessor = builder.add_floats(times, "SCALAR")
    hips_node = node_indices[0]
    samplers: list[dict[str, Any]] = []
    channels: list[dict[str, Any]] = []
    if clip.hips_translations:
        translation_accessor = builder.add_floats(clip.hips_translations, "VEC3")
        samplers.append(
            {"input": time_accessor, "output": translation_accessor, "interpolation": "LINEAR"}
        )
        channels.append({"sampler": 0, "target": {"node": hips_node, "path": "translation"}})
    for bone_index in ordered_indices:
        bone = skeleton.bones[bone_index]
        track = clip.rotations.get(bone.name)
        if track is None:
            continue
        output_accessor = builder.add_floats(track, "VEC4")
        sampler_index = len(samplers)
        samplers.append(
            {"input": time_accessor, "output": output_accessor, "interpolation": "LINEAR"}
        )
        channels.append(
            {
                "sampler": sampler_index,
                "target": {"node": node_indices[bone_index], "path": "rotation"},
            }
        )

    human_bones = {
        skeleton.bones[bone_index].name: {"node": node_indices[bone_index]}
        for bone_index in ordered_indices
    }
    document: dict[str, Any] = {
        "asset": {"version": "2.0", "generator": "vroom 0.1.0"},
        "scene": 0,
        "scenes": [{"nodes": [hips_node]}],
        "nodes": nodes,
        "animations": [{"name": clip.name, "samplers": samplers, "channels": channels}],
        "extensionsUsed": [VRMA_EXTENSION],
        "extensions": {
            VRMA_EXTENSION: {
                "specVersion": "1.0",
                "humanoid": {"humanBones": human_bones},
            }
        },
        "buffers": [{"byteLength": len(builder.data)}],
        "bufferViews": builder.views,
        "accessors": builder.accessors,
    }
    return Glb(document=document, binary=bytes(builder.data))


def write_vrma(path: str | Path, clip: MotionClip, skeleton: Skeleton) -> Path:
    target = Path(path)
    if target.suffix.lower() != ".vrma":
        raise VrmaError("VRM Animation output must use the .vrma extension")
    return build_vrma(clip, skeleton).write(target)


def _coerce_glb(source: str | Path | bytes | Glb) -> Glb:
    if isinstance(source, Glb):
        return source
    if isinstance(source, bytes):
        return Glb.from_bytes(source)
    return Glb.read(source)


@dataclass(frozen=True, slots=True)
class VrmaFile:
    glb: Glb
    humanoid_bones: dict[str, int]

    @property
    def animation(self) -> dict[str, Any]:
        animations = self.glb.document.get("animations")
        if not isinstance(animations, list) or not animations:
            raise VrmaError("VRMA has no glTF animations")
        animation = animations[0]
        if not isinstance(animation, dict):
            raise VrmaError("VRMA first animation must be an object")
        return animation

    @property
    def times(self) -> tuple[float, ...]:
        samplers = self.animation.get("samplers")
        if not isinstance(samplers, list) or not samplers:
            raise VrmaError("VRMA first animation has no samplers")
        sampler = samplers[0]
        if not isinstance(sampler, dict):
            raise VrmaError("VRMA first animation sampler must be an object")
        values = AccessorReader(self.glb).read(sampler.get("input"))
        return tuple(float(value[0]) for value in values)

    @property
    def frame_count(self) -> int:
        return len(self.times)

    @property
    def duration(self) -> float:
        times = self.times
        return 0.0 if not times else times[-1] - times[0]

    @property
    def nominal_fps(self) -> float | None:
        times = self.times
        intervals = [later - earlier for earlier, later in pairwise(times) if later > earlier]
        return None if not intervals else 1.0 / statistics.median(intervals)

    def summary(self) -> dict[str, Any]:
        return {
            "format": "VRMA",
            "specVersion": "1.0",
            "nodes": len(self.glb.document.get("nodes", [])),
            "humanoidBones": len(self.humanoid_bones),
            "animations": len(self.glb.document.get("animations", [])),
            "frames": self.frame_count,
            "durationSeconds": self.duration,
            "nominalFps": self.nominal_fps,
        }


def read_vrma(source: str | Path | bytes | Glb) -> VrmaFile:
    glb = _coerce_glb(source)
    extensions = glb.document.get("extensions", {})
    extension = extensions.get(VRMA_EXTENSION) if isinstance(extensions, dict) else None
    if not isinstance(extension, dict):
        raise VrmaError(f"GLB has no {VRMA_EXTENSION} extension")
    if extension.get("specVersion") != "1.0":
        raise VrmaError(
            f"unsupported {VRMA_EXTENSION} specVersion {extension.get('specVersion')!r}"
        )
    humanoid = extension.get("humanoid", {})
    if not isinstance(humanoid, dict):
        raise VrmaError("VRMA humanoid must be an object")
    raw_bones = humanoid.get("humanBones", {})
    if not isinstance(raw_bones, dict):
        raise VrmaError("VRMA humanoid humanBones must be an object")
    mapping = {
        name: value.get("node")
        for name, value in raw_bones.items()
        if (
            isinstance(name, str)
            and isinstance(value, dict)
            and isinstance(value.get("node"), int)
            and not isinstance(value.get("node"), bool)
        )
    }
    animations = glb.document.get("animations")
    if not isinstance(animations, list) or not animations:
        raise VrmaError("VRMA has no glTF animations")
    if not isinstance(animations[0], dict):
        raise VrmaError("VRMA first animation must be an object")
    animation = animations[0]
    samplers = animation.get("samplers")
    if not isinstance(samplers, list) or not samplers:
        raise VrmaError("VRMA first animation has no samplers")
    if not isinstance(samplers[0], dict):
        raise VrmaError("VRMA first animation sampler must be an object")
    channels = animation.get("channels")
    if not isinstance(channels, list) or not channels:
        raise VrmaError("VRMA first animation has no channels")
    return VrmaFile(glb=glb, humanoid_bones=mapping)


@dataclass(slots=True)
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def require_valid(self) -> None:
        if self.errors:
            raise VrmaError("; ".join(self.errors))


def _validate_accessor_metadata(
    report: ValidationReport,
    document: dict[str, Any],
    accessor_index: Any,
    expected_type: str,
    label: str,
    *,
    allow_normalized_integer: bool = False,
) -> dict[str, Any] | None:
    accessors = document.get("accessors", [])
    if (
        not isinstance(accessor_index, int)
        or isinstance(accessor_index, bool)
        or accessor_index < 0
        or not isinstance(accessors, list)
    ):
        report.errors.append(f"{label} has an invalid accessor")
        return None
    try:
        accessor = accessors[accessor_index]
    except IndexError:
        report.errors.append(f"{label} points to missing accessor {accessor_index}")
        return None
    if not isinstance(accessor, dict):
        report.errors.append(f"{label} accessor is not an object")
        return None
    component_type = accessor.get("componentType")
    component_valid = component_type == 5126 or (
        allow_normalized_integer
        and component_type in {5120, 5121, 5122, 5123}
        and accessor.get("normalized") is True
    )
    if not component_valid or accessor.get("type") != expected_type:
        allowed = "FLOAT or normalized BYTE/SHORT" if allow_normalized_integer else "FLOAT"
        report.errors.append(f"{label} must use {allowed} {expected_type}")
    return accessor


def _validate_time_bounds(
    report: ValidationReport,
    accessor: dict[str, Any],
    times: list[float],
    label: str,
) -> None:
    bounds: dict[str, float] = {}
    for bound_name in ("min", "max"):
        value = accessor.get(bound_name)
        if (
            not isinstance(value, list)
            or len(value) != 1
            or not isinstance(value[0], (int, float))
            or isinstance(value[0], bool)
            or not math.isfinite(float(value[0]))
        ):
            report.errors.append(f"{label} accessor must define a finite one-value {bound_name}")
            continue
        bounds[bound_name] = float(value[0])

    if len(bounds) != 2 or not times or any(not math.isfinite(value) for value in times):
        return
    actual_min = min(times)
    actual_max = max(times)
    if bounds["min"] != actual_min:
        report.errors.append(
            f"{label} accessor min {bounds['min']!r} does not match {actual_min!r}"
        )
    if bounds["max"] != actual_max:
        report.errors.append(
            f"{label} accessor max {bounds['max']!r} does not match {actual_max!r}"
        )


def _node_parents(
    report: ValidationReport,
    nodes: list[Any],
) -> list[set[int]]:
    parents: list[set[int]] = [set() for _ in nodes]
    children_by_node: list[set[int]] = [set() for _ in nodes]
    for parent_index, node in enumerate(nodes):
        if not isinstance(node, dict):
            report.errors.append(f"node {parent_index} is not an object")
            continue
        children = node.get("children", [])
        if not isinstance(children, list):
            report.errors.append(f"node {parent_index} children must be an array")
            continue
        seen_children: set[int] = set()
        for child_index in children:
            if (
                not isinstance(child_index, int)
                or isinstance(child_index, bool)
                or not 0 <= child_index < len(nodes)
            ):
                report.errors.append(f"node {parent_index} has invalid child {child_index!r}")
                continue
            if child_index in seen_children:
                report.errors.append(
                    f"node {parent_index} lists child {child_index} more than once"
                )
                continue
            seen_children.add(child_index)
            children_by_node[parent_index].add(child_index)
            parents[child_index].add(parent_index)
    for node_index, values in enumerate(parents):
        if len(values) > 1:
            report.errors.append(f"node {node_index} has multiple parents")

    in_degree = [len(values) for values in parents]
    pending = [index for index, degree in enumerate(in_degree) if degree == 0]
    visited = 0
    while pending:
        parent_index = pending.pop()
        visited += 1
        for child_index in children_by_node[parent_index]:
            in_degree[child_index] -= 1
            if in_degree[child_index] == 0:
                pending.append(child_index)
    if visited != len(nodes):
        report.errors.append("glTF node hierarchy contains a cycle")
    return parents


def _has_ancestor(node_index: int, ancestor_index: int, parents: list[set[int]]) -> bool:
    pending = list(parents[node_index])
    visited: set[int] = set()
    while pending:
        current = pending.pop()
        if current == ancestor_index:
            return True
        if current in visited:
            continue
        visited.add(current)
        pending.extend(parents[current])
    return False


def validate_vrma(source: str | Path | bytes | Glb) -> ValidationReport:
    """Validate the VRMA rules relevant to portable humanoid animation."""

    report = ValidationReport()
    try:
        vrma = read_vrma(source)
    except (GlbError, VrmaError) as exc:
        report.errors.append(str(exc))
        return report

    glb = vrma.glb
    document = glb.document
    nodes = document.get("nodes", [])
    asset = document.get("asset")
    if not isinstance(asset, dict) or asset.get("version") != "2.0":
        report.errors.append("glTF asset.version must be 2.0")
    extensions_used = document.get("extensionsUsed")
    if not isinstance(extensions_used, list) or VRMA_EXTENSION not in extensions_used:
        report.errors.append(f"extensionsUsed does not include {VRMA_EXTENSION}")
    if not isinstance(nodes, list):
        report.errors.append("glTF nodes must be an array")
        return report

    extension = document["extensions"][VRMA_EXTENSION]
    raw_bones = extension["humanoid"]["humanBones"]
    for bone_name, value in raw_bones.items():
        if bone_name not in HUMANOID_PARENTS:
            report.errors.append(f"unknown humanoid bone {bone_name!r}")
        if (
            not isinstance(value, dict)
            or not isinstance(value.get("node"), int)
            or isinstance(value.get("node"), bool)
        ):
            report.errors.append(f"humanoid bone {bone_name!r} has an invalid node")

    parents = _node_parents(report, nodes)
    mapped_nodes: dict[int, str] = {}
    for bone_name, node_index in vrma.humanoid_bones.items():
        if bone_name in {"leftEye", "rightEye"}:
            report.errors.append(f"{bone_name} must be animated through VRMA LookAt, not humanoid")
        if not 0 <= node_index < len(nodes):
            report.errors.append(f"{bone_name} points to missing node {node_index}")
        elif node_index in mapped_nodes:
            report.errors.append(
                f"{bone_name} and {mapped_nodes[node_index]} both point to node {node_index}"
            )
        else:
            mapped_nodes[node_index] = bone_name
            node = nodes[node_index]
            if not isinstance(node, dict):
                continue
            scale = node.get("scale", [1.0, 1.0, 1.0])
            valid_scale = (
                isinstance(scale, list)
                and len(scale) == 3
                and all(
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(float(value))
                    and value > 0
                    for value in scale
                )
            )
            uniform_scale = valid_scale and all(
                math.isclose(
                    float(value),
                    float(scale[0]),
                    rel_tol=1e-6,
                    abs_tol=1e-8,
                )
                for value in scale[1:]
            )
            if not uniform_scale:
                report.errors.append(
                    f"{bone_name} rest scale must have three equal positive finite values"
                )

    missing = REQUIRED_HUMANOID_BONES - vrma.humanoid_bones.keys()
    if missing:
        report.errors.append(f"missing required humanoid bones: {', '.join(sorted(missing))}")

    for bone_name, node_index in vrma.humanoid_bones.items():
        parent_name = HUMANOID_PARENTS.get(bone_name)
        if (
            bone_name in REQUIRED_OPTIONAL_PARENTS
            and parent_name is not None
            and parent_name not in vrma.humanoid_bones
        ):
            report.errors.append(f"{bone_name} requires mapped parent {parent_name}")
        while parent_name is not None and parent_name not in vrma.humanoid_bones:
            parent_name = HUMANOID_PARENTS[parent_name]
        if parent_name is None:
            continue
        parent_index = vrma.humanoid_bones[parent_name]
        if (
            0 <= node_index < len(nodes)
            and 0 <= parent_index < len(nodes)
            and not _has_ancestor(node_index, parent_index, parents)
        ):
            report.errors.append(f"{bone_name} node is not a descendant of mapped {parent_name}")

    animation = vrma.animation
    samplers = animation.get("samplers", [])
    channels = animation.get("channels", [])
    if not isinstance(samplers, list) or not isinstance(channels, list):
        report.errors.append("animation samplers and channels must be arrays")
        return report

    reader = AccessorReader(glb)
    seen_rotations: set[str] = set()
    has_hips_translation = False
    seen_targets: set[tuple[int, str]] = set()
    time_cache: dict[int, tuple[tuple[int | float, ...], ...]] = {}
    for channel_index, channel in enumerate(channels):
        label = f"animation channel {channel_index}"
        if not isinstance(channel, dict) or not isinstance(channel.get("target"), dict):
            report.errors.append(f"{label} is malformed")
            continue
        sampler_index = channel.get("sampler")
        if (
            not isinstance(sampler_index, int)
            or isinstance(sampler_index, bool)
            or not 0 <= sampler_index < len(samplers)
        ):
            report.errors.append(f"{label} has invalid sampler {sampler_index!r}")
            continue
        sampler = samplers[sampler_index]
        if not isinstance(sampler, dict):
            report.errors.append(f"{label} sampler is malformed")
            continue

        interpolation = sampler.get("interpolation", "LINEAR")
        if interpolation not in {"LINEAR", "STEP", "CUBICSPLINE"}:
            report.errors.append(f"{label} uses unsupported interpolation {interpolation!r}")
        target = channel["target"]
        node_index = target.get("node")
        path = target.get("path")
        if (
            not isinstance(node_index, int)
            or isinstance(node_index, bool)
            or not 0 <= node_index < len(nodes)
        ):
            report.errors.append(f"{label} targets an invalid node")
            continue
        if path not in {"translation", "rotation", "scale", "weights"}:
            report.errors.append(f"{label} uses invalid target path {path!r}")
            continue
        target_key = (node_index, path)
        if target_key in seen_targets:
            report.errors.append(f"{label} duplicates node {node_index} {path}")
        seen_targets.add(target_key)

        bone_name = mapped_nodes.get(node_index)
        if bone_name is not None:
            if path == "scale":
                report.errors.append(f"{bone_name} must not contain scale animation")
            elif path == "translation" and bone_name != "hips":
                report.errors.append(f"{bone_name} must not contain translation animation")
            elif path == "translation":
                has_hips_translation = True
            elif path == "rotation":
                seen_rotations.add(bone_name)

        input_index = sampler.get("input")
        input_accessor = _validate_accessor_metadata(
            report, document, input_index, "SCALAR", f"{label} input"
        )
        expected_output_type = {"translation": "VEC3", "rotation": "VEC4"}.get(path)
        output_accessor = None
        if expected_output_type is not None:
            output_accessor = _validate_accessor_metadata(
                report,
                document,
                sampler.get("output"),
                expected_output_type,
                f"{label} output",
                allow_normalized_integer=path == "rotation",
            )
        if input_accessor is None or output_accessor is None:
            continue

        try:
            if input_index not in time_cache:
                time_cache[input_index] = reader.read(input_index)
            time_values = time_cache[input_index]
            output_values = reader.read(sampler["output"])
        except GlbError as exc:
            report.errors.append(f"{label}: {exc}")
            continue

        times = [float(value[0]) for value in time_values]
        _validate_time_bounds(report, input_accessor, times, f"{label} input")
        if not times:
            report.errors.append(f"{label} has no keyframes")
        if any(value < 0 for value in times):
            report.errors.append(f"{label} has negative keyframe times")
        if any(not math.isfinite(value) for value in times):
            report.errors.append(f"{label} has non-finite keyframe times")
        if any(later <= earlier for earlier, later in pairwise(times)):
            report.errors.append(f"{label} keyframe times are not strictly increasing")
        multiplier = 3 if interpolation == "CUBICSPLINE" else 1
        if interpolation == "CUBICSPLINE" and len(time_values) < 2:
            report.errors.append(f"{label} CUBICSPLINE requires at least two keyframes")
        if len(output_values) != len(time_values) * multiplier:
            report.errors.append(f"{label} input and output keyframe counts do not match")
        if any(
            not math.isfinite(float(component)) for value in output_values for component in value
        ):
            report.errors.append(f"{label} has non-finite output values")

        if path == "rotation":
            quaternion_values = (
                output_values[1::3] if interpolation == "CUBICSPLINE" else output_values
            )
            for keyframe, quaternion in enumerate(quaternion_values):
                norm = math.sqrt(sum(float(value) ** 2 for value in quaternion))
                if not math.isfinite(norm) or abs(norm - 1.0) > 2e-3:
                    report.errors.append(
                        f"{label} quaternion {keyframe} is not normalized ({norm:.6g})"
                    )
                    break
            for previous, current in pairwise(quaternion_values):
                dot = sum(float(a) * float(b) for a, b in zip(previous, current, strict=True))
                if dot < 0:
                    report.warnings.append(f"{label} has a quaternion sign discontinuity")
                    break

    if not has_hips_translation:
        report.warnings.append("hips has no translation track")
    for bone_name in sorted(REQUIRED_HUMANOID_BONES - seen_rotations):
        report.warnings.append(f"{bone_name} has no rotation track")
    return report

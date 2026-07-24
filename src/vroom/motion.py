"""Backend-neutral humanoid motion data and HY-Motion conversion."""

from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import HyMotionError, VrmaError

type Vec3 = tuple[float, float, float]
type Quat = tuple[float, float, float, float]  # glTF order: x, y, z, w
type Matrix3 = tuple[Vec3, Vec3, Vec3]  # rows

HY_JOINT_NAMES = (
    "Pelvis",
    "L_Hip",
    "R_Hip",
    "Spine1",
    "L_Knee",
    "R_Knee",
    "Spine2",
    "L_Ankle",
    "R_Ankle",
    "Spine3",
    "L_Foot",
    "R_Foot",
    "Neck",
    "L_Collar",
    "R_Collar",
    "Head",
    "L_Shoulder",
    "R_Shoulder",
    "L_Elbow",
    "R_Elbow",
    "L_Wrist",
    "R_Wrist",
    "L_Index1",
    "L_Index2",
    "L_Index3",
    "L_Middle1",
    "L_Middle2",
    "L_Middle3",
    "L_Pinky1",
    "L_Pinky2",
    "L_Pinky3",
    "L_Ring1",
    "L_Ring2",
    "L_Ring3",
    "L_Thumb1",
    "L_Thumb2",
    "L_Thumb3",
    "R_Index1",
    "R_Index2",
    "R_Index3",
    "R_Middle1",
    "R_Middle2",
    "R_Middle3",
    "R_Pinky1",
    "R_Pinky2",
    "R_Pinky3",
    "R_Ring1",
    "R_Ring2",
    "R_Ring3",
    "R_Thumb1",
    "R_Thumb2",
    "R_Thumb3",
)

HY_TO_VRM_BONES = (
    "hips",
    "leftUpperLeg",
    "rightUpperLeg",
    "spine",
    "leftLowerLeg",
    "rightLowerLeg",
    "chest",
    "leftFoot",
    "rightFoot",
    "upperChest",
    "leftToes",
    "rightToes",
    "neck",
    "leftShoulder",
    "rightShoulder",
    "head",
    "leftUpperArm",
    "rightUpperArm",
    "leftLowerArm",
    "rightLowerArm",
    "leftHand",
    "rightHand",
    "leftIndexProximal",
    "leftIndexIntermediate",
    "leftIndexDistal",
    "leftMiddleProximal",
    "leftMiddleIntermediate",
    "leftMiddleDistal",
    "leftLittleProximal",
    "leftLittleIntermediate",
    "leftLittleDistal",
    "leftRingProximal",
    "leftRingIntermediate",
    "leftRingDistal",
    "leftThumbMetacarpal",
    "leftThumbProximal",
    "leftThumbDistal",
    "rightIndexProximal",
    "rightIndexIntermediate",
    "rightIndexDistal",
    "rightMiddleProximal",
    "rightMiddleIntermediate",
    "rightMiddleDistal",
    "rightLittleProximal",
    "rightLittleIntermediate",
    "rightLittleDistal",
    "rightRingProximal",
    "rightRingIntermediate",
    "rightRingDistal",
    "rightThumbMetacarpal",
    "rightThumbProximal",
    "rightThumbDistal",
)


@dataclass(frozen=True, slots=True)
class Bone:
    """One VRM humanoid bone in source-skeleton order."""

    name: str
    parent: int | None
    translation: Vec3


@dataclass(frozen=True, slots=True)
class Skeleton:
    """A VRM T-pose hierarchy and the corresponding HY source positions."""

    bones: tuple[Bone, ...]
    source_positions: tuple[Vec3, ...]
    source_floor_y: float

    def __post_init__(self) -> None:
        if not self.bones or len(self.bones) != len(self.source_positions):
            raise VrmaError(
                "skeleton bones and source positions must have the same non-zero length"
            )
        if not math.isfinite(self.source_floor_y):
            raise VrmaError("skeleton floor height must be finite")
        names: set[str] = set()
        for index, bone in enumerate(self.bones):
            if bone.name in names:
                raise VrmaError(f"duplicate humanoid bone {bone.name!r}")
            names.add(bone.name)
            if bone.parent is not None and not 0 <= bone.parent < index:
                raise VrmaError(f"bone {bone.name!r} has invalid parent {bone.parent}")

    @property
    def bone_names(self) -> tuple[str, ...]:
        return tuple(bone.name for bone in self.bones)

    def index(self, bone_name: str) -> int:
        try:
            return self.bone_names.index(bone_name)
        except ValueError as exc:
            raise VrmaError(f"skeleton has no {bone_name!r} bone") from exc


@dataclass(frozen=True, slots=True)
class MotionClip:
    """A sampled humanoid animation ready to serialize as VRMA."""

    fps: float
    hips_translations: tuple[Vec3, ...]
    rotations: dict[str, tuple[Quat, ...]]
    name: str = "HY-Motion"

    def __post_init__(self) -> None:
        if not math.isfinite(self.fps) or self.fps <= 0:
            raise VrmaError("motion fps must be a positive finite number")
        frame_count = len(self.hips_translations)
        if frame_count == 0:
            raise VrmaError("motion clip has no frames")
        for frame, value in enumerate(self.hips_translations):
            _finite_vector(value, 3, f"hips translation at frame {frame}")
        for bone_name, values in self.rotations.items():
            if len(values) != frame_count:
                raise VrmaError(
                    f"{bone_name} has {len(values)} rotations for a {frame_count}-frame clip"
                )
            for frame, value in enumerate(values):
                _finite_vector(value, 4, f"{bone_name} rotation at frame {frame}")
                norm = math.sqrt(sum(component * component for component in value))
                if abs(norm - 1.0) > 1e-3:
                    raise VrmaError(
                        f"{bone_name} rotation at frame {frame} is not normalized ({norm:.6f})"
                    )

    @property
    def frame_count(self) -> int:
        return len(self.hips_translations)

    @property
    def duration(self) -> float:
        return 0.0 if self.frame_count == 1 else (self.frame_count - 1) / self.fps


def _finite_vector(value: tuple[float, ...], width: int, label: str) -> None:
    if len(value) != width or any(not math.isfinite(component) for component in value):
        raise VrmaError(f"{label} must contain {width} finite numbers")


def load_hy_motion_skeleton(hy_motion_root: str | Path) -> Skeleton:
    """Load the 52-joint wooden T-pose bundled with HY-Motion."""

    assets = Path(hy_motion_root) / "scripts" / "gradio" / "static" / "assets" / "dump_wooden"
    names_path = assets / "joint_names.json"
    positions_path = assets / "j_template.bin"
    parents_path = assets / "kintree.bin"
    vertices_path = assets / "v_template.bin"
    for path in (names_path, positions_path, parents_path, vertices_path):
        if not path.is_file():
            raise HyMotionError(
                f"missing HY-Motion skeleton asset {path}; run scripts/bootstrap_hy_motion.sh"
            )

    try:
        names = json.loads(names_path.read_text(encoding="utf-8"))
        position_bytes = positions_path.read_bytes()
        parent_bytes = parents_path.read_bytes()
        vertex_bytes = vertices_path.read_bytes()
    except (OSError, json.JSONDecodeError) as exc:
        raise HyMotionError(f"cannot load HY-Motion skeleton: {exc}") from exc

    if tuple(names) != HY_JOINT_NAMES:
        raise HyMotionError("the vendored HY-Motion joint order is not the supported SMPL-H order")
    if len(position_bytes) != len(names) * 3 * 4 or len(parent_bytes) != len(names) * 4:
        raise HyMotionError("HY-Motion skeleton assets have unexpected dimensions")

    flat_positions = struct.unpack(f"<{len(names) * 3}f", position_bytes)
    if len(vertex_bytes) % 12:
        raise HyMotionError("HY-Motion vertex template has unexpected dimensions")
    flat_vertices = struct.unpack(f"<{len(vertex_bytes) // 4}f", vertex_bytes)
    source_floor_y = min(flat_vertices[1::3])
    source_positions: tuple[Vec3, ...] = tuple(
        (
            float(flat_positions[index * 3]),
            float(flat_positions[index * 3 + 1]),
            float(flat_positions[index * 3 + 2]),
        )
        for index in range(len(names))
    )
    parent_values = struct.unpack(f"<{len(names)}i", parent_bytes)
    bones: list[Bone] = []
    for index, (name, parent_value) in enumerate(zip(HY_TO_VRM_BONES, parent_values, strict=True)):
        parent = None if parent_value < 0 else int(parent_value)
        position = source_positions[index]
        if parent is None:
            # The source template is centered above its feet. Ground the static T-pose.
            translation = (position[0], position[1] - source_floor_y, position[2])
        else:
            parent_position = source_positions[parent]
            translation = tuple(position[axis] - parent_position[axis] for axis in range(3))
        bones.append(Bone(name=name, parent=parent, translation=translation))
    return Skeleton(
        bones=tuple(bones),
        source_positions=source_positions,
        source_floor_y=source_floor_y,
    )


def _dot(left: Vec3, right: Vec3) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def _cross(left: Vec3, right: Vec3) -> Vec3:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _normalize(value: Vec3) -> Vec3:
    norm = math.sqrt(_dot(value, value))
    if norm < 1e-12:
        raise VrmaError("HY-Motion emitted a degenerate 6-D rotation")
    return tuple(component / norm for component in value)


def rot6d_to_matrix(value: tuple[float, ...] | list[float]) -> Matrix3:
    """Match HY-Motion's flattened-first-two-columns 6-D conversion."""

    if len(value) != 6 or any(not math.isfinite(float(component)) for component in value):
        raise VrmaError("a 6-D rotation must contain six finite values")
    first = _normalize((float(value[0]), float(value[2]), float(value[4])))
    second_raw = (float(value[1]), float(value[3]), float(value[5]))
    projection = _dot(first, second_raw)
    second = _normalize(tuple(second_raw[axis] - projection * first[axis] for axis in range(3)))
    third = _cross(first, second)
    return (
        (first[0], second[0], third[0]),
        (first[1], second[1], third[1]),
        (first[2], second[2], third[2]),
    )


def matrix_to_quaternion(matrix: Matrix3) -> Quat:
    """Convert a rotation matrix to a normalized glTF XYZW quaternion."""

    m00, m01, m02 = matrix[0]
    m10, m11, m12 = matrix[1]
    m20, m21, m22 = matrix[2]
    trace = m00 + m11 + m22
    if trace > 0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion = ((m21 - m12) / scale, (m02 - m20) / scale, (m10 - m01) / scale, 0.25 * scale)
    elif m00 > m11 and m00 > m22:
        scale = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        quaternion = (0.25 * scale, (m01 + m10) / scale, (m02 + m20) / scale, (m21 - m12) / scale)
    elif m11 > m22:
        scale = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        quaternion = ((m01 + m10) / scale, 0.25 * scale, (m12 + m21) / scale, (m02 - m20) / scale)
    else:
        scale = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        quaternion = ((m02 + m20) / scale, (m12 + m21) / scale, 0.25 * scale, (m10 - m01) / scale)

    norm = math.sqrt(sum(component * component for component in quaternion))
    result = tuple(component / norm for component in quaternion)
    if result[3] < 0:
        result = tuple(-component for component in result)
    return result


def rot6d_to_quaternion(value: tuple[float, ...] | list[float]) -> Quat:
    return matrix_to_quaternion(rot6d_to_matrix(value))


def _matrix_multiply(left: Matrix3, right: Matrix3) -> Matrix3:
    return tuple(
        tuple(sum(left[row][k] * right[k][column] for k in range(3)) for column in range(3))
        for row in range(3)
    )


def _matrix_vector(matrix: Matrix3, vector: Vec3) -> Vec3:
    return tuple(
        sum(matrix[row][column] * vector[column] for column in range(3)) for row in range(3)
    )


def _to_list(value: Any) -> Any:
    for method_name in ("detach", "cpu"):
        method = getattr(value, method_name, None)
        if callable(method):
            value = method()
    method = getattr(value, "tolist", None)
    return method() if callable(method) else value


def motion_from_hy_output(
    output: dict[str, Any],
    skeleton: Skeleton,
    *,
    sample: int = 0,
    fps: float = 30.0,
    name: str = "HY-Motion",
    ground: bool = True,
) -> MotionClip:
    """Convert HY-Motion's CPU tensors into a backend-neutral clip.

    Grounding uses the source skeleton joints rather than HY-Motion's wooden
    mesh. It preserves horizontal root motion and shifts the entire clip once,
    so the lowest joint over the animation touches Y=0.
    """

    if "rot6d" not in output or "transl" not in output:
        raise HyMotionError("HY-Motion output is missing rot6d or transl")
    rotations_batch = _to_list(output["rot6d"])
    translations_batch = _to_list(output["transl"])
    try:
        rotations_frames = rotations_batch[sample]
        translation_frames = translations_batch[sample]
    except (IndexError, TypeError) as exc:
        raise HyMotionError(f"HY-Motion output has no sample {sample}") from exc

    if not isinstance(rotations_frames, list) or not rotations_frames:
        raise HyMotionError("HY-Motion returned no rotation frames")
    if len(rotations_frames) != len(translation_frames):
        raise HyMotionError("HY-Motion rotation and translation frame counts differ")

    joint_count = len(rotations_frames[0])
    if joint_count not in (22, 52) or joint_count > len(skeleton.bones):
        raise HyMotionError(f"unsupported HY-Motion joint count {joint_count}")
    for frame in rotations_frames:
        if len(frame) != joint_count:
            raise HyMotionError("HY-Motion joint count changes between frames")

    frame_matrices: list[list[Matrix3]] = []
    rotations_by_bone: dict[str, list[Quat]] = {
        skeleton.bones[joint].name: [] for joint in range(joint_count)
    }
    previous: dict[str, Quat] = {}
    for frame in rotations_frames:
        matrices: list[Matrix3] = []
        for joint, value in enumerate(frame):
            matrix = rot6d_to_matrix(value)
            matrices.append(matrix)
            bone_name = skeleton.bones[joint].name
            quaternion = matrix_to_quaternion(matrix)
            prior = previous.get(bone_name)
            if prior is not None and sum(a * b for a, b in zip(prior, quaternion, strict=True)) < 0:
                quaternion = tuple(-component for component in quaternion)
            rotations_by_bone[bone_name].append(quaternion)
            previous[bone_name] = quaternion
        frame_matrices.append(matrices)

    source_root = skeleton.source_positions[0]
    raw_translations: list[Vec3] = []
    for frame, value in enumerate(translation_frames):
        if not isinstance(value, list) or len(value) != 3:
            raise HyMotionError(f"HY-Motion translation at frame {frame} is not a 3-vector")
        raw_translations.append(tuple(float(component) for component in value))

    ground_offset = 0.0
    if ground:
        minimum_y = math.inf
        for local_matrices, translation in zip(frame_matrices, raw_translations, strict=True):
            world_rotations: list[Matrix3] = []
            world_positions: list[Vec3] = []
            for joint in range(joint_count):
                bone = skeleton.bones[joint]
                if bone.parent is None:
                    world_rotations.append(local_matrices[joint])
                    world_positions.append(
                        tuple(source_root[axis] + translation[axis] for axis in range(3))
                    )
                    continue
                if bone.parent >= joint_count:
                    raise HyMotionError("HY-Motion output omits a required parent joint")
                parent_rotation = world_rotations[bone.parent]
                parent_position = world_positions[bone.parent]
                offset = _matrix_vector(parent_rotation, bone.translation)
                world_positions.append(
                    tuple(parent_position[axis] + offset[axis] for axis in range(3))
                )
                world_rotations.append(_matrix_multiply(parent_rotation, local_matrices[joint]))
            minimum_y = min(minimum_y, *(position[1] for position in world_positions))
        joint_floor_y = min(position[1] for position in skeleton.source_positions[:joint_count])
        ground_offset = -minimum_y + joint_floor_y - skeleton.source_floor_y

    hips_translations = tuple(
        (
            source_root[0] + translation[0],
            source_root[1] + translation[1] + ground_offset,
            source_root[2] + translation[2],
        )
        for translation in raw_translations
    )
    return MotionClip(
        fps=fps,
        hips_translations=hips_translations,
        rotations={name: tuple(values) for name, values in rotations_by_bone.items()},
        name=name,
    )

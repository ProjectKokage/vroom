"""Convert ARDY Core27 motion into VRM humanoid tracks without importing Torch."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any
from zipfile import BadZipFile, ZipFile

from .errors import ArdyError
from .motion import (
    Bone,
    Matrix3,
    MotionClip,
    Quat,
    Skeleton,
    Vec3,
    _matrix_multiply,
    _matrix_vector,
    _to_list,
    matrix_to_quaternion,
)

# Exact Core27 hierarchy at nv-tlabs/ardy revision 693f74d. End markers have
# no VRM semantic; Spine1 is folded into Spine2, including its rotation.
CORE_JOINT_NAMES = (
    "Hips",
    "Spine",
    "Spine1",
    "Spine2",
    "Spine3",
    "Neck",
    "Head",
    "RightShoulder",
    "RightArm",
    "RightForeArm",
    "RightHand",
    "RightHandEnd",
    "RightHandThumb1",
    "LeftShoulder",
    "LeftArm",
    "LeftForeArm",
    "LeftHand",
    "LeftHandEnd",
    "LeftHandThumb1",
    "RightUpLeg",
    "RightLeg",
    "RightFoot",
    "RightToeBase",
    "LeftUpLeg",
    "LeftLeg",
    "LeftFoot",
    "LeftToeBase",
)
CORE_PARENTS = (
    -1,
    0,
    1,
    2,
    3,
    4,
    5,
    4,
    7,
    8,
    9,
    10,
    10,
    4,
    13,
    14,
    15,
    16,
    16,
    0,
    19,
    20,
    21,
    0,
    23,
    24,
    25,
)
_VRM_NAMES = (
    "hips",
    "spine",
    None,
    "chest",
    "upperChest",
    "neck",
    "head",
    "rightShoulder",
    "rightUpperArm",
    "rightLowerArm",
    "rightHand",
    None,
    "rightThumbMetacarpal",
    "leftShoulder",
    "leftUpperArm",
    "leftLowerArm",
    "leftHand",
    None,
    "leftThumbMetacarpal",
    "rightUpperLeg",
    "rightLowerLeg",
    "rightFoot",
    "rightToes",
    "leftUpperLeg",
    "leftLowerLeg",
    "leftFoot",
    "leftToes",
)
CORE_TO_VRM = {
    source: target
    for source, target in zip(CORE_JOINT_NAMES, _VRM_NAMES, strict=True)
    if target is not None
}
_MAPPED = tuple(index for index, name in enumerate(_VRM_NAMES) if name is not None)
_IDENTITY: Matrix3 = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def _sequence(value: Any, length: int, label: str) -> list | tuple:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ArdyError(f"{label} must have {length} entries")
    return value


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ArdyError(f"{label} must be finite numeric data")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ArdyError(f"{label} must be finite numeric data") from exc
    if not math.isfinite(result):
        raise ArdyError(f"{label} must be finite numeric data")
    return result


def _vector(value: Any, label: str) -> Vec3:
    return tuple(_number(v, label) for v in _sequence(value, 3, label))


def _transpose(matrix: Matrix3) -> Matrix3:
    return tuple(tuple(matrix[column][row] for column in range(3)) for row in range(3))


def _rotation(value: Any, label: str) -> Matrix3:
    matrix = tuple(_vector(row, label) for row in _sequence(value, 3, label))
    gram = _matrix_multiply(matrix, _transpose(matrix))
    determinant = (
        matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
    )
    # Float32 inference and bind assets introduce rounding, not scale/shear.
    if abs(determinant - 1.0) > 1e-3 or any(
        abs(gram[r][c] - _IDENTITY[r][c]) > 1e-3 for r in range(3) for c in range(3)
    ):
        raise ArdyError(f"{label} must be a proper rotation matrix")
    return matrix


def _bind_rotations(source: Any) -> tuple[Matrix3, ...]:
    offsets = getattr(source, "global_rot_offsets", None)
    if offsets is not None:
        values = _to_list(offsets)
    else:
        # Core's standard skeleton has no offsets .p file. Its skin archive
        # carries the actual bind basis, particularly the nonidentity thumbs.
        try:
            import numpy as np
        except ImportError as exc:
            raise ArdyError("reading the ARDY Core bind pose requires NumPy") from exc
        try:
            path = Path(source.folder) / "skin_standard.npz"
            with ZipFile(path) as archive:
                # Only two arrays are needed. Bound decompression by their
                # expected float64/UTF-32 payload plus NumPy's header limit.
                for entry, maximum in (
                    ("bind_rig_transform.npy", 27 * 16 * 8 + 10_000),
                    ("rig_joint_names.npy", 27 * max(map(len, CORE_JOINT_NAMES)) * 4 + 10_000),
                ):
                    if archive.getinfo(entry).file_size > maximum:
                        raise ArdyError(f"ARDY bind-pose array {entry} exceeds its dimensions")
            with np.load(path, allow_pickle=False) as data:
                names = data["rig_joint_names"]
                transforms = data["bind_rig_transform"]
                if names.shape != (27,) or tuple(names.tolist()) != CORE_JOINT_NAMES:
                    raise ArdyError("ARDY bind pose has an unsupported joint order")
                if transforms.shape != (27, 4, 4) or transforms.dtype.kind != "f":
                    raise ArdyError("ARDY bind pose must contain 27 floating-point 4x4 transforms")
                if not np.isfinite(transforms).all():
                    raise ArdyError("ARDY bind pose must contain finite transforms")
                if not np.allclose(transforms[:, 3, :], (0, 0, 0, 1), atol=1e-6):
                    raise ArdyError("ARDY bind pose contains a non-affine transform")
                values = transforms[:, :3, :3].tolist()
        except (OSError, ValueError, KeyError, BadZipFile, AttributeError, TypeError) as exc:
            raise ArdyError(f"cannot read ARDY Core bind pose: {exc}") from exc
    return tuple(
        _rotation(value, f"ARDY bind rotation {index}")
        for index, value in enumerate(_sequence(values, 27, "ARDY bind rotations"))
    )


def _source_data(source: Any) -> tuple[tuple[Vec3, ...], tuple[Matrix3, ...]]:
    if getattr(source, "name", None) != "cskel27":
        raise ArdyError("ARDY VRMA conversion requires the Core27 skeleton")
    names = getattr(source, "bone_order_names", None)
    if not isinstance(names, (list, tuple)) or tuple(names) != CORE_JOINT_NAMES:
        raise ArdyError("ARDY Core skeleton has an unsupported joint order")
    parents = _to_list(getattr(source, "joint_parents", None))
    if (
        not isinstance(parents, (list, tuple))
        or tuple(parents) != CORE_PARENTS
        or any(type(parent) is not int for parent in parents)
        or type(getattr(source, "root_idx", None)) is not int
        or source.root_idx != 0
    ):
        raise ArdyError("ARDY Core skeleton has an unsupported hierarchy")
    positions = tuple(
        _vector(value, f"ARDY neutral joint {index}")
        for index, value in enumerate(
            _sequence(_to_list(getattr(source, "neutral_joints", None)), 27, "ARDY neutral joints")
        )
    )
    return positions, _bind_rotations(source)


def _rest_positions(positions: tuple[Vec3, ...], bind: tuple[Matrix3, ...]) -> tuple[Vec3, ...]:
    rest = [positions[0]]
    for index, parent in enumerate(CORE_PARENTS[1:], 1):
        offset = _matrix_vector(
            bind[parent], tuple(positions[index][a] - positions[parent][a] for a in range(3))
        )
        rest.append(tuple(rest[parent][a] + offset[a] for a in range(3)))
    return tuple(rest)


def _skeleton(positions: tuple[Vec3, ...], bind: tuple[Matrix3, ...]) -> Skeleton:
    rest = _rest_positions(positions, bind)
    floor = min(position[1] for position in rest)
    bones = []
    for index in _MAPPED:
        parent = CORE_PARENTS[index]
        while parent >= 0 and _VRM_NAMES[parent] is None:
            parent = CORE_PARENTS[parent]
        translation = (
            (rest[index][0], rest[index][1] - floor, rest[index][2])
            if parent < 0
            else tuple(rest[index][a] - rest[parent][a] for a in range(3))
        )
        bones.append(
            Bone(_VRM_NAMES[index], None if parent < 0 else _MAPPED.index(parent), translation)
        )
    return Skeleton(tuple(bones), tuple(rest[index] for index in _MAPPED), floor)


def skeleton_from_ardy(source_skeleton: Any) -> Skeleton:
    """Build Core's normalized VRM rest hierarchy, retaining both thumb roots."""
    return _skeleton(*_source_data(source_skeleton))


def motion_from_ardy_output(
    output: dict[str, Any],
    source_skeleton: Any,
    *,
    sample: int = 0,
    fps: float = 20.0,
    name: str = "ARDY",
    ground: bool = True,
) -> MotionClip:
    """Convert batched [B,T,27,3,3] rotations and [B,T,3] absolute roots.

    Global rotations are expressed relative to the source bind basis before
    deriving VRM-local rotations. The omitted Spine1's rotation is composed
    into chest. Grounding applies one vertical shift, preserving root travel.
    """
    fps = _number(fps, "ARDY fps")
    if fps <= 0:
        raise ArdyError("ARDY fps must be positive")
    if type(sample) is not int or sample < 0:
        raise ArdyError("ARDY sample must be a non-negative integer")
    if not isinstance(output, dict) or not {"local_rot_mats", "root_positions"} <= output.keys():
        raise ArdyError("ARDY output is missing local_rot_mats or root_positions")
    rotations = _to_list(output["local_rot_mats"])
    roots = _to_list(output["root_positions"])
    if not isinstance(rotations, (list, tuple)) or not rotations:
        raise ArdyError("ARDY output contains no rotation samples")
    _sequence(roots, len(rotations), "ARDY root samples")
    if sample >= len(rotations):
        raise ArdyError(f"ARDY output has no sample {sample}")
    frames = rotations[sample]
    if not isinstance(frames, (list, tuple)) or not frames:
        raise ArdyError("ARDY output contains no rotation frames")
    root_frames = _sequence(roots[sample], len(frames), "ARDY root frames")
    positions, bind = _source_data(source_skeleton)
    skeleton = _skeleton(positions, bind)
    tracks: dict[str, list[Quat]] = {bone.name: [] for bone in skeleton.bones}
    translations = []
    minimum_y = math.inf
    for frame_index, (frame, root_value) in enumerate(zip(frames, root_frames, strict=True)):
        root = _vector(root_value, f"ARDY root at frame {frame_index}")
        translations.append(root)
        globals_: list[Matrix3] = []
        posed: list[Vec3] = []
        for index, value in enumerate(_sequence(frame, 27, f"ARDY frame {frame_index}")):
            matrix = _rotation(value, f"ARDY rotation at frame {frame_index}, joint {index}")
            parent = CORE_PARENTS[index]
            globals_.append(matrix if parent < 0 else _matrix_multiply(globals_[parent], matrix))
            if ground:
                offset = (
                    (0.0, 0.0, 0.0)
                    if parent < 0
                    else _matrix_vector(
                        globals_[parent],
                        tuple(positions[index][a] - positions[parent][a] for a in range(3)),
                    )
                )
                posed.append(
                    root if parent < 0 else tuple(posed[parent][a] + offset[a] for a in range(3))
                )
        if ground:
            minimum_y = min(minimum_y, *(position[1] for position in posed))
        normalized = [
            _matrix_multiply(globals_[index], _transpose(bind[index])) for index in _MAPPED
        ]
        for index, bone in enumerate(skeleton.bones):
            matrix = (
                normalized[index]
                if bone.parent is None
                else _matrix_multiply(_transpose(normalized[bone.parent]), normalized[index])
            )
            quaternion = matrix_to_quaternion(matrix)
            track = tracks[bone.name]
            if track and sum(a * b for a, b in zip(track[-1], quaternion, strict=True)) < 0:
                quaternion = tuple(-component for component in quaternion)
            track.append(quaternion)
    shift = -minimum_y if ground else 0.0
    return MotionClip(
        fps=fps,
        hips_translations=tuple((x, y + shift, z) for x, y, z in translations),
        rotations={bone: tuple(values) for bone, values in tracks.items()},
        name=name,
    )

from __future__ import annotations

import copy
import math
import subprocess
import sys
from itertools import pairwise
from types import SimpleNamespace

import pytest

from vroom.ardy_motion import motion_from_ardy_output, skeleton_from_ardy
from vroom.errors import ArdyError

NAMES = (
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
PARENTS = [
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
]
IDENTITY = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


def rz(degrees: float) -> list[list[float]]:
    c, s = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))
    return [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]


def ry(degrees: float) -> list[list[float]]:
    c, s = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))
    return [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]]


def rx(degrees: float) -> list[list[float]]:
    c, s = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))
    return [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]]


@pytest.fixture
def source() -> SimpleNamespace:
    # Deliberately simple, independent T-pose: +X is anatomical left, +Z forward.
    positions = [
        [0, 0, 0],
        [0, 0.1, 0],
        [0, 0.2, 0],
        [0, 0.3, 0],
        [0, 0.4, 0],
        [0, 0.6, 0],
        [0, 0.8, 0],
        [-0.1, 0.5, 0],
        [-0.2, 0.5, 0],
        [-0.5, 0.5, 0],
        [-0.8, 0.5, 0],
        [-0.9, 0.5, 0],
        [-0.85, 0.5, 0.05],
        [0.1, 0.5, 0],
        [0.2, 0.5, 0],
        [0.5, 0.5, 0],
        [0.8, 0.5, 0],
        [0.9, 0.5, 0],
        [0.85, 0.5, 0.05],
        [-0.1, 0, 0],
        [-0.1, -0.4, 0],
        [-0.1, -0.9, 0],
        [-0.1, -1, 0.2],
        [0.1, 0, 0],
        [0.1, -0.4, 0],
        [0.1, -0.9, 0],
        [0.1, -1, 0.2],
    ]
    return SimpleNamespace(
        name="cskel27",
        bone_order_names=NAMES,
        joint_parents=PARENTS.copy(),
        root_idx=0,
        neutral_joints=positions,
        global_rot_offsets=[copy.deepcopy(IDENTITY) for _ in NAMES],
    )


def output(frames: int = 1) -> dict:
    return {
        "local_rot_mats": [[[copy.deepcopy(IDENTITY) for _ in NAMES] for _ in range(frames)]],
        "root_positions": [[[0, 1, 0] for _ in range(frames)]],
    }


def test_import_does_not_load_inference_modules() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import vroom.ardy_motion; "
            "assert not {'torch', 'numpy', 'ardy'} & sys.modules.keys()",
        ],
        check=True,
    )


def test_rest_skeleton_semantics_positions_and_thumb_basis(source) -> None:
    source.global_rot_offsets[12] = rx(90)
    source.global_rot_offsets[18] = ry(-90)
    skeleton = skeleton_from_ardy(source)
    assert len(skeleton.bones) == 24
    assert skeleton.bones[skeleton.index("hips")].translation == (0, 1, 0)
    assert skeleton.bones[skeleton.index("chest")].translation == pytest.approx((0, 0.2, 0))
    assert skeleton.bones[skeleton.index("leftUpperLeg")].translation[0] > 0
    assert skeleton.bones[skeleton.index("rightUpperLeg")].translation[0] < 0
    assert skeleton.bones[skeleton.index("leftToes")].translation[2] > 0
    thumb = skeleton.bones[skeleton.index("leftThumbMetacarpal")]
    assert thumb.parent == skeleton.index("leftHand")
    assert thumb.translation == pytest.approx((0.05, 0, 0.05))

    data = output()
    data["local_rot_mats"][0][0][12] = rx(90)
    data["local_rot_mats"][0][0][18] = ry(-90)
    clip = motion_from_ardy_output(data, source)
    assert all(track[0] == pytest.approx((0, 0, 0, 1)) for track in clip.rotations.values())


def test_parent_bind_rotation_transforms_child_rest_offset(source) -> None:
    # A nonidentity hand basis rotates its child offset in the actual bind pose.
    source.global_rot_offsets[16] = rz(90)
    source.global_rot_offsets[17] = rz(90)
    source.global_rot_offsets[18] = rz(90)
    skeleton = skeleton_from_ardy(source)
    thumb = skeleton.bones[skeleton.index("leftThumbMetacarpal")]
    assert thumb.translation == pytest.approx((0, 0.05, 0.05))
    data = output()
    data["local_rot_mats"][0][0][16] = rz(90)
    clip = motion_from_ardy_output(data, source)
    assert clip.rotations["leftHand"][0] == pytest.approx((0, 0, 0, 1))
    assert clip.rotations["leftThumbMetacarpal"][0] == pytest.approx((0, 0, 0, 1))


def test_spine_collapse_preserves_noncommuting_rotations(source) -> None:
    data = output()
    data["local_rot_mats"][0][0][2] = rx(90)  # omitted Spine1
    data["local_rot_mats"][0][0][3] = ry(90)  # retained chest
    data["local_rot_mats"][0][0][4] = rz(90)  # upper chest remains separate
    clip = motion_from_ardy_output(data, source, ground=False)
    assert clip.rotations["chest"][0] == pytest.approx((0.5, 0.5, 0.5, 0.5))
    assert clip.rotations["upperChest"][0] == pytest.approx((0, 0, math.sqrt(0.5), math.sqrt(0.5)))
    assert clip.rotations["neck"][0] == pytest.approx((0, 0, 0, 1))


def test_bind_normalization_keeps_motion_in_global_axes(source) -> None:
    source.global_rot_offsets[18] = rx(90)
    # Ry(90) @ Rx(90): animated global basis relative to a rotated thumb bind.
    data = output()
    data["local_rot_mats"][0][0][18] = [[0, 1, 0], [0, 0, -1], [-1, 0, 0]]
    clip = motion_from_ardy_output(data, source, ground=False)
    assert clip.rotations["leftThumbMetacarpal"][0] == pytest.approx(
        (0, math.sqrt(0.5), 0, math.sqrt(0.5))
    )
    assert clip.rotations["rightThumbMetacarpal"][0] == pytest.approx((0, 0, 0, 1))


def test_absolute_roots_and_single_grounding_offset(source) -> None:
    data = output(2)
    data["root_positions"][0] = [[2, 4, 5], [3, 4.5, 7]]
    raw = motion_from_ardy_output(data, source, ground=False)
    grounded = motion_from_ardy_output(data, source)
    assert raw.hips_translations == ((2, 4, 5), (3, 4.5, 7))
    assert grounded.hips_translations == ((2, 1, 5), (3, 1.5, 7))
    assert raw.fps == 20
    assert raw.duration == pytest.approx(0.05)


def test_grounding_uses_animated_fk(source) -> None:
    data = output()
    # Point the left arm down. The hand endpoint reaches below the planted toes.
    data["local_rot_mats"][0][0][14] = rz(-90)
    source.neutral_joints[17] = [2, 0.5, 0]
    clip = motion_from_ardy_output(data, source)
    # Arm starts at Y=1.5; endpoint is 1.8 m below it -> minimum Y=-.3.
    assert clip.hips_translations[0] == pytest.approx((0, 1.3, 0))


def test_sample_selection_and_quaternion_continuity(source) -> None:
    data = output(3)
    data["local_rot_mats"].append(copy.deepcopy(data["local_rot_mats"][0]))
    data["root_positions"].append([[2, 1, 0]] * 3)
    for frame, angle in zip(data["local_rot_mats"][1], (170, 180, 190), strict=True):
        frame[0] = ry(angle)
    clip = motion_from_ardy_output(data, source, sample=1, ground=False)
    assert clip.hips_translations[0] == (2, 1, 0)
    quaternions = clip.rotations["hips"]
    assert all(
        sum(a * b for a, b in zip(left, right, strict=True)) > 0.99
        for left, right in pairwise(quaternions)
    )
    assert quaternions[-1][3] < 0


@pytest.mark.parametrize("value", [math.nan, math.inf, "1", True, 10**400])
def test_rejects_nonfinite_or_nonnumeric_root(source, value) -> None:
    data = output()
    data["root_positions"][0][0][0] = value
    with pytest.raises(ArdyError, match="finite numeric"):
        motion_from_ardy_output(data, source)


@pytest.mark.parametrize(
    "matrix",
    [
        [[-1, 0, 0], [0, 1, 0], [0, 0, 1]],  # reflection
        [[2, 0, 0], [0, 1, 0], [0, 0, 1]],  # scale
        [[1, 0.2, 0], [0, 1, 0], [0, 0, 1]],  # shear
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
    ],
)
def test_rejects_improper_rotations(source, matrix) -> None:
    data = output()
    data["local_rot_mats"][0][0][0] = matrix
    with pytest.raises(ArdyError, match="proper rotation"):
        motion_from_ardy_output(data, source)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda d: d.pop("root_positions"),
        lambda d: d["root_positions"].append([[0, 0, 0]]),
        lambda d: d["root_positions"][0].clear(),
        lambda d: d["local_rot_mats"][0][0].pop(),
        lambda d: d["local_rot_mats"][0][0][0].pop(),
        lambda d: d["local_rot_mats"][0].clear(),
    ],
)
def test_rejects_malformed_output(source, mutation) -> None:
    data = output()
    mutation(data)
    with pytest.raises(ArdyError):
        motion_from_ardy_output(data, source)


@pytest.mark.parametrize("sample", [-1, True, 1, 1.5])
def test_rejects_invalid_sample(source, sample) -> None:
    with pytest.raises(ArdyError, match="sample"):
        motion_from_ardy_output(output(), source, sample=sample)


@pytest.mark.parametrize("fps", [0, -1, math.nan, math.inf, True])
def test_rejects_invalid_fps(source, fps) -> None:
    with pytest.raises(ArdyError, match="fps"):
        motion_from_ardy_output(output(), source, fps=fps)


def test_rejects_foreign_skeleton_order_hierarchy_and_bind(source) -> None:
    source.name = "g1skel34"
    with pytest.raises(ArdyError, match="Core27"):
        skeleton_from_ardy(source)
    source.name = "cskel27"
    source.joint_parents[3] = 1
    with pytest.raises(ArdyError, match="hierarchy"):
        skeleton_from_ardy(source)
    source.joint_parents[3] = 2
    source.bone_order_names = tuple(reversed(NAMES))
    with pytest.raises(ArdyError, match="joint order"):
        skeleton_from_ardy(source)
    source.bone_order_names = NAMES
    source.global_rot_offsets[18] = [[0, 0, 0]] * 3
    with pytest.raises(ArdyError, match="bind rotation"):
        skeleton_from_ardy(source)


@pytest.mark.parametrize(
    "attribute,value",
    [
        ("bone_order_names", None),
        ("joint_parents", None),
        ("root_idx", False),
        ("neutral_joints", None),
        ("neutral_joints", [[0, 0, 0]]),
        ("global_rot_offsets", []),
    ],
)
def test_rejects_malformed_source_metadata(source, attribute, value) -> None:
    setattr(source, attribute, value)
    with pytest.raises(ArdyError):
        skeleton_from_ardy(source)


def test_rejects_nonfinite_rotation_and_neutral_joint(source) -> None:
    data = output()
    data["local_rot_mats"][0][0][18][0][0] = math.nan
    with pytest.raises(ArdyError, match="finite numeric"):
        motion_from_ardy_output(data, source)
    source.neutral_joints[1][0] = math.inf
    with pytest.raises(ArdyError, match="neutral joint"):
        skeleton_from_ardy(source)

from __future__ import annotations

import math

import pytest

from vroom.motion import (
    HY_TO_VRM_BONES,
    Skeleton,
    motion_from_hy_output,
    rot6d_to_quaternion,
)

from .conftest import IDENTITY_ROT6D


def test_interleaved_rot6d_identity() -> None:
    assert rot6d_to_quaternion(IDENTITY_ROT6D) == pytest.approx((0.0, 0.0, 0.0, 1.0))


def test_interleaved_rot6d_quarter_turn_around_z() -> None:
    # Flattened first two columns of Rz(90°), interleaved row-by-row.
    quaternion = rot6d_to_quaternion([0.0, -1.0, 1.0, 0.0, 0.0, 0.0])
    root_half = math.sqrt(0.5)
    assert quaternion == pytest.approx((0.0, 0.0, root_half, root_half))


def test_hy_skeleton_has_vrm1_basis_and_complete_mapping(hy_skeleton: Skeleton) -> None:
    assert hy_skeleton.bone_names == HY_TO_VRM_BONES
    assert hy_skeleton.bones[0].name == "hips"
    assert hy_skeleton.bones[1].translation[0] > 0  # anatomical left is +X in VRM 1
    assert hy_skeleton.bones[2].translation[0] < 0
    assert hy_skeleton.bones[10].translation[2] > 0  # toes point toward +Z
    assert hy_skeleton.bones[0].translation[1] == pytest.approx(0.97017, abs=1e-3)


def test_hy_output_becomes_absolute_grounded_hips(hy_skeleton: Skeleton) -> None:
    output = {
        "rot6d": [[[IDENTITY_ROT6D.copy() for _ in range(22)]]],
        "transl": [[[0.0, 0.0, 0.0]]],
    }

    clip = motion_from_hy_output(output, hy_skeleton)

    assert clip.hips_translations[0] == pytest.approx(
        hy_skeleton.bones[0].translation,
        abs=1e-5,
    )
    assert set(clip.rotations) == set(HY_TO_VRM_BONES[:22])
    assert all(track[0] == pytest.approx((0.0, 0.0, 0.0, 1.0)) for track in clip.rotations.values())


def test_hy_output_can_preserve_raw_vertical_position(hy_skeleton: Skeleton) -> None:
    output = {
        "rot6d": [[[IDENTITY_ROT6D.copy() for _ in range(22)]]],
        "transl": [[[0.0, 0.0, 0.0]]],
    }

    clip = motion_from_hy_output(output, hy_skeleton, ground=False)

    assert clip.hips_translations[0] == pytest.approx(hy_skeleton.source_positions[0])

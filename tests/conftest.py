from __future__ import annotations

from pathlib import Path

import pytest

from vroom.motion import MotionClip, Skeleton, load_hy_motion_skeleton, motion_from_hy_output

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HY_ROOT = PROJECT_ROOT / "vendor" / "HY-Motion-1.0"
IDENTITY_ROT6D = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]


@pytest.fixture(scope="session")
def hy_skeleton() -> Skeleton:
    return load_hy_motion_skeleton(HY_ROOT)


@pytest.fixture
def identity_clip(hy_skeleton: Skeleton) -> MotionClip:
    frames = 4
    output = {
        "rot6d": [[[IDENTITY_ROT6D.copy() for _ in range(22)] for _ in range(frames)]],
        "transl": [[[0.0, 0.0, frame * 0.01] for frame in range(frames)]],
    }
    return motion_from_hy_output(output, hy_skeleton, fps=30.0, name="identity")

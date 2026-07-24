from __future__ import annotations

import zipfile

import pytest

from vroom.vrm import VrmAvatar
from vroom.vrma import read_vrma, validate_vrma

from .conftest import PROJECT_ROOT


def test_alicia_legacy_vrm_humanoid() -> None:
    path = PROJECT_ROOT / "AliciaSolid_vrm-0.51.vrm"
    if not path.is_file():
        pytest.skip("local Alicia VRM fixture is not present")

    avatar = VrmAvatar.read(path)

    assert avatar.generation == "0.x"
    assert len(avatar.humanoid_bones) == 55
    assert avatar.node_name("hips") == "Hips"
    assert "leftThumbMetacarpal" in avatar.humanoid_bones
    assert "leftThumbIntermediate" not in avatar.humanoid_bones


def test_reference_motion_pack_is_structurally_valid() -> None:
    path = PROJECT_ROOT / "VRMA_MotionPack.zip"
    if not path.is_file():
        pytest.skip("local VRMA motion-pack fixture is not present")

    with zipfile.ZipFile(path) as archive:
        members = sorted(name for name in archive.namelist() if name.endswith(".vrma"))
        assert len(members) == 7
        for member in members:
            payload = archive.read(member)
            report = validate_vrma(payload)
            assert report.errors == [], f"{member}: {report.errors}"
            vrma = read_vrma(payload)
            assert vrma.nominal_fps == pytest.approx(60.0, rel=1e-3)

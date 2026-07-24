from __future__ import annotations

import os
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

from hatchling.build import build_sdist, build_wheel

from .conftest import PROJECT_ROOT


def test_sdist_rebuilds_a_clean_working_wheel(tmp_path, monkeypatch) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()

    monkeypatch.chdir(PROJECT_ROOT)
    sdist = artifacts / build_sdist(str(artifacts))
    with tarfile.open(sdist) as archive:
        source_names = archive.getnames()
        assert not any(
            forbidden in Path(name).parts
            for name in source_names
            for forbidden in (
                ".gitmodules",
                ".python-version",
                "output",
                "scripts",
                "tests",
                "uv.lock",
                "vendor",
            )
        )
        assert not any(
            Path(name).name in {"AliciaSolid_vrm-0.51.vrm", "VRMA_MotionPack.zip"}
            for name in source_names
        )
        archive.extractall(tmp_path / "source", filter="data")

    source_root = next((tmp_path / "source").iterdir())
    monkeypatch.chdir(source_root)
    wheel = artifacts / build_wheel(str(artifacts))
    with zipfile.ZipFile(wheel) as archive:
        wheel_names = archive.namelist()
        assert "vroom/__init__.py" in wheel_names
        entry_points_name = next(
            name for name in wheel_names if name.endswith(".dist-info/entry_points.txt")
        )
        entry_points = archive.read(entry_points_name).decode()
        assert "vroom = vroom.cli:main" in entry_points
        assert not any("vendor" in Path(name).parts for name in wheel_names)

    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(wheel)
    completed = subprocess.run(
        [sys.executable, "-S", "-m", "vroom", "--help"],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "generate" in completed.stdout
    assert "validate" in completed.stdout
    assert "inspect" in completed.stdout

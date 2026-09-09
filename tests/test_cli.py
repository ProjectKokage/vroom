from __future__ import annotations

import copy

from vroom.cli import _build_parser, main
from vroom.glb import Glb
from vroom.motion import MotionClip, Skeleton
from vroom.vrma import build_vrma, write_vrma


def test_inspect_and_validate_generated_vrma(
    tmp_path,
    capsys,
    identity_clip: MotionClip,
    hy_skeleton: Skeleton,
) -> None:
    path = tmp_path / "test.vrma"
    write_vrma(path, identity_clip, hy_skeleton)

    assert main(["inspect", str(path), "--json"]) == 0
    summary = capsys.readouterr().out
    assert '"format": "VRMA"' in summary
    assert '"frames": 4' in summary

    assert main(["validate", str(path), "--quiet"]) == 0
    assert capsys.readouterr().out == ""


def test_generate_reports_missing_checkpoint(tmp_path, capsys) -> None:
    code = main(
        [
            "generate",
            "--backend",
            "hy-motion",
            "a person waves",
            "--model-path",
            str(tmp_path),
            "--output",
            str(tmp_path / "wave.vrma"),
        ]
    )

    assert code == 2
    assert "missing HY-Motion runtime files" in capsys.readouterr().err


def test_generate_defaults_to_ardy_core(monkeypatch) -> None:
    monkeypatch.delenv("HY_MOTION_MODEL_PATH", raising=False)

    args = _build_parser().parse_args(["generate", "a person waves", "--output", "wave.vrma"])

    assert args.backend == "ardy"
    assert args.model == "core"
    assert args.model_path is None


def test_inspect_reports_malformed_animation_cleanly(
    tmp_path,
    capsys,
    identity_clip: MotionClip,
    hy_skeleton: Skeleton,
) -> None:
    original = build_vrma(identity_clip, hy_skeleton)
    document = copy.deepcopy(original.document)
    document["animations"] = [{}]
    path = tmp_path / "malformed.vrma"
    Glb(document, original.binary).write(path)

    assert main(["inspect", str(path)]) == 2
    assert "first animation has no samplers" in capsys.readouterr().err

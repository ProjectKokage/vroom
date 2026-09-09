"""Explicitly enabled, full-weight ARDY generation and continuation checks."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from vroom.ardy_source import read_source
from vroom.cli import main
from vroom.vrma import read_vrma, validate_vrma

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("VROOM_TEST_ARDY_MODELS") != "1",
        reason="set VROOM_TEST_ARDY_MODELS=1 for full-weight ARDY inference",
    ),
]


def test_real_prompt_plan_variants_and_continuation(tmp_path: Path) -> None:
    import numpy as np

    output = Path(os.environ.get("ARDY_TEST_OUTPUT_DIR", tmp_path))
    output.mkdir(parents=True, exist_ok=True)
    shared = [
        "--device",
        os.environ.get("ARDY_TEST_DEVICE", "auto"),
        "--text-device",
        os.environ.get("ARDY_TEST_TEXT_DEVICE", "auto"),
        "--model",
        os.environ.get("ARDY_TEST_MODEL", "core40"),
    ]
    base = output / "wave.vrma"
    source = output / "wave.npz"
    assert (
        main(
            [
                "generate",
                "A person waves their right hand in greeting.",
                "--duration",
                "3",
                "--seed",
                "42",
                "--output",
                str(base),
                "--source-output",
                str(source),
                *shared,
            ]
        )
        == 0
    )
    assert validate_vrma(base).errors == []
    clip = read_vrma(base)
    assert clip.frame_count == 61
    assert clip.duration == pytest.approx(3)
    prefix = read_source(source, max_bytes=256 * 1024**2)
    assert prefix.frame_count == 61
    with np.load(source, allow_pickle=False) as arrays:
        assert "text" not in arrays and "prompt" not in arrays

    plan = output / "continuation.json"
    plan.write_text(
        json.dumps(
            {
                "duration": 5,
                "prompts": [
                    {"time": 0, "text": "A person waves their right hand in greeting."},
                    {"time": 3.5, "text": "The person lowers their hand and stands calmly."},
                ],
                "seeds": [7, 8],
                "continuation": "wave.npz",
                "constraints": [
                    {
                        "type": "pose",
                        "time": 5,
                        "source": "wave.npz",
                        "sourceTime": 0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    report = output / "continuation-report.json"
    assert (
        main(
            [
                "generate",
                "--plan",
                str(plan),
                "--output",
                str(output / "extended.vrma"),
                "--source-output",
                str(output / "extended.npz"),
                "--report",
                str(report),
                *shared,
            ]
        )
        == 0
    )
    for seed in (7, 8):
        path = output / f"extended-{seed}.vrma"
        assert validate_vrma(path).errors == []
        extended = read_vrma(path)
        assert extended.frame_count == 101
        assert extended.duration == pytest.approx(5)
        continued = read_source(output / f"extended-{seed}.npz", max_bytes=256 * 1024**2)
        np.testing.assert_array_equal(continued.local_rot_mats[:61], prefix.local_rot_mats)
        np.testing.assert_array_equal(continued.root_positions[:61], prefix.root_positions)
    receipt = json.loads(report.read_text())
    assert len(receipt["candidates"]) == 2
    for candidate in receipt["candidates"]:
        assert candidate["measurements"]
        assert all(np.isfinite(item["error"]) for item in candidate["measurements"])

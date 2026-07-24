from __future__ import annotations

import os
from pathlib import Path

import pytest

from vroom.cli import main
from vroom.vrma import read_vrma, validate_vrma

from .conftest import HY_ROOT


@pytest.mark.integration
def test_real_text_to_vrma_when_configured(tmp_path: Path) -> None:
    configured = os.environ.get("HY_MOTION_MODEL_PATH")
    if not configured:
        pytest.skip("set HY_MOTION_MODEL_PATH to run real model inference")
    model_path = Path(configured)
    if not (model_path / "latest.ckpt").is_file():
        pytest.skip("configured HY-Motion checkpoint is absent")

    output_path = tmp_path / "real-smoke.vrma"
    code = main(
        [
            "generate",
            "a person raises one hand",
            "--model-path",
            str(model_path),
            "--hy-motion-root",
            str(HY_ROOT),
            "--duration",
            "1",
            "--seed",
            "7",
            "--validation-steps",
            "1",
            "--device",
            os.environ.get("HY_MOTION_TEST_DEVICE", "auto"),
            "--text-device",
            os.environ.get("HY_MOTION_TEST_TEXT_DEVICE", "auto"),
            "--text-models",
            os.environ.get("HY_MOTION_TEST_TEXT_MODELS", "auto"),
            "--output",
            str(output_path),
        ]
    )

    assert code == 0
    assert output_path.is_file()
    assert read_vrma(output_path).frame_count == 30
    assert validate_vrma(output_path).errors == []

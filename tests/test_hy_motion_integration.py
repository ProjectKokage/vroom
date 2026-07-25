from __future__ import annotations

import os
from pathlib import Path

import pytest

from vroom.cli import main
from vroom.hy_motion import HyMotionBackend, HyMotionOptions
from vroom.motion import load_hy_motion_skeleton
from vroom.vrma import build_vrma, read_vrma, validate_vrma

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


@pytest.mark.integration
def test_real_same_prompt_seed_batch_when_configured(tmp_path: Path) -> None:
    configured = os.environ.get("HY_MOTION_MODEL_PATH")
    if not configured:
        pytest.skip("set HY_MOTION_MODEL_PATH to run real model inference")
    model_path = Path(configured)
    if not (model_path / "latest.ckpt").is_file():
        pytest.skip("configured HY-Motion checkpoint is absent")

    backend = HyMotionBackend(
        HyMotionOptions(
            model_path=model_path,
            hy_motion_root=HY_ROOT,
            device=os.environ.get("HY_MOTION_TEST_DEVICE", "auto"),
            text_device=os.environ.get("HY_MOTION_TEST_TEXT_DEVICE", "auto"),
            text_models=os.environ.get("HY_MOTION_TEST_TEXT_MODELS", "auto"),
            validation_steps=1,
        )
    )
    clips = backend.generate_variants(
        "a person raises one hand",
        duration=1.0,
        seeds=(7, 8),
    )
    skeleton = load_hy_motion_skeleton(HY_ROOT)

    assert len(clips) == 2
    for index, clip in enumerate(clips):
        output = tmp_path / f"real-batch-{index}.vrma"
        build_vrma(clip, skeleton).write(output)
        assert read_vrma(output).frame_count == 30
        assert validate_vrma(output).errors == []

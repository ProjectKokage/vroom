from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from vroom.errors import HyMotionError
from vroom.hy_motion import (
    HyMotionBackend,
    HyMotionOptions,
    _configure_text_models,
    _move_text_conditioning,
    _use_device_random_generator,
)


class _Tensor:
    def __init__(self, label: str, calls: list[tuple[str, str]]):
        self.label = label
        self.calls = calls

    def float(self) -> _Tensor:
        self.calls.append((self.label, "float"))
        return self

    def to(self, device_name: str) -> _Tensor:
        self.calls.append((self.label, f"to:{device_name}"))
        return self


def test_text_conditioning_casts_before_accelerator_transfer() -> None:
    calls: list[tuple[str, str]] = []
    vector = _Tensor("vector", calls)
    context = _Tensor("context", calls)
    length = _Tensor("length", calls)

    hidden = _move_text_conditioning(vector, context, length, "mps")

    assert hidden == {
        "text_vec_raw": vector,
        "text_ctxt_raw": context,
        "text_ctxt_raw_length": length,
    }
    assert calls == [
        ("vector", "float"),
        ("vector", "to:mps"),
        ("context", "float"),
        ("context", "to:mps"),
        ("length", "to:mps"),
    ]


def test_noise_generator_preserves_checkpoint_choice_only_on_cuda() -> None:
    assert _use_device_random_generator(True, "cuda:0")
    assert not _use_device_random_generator(False, "cuda:0")
    assert not _use_device_random_generator(True, "mps")
    assert not _use_device_random_generator(True, "cpu")


def test_text_model_source_can_change_after_upstream_import(tmp_path, monkeypatch) -> None:
    qwen = tmp_path / "ckpts" / "Qwen3-8B"
    clip = tmp_path / "ckpts" / "clip-vit-large-patch14"
    qwen.mkdir(parents=True)
    clip.mkdir()
    loaded = SimpleNamespace(
        LLM_ENCODER_LAYOUT={"qwen3": {"module_path": "old-qwen"}},
        SENTENCE_EMB_LAYOUT={"clipl": {"module_path": "old-clip"}},
    )
    module_name = "hymotion.network.text_encoders.text_encoder"
    monkeypatch.setitem(sys.modules, module_name, loaded)
    monkeypatch.delenv("USE_HF_MODELS", raising=False)

    assert _configure_text_models("local", tmp_path) == "local"
    assert os.environ["USE_HF_MODELS"] == "0"
    assert loaded.LLM_ENCODER_LAYOUT["qwen3"]["module_path"] == str(qwen)
    assert loaded.SENTENCE_EMB_LAYOUT["clipl"]["module_path"] == str(clip)

    assert _configure_text_models("hugging-face", tmp_path) == "hugging-face"
    assert os.environ["USE_HF_MODELS"] == "1"
    assert loaded.LLM_ENCODER_LAYOUT["qwen3"]["module_path"] == "Qwen/Qwen3-8B"
    assert loaded.SENTENCE_EMB_LAYOUT["clipl"]["module_path"] == "openai/clip-vit-large-patch14"


@pytest.mark.parametrize(
    ("seeds", "message"),
    [
        ((), "between 1 and 16"),
        (tuple(range(17)), "between 1 and 16"),
        ((7, 7), "unique"),
        ((-1,), "each seed"),
        ((2**63,), "each seed"),
    ],
)
def test_generate_variants_rejects_invalid_seed_batches_before_loading(
    seeds: tuple[int, ...],
    message: str,
) -> None:
    backend = HyMotionBackend(
        HyMotionOptions(
            model_path=Path("/missing/model"),
            hy_motion_root=Path("/missing/runtime"),
        )
    )

    with pytest.raises(HyMotionError, match=message):
        backend.generate_variants("a person waves", duration=1.0, seeds=seeds)

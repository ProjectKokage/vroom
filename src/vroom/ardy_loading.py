"""Pinned upstream artifacts and local, single-device ARDY text encoding."""

from __future__ import annotations

import hashlib
import json
from importlib.resources import files
from pathlib import Path
from typing import Any

from .errors import ArdyError

BASE_REPO = "meta-llama/Meta-Llama-3-8B-Instruct"
BASE_REVISION = "8afb486c1db24fe5011ec46dfbe5b5dccdb575c2"
MNTP_REPO = "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp"
MNTP_REVISION = "31474e395ada192e8ed1586db6be79fb3b70c9c0"
SUPERVISED_REPO = "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised"
SUPERVISED_REVISION = "baa8ebf04a1c2500e61288e7dad65e8ae42601a7"


def load_motion_model(name: str, checkpoints_dir: Path | None) -> Any:
    from ardy.model.load_model import load_model
    from huggingface_hub import snapshot_download

    manifests = json.loads(files("vroom").joinpath("ardy_models.json").read_text())
    manifest = next(
        (item for item in manifests.values() if item["repo_id"].endswith("/" + name)), None
    )
    if manifest is None:
        raise ArdyError("VRMA generation requires an ARDY Core human skeleton model")
    folder = (
        Path(
            snapshot_download(
                manifest["repo_id"],
                revision=manifest["revision"],
                allow_patterns=[item["path"] for item in manifest["files"]],
            )
        )
        if checkpoints_dir is None
        else checkpoints_dir / name
    )
    for item in manifest["files"]:
        path = folder / item["path"]
        try:
            if path.stat().st_size != item["size_bytes"]:
                raise ArdyError(f"ARDY artifact size mismatch: {item['path']}")
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest != item["sha256"]:
                raise ArdyError(f"ARDY artifact SHA-256 mismatch: {item['path']}")
        except OSError as exc:
            raise ArdyError(f"ARDY artifact missing/unreadable: {item['path']}") from exc
    # Upstream takes a parent directory plus a released folder name. A pinned
    # HF snapshot has a hash as its name, so load its verified config directly.
    from ardy.model.loading import instantiate_from_dict
    from omegaconf import OmegaConf

    if checkpoints_dir is not None:
        return load_model(
            name, device="cpu", text_encoder=False, checkpoints_dir=str(checkpoints_dir)
        )
    config = OmegaConf.load(folder / "config.yaml")
    merged = OmegaConf.merge(config, {"checkpoint_dir": str(folder)})
    values = OmegaConf.to_container(merged, resolve=True)
    values.pop("checkpoint_dir", None)
    values["text_encoder"] = None
    return instantiate_from_dict(values, overrides={"device": "cpu"}).eval()


def load_local_encoder(device: str) -> Any:
    """Use upstream LLM2Vec with the same two adapters, all revisions pinned."""
    import torch
    from ardy.model.llm2vec import LLM2Vec
    from huggingface_hub import snapshot_download
    from peft import PeftModel
    from transformers import AutoConfig, AutoTokenizer

    adapters = [
        snapshot_download(repo, revision=revision, allow_patterns=["*.json", "*.safetensors"])
        for repo, revision in ((MNTP_REPO, MNTP_REVISION), (SUPERVISED_REPO, SUPERVISED_REVISION))
    ]
    tokenizer = AutoTokenizer.from_pretrained(BASE_REPO, revision=BASE_REVISION)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    config = AutoConfig.from_pretrained(BASE_REPO, revision=BASE_REVISION)
    model_class = LLM2Vec._get_model_class(config.__class__.__name__, True)
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    model = model_class.from_pretrained(BASE_REPO, revision=BASE_REVISION, torch_dtype=dtype)
    # Match upstream's MNTP merge followed by its supervised adapter. Loading
    # the base explicitly prevents adapter metadata from choosing an unpinned
    # model revision or loading it onto other GPUs.
    model = PeftModel.from_pretrained(model, adapters[0]).merge_and_unload()
    model = PeftModel.from_pretrained(model, adapters[1])
    model.config._name_or_path = BASE_REPO
    settings_path = Path(adapters[1]) / "llm2vec_config.json"
    settings = json.loads(settings_path.read_text()) if settings_path.is_file() else {}
    allowed = {"pooling_mode", "max_length", "doc_max_length", "skip_instruction"}
    if not set(settings) <= allowed:
        raise ArdyError("unsupported LLM2Vec encoder configuration")
    encoder = LLM2Vec(model=model, tokenizer=tokenizer, **settings).to(device, dtype=dtype).eval()
    return encoder

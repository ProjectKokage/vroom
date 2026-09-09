"""Artifact and encoder loading contracts without optional native dependencies."""

from __future__ import annotations

import hashlib
import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

import vroom.ardy_loading as loading
from vroom.errors import ArdyError


def _module(monkeypatch, name, **attributes):
    module = ModuleType(name)
    module.__path__ = []
    for key, value in attributes.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)
    return module


@pytest.fixture
def motion_loader(tmp_path, monkeypatch):
    name = "ARDY-Core-RP-20FPS-Horizon40"
    folder = tmp_path / name
    folder.mkdir()
    artifacts = {"config.yaml": b"verified config", "stats/mean.npy": b"verified statistics"}
    manifest = {
        "repo_id": "nvidia/" + name,
        "revision": "1" * 40,
        "files": [],
    }
    for filename, data in artifacts.items():
        path = folder / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        manifest["files"].append(
            {
                "path": filename,
                "size_bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    (tmp_path / "ardy_models.json").write_text(json.dumps({"core40": manifest}))
    monkeypatch.setattr(loading, "files", lambda package: tmp_path)
    events = []
    model = SimpleNamespace(eval=lambda: events.append("eval") or "evaluated model")

    def download(repo, **kwargs):
        events.append(("download", repo, kwargs))
        return str(folder)

    def load_model(*args, **kwargs):
        events.append(("load_model", args, kwargs))
        return model

    def load_config(path):
        events.append(("load_config", path))
        return {"_target_": "trusted.ARDY", "text_encoder": "must be disabled"}

    def instantiate(values, **kwargs):
        events.append(("instantiate", values, kwargs))
        return model

    _module(monkeypatch, "ardy")
    _module(monkeypatch, "ardy.model")
    _module(monkeypatch, "ardy.model.load_model", load_model=load_model)
    _module(monkeypatch, "ardy.model.loading", instantiate_from_dict=instantiate)
    _module(monkeypatch, "huggingface_hub", snapshot_download=download)
    _module(
        monkeypatch,
        "omegaconf",
        OmegaConf=SimpleNamespace(
            load=load_config,
            merge=lambda base, override: base | override,
            to_container=lambda value, resolve: value.copy(),
        ),
    )
    return SimpleNamespace(
        name=name, parent=tmp_path, folder=folder, manifest=manifest, events=events, model=model
    )


@pytest.mark.parametrize("local", [False, True])
def test_motion_loading_verifies_artifacts_and_disables_text_encoder(motion_loader, local):
    fixture = motion_loader
    result = loading.load_motion_model(fixture.name, fixture.parent if local else None)

    if local:
        assert result is fixture.model
        assert fixture.events == [
            (
                "load_model",
                (fixture.name,),
                {"device": "cpu", "text_encoder": False, "checkpoints_dir": str(fixture.parent)},
            )
        ]
    else:
        assert result == "evaluated model"
        assert fixture.events == [
            (
                "download",
                fixture.manifest["repo_id"],
                {"revision": "1" * 40, "allow_patterns": ["config.yaml", "stats/mean.npy"]},
            ),
            ("load_config", fixture.folder / "config.yaml"),
            (
                "instantiate",
                {"_target_": "trusted.ARDY", "text_encoder": None},
                {"overrides": {"device": "cpu"}},
            ),
            "eval",
        ]


@pytest.mark.parametrize(
    ("damage", "message"),
    [
        ("missing", "missing/unreadable"),
        ("short", "size mismatch"),
        ("changed", "SHA-256 mismatch"),
    ],
)
@pytest.mark.parametrize("local", [False, True])
def test_invalid_artifacts_never_reach_config_or_model_loading(
    motion_loader, damage, message, local
):
    fixture = motion_loader
    path = fixture.folder / "stats/mean.npy"
    if damage == "missing":
        path.unlink()
    elif damage == "short":
        path.write_bytes(b"short")
    else:
        path.write_bytes(b"X" * path.stat().st_size)

    with pytest.raises(ArdyError, match=message):
        loading.load_motion_model(fixture.name, fixture.parent if local else None)

    assert all(event[0] == "download" for event in fixture.events)


def test_unknown_motion_model_never_downloads(motion_loader):
    with pytest.raises(ArdyError, match="Core human skeleton"):
        loading.load_motion_model("ARDY-G1-RP-50FPS-Horizon52", None)
    assert motion_loader.events == []


@pytest.mark.parametrize("device", ["cpu", "mps", "cuda:1"])
def test_encoder_pins_base_and_adapters_and_merges_only_mntp(tmp_path, monkeypatch, device):
    events = []
    paths = {}
    settings = {"pooling_mode": "mean", "max_length": 512, "doc_max_length": 400}
    for label, repo in (("mntp", loading.MNTP_REPO), ("supervised", loading.SUPERVISED_REPO)):
        folder = tmp_path / label
        folder.mkdir()
        paths[repo] = str(folder)
        if label == "supervised":
            (folder / "llm2vec_config.json").write_text(json.dumps(settings))

    def download(repo, **kwargs):
        events.append(("download", repo, kwargs))
        return paths[repo]

    tokenizer = SimpleNamespace(eos_token="EOS", pad_token=None, padding_side="right")
    config = type("LlamaConfig", (), {})()

    def tokenizer_load(repo, **kwargs):
        events.append(("tokenizer", repo, kwargs))
        return tokenizer

    def config_load(repo, **kwargs):
        events.append(("config", repo, kwargs))
        return config

    base = SimpleNamespace(config=SimpleNamespace(_name_or_path="unused"))

    def base_load(repo, **kwargs):
        events.append(("base", repo, kwargs))
        return base

    merged = SimpleNamespace(config=base.config)
    supervised = SimpleNamespace(config=base.config)

    def merge():
        events.append("merge MNTP")
        return merged

    def adapter_load(model, path):
        events.append(("adapter", path))
        if path == paths[loading.MNTP_REPO]:
            assert model is base
            return SimpleNamespace(merge_and_unload=merge)
        assert model is merged
        return supervised

    class Encoder:
        @staticmethod
        def _get_model_class(name, bidirectional):
            assert name == "LlamaConfig" and bidirectional is True
            return SimpleNamespace(from_pretrained=base_load)

        def __init__(self, **kwargs):
            assert kwargs == {"model": supervised, "tokenizer": tokenizer, **settings}

        def to(self, selected, *, dtype):
            assert selected == device
            assert dtype == ("bf16" if device.startswith("cuda") else "f32")
            return self

        def eval(self):
            return self

    _module(monkeypatch, "torch", bfloat16="bf16", float32="f32")
    _module(monkeypatch, "ardy")
    _module(monkeypatch, "ardy.model")
    _module(monkeypatch, "ardy.model.llm2vec", LLM2Vec=Encoder)
    _module(monkeypatch, "huggingface_hub", snapshot_download=download)
    _module(monkeypatch, "peft", PeftModel=SimpleNamespace(from_pretrained=adapter_load))
    _module(
        monkeypatch,
        "transformers",
        AutoTokenizer=SimpleNamespace(from_pretrained=tokenizer_load),
        AutoConfig=SimpleNamespace(from_pretrained=config_load),
    )

    assert isinstance(loading.load_local_encoder(device), Encoder)
    assert tokenizer.pad_token == "EOS" and tokenizer.padding_side == "left"
    assert supervised.config._name_or_path == loading.BASE_REPO
    assert events == [
        (
            "download",
            repo,
            {"revision": revision, "allow_patterns": ["*.json", "*.safetensors"]},
        )
        for repo, revision in (
            (loading.MNTP_REPO, loading.MNTP_REVISION),
            (loading.SUPERVISED_REPO, loading.SUPERVISED_REVISION),
        )
    ] + [
        ("tokenizer", loading.BASE_REPO, {"revision": loading.BASE_REVISION}),
        ("config", loading.BASE_REPO, {"revision": loading.BASE_REVISION}),
        (
            "base",
            loading.BASE_REPO,
            {
                "revision": loading.BASE_REVISION,
                "torch_dtype": "bf16" if device.startswith("cuda") else "f32",
            },
        ),
        ("adapter", paths[loading.MNTP_REPO]),
        "merge MNTP",
        ("adapter", paths[loading.SUPERVISED_REPO]),
    ]

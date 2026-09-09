"""Sampling and ownership checks using small NumPy-backed fake tensors."""

from __future__ import annotations

import sys
import weakref
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

import vroom.ardy as runtime
from vroom.ardy import ArdyBackend, ArdyOptions
from vroom.ardy_constraints import ConstraintDeviation
from vroom.ardy_plan import ArdyPlan, PromptCue
from vroom.ardy_source import ArdySource
from vroom.errors import ArdyError


class _Tensor(np.ndarray):
    def __new__(cls, values):
        return np.asarray(values).view(cls)

    def to(self, *args, **kwargs):
        return self

    def cpu(self):
        return self

    def detach(self):
        return self

    def clone(self):
        return _Tensor(self.copy())

    def numpy(self):
        return np.asarray(self)

    def float(self):
        return _Tensor(self.astype(np.float32))


@pytest.fixture
def fake_modules(monkeypatch):
    seeded = []
    torch = SimpleNamespace(
        tensor=lambda value: _Tensor(np.array(value)),
        cat=lambda values, dim: _Tensor(np.concatenate(values, axis=dim)),
        ones=lambda shape, **kwargs: _Tensor(np.ones(shape, dtype=kwargs.get("dtype"))),
        arange=lambda count: _Tensor(np.arange(count)),
        isfinite=lambda value: _Tensor(np.isfinite(value)),
        bool=np.bool_,
        inference_mode=nullcontext,
        cuda=SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 2,
            device=lambda name: nullcontext(),
            empty_cache=lambda: None,
        ),
        mps=SimpleNamespace(empty_cache=lambda: None),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True)),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    for name in (
        "ardy",
        "ardy.model",
        "ardy.model.registry",
        "ardy.tools",
        "ardy.postprocess",
        "ardy.constraints",
    ):
        module = ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    sys.modules["ardy.model.registry"].resolve_model_name = lambda name: (
        "ARDY-Core-RP-20FPS-Horizon40"
    )
    sys.modules["ardy.tools"].seed_everything = seeded.append
    sys.modules["ardy.postprocess"].post_process_motion = lambda *args, **kwargs: {}
    sys.modules["ardy.constraints"].FullBodyConstraintSet = SimpleNamespace
    return SimpleNamespace(torch=torch, seeded=seeded, postprocess=sys.modules["ardy.postprocess"])


class _Representation:
    fps = 20
    motion_rep_dim = 3

    def __init__(self):
        self.conditions = []
        self.decoded = []

    def __call__(self, rotations, roots, *, to_normalize):
        assert to_normalize
        return roots.clone()

    def create_conditions(self, indices, values, frames, normalize, device):
        assert normalize and device == "cpu"
        self.conditions.append((indices, values, frames))
        return _Tensor(np.zeros((frames, 3))), _Tensor(np.zeros((frames, 3)))

    def inverse(self, motion, *, is_normalized):
        assert is_normalized
        self.decoded.append(motion)
        return {"features": motion}


class _Model:
    num_frames_per_token = 4
    gen_horizon_len = 40
    device = "cpu"

    def __init__(self):
        self.motion_rep = _Representation()
        self.diffusion = SimpleNamespace(num_base_steps=10)
        self.skeleton = SimpleNamespace(
            fk=lambda local, roots: (local, roots, None), cpu=lambda: None
        )
        self.calls = []

    def autoregressive_step(self, **kwargs):
        self.calls.append(kwargs)
        history = kwargs["init_history_sequence"]
        cursor = 0 if history is None else int(history[0, -1, 0]) + 1
        label = float(kwargs["text_feat"].reshape(-1)[0])
        chunk = np.zeros((1, self.gen_horizon_len, 3), dtype=np.float32)
        chunk[0, :, 0] = np.arange(cursor, cursor + self.gen_horizon_len)
        chunk[0, :, 1] = label
        if history is not None:
            chunk = np.concatenate((history, chunk), axis=1)
        # Match upstream: conditioning may include future frames, but returned
        # explicit motion contains history plus one generation horizon only.
        return _Tensor(chunk)


def _targets(*frames):
    return SimpleNamespace(
        indices={"root_2d": [_Tensor([frame]) for frame in frames]},
        values={"root_2d": [_Tensor([[frame, 0]]) for frame in frames]},
        goals=tuple(SimpleNamespace(frame=frame) for frame in frames),
        correction=(),
    )


def _prefix(frames=7, fps=20):
    roots = np.zeros((frames, 3), dtype=np.float32)
    roots[:, 0] = np.arange(frames)
    roots[:, 1] = -5
    return ArdySource(
        np.broadcast_to(np.eye(3, dtype=np.float32), (frames, 27, 3, 3)).copy(),
        roots,
        fps,
    )


@pytest.mark.parametrize("horizon", [8, 40])
def test_long_sampling_bounds_windows_shifts_targets_and_keeps_every_frame(
    fake_modules, horizon
) -> None:
    model = _Model()
    model.gen_horizon_len = horizon
    backend = ArdyBackend()
    plan = ArdyPlan(22.45, (PromptCue(0, "Walk."),))
    targets = _targets(0, 39, 40, 180, 400)
    output = backend._sample(plan, model, (_Tensor([[[11]]]),), targets, 450, 8, None)

    np.testing.assert_array_equal(output["features"][0, :, 0], np.arange(450))
    assert output["features"].shape == (1, 450, 3)
    assert all(call["num_frames"] <= 200 and call["num_frames"] % 4 == 0 for call in model.calls)
    assert all(call["num_denoising_steps"] == 8 for call in model.calls)
    assert all(call["cfg_weight"] == (2, 2) for call in model.calls)
    seen_goals = set()
    cursor = 0
    for call, (indices, values, window) in zip(
        model.calls, model.motion_rep.conditions, strict=True
    ):
        history = call["init_history_sequence"]
        count = 0 if history is None else history.shape[1]
        assert count <= 200 - horizon and count % 4 == 0
        for shifted, value in zip(
            indices.get("root_2d", []), values.get("root_2d", []), strict=True
        ):
            original = int(value[0, 0])
            assert shifted[0] == original - (cursor - count)
            assert count <= shifted[0] < window
            seen_goals.add(original)
        cursor += min(horizon, 450 - cursor)
    assert seen_goals == {0, 39, 40, 180, 400}
    assert [int(value[0]) for value in targets.indices["root_2d"]] == [0, 39, 40, 180, 400]


def test_joint_conditioning_shifts_only_frame_column(fake_modules) -> None:
    model = _Model()
    original = _Tensor([[340, 16]])
    targets = SimpleNamespace(
        indices={"global_joints_positions": [original]},
        values={"global_joints_positions": [_Tensor([[1, 2, 3]])]},
        goals=(SimpleNamespace(frame=340),),
    )
    ArdyBackend()._sample(
        ArdyPlan(20, (PromptCue(0, "Walk."),)),
        model,
        (_Tensor([[[1]]]),),
        targets,
        401,
        10,
        None,
    )
    shifted = [
        index
        for indices, _, _ in model.motion_rep.conditions
        for index in indices.get("global_joints_positions", [])
    ]
    assert shifted
    assert any(index[0, 0] != 340 for index in shifted)
    assert all(index[0, 1] == 16 for index in shifted)
    np.testing.assert_array_equal(original, [[340, 16]])


def test_prompt_changes_apply_at_exact_frame_without_losing_samples(fake_modules) -> None:
    model = _Model()
    plan = ArdyPlan(1, (PromptCue(0, "Walk."), PromptCue(0.4, "Wave."), PromptCue(0.65, "Stop.")))
    output = ArdyBackend()._sample(
        plan,
        model,
        tuple(_Tensor([[[value]]]) for value in (1, 2, 3)),
        _targets(),
        21,
        10,
        None,
    )
    np.testing.assert_array_equal(output["features"][0, :, 0], np.arange(21))
    np.testing.assert_array_equal(output["features"][0, :, 1], [1] * 8 + [2] * 5 + [3] * 8)
    assert all(call["num_frames"] % 4 == 0 for call in model.calls)


def test_prefix_preserves_history_and_selects_current_prompt(fake_modules) -> None:
    model = _Model()
    plan = ArdyPlan(1, (PromptCue(0, "Old."), PromptCue(0.2, "New.")))
    prefix = _prefix()
    output = ArdyBackend()._sample(
        plan,
        model,
        (_Tensor([[[1]]]), _Tensor([[[2]]])),
        _targets(2, 10),
        21,
        10,
        prefix,
    )
    np.testing.assert_array_equal(output["features"][0, :7], prefix.root_positions)
    np.testing.assert_array_equal(output["features"][0, :, 0], np.arange(21))
    np.testing.assert_array_equal(output["features"][0, 7:, 1], [2] * 14)
    np.testing.assert_array_equal(
        model.calls[0]["init_history_sequence"][0], prefix.root_positions[-4:]
    )
    assert [value[0, 0] for value in model.motion_rep.conditions[0][1]["root_2d"]] == [10]


@pytest.mark.parametrize("bad", ["short", "nan", "batch", "features"])
def test_malformed_samples_stop_before_motion_decode(fake_modules, bad) -> None:
    model = _Model()
    original = model.autoregressive_step

    def malformed(**kwargs):
        chunk = original(**kwargs)
        if bad == "short":
            return chunk[:, :2]
        if bad == "nan":
            chunk[0, 0, 0] = np.nan
            return chunk
        if bad == "batch":
            return _Tensor(np.repeat(chunk, 2, axis=0))
        return _Tensor(np.repeat(chunk, 2, axis=2))

    model.autoregressive_step = malformed
    with pytest.raises(ArdyError, match=r"malformed|shape|non-finite"):
        ArdyBackend()._sample(
            ArdyPlan(1, (PromptCue(0, "Walk."),)),
            model,
            (_Tensor([[[1]]]),),
            _targets(),
            21,
            10,
            None,
        )
    assert not model.motion_rep.decoded


def _decoded_motion(frames):
    return {
        "local_rot_mats": _Tensor(
            np.broadcast_to(np.eye(3, dtype=np.float32), (1, frames, 27, 3, 3)).copy()
        ),
        "root_positions": _Tensor(np.zeros((1, frames, 3), dtype=np.float32)),
        "foot_contacts": _Tensor(np.zeros((1, frames, 4), dtype=np.float32)),
    }


@pytest.fixture
def prepared_runtime(monkeypatch, fake_modules):
    monkeypatch.setattr(runtime, "build_targets", lambda *args, **kwargs: _targets())
    monkeypatch.setattr(runtime, "measure_targets", lambda *args: ())
    monkeypatch.setattr(runtime, "skeleton_from_ardy", lambda skeleton: "converted skeleton")
    monkeypatch.setattr(
        runtime, "motion_from_ardy_output", lambda *args, **kwargs: "converted clip"
    )
    return fake_modules


def test_seed_variants_restore_exact_prefix_after_correction_before_measurement(
    monkeypatch,
    prepared_runtime,
) -> None:
    model = _Model()
    backend = ArdyBackend()
    prefix = _prefix()
    targets = _targets()
    user_constraint = object()
    targets.correction = (user_constraint,)
    monkeypatch.setattr(runtime, "build_targets", lambda *args, **kwargs: targets)
    monkeypatch.setattr(runtime, "read_source", lambda *args, **kwargs: prefix)
    monkeypatch.setattr(backend, "_sample", lambda *args: _decoded_motion(21))
    joint_offsets = np.arange(81).reshape(27, 3) / 100
    world_basis = np.asarray([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    model.skeleton.fk = lambda local, roots: (
        _Tensor(local @ world_basis),
        _Tensor(roots[..., None, :] + joint_offsets),
        None,
    )
    supplied = []

    def correct(local, roots, contacts, skeleton, *, constraint_lst):
        assert len(constraint_lst) == 2
        assert constraint_lst[0] is user_constraint
        anchor = constraint_lst[1]
        assert anchor.skeleton is model.skeleton
        np.testing.assert_array_equal(anchor.frame_indices, np.arange(prefix.frame_count))
        np.testing.assert_array_equal(
            anchor.global_joints_positions, prefix.root_positions[:, None, :] + joint_offsets
        )
        np.testing.assert_array_equal(
            anchor.global_joints_rots, prefix.local_rot_mats @ world_basis
        )
        supplied.append(anchor)
        return {"root_positions": _Tensor(roots + 100)}

    prepared_runtime.postprocess.post_process_motion = correct
    measured = []

    def measure(targets, output, model):
        measured.append(output["root_positions"][0].copy())
        return ()

    monkeypatch.setattr(runtime, "measure_targets", measure)
    results = backend._generate_loaded(
        ArdyPlan(1, (PromptCue(0, "Continue."),), seeds=(7, 8), continuation="prefix.npz"),
        model,
        (_Tensor([[[1]]]),),
    )
    assert prepared_runtime.seeded == [7, 8]
    assert supplied[0] is supplied[1]
    assert targets.correction == (user_constraint,)
    assert [result.seed for result in results] == [7, 8]
    for roots, result in zip(measured, results, strict=True):
        np.testing.assert_array_equal(roots[:7], prefix.root_positions)
        np.testing.assert_array_equal(result.source.root_positions[:7], prefix.root_positions)
        np.testing.assert_array_equal(result.source.root_positions[7:], np.full((14, 3), 100))


def test_disabled_correction_skips_prefix_anchor_but_restores_exact_prefix(
    monkeypatch, prepared_runtime
) -> None:
    backend = ArdyBackend(ArdyOptions(postprocess=False))
    prefix = _prefix()
    monkeypatch.setattr(runtime, "read_source", lambda *args, **kwargs: prefix)
    monkeypatch.setattr(backend, "_sample", lambda *args: _decoded_motion(21))
    monkeypatch.setattr(
        sys.modules["ardy.constraints"],
        "FullBodyConstraintSet",
        lambda **kwargs: pytest.fail("disabled correction built a prefix anchor"),
    )
    prepared_runtime.postprocess.post_process_motion = lambda *args, **kwargs: pytest.fail(
        "disabled correction ran"
    )
    (result,) = backend._generate_loaded(
        ArdyPlan(1, (PromptCue(0, "Continue."),), continuation="prefix.npz"),
        _Model(),
        (_Tensor([[[1]]]),),
    )
    np.testing.assert_array_equal(result.source.local_rot_mats[:7], prefix.local_rot_mats)
    np.testing.assert_array_equal(result.source.root_positions[:7], prefix.root_positions)


@pytest.mark.parametrize("frames,fps", [(3, 20), (21, 20), (7, 30)])
def test_invalid_continuation_stops_before_sampling(
    monkeypatch, prepared_runtime, frames, fps
) -> None:
    backend = ArdyBackend()
    monkeypatch.setattr(runtime, "read_source", lambda *args, **kwargs: _prefix(frames, fps))
    monkeypatch.setattr(backend, "_sample", lambda *args: pytest.fail("invalid prefix was sampled"))
    with pytest.raises(ArdyError, match="continuation"):
        backend._generate_loaded(
            ArdyPlan(1, (PromptCue(0, "Continue."),), continuation="prefix.npz"),
            _Model(),
            (_Tensor([[[1]]]),),
        )
    assert not prepared_runtime.seeded


@pytest.mark.parametrize("bad", ["rotation", "contact-count", "contact-nan", "contact-range"])
def test_invalid_decoded_motion_never_reaches_correction(
    monkeypatch, prepared_runtime, bad
) -> None:
    backend = ArdyBackend()
    output = _decoded_motion(21)
    if bad == "rotation":
        output["local_rot_mats"][0, 0, 0, 0, 0] = -1
    elif bad == "contact-count":
        output["foot_contacts"] = output["foot_contacts"][:, :2]
    else:
        output["foot_contacts"][0, 0, 0] = np.nan if bad == "contact-nan" else 2
    monkeypatch.setattr(backend, "_sample", lambda *args: output)
    prepared_runtime.postprocess.post_process_motion = lambda *args, **kwargs: pytest.fail(
        "invalid motion reached C++ correction"
    )
    with pytest.raises(ArdyError, match=r"rotation|contact"):
        backend._generate_loaded(
            ArdyPlan(1, (PromptCue(0, "Walk."),)), _Model(), (_Tensor([[[1]]]),)
        )


def test_strict_failure_stops_remaining_seed_variants(monkeypatch, prepared_runtime) -> None:
    backend = ArdyBackend(ArdyOptions(postprocess=False))
    monkeypatch.setattr(backend, "_sample", lambda *args: _decoded_motion(21))
    monkeypatch.setattr(
        runtime, "measure_targets", lambda *args: (ConstraintDeviation(0, 1, "root", 1, 0.05),)
    )
    with pytest.raises(ArdyError, match="no output written"):
        backend._generate_loaded(
            ArdyPlan(1, (PromptCue(0, "Walk."),), seeds=(7, 8), strict=True),
            _Model(),
            (_Tensor([[[1]]]),),
        )
    assert prepared_runtime.seeded == [7]


@pytest.fixture
def lifecycle(monkeypatch, prepared_runtime):
    events = []
    references = {}

    class Model(_Model):
        def float(self):
            events.append("float")

        def to(self, device):
            events.append(f"model to {device}")

    class Encoder:
        pass

    def load_model(*args):
        model = Model()
        references["model"] = weakref.ref(model)
        return model

    def load_encoder(device):
        events.append(f"encoder on {device}")
        encoder = Encoder()
        references["encoder"] = weakref.ref(encoder)
        return encoder

    monkeypatch.setattr(runtime, "load_motion_model", load_model)
    monkeypatch.setattr(runtime, "load_local_encoder", load_encoder)
    empty_cache = ArdyBackend._empty_cache
    monkeypatch.setattr(
        ArdyBackend,
        "_empty_cache",
        staticmethod(lambda torch, device: events.append(f"clear {device}")),
    )
    return SimpleNamespace(
        events=events,
        references=references,
        empty_cache=empty_cache,
        torch=prepared_runtime.torch,
    )


def test_success_releases_encoder_before_motion_and_model_after_return(
    monkeypatch, lifecycle
) -> None:
    backend = ArdyBackend(ArdyOptions(device="mps", text_device="cuda:1"))
    monkeypatch.setattr(backend, "_encode_prompts", lambda *args: (_Tensor([[[1]]]),))

    def generate(*args, **kwargs):
        assert lifecycle.references["encoder"]() is None
        return ()

    monkeypatch.setattr(backend, "_generate_loaded", generate)
    assert backend.generate_plan(ArdyPlan(1, (PromptCue(0, "Walk."),))) == ()
    assert lifecycle.references["model"]() is None
    assert lifecycle.events.index("float") < lifecycle.events.index("model to mps")
    assert "clear cuda:1" in lifecycle.events
    assert "clear mps" in lifecycle.events


@pytest.mark.parametrize("failure", [RuntimeError("encoder failed"), KeyboardInterrupt()])
def test_encoding_failure_clears_both_device_caches(monkeypatch, lifecycle, failure) -> None:
    backend = ArdyBackend(ArdyOptions(device="mps", text_device="cuda:1"))

    def encode(*args):
        raise failure

    monkeypatch.setattr(backend, "_encode_prompts", encode)
    error = KeyboardInterrupt if isinstance(failure, KeyboardInterrupt) else ArdyError
    with pytest.raises(error):
        backend.generate_plan(ArdyPlan(1, (PromptCue(0, "Walk."),)))
    assert "model to mps" not in lifecycle.events
    assert "clear cuda:1" in lifecycle.events
    assert "clear mps" in lifecycle.events


def test_missing_continuation_fails_before_text_encoder_load(monkeypatch, lifecycle) -> None:
    def missing(*args, **kwargs):
        raise ArdyError("missing continuation")

    monkeypatch.setattr(runtime, "read_source", missing)
    with pytest.raises(ArdyError, match="missing continuation"):
        ArdyBackend().generate_plan(
            ArdyPlan(1, (PromptCue(0, "Continue."),), continuation="missing.npz"),
        )
    assert "encoder" not in lifecycle.references


def test_partial_model_load_failure_clears_caches_without_loading_encoder(
    monkeypatch, lifecycle
) -> None:
    def missing(*args, **kwargs):
        raise ArdyError("missing motion checkpoint")

    monkeypatch.setattr(runtime, "load_motion_model", missing)
    with pytest.raises(ArdyError, match="missing motion checkpoint"):
        ArdyBackend(ArdyOptions(device="mps", text_device="cuda:1")).generate_plan(
            ArdyPlan(1, (PromptCue(0, "Walk."),)),
        )
    assert "encoder" not in lifecycle.references
    assert "clear cuda:1" in lifecycle.events
    assert "clear mps" in lifecycle.events


def test_cache_cleanup_failure_does_not_mask_cancellation(monkeypatch, lifecycle) -> None:
    backend = ArdyBackend(ArdyOptions(device="mps"))

    def interrupt(*args):
        raise KeyboardInterrupt

    def broken_cache():
        raise RuntimeError("device cleanup failed")

    monkeypatch.setattr(backend, "_encode_prompts", interrupt)
    monkeypatch.setattr(backend, "_empty_cache", lifecycle.empty_cache)
    monkeypatch.setattr(lifecycle.torch.mps, "empty_cache", broken_cache)
    with pytest.raises(KeyboardInterrupt):
        backend.generate_plan(ArdyPlan(1, (PromptCue(0, "Walk."),)))

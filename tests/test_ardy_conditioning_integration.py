"""Opt-in CPU contract checks with upstream ARDY, without model weights or inference.

Run VROOM_TEST_ARDY_RUNTIME=1 ARDY_ROOT=/path/to/ardy .venv-ardy/bin/python
-m pytest tests/test_ardy_conditioning_integration.py. Normal tests collect this
module without importing Torch, NumPy, or upstream ARDY.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from vroom.ardy_constraints import build_targets, frame_at, measure_targets
from vroom.ardy_plan import ArdyPlan, JointTarget, PoseTarget, PromptCue, RootTarget
from vroom.errors import ArdyError

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("VROOM_TEST_ARDY_RUNTIME") != "1",
        reason="set VROOM_TEST_ARDY_RUNTIME=1 for no-weight ARDY CPU contract checks",
    ),
]


@pytest.fixture
def runtime(tmp_path: Path, monkeypatch) -> SimpleNamespace:
    root = Path(os.environ.get("ARDY_ROOT", Path(__file__).parents[1] / "vendor" / "ARDY"))
    assert (root / "ardy" / "skeleton" / "definitions.py").is_file(), (
        "enabled ARDY runtime tests require the trusted ARDY_ROOT source checkout"
    )
    monkeypatch.syspath_prepend(str(root))
    import numpy as np
    import torch
    from ardy.motion_rep import ArdyMotionRep
    from ardy.skeleton import CoreSkeleton27

    skeleton = CoreSkeleton27().to(device="cpu", dtype=torch.float32)
    # Stored statistics cover 5 global-root + 4 local-root + 325 body features.
    # Nonzero means/stds test real masking after upstream normalization.
    stats = tmp_path / "synthetic_stats"
    stats.mkdir()
    np.save(stats / "mean.npy", np.full(334, 0.25, dtype=np.float32))
    np.save(stats / "std.npy", np.full(334, 2.0, dtype=np.float32))
    model = SimpleNamespace(
        skeleton=skeleton,
        motion_rep=ArdyMotionRep(skeleton, fps=20, stats_path=str(stats)),
        num_frames_per_token=4,
        diffusion=SimpleNamespace(num_base_steps=10),
    )
    return SimpleNamespace(np=np, torch=torch, model=model)


def _plan(*targets, strict: bool = False) -> ArdyPlan:
    return ArdyPlan(
        duration=1,
        prompts=(PromptCue(0, "A person reaches forward."),),
        constraints=targets,
        strict=strict,
    )


def _condition(runtime, targets):
    rep = runtime.model.motion_rep
    normalized, mask = rep.create_conditions(
        targets.indices, targets.values, length=21, to_normalize=True, device="cpu"
    )
    assert normalized.device.type == "cpu"
    assert runtime.torch.isfinite(normalized).all()
    assert runtime.torch.count_nonzero(normalized[mask == 0]) == 0
    return rep.unnormalize(normalized), mask


def _motion(runtime, *, frames: int = 21):
    torch = runtime.torch
    local = torch.eye(3).repeat(1, frames, 27, 1, 1)
    roots = torch.zeros(1, frames, 3)
    roots[..., 1] = 1
    return local, roots


def _output(runtime, local, roots):
    rotations, positions, _ = runtime.model.skeleton.fk(local, roots)
    return {"global_rot_mats": rotations, "posed_joints": positions, "root_positions": roots}


def _pose_file(runtime, tmp_path: Path) -> Path:
    np = runtime.np
    rotations = np.broadcast_to(np.eye(3, dtype=np.float32), (2, 27, 3, 3)).copy()
    # Source frame 1 rotates the left forearm 90 degrees around +Z.
    rotations[1, 15] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    roots = np.asarray([[0, 1, 0], [1, 2, 3]], dtype=np.float32)
    path = tmp_path / "reference.npz"
    np.savez(path, local_rot_mats=rotations, root_positions=roots, fps=np.asarray(20.0))
    return path


def test_real_root_and_heading_conditioning_preserves_sparse_frame_masks(runtime) -> None:
    targets = build_targets(
        _plan(RootTarget(0.5, (2, -3), math.pi / 2)), runtime.model, max_bytes=1_000_000
    )
    values, mask = _condition(runtime, targets)
    assert values[10, :5].tolist() == pytest.approx([2, 0.25, -3, 0, 1])
    assert mask[10, :5].tolist() == [1, 0, 1, 1, 1]
    assert mask.sum().item() == 4
    assert targets.correction[0].name == "root2d"


def test_real_full_pose_reference_conditions_all_joints_at_requested_time(
    runtime, tmp_path
) -> None:
    path = _pose_file(runtime, tmp_path)
    targets = build_targets(_plan(PoseTarget(0.5, path, 0.05)), runtime.model, max_bytes=1_000_000)
    values, mask = _condition(runtime, targets)
    rep = runtime.model.motion_rep
    assert values[10, :5].tolist() == pytest.approx([1, 2, 3, 1, 0])
    position_values = values[10, rep.slice_dict["local_joints_positions"]].reshape(26, 3)
    forearm = runtime.model.skeleton.neutral_joints[15]
    hand = runtime.model.skeleton.neutral_joints[16]
    offset = hand - forearm
    expected_hand = [forearm[0] - offset[1], 2 + forearm[1] + offset[0], hand[2]]
    # Float32 FK adds a 3 m root then conditioning subtracts it again.
    assert position_values[15].tolist() == pytest.approx(
        [float(v) for v in expected_hand], abs=1e-6
    )
    rotation_mask = mask[10, rep.slice_dict["global_rot_data"]].reshape(27, 6)
    assert rotation_mask.sum().item() == 27 * 6
    assert mask.sum().item() == 5 + 26 * 3 + 27 * 6
    assert targets.correction[0].name == "fullbody"


def test_real_subset_pose_maps_vrm_semantics_and_supplies_required_hips(runtime, tmp_path) -> None:
    path = _pose_file(runtime, tmp_path)
    target = PoseTarget(0.5, path, 0.05, joints=("leftHand", "leftFoot"))
    targets = build_targets(_plan(target), runtime.model, max_bytes=1_000_000)
    _, mask = _condition(runtime, targets)
    rep = runtime.model.motion_rep
    positions = mask[10, rep.slice_dict["local_joints_positions"]].reshape(26, 3)
    rotations = mask[10, rep.slice_dict["global_rot_data"]].reshape(27, 6)
    assert positions.sum().item() == 6
    assert rotations.sum().item() == 12
    assert positions[15].all() and positions[24].all()
    assert rotations[16].all() and rotations[25].all()
    assert {item.name for item in targets.correction} == {"left-hand", "left-foot"}


def test_real_arbitrary_joint_position_and_rotation_use_world_coordinates(runtime) -> None:
    target = JointTarget(
        0.5,
        "leftLowerArm",
        (1, 1.5, -2),
        math.pi / 2,
        position=(2, 3, 4),
        rotation=(0, 0, math.sqrt(0.5), math.sqrt(0.5)),
    )
    targets = build_targets(_plan(target), runtime.model, max_bytes=1_000_000)
    values, mask = _condition(runtime, targets)
    rep = runtime.model.motion_rep
    assert values[10, :5].tolist() == pytest.approx([1, 1.5, -2, 0, 1])
    positions = values[10, rep.slice_dict["local_joints_positions"]].reshape(26, 3)
    assert positions[14].tolist() == pytest.approx([1, 3, 6])
    from ardy.geometry import cont6d_to_matrix

    encoded = values[10, rep.slice_dict["global_rot_data"]].reshape(27, 6)[15]
    actual = cont6d_to_matrix(encoded).numpy()
    runtime.np.testing.assert_allclose(actual, [[0, -1, 0], [1, 0, 0], [0, 0, 1]], atol=1e-6)
    assert mask.sum().item() == 5 + 3 + 6


def test_real_rotation_only_target_does_not_pin_joint_positions(runtime) -> None:
    target = JointTarget(0.5, "LeftHandThumb1", (0, 1, 0), 0, rotation=(0, 0, 0, 1))
    targets = build_targets(_plan(target), runtime.model, max_bytes=1_000_000)
    _, mask = _condition(runtime, targets)
    rep = runtime.model.motion_rep
    assert mask[:, rep.slice_dict["local_joints_positions"]].sum().item() == 0
    assert mask.sum().item() == 5 + 6


@pytest.mark.parametrize(
    "targets",
    [
        (RootTarget(0.5, (0, 0)), RootTarget(0.5, (1, 0))),
        (RootTarget(0.5, (0, 0), 0), RootTarget(0.5, (0, 0), math.pi)),
        (JointTarget(0.5, "hips", (0, 1, 0), 0, position=(0, 2, 0)),),
        (
            JointTarget(0.5, "leftHand", (0, 1, 0), 0, position=(0, 1, 1)),
            JointTarget(0.5, "LeftHand", (0, 1, 0), 0, position=(0, 1, 2)),
        ),
    ],
)
def test_conflicting_targets_fail_before_upstream_conditioning(runtime, targets) -> None:
    with pytest.raises(ArdyError, match="conflicting"):
        build_targets(_plan(*targets), runtime.model, max_bytes=1_000_000)


def test_equivalent_quaternion_signs_do_not_conflict(runtime) -> None:
    positive = JointTarget(0.5, "leftHand", (0, 1, 0), 0, rotation=(0, 0, 0, 1))
    negative = JointTarget(0.5, "LeftHand", (0, 1, 0), 0, rotation=(0, 0, 0, -1))
    targets = build_targets(_plan(positive, negative), runtime.model, max_bytes=1_000_000)
    _, mask = _condition(runtime, targets)
    assert mask.sum().item() == 11


def test_measured_root_heading_wraps_at_pi_and_exposes_tolerance_failures(runtime) -> None:
    target = RootTarget(
        0.5, (0, 0), math.pi - 0.01, position_tolerance=0.04, rotation_tolerance=0.03
    )
    targets = build_targets(_plan(target, strict=True), runtime.model, max_bytes=1_000_000)
    local, roots = _motion(runtime)
    roots[0, 10, 0] = 0.05
    angle = -math.pi + 0.01
    c, s = math.cos(angle), math.sin(angle)
    local[0, 10, 0] = runtime.torch.tensor([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    measured = measure_targets(targets, _output(runtime, local, roots), runtime.model)
    by_component = {deviation.component: deviation for deviation in measured}
    assert by_component["root_2d"].error == pytest.approx(0.05)
    assert not by_component["root_2d"].satisfied
    assert by_component["global_root_heading"].error == pytest.approx(0.02, abs=1e-6)
    assert by_component["global_root_heading"].satisfied
    assert all(deviation.target == 0 and deviation.time == 0.5 for deviation in measured)


def test_measured_joint_position_and_rotation_use_fk_and_angular_distance(runtime) -> None:
    local, roots = _motion(runtime)
    baseline = _output(runtime, local, roots)
    position = tuple(baseline["posed_joints"][0, 10, 15].tolist())
    target = JointTarget(
        0.5,
        "leftLowerArm",
        (0, 1, 0),
        0,
        position=position,
        rotation=(0, 0, 0, 1),
        position_tolerance=0.02,
        rotation_tolerance=0.1,
    )
    targets = build_targets(_plan(target), runtime.model, max_bytes=1_000_000)
    angle = 0.2
    c, s = math.cos(angle), math.sin(angle)
    local[0, 10, 15] = runtime.torch.tensor([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    measured = measure_targets(targets, _output(runtime, local, roots), runtime.model)
    by_component = {deviation.component: deviation for deviation in measured}
    assert by_component["global_joints_positions"].error == pytest.approx(0, abs=1e-6)
    assert by_component["global_joints_positions"].satisfied
    assert by_component["global_joints_rots"].error == pytest.approx(0.2, abs=1e-6)
    assert not by_component["global_joints_rots"].satisfied


def test_pose_source_time_and_model_frame_grid_are_enforced(runtime, tmp_path) -> None:
    path = _pose_file(runtime, tmp_path)
    with pytest.raises(ArdyError, match="frame grid"):
        build_targets(_plan(PoseTarget(0.51, path, 0)), runtime.model, max_bytes=1_000_000)
    with pytest.raises(ArdyError, match="frame grid"):
        build_targets(_plan(PoseTarget(0.5, path, 0.01)), runtime.model, max_bytes=1_000_000)
    with pytest.raises(ArdyError, match="outside"):
        build_targets(_plan(PoseTarget(0.5, path, 0.1)), runtime.model, max_bytes=1_000_000)
    assert frame_at(0.5, 20, "time") == 10


@pytest.mark.parametrize("strict", [False, True])
def test_loaded_backend_enforces_strict_diagnostics_on_injected_motion(
    runtime, monkeypatch, strict
):
    """Exercise real conditioning/measurement; only denoising is substituted."""
    from vroom.ardy import ArdyBackend, ArdyOptions

    target = RootTarget(1, (2, 0), position_tolerance=0.01)
    plan = _plan(target, strict=strict)
    local, roots = _motion(runtime)
    motion = _output(runtime, local, roots)
    motion["local_rot_mats"] = local
    motion["foot_contacts"] = runtime.torch.zeros(1, 21, 4, dtype=runtime.torch.bool)
    backend = ArdyBackend(ArdyOptions(device="cpu", postprocess=False))
    called = []

    def sample(plan, model, embeddings, targets, frames, steps, prefix):
        _condition(runtime, targets)
        called.append((frames, steps))
        return motion

    monkeypatch.setattr(backend, "_sample", sample)
    if strict:
        with pytest.raises(ArdyError, match=r"target 0 at 1s misses root_2d.*no output written"):
            backend._generate_loaded(plan, runtime.model, ())
    else:
        results = backend._generate_loaded(plan, runtime.model, ())
        assert len(results) == 1
        assert results[0].clip.frame_count == 21
        assert results[0].deviations[0].error == pytest.approx(2)
        assert not results[0].deviations[0].satisfied
    assert called == [(21, 10)]


def test_hips_target_rejects_conflicting_root_context(runtime) -> None:
    target = JointTarget(0, "hips", (0, 1, 0), 0, position=(1, 1, 0))
    with pytest.raises(ArdyError, match="conflicting global_joints_positions"):
        build_targets(_plan(target), runtime.model, max_bytes=1_000_000)


def test_llama_encoder_retains_bidirectional_attention() -> None:
    """Catch Transformers upgrades that silently restore a causal mask."""
    import torch
    from ardy.model.llm2vec.models.bidirectional_llama import LlamaBiModel
    from transformers import LlamaConfig

    torch.manual_seed(42)
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
    )
    model = LlamaBiModel(config).eval()
    with torch.no_grad():
        original = model(torch.tensor([[1, 2, 3]]), use_cache=False).last_hidden_state
        changed = model(torch.tensor([[1, 2, 4]]), use_cache=False).last_hidden_state
    assert torch.isfinite(original).all()
    assert not torch.allclose(original[0, 0], changed[0, 0])

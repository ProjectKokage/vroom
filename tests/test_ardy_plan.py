from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

import vroom.ardy_plan as plans
from vroom.ardy_plan import ArdyPlan, JointTarget, PoseTarget, PromptCue, RootTarget, load_plan
from vroom.errors import ArdyError


def _write_plan(tmp_path: Path, **changes: object) -> Path:
    data = {"duration": 4, "prompts": [{"time": 0, "text": "Walk forward."}], **changes}
    path = tmp_path / "motion.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_load_plan_preserves_targets_and_resolves_native_sources(tmp_path: Path) -> None:
    path = _write_plan(
        tmp_path,
        prompts=[{"time": 0, "text": "Walk."}, {"time": 2, "text": "Wave."}],
        seeds=[0, 2**32 - 1],
        continuation="prefix.npz",
        strict=True,
        constraints=[
            {"type": "root", "time": 2, "position": [1, 2], "heading": 0.5},
            {
                "type": "pose",
                "time": 3,
                "source": "poses/stand.npz",
                "sourceTime": 1,
                "joints": ["LeftHand", "Hips"],
                "positionTolerance": 0.02,
            },
            {
                "type": "joint",
                "time": 4,
                "joint": "LeftHand",
                "rootPosition": [0, 1, 0],
                "rootHeading": 0,
                "position": [0.5, 1.5, 0.2],
                "rotation": [0, 0, 0, 1],
                "rotationTolerance": 0.1,
            },
        ],
    )

    plan = load_plan(path)

    assert plan == ArdyPlan(
        duration=4,
        prompts=(PromptCue(0, "Walk."), PromptCue(2, "Wave.")),
        seeds=(0, 2**32 - 1),
        continuation=tmp_path / "prefix.npz",
        strict=True,
        constraints=(
            RootTarget(2, (1, 2), heading=0.5),
            PoseTarget(
                3,
                tmp_path / "poses/stand.npz",
                1,
                joints=("LeftHand", "Hips"),
                position_tolerance=0.02,
            ),
            JointTarget(
                4,
                "LeftHand",
                (0, 1, 0),
                0,
                position=(0.5, 1.5, 0.2),
                rotation=(0, 0, 0, 1),
                rotation_tolerance=0.1,
            ),
        ),
    )


def test_plan_copies_mutable_constructor_inputs() -> None:
    position = [1, 2]
    joints = ["Head"]
    cues = [PromptCue(0, "Stand.")]
    targets = [RootTarget(1, position), PoseTarget(1, Path("pose.npz"), 0, joints=joints)]
    seeds = [7]
    plan = ArdyPlan(1, cues, seeds=seeds, constraints=targets)

    position[0] = 99
    joints[0] = "Hips"
    cues.clear()
    targets.clear()
    seeds.clear()

    assert plan.prompts == (PromptCue(0, "Stand."),)
    assert plan.constraints[0].position == (1, 2)
    assert plan.constraints[1].joints == ("Head",)
    assert plan.seeds == (7,)
    with pytest.raises(FrozenInstanceError):
        plan.constraints[0].heading = 1


@pytest.mark.parametrize(
    "changes",
    [
        {"unknown": True},
        {"duration": True},
        {"duration": 0},
        {"duration": float("inf")},
        {"prompts": []},
        {"prompts": {}},
        {"prompts": [{"time": 1, "text": "Walk."}]},
        {"prompts": [{"time": 0, "text": " "}]},
        {"prompts": [{"time": 0, "text": "Walk.", "typo": 1}]},
        {"prompts": [{"time": 0, "text": "Walk."}, {"time": 0, "text": "Run."}]},
        {"prompts": [{"time": 0, "text": "Walk."}, {"time": 5, "text": "Run."}]},
        {"seeds": []},
        {"seeds": [1, 1]},
        {"seeds": [True]},
        {"seeds": [2**32]},
        {"seeds": list(range(17))},
        {"strict": "true"},
        {"continuation": "pose.vrma"},
        {"continuation": []},
        {"constraints": {}},
        {"constraints": [{"type": "fullbody", "time": 0}]},
        {"constraints": [{"type": "root", "time": 0}]},
        {"constraints": [{"type": "root", "time": 5, "position": [0, 0]}]},
        {"constraints": [{"type": "root", "time": 1, "position": [0, float("nan")]}]},
        {"constraints": [{"type": "root", "time": 1, "position": [0, 0], "typo": 1}]},
        {"constraints": [{"type": "pose", "time": 1, "source": "a.npz"}]},
        {"constraints": [{"type": "joint", "time": 1, "joint": "LeftHand"}]},
    ],
)
def test_json_plan_rejects_invalid_authoring_data(tmp_path: Path, changes: dict) -> None:
    with pytest.raises(ArdyError):
        load_plan(_write_plan(tmp_path, **changes))


@pytest.mark.parametrize(
    "make_value",
    [
        lambda: PromptCue(True, "Walk."),
        lambda: PromptCue(-1, "Walk."),
        lambda: PromptCue(0, "\ud800"),
        lambda: RootTarget(0, (1, 2, 3)),
        lambda: RootTarget(0, (1, True)),
        lambda: RootTarget(0, (1, 10**1000)),
        lambda: RootTarget(0, (1, 2), heading=float("inf")),
        lambda: RootTarget(0, (1, 2), position_tolerance=-0.1),
        lambda: PoseTarget(0, Path("pose.vrma"), 0),
        lambda: PoseTarget(0, Path("pose.npz"), -1),
        lambda: PoseTarget(0, Path("pose.npz"), 0, joints="Head"),
        lambda: PoseTarget(0, Path("pose.npz"), 0, joints=[]),
        lambda: PoseTarget(0, Path("pose.npz"), 0, joints=["Head", "Head"]),
        lambda: JointTarget(0, "Head", (0, 1, 0), 0),
        lambda: JointTarget(0, "Head", (0, 1, 0), 0, rotation=(0, 0, 0, 2)),
        lambda: JointTarget(0, "Head", (0, 1, 0), True, position=(0, 1, 2)),
        lambda: ArdyPlan(1, (PromptCue(0, "Walk."),), constraints=({"time": 0},)),
        lambda: ArdyPlan(1, (PromptCue(0, "Walk."),), seeds="42"),
    ],
)
def test_python_constructors_validate_boundaries(make_value) -> None:
    with pytest.raises(ArdyError):
        make_value()


@pytest.mark.parametrize(
    "payload",
    [
        b'{"duration":1,"duration":2,"prompts":[]}',
        b'{"duration":1,"prompts":[{"time":0,"text":"a","text":"b"}]}',
        b'{"duration":',
        b"\xff",
        b"[" * 2000,
    ],
)
def test_load_plan_reports_malformed_json_cleanly(tmp_path: Path, payload: bytes) -> None:
    path = tmp_path / "motion.json"
    path.write_bytes(payload)
    with pytest.raises(ArdyError):
        load_plan(path)


def test_load_plan_bounds_read_before_json_decode(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(plans, "MAX_PLAN_BYTES", 32)
    path = tmp_path / "motion.json"
    path.write_bytes(b" " * 33)

    def reject_decode(*args, **kwargs):
        pytest.fail("oversized input reached JSON decoding")

    monkeypatch.setattr(plans.json, "loads", reject_decode)
    with pytest.raises(ArdyError, match="byte"):
        load_plan(path)


def test_python_plan_bounds_total_text_and_items(monkeypatch) -> None:
    cues = (PromptCue(0, "12345"), PromptCue(1, "67890"))
    monkeypatch.setattr(plans, "MAX_PLAN_BYTES", 8)
    with pytest.raises(ArdyError, match="byte budget"):
        ArdyPlan(1, cues)
    monkeypatch.setattr(plans, "MAX_PLAN_ITEMS", 1)
    with pytest.raises(ArdyError, match="item budget"):
        ArdyPlan(1, (PromptCue(0, "Go."),), constraints=(RootTarget(1, (0, 0)),))


def test_endpoint_time_is_accepted_without_model_grid_rounding(tmp_path: Path) -> None:
    plan = load_plan(
        _write_plan(
            tmp_path,
            duration=1.023,
            prompts=[{"time": 0, "text": "Stand."}, {"time": 1.023, "text": "Wave."}],
            constraints=[{"type": "root", "time": 1.023, "position": [0, 0]}],
        )
    )
    assert plan.prompts[-1].time == plan.constraints[0].time == plan.duration == 1.023

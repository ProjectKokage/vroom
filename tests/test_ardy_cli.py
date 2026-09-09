from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import vroom.cli as cli
from vroom.ardy import ArdyGeneration
from vroom.ardy_constraints import ConstraintDeviation
from vroom.ardy_plan import ArdyPlan, PromptCue
from vroom.ardy_source import ArdySource, read_source
from vroom.errors import ArdyError
from vroom.motion import Bone, MotionClip, Skeleton
from vroom.vrm import HUMANOID_PARENTS, REQUIRED_HUMANOID_BONES
from vroom.vrma import read_vrma, validate_vrma


@pytest.fixture
def generation() -> ArdyGeneration:
    indices = {}
    bones = []
    positions = []
    for name in HUMANOID_PARENTS:
        if name not in REQUIRED_HUMANOID_BONES:
            continue
        parent = HUMANOID_PARENTS[name]
        while parent is not None and parent not in indices:
            parent = HUMANOID_PARENTS[parent]
        parent_index = None if parent is None else indices[parent]
        translation = (0.0, 1.0 if parent is None else 0.1, 0.0)
        indices[name] = len(bones)
        bones.append(Bone(name, parent_index, translation))
        positions.append(
            translation if parent_index is None else (0.0, positions[parent_index][1] + 0.1, 0.0)
        )
    skeleton = Skeleton(tuple(bones), tuple(positions), 0.0)
    clip = MotionClip(
        fps=20,
        hips_translations=((0.0, 1.0, 0.0), (0.0, 1.0, 0.05)),
        rotations={name: ((0.0, 0.0, 0.0, 1.0),) * 2 for name in indices},
        name="Synthetic motion",
    )
    source = ArdySource(
        np.broadcast_to(np.eye(3, dtype=np.float32), (2, 27, 3, 3)),
        np.asarray(clip.hips_translations, dtype=np.float32),
        20,
    )
    return ArdyGeneration(clip, skeleton, source, (), 42)


@pytest.fixture
def fake_ardy(monkeypatch, generation):
    calls = []

    class Backend:
        def __init__(self, options):
            self.options = options

        def generate_plan(self, plan):
            calls.append((self.options, plan))
            return tuple(replace(generation, seed=seed) for seed in plan.seeds)

    monkeypatch.setattr(cli, "ArdyBackend", Backend)
    return calls


def _plan_file(tmp_path: Path, **changes) -> Path:
    data = {
        "duration": 3,
        "prompts": [{"time": 0, "text": "Walk."}, {"time": 2, "text": "Stop."}],
        **changes,
    }
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_default_cli_generates_ardy_with_documented_defaults(tmp_path, fake_ardy, capsys) -> None:
    output = tmp_path / "motion.vrma"

    assert cli.main(["generate", "Wave.", "--output", str(output)]) == 0

    options, plan = fake_ardy[0]
    assert options.model == "core"
    assert options.cfg_scale == options.constraint_cfg_scale == 2
    assert options.device == options.text_device == "auto"
    assert options.postprocess
    assert not options.ground
    assert plan == ArdyPlan(4, (PromptCue(0, "Wave."),))
    assert read_vrma(output).frame_count == 2
    assert validate_vrma(output).ok
    assert "Wrote" in capsys.readouterr().out


def test_simple_cli_passes_native_authoring_options(tmp_path, fake_ardy) -> None:
    output = tmp_path / "motion.vrma"
    continuation = tmp_path / "prefix.npz"
    checkpoints = tmp_path / "models"
    assert (
        cli.main(
            [
                "generate",
                "Wave.",
                "--output",
                str(output),
                "--duration",
                "7.5",
                "--seed",
                "9",
                "--model",
                "core8",
                "--checkpoints-dir",
                str(checkpoints),
                "--device",
                "mps",
                "--text-device",
                "cpu",
                "--cfg-scale",
                "3",
                "--constraint-cfg-scale",
                "4",
                "--steps",
                "8",
                "--no-postprocess",
                "--ground",
                "--motion-memory-mib",
                "512",
                "--continue-from",
                str(continuation),
                "--strict",
            ]
        )
        == 0
    )

    options, plan = fake_ardy[0]
    assert plan.duration == 7.5
    assert plan.seeds == (9,)
    assert plan.continuation == continuation
    assert plan.strict
    assert options.model == "core8"
    assert options.checkpoints_dir == checkpoints
    assert options.device == "mps"
    assert options.text_device == "cpu"
    assert options.cfg_scale == 3
    assert options.constraint_cfg_scale == 4
    assert options.steps == 8
    assert not options.postprocess
    assert options.ground
    assert options.motion_memory_mib == 512


@pytest.mark.parametrize(("saved_strict", "flag"), [(False, False), (False, True), (True, False)])
def test_json_plan_preserves_timeline_sources_and_strictness(
    tmp_path,
    fake_ardy,
    saved_strict: bool,
    flag: bool,
) -> None:
    plan_path = _plan_file(
        tmp_path,
        seeds=[7],
        strict=saved_strict,
        continuation="prefix.npz",
        constraints=[{"type": "pose", "time": 3, "source": "pose.npz", "sourceTime": 1}],
    )
    args = ["generate", "--plan", str(plan_path), "--output", str(tmp_path / "motion.vrma")]
    assert cli.main(args + (["--strict"] if flag else [])) == 0

    _, plan = fake_ardy[0]
    assert plan.prompts == (PromptCue(0, "Walk."), PromptCue(2, "Stop."))
    assert plan.duration == 3
    assert plan.seeds == (7,)
    assert plan.continuation == tmp_path / "prefix.npz"
    assert plan.constraints[0].source == tmp_path / "pose.npz"
    assert plan.strict == (saved_strict or flag)


def test_multiple_candidates_write_vrma_source_sidecars_and_one_report(
    tmp_path,
    monkeypatch,
    generation,
    capsys,
) -> None:
    measurements = (
        ConstraintDeviation(0, 1.0, "root position", 0.03, 0.05),
        ConstraintDeviation(1, 2.0, "heading", 0.2, 0.1),
    )

    class Backend:
        def __init__(self, options):
            pass

        def generate_plan(self, plan):
            return tuple(
                replace(generation, seed=seed, deviations=measurements) for seed in plan.seeds
            )

    monkeypatch.setattr(cli, "ArdyBackend", Backend)
    report = tmp_path / "metrics" / "report.json"
    assert (
        cli.main(
            [
                "generate",
                "Private prompt.",
                "--seeds",
                "7",
                "8",
                "--output",
                str(tmp_path / "motion.vrma"),
                "--source-output",
                str(tmp_path / "source.npz"),
                "--report",
                str(report),
            ]
        )
        == 0
    )

    for seed in (7, 8):
        assert validate_vrma(tmp_path / f"motion-{seed}.vrma").ok
        source = read_source(tmp_path / f"source-{seed}.npz", max_bytes=128 * 1024)
        assert source.frame_count == 2
    assert not (tmp_path / "motion.vrma").exists()
    assert not (tmp_path / "source.npz").exists()
    data = json.loads(report.read_text())
    assert [candidate["seed"] for candidate in data["candidates"]] == [7, 8]
    assert data["candidates"][0]["measurements"][1]["error"] == 0.2
    assert "before optional export grounding" in data["coordinateSpace"]
    assert "Private prompt" not in report.read_text()
    assert "1 of 2 constraint measurements exceed tolerance" in capsys.readouterr().out


def test_single_candidate_keeps_exact_output_and_source_names(tmp_path, fake_ardy) -> None:
    output = tmp_path / "motion.vrma"
    source = tmp_path / "source.npz"
    assert (
        cli.main(
            [
                "generate",
                "Stand.",
                "--seed",
                "7",
                "--output",
                str(output),
                "--source-output",
                str(source),
            ]
        )
        == 0
    )
    assert output.exists()
    assert read_source(source, max_bytes=128 * 1024).frame_count == 2
    assert not (tmp_path / "motion-7.vrma").exists()
    assert not (tmp_path / "source-7.npz").exists()


@pytest.mark.parametrize(
    "collision", ["output-report", "source-report", "continuation", "pose", "plan"]
)
def test_sidecar_and_source_collisions_fail_before_runtime(
    tmp_path,
    fake_ardy,
    capsys,
    collision: str,
) -> None:
    output = tmp_path / "motion.vrma"
    extras = []
    positional = ["Wave."]
    if collision == "output-report":
        extras = ["--report", str(output)]
    elif collision == "source-report":
        extras = [
            "--seeds",
            "7",
            "8",
            "--source-output",
            str(tmp_path / "source.npz"),
            "--report",
            str(tmp_path / "source-7.npz"),
        ]
    elif collision == "continuation":
        source = tmp_path / "source.npz"
        source.write_bytes(b"preserved original")
        extras = ["--continue-from", str(source), "--source-output", str(source)]
    else:
        plan = _plan_file(
            tmp_path,
            constraints=[{"type": "pose", "time": 2, "source": "pose.npz", "sourceTime": 0}],
        )
        positional = []
        extras = ["--plan", str(plan)]
        extras += (
            ["--source-output", str(tmp_path / "pose.npz")]
            if collision == "pose"
            else ["--report", str(plan)]
        )
    existing = {path: path.read_bytes() for path in tmp_path.iterdir() if path.is_file()}

    assert cli.main(["generate", *positional, "--output", str(output), *extras]) == 2

    assert not fake_ardy
    assert "output paths must be distinct" in capsys.readouterr().err
    assert not output.exists()
    assert all(path.read_bytes() == contents for path, contents in existing.items())


def test_symlink_report_cannot_overwrite_plan_input(tmp_path, fake_ardy, capsys) -> None:
    plan = _plan_file(tmp_path)
    original = plan.read_bytes()
    alias = tmp_path / "report.json"
    alias.symlink_to(plan.name)
    assert (
        cli.main(
            [
                "generate",
                "--plan",
                str(plan),
                "--output",
                str(tmp_path / "motion.vrma"),
                "--report",
                str(alias),
            ]
        )
        == 2
    )
    assert not fake_ardy
    assert plan.read_bytes() == original
    assert "must not overwrite" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--seed", "7", "--seeds", "8"], "choose --seed or --seeds"),
        (["--model-path", "missing"], "--backend hy-motion"),
        (["--hy-motion-root", "missing"], "--backend hy-motion"),
        (["--validation-steps", "1"], "--backend hy-motion"),
        (["--text-models", "local"], "HF_HUB_OFFLINE"),
        (["--ground", "--no-ground"], "choose --ground or --no-ground"),
        (["--source-output", "source.vrma"], ".npz extension"),
        (["--duration", "0"], "duration"),
        (["--steps", "0"], "steps"),
        (["--motion-memory-mib", "0"], "motion_memory_mib"),
        (["--cfg-scale", "nan"], "cfg_scale"),
        (["--constraint-cfg-scale", "-1"], "constraint_cfg_scale"),
    ],
)
def test_bad_ardy_options_fail_cleanly_before_runtime(
    tmp_path, fake_ardy, capsys, extra, message
) -> None:
    output = tmp_path / "motion.vrma"
    assert cli.main(["generate", "Wave.", "--output", str(output), *extra]) == 2
    assert not fake_ardy
    assert message in capsys.readouterr().err
    assert not output.exists()


@pytest.mark.parametrize(
    "extra",
    [
        ["Wave."],
        ["--duration", "2"],
        ["--seed", "1"],
        ["--seeds", "1", "2"],
        ["--continue-from", "a.npz"],
    ],
)
def test_json_plan_rejects_conflicting_simple_inputs(tmp_path, fake_ardy, capsys, extra) -> None:
    plan = _plan_file(tmp_path)
    assert (
        cli.main(
            [
                "generate",
                "--plan",
                str(plan),
                "--output",
                str(tmp_path / "motion.vrma"),
                *extra,
            ]
        )
        == 2
    )
    assert not fake_ardy
    assert (
        "put prompt, duration, seeds, and continuation in the motion plan"
        in capsys.readouterr().err
    )


def test_missing_prompt_or_plan_is_actionable(tmp_path, fake_ardy, capsys) -> None:
    assert cli.main(["generate", "--output", str(tmp_path / "motion.vrma")]) == 2
    assert not fake_ardy
    assert "provide a motion prompt or --plan" in capsys.readouterr().err


@pytest.mark.parametrize(
    "extra",
    [
        ["--plan", "plan.json"],
        ["--continue-from", "source.npz"],
        ["--source-output", "source.npz"],
        ["--report", "report.json"],
        ["--strict"],
        ["--steps", "4"],
        ["--no-postprocess"],
        ["--ground"],
        ["--model", "core8"],
        ["--checkpoints-dir", "models"],
        ["--constraint-cfg-scale", "3"],
        ["--motion-memory-mib", "128"],
    ],
)
def test_hy_rejects_ardy_options_before_loading_model(tmp_path, monkeypatch, capsys, extra) -> None:
    monkeypatch.setattr(
        cli,
        "find_hy_motion_root",
        lambda *args: pytest.fail("incompatible options reached HY"),
    )
    assert (
        cli.main(
            [
                "generate",
                "Wave.",
                "--backend",
                "hy-motion",
                "--output",
                str(tmp_path / "motion.vrma"),
                *extra,
            ]
        )
        == 2
    )
    assert "require --backend ardy" in capsys.readouterr().err


@pytest.mark.parametrize("configured", [None, "/configured/hy-model"])
def test_explicit_hy_preserves_default_checkpoint_and_generation_options(
    tmp_path,
    monkeypatch,
    generation,
    configured,
) -> None:
    if configured is None:
        monkeypatch.delenv("HY_MOTION_MODEL_PATH", raising=False)
    else:
        monkeypatch.setenv("HY_MOTION_MODEL_PATH", configured)
    calls = []

    class Backend:
        def __init__(self, options):
            self.options = options

        def generate_variants(self, prompt, *, duration, seeds):
            calls.append((self.options, prompt, duration, seeds))
            return (generation.clip,)

    monkeypatch.setattr(cli, "HyMotionBackend", Backend)
    monkeypatch.setattr(cli, "find_hy_motion_root", lambda path: tmp_path / "hy")
    monkeypatch.setattr(cli, "load_hy_motion_skeleton", lambda path: generation.skeleton)
    assert (
        cli.main(
            [
                "generate",
                "Wave.",
                "--backend",
                "hy-motion",
                "--output",
                str(tmp_path / "motion.vrma"),
            ]
        )
        == 0
    )

    options, prompt, duration, seeds = calls[0]
    assert options.model_path == Path(
        configured or "vendor/HY-Motion-1.0/ckpts/tencent/HY-Motion-1.0"
    )
    assert options.hy_motion_root == tmp_path / "hy"
    assert options.cfg_scale == 5
    assert options.ground
    assert options.device == options.text_device == options.text_models == "auto"
    assert prompt == "Wave."
    assert duration == 4
    assert seeds == (42,)


def test_explicit_hy_passes_original_backend_options(tmp_path, monkeypatch, generation) -> None:
    calls = []
    root = tmp_path / "hy"

    class Backend:
        def __init__(self, options):
            self.options = options

        def generate_variants(self, prompt, *, duration, seeds):
            calls.append((self.options, prompt, duration, seeds))
            return (generation.clip,)

    monkeypatch.setattr(cli, "HyMotionBackend", Backend)
    monkeypatch.setattr(cli, "find_hy_motion_root", lambda path: path)
    monkeypatch.setattr(cli, "load_hy_motion_skeleton", lambda path: generation.skeleton)
    assert (
        cli.main(
            [
                "generate",
                "Wave.",
                "--backend",
                "hy-motion",
                "--output",
                str(tmp_path / "motion.vrma"),
                "--model-path",
                str(tmp_path / "weights"),
                "--hy-motion-root",
                str(root),
                "--duration",
                "2.5",
                "--seed",
                "7",
                "--device",
                "cuda:1",
                "--text-device",
                "cpu",
                "--text-models",
                "local",
                "--cfg-scale",
                "6",
                "--validation-steps",
                "4",
                "--no-ground",
            ]
        )
        == 0
    )

    options, _, duration, seeds = calls[0]
    assert options.model_path == tmp_path / "weights"
    assert options.hy_motion_root == root
    assert options.device == "cuda:1"
    assert options.text_device == "cpu"
    assert options.text_models == "local"
    assert options.cfg_scale == 6
    assert options.validation_steps == 4
    assert not options.ground
    assert duration == 2.5
    assert seeds == (7,)


@pytest.mark.parametrize("failure", [KeyboardInterrupt(), ArdyError("target 0 exceeds tolerance")])
def test_generation_failure_and_interrupt_leave_no_output(
    tmp_path, monkeypatch, capsys, failure
) -> None:
    class Backend:
        def __init__(self, options):
            pass

        def generate_plan(self, plan):
            raise failure

    monkeypatch.setattr(cli, "ArdyBackend", Backend)
    expected = 130 if isinstance(failure, KeyboardInterrupt) else 2
    output = tmp_path / "motion.vrma"
    assert cli.main(["generate", "Wave.", "--output", str(output)]) == expected
    assert not output.exists()
    assert ("cancelled" if expected == 130 else "exceeds tolerance") in capsys.readouterr().err

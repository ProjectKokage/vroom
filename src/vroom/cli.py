"""Command-line interface for generation, curation, inspection, and validation."""

from __future__ import annotations

import argparse
import json
import os
import sys
import zipfile
from collections.abc import Sequence
from dataclasses import asdict, replace
from pathlib import Path

from .ardy import ArdyBackend, ArdyOptions
from .ardy_plan import ArdyPlan, PromptCue, load_plan
from .ardy_source import write_source
from .curation import (
    HIPS_TRANSLATION_MODES,
    MAX_ROTATION_GAIN,
    ROTATION_PROJECTIONS,
    BoneCurationPolicy,
    HipsTranslationPolicy,
    write_curated_vrma,
)
from .errors import ArdyError, VrmaError, VroomError
from .glb import Glb
from .hy_motion import HyMotionBackend, HyMotionOptions, find_hy_motion_root
from .motion import MotionClip, load_hy_motion_skeleton
from .vrm import VrmAvatar
from .vrma import VRMA_EXTENSION, build_vrma, read_vrma, validate_vrma


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vroom",
        description="Generate portable VRM Animation (.vrma) with ARDY or HY-Motion.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="generate one VRMA clip from text")
    generate.add_argument("prompt", nargs="?", help="English description of the desired motion")
    generate.add_argument("--backend", choices=("ardy", "hy-motion"), default="ardy")
    generate.add_argument("--plan", type=Path, help="ARDY JSON motion plan")
    generate.add_argument("--model", default="core", help="ARDY Core checkpoint: core40 or core8")
    generate.add_argument("--checkpoints-dir", type=Path, help="local ARDY checkpoint directories")
    generate.add_argument(
        "--continue-from", type=Path, help="preserve and extend an ARDY source NPZ"
    )
    generate.add_argument(
        "--source-output", type=Path, help="save reusable ARDY source motion as NPZ"
    )
    generate.add_argument("--report", type=Path, help="save ARDY constraint measurements as JSON")
    generate.add_argument(
        "--strict", action="store_true", help="fail on missed ARDY target tolerances"
    )
    generate.add_argument("--seeds", type=int, nargs="+", help="generate one candidate per seed")
    generate.add_argument(
        "--steps", type=int, help="ARDY denoising steps within checkpoint schedule"
    )
    generate.add_argument("--constraint-cfg-scale", type=float, default=2.0)
    generate.add_argument(
        "--no-postprocess", action="store_true", help="disable ARDY motion correction"
    )
    generate.add_argument(
        "--ground", action="store_true", help="ground ARDY export using source joints"
    )
    generate.add_argument(
        "--motion-memory-mib",
        type=int,
        default=256,
        help="ARDY working-data budget, excluding model weights (default: 256)",
    )
    generate.add_argument("-o", "--output", type=Path, required=True, help="output .vrma path")
    generate.add_argument(
        "--model-path",
        type=Path,
        help="directory containing HY-Motion config.yml and latest.ckpt",
    )
    generate.add_argument(
        "--hy-motion-root",
        type=Path,
        help="HY-Motion submodule root (or set VROOM_HY_MOTION_ROOT)",
    )
    generate.add_argument("--duration", type=float, help="duration in seconds (default: 4)")
    generate.add_argument("--seed", type=int, help="random seed (default: 42)")
    generate.add_argument(
        "--device",
        default="auto",
        help="motion device: auto, cpu, mps, cuda, or cuda:N",
    )
    generate.add_argument(
        "--text-device",
        default="auto",
        help="text encoder device: auto, same, cpu, mps, cuda, or cuda:N",
    )
    generate.add_argument(
        "--text-models",
        choices=("auto", "local", "hugging-face"),
        default="auto",
        help="where Qwen and CLIP weights are loaded from",
    )
    generate.add_argument("--cfg-scale", type=float, help="text guidance (ARDY: 2; HY-Motion: 5)")
    generate.add_argument(
        "--validation-steps",
        type=int,
        help="override the checkpoint's ODE inference steps",
    )
    generate.add_argument(
        "--no-ground",
        action="store_true",
        help="preserve HY-Motion's raw vertical root position",
    )

    curate = subparsers.add_parser(
        "curate",
        help="make a bounded clip from sampled LINEAR VRMA",
    )
    curate.add_argument("input", type=Path, help="source .vrma path")
    curate.add_argument("-o", "--output", type=Path, required=True, help="output .vrma path")
    curate.add_argument(
        "--bones",
        nargs="+",
        required=True,
        metavar="BONE",
        help="exact humanoid rotation bones (space- or comma-separated)",
    )
    curate.add_argument(
        "--duration",
        type=float,
        required=True,
        help="main action duration; relative-return adds its hold and tail",
    )
    curate.add_argument("--fade-in", type=float, required=True, help="smoothstep fade-in duration")
    curate.add_argument(
        "--fade-out", type=float, required=True, help="smoothstep fade-out duration"
    )
    curate.add_argument(
        "--rotation-gain",
        type=float,
        default=1.0,
        help=(f"relative rotation multiplier from 0 to {MAX_ROTATION_GAIN:g} (default: 1)"),
    )
    curate.add_argument(
        "--rotation-projection",
        choices=ROTATION_PROJECTIONS,
        default="none",
        help="project each local rotation before gain and fades (default: none)",
    )
    curate.add_argument(
        "--bone-rotation-gain",
        action="append",
        default=[],
        metavar="BONE=GAIN",
        help="override rotation gain for one selected bone; repeat as needed",
    )
    curate.add_argument(
        "--bone-rotation-projection",
        action="append",
        default=[],
        metavar="BONE=PROJECTION",
        help="override rotation projection for one selected bone; repeat as needed",
    )
    curate.add_argument(
        "--hips-translation",
        choices=HIPS_TRANSLATION_MODES,
        default="strip",
        help="strip root motion or preserve it with a rest-to-rest return tail",
    )
    curate.add_argument(
        "--hips-return-duration",
        type=float,
        default=0.0,
        help="return-tail duration required by --hips-translation relative-return",
    )
    curate.add_argument("--start-frame", type=int, default=0, help="first source frame")
    curate.add_argument(
        "--end-frame",
        type=int,
        help="last source frame, inclusive (defaults to the final frame)",
    )
    curate.add_argument("--name", help="output animation name")

    validate = subparsers.add_parser("validate", help="validate .vrma files or a motion-pack ZIP")
    validate.add_argument("paths", type=Path, nargs="+")
    validate.add_argument("--quiet", action="store_true", help="only print invalid files")

    inspect = subparsers.add_parser("inspect", help="summarize a VRM or VRMA GLB")
    inspect.add_argument("path", type=Path)
    inspect.add_argument("--json", action="store_true", dest="as_json")
    return parser


def _generate(args: argparse.Namespace) -> int:
    if args.output.suffix.lower() != ".vrma":
        raise VrmaError("output path must end in .vrma")
    if args.backend == "ardy":
        return _generate_ardy(args)
    if (
        args.plan
        or args.continue_from
        or args.source_output
        or args.report
        or args.strict
        or args.checkpoints_dir
        or args.steps is not None
        or args.no_postprocess
        or args.ground
        or args.model != "core"
        or args.constraint_cfg_scale != 2.0
        or args.motion_memory_mib != 256
    ):
        raise ArdyError("ARDY authoring options require --backend ardy")
    if not args.prompt:
        raise VrmaError("HY-Motion generation requires a prompt")
    if args.seed is not None and args.seeds is not None:
        raise VrmaError("choose --seed or --seeds")
    hy_root = find_hy_motion_root(args.hy_motion_root)
    options = HyMotionOptions(
        model_path=args.model_path
        or Path(
            os.environ.get(
                "HY_MOTION_MODEL_PATH", "vendor/HY-Motion-1.0/ckpts/tencent/HY-Motion-1.0"
            )
        ),
        hy_motion_root=hy_root,
        device=args.device,
        text_device=args.text_device,
        text_models=args.text_models,
        cfg_scale=5.0 if args.cfg_scale is None else args.cfg_scale,
        validation_steps=args.validation_steps,
        ground=not args.no_ground,
    )
    clips = HyMotionBackend(options).generate_variants(
        args.prompt,
        duration=4.0 if args.duration is None else args.duration,
        seeds=args.seeds or (42 if args.seed is None else args.seed,),
    )
    skeleton = load_hy_motion_skeleton(hy_root)
    seeds = args.seeds or (42 if args.seed is None else args.seed,)
    for clip, seed in zip(clips, seeds, strict=True):
        output = _candidate_path(args.output, seed, len(clips))
        glb = build_vrma(clip, skeleton)
        validate_vrma(glb).require_valid()
        glb.write(output)
        _print_written(output, clip)
    return 0


def _candidate_path(path: Path, seed: int, count: int) -> Path:
    return path if count == 1 else path.with_name(f"{path.stem}-{seed}{path.suffix}")


def _print_written(path: Path, clip: MotionClip) -> None:
    print(f"Wrote {path} ({clip.frame_count} frames, {clip.duration:.3f}s at {clip.fps:g} fps)")


def _generate_ardy(args: argparse.Namespace) -> int:
    if args.model_path or args.hy_motion_root or args.validation_steps is not None:
        raise ArdyError("HY-Motion checkpoint/ODE options require --backend hy-motion")
    if args.text_models != "auto":
        raise ArdyError(
            "ARDY uses its local Llama encoder; use HF_HUB_OFFLINE=1 for cached-only loading"
        )
    if args.ground and args.no_ground:
        raise ArdyError("choose --ground or --no-ground")
    if args.plan:
        if (
            args.prompt is not None
            or args.duration is not None
            or args.seed is not None
            or args.seeds is not None
            or args.continue_from is not None
        ):
            raise ArdyError("put prompt, duration, seeds, and continuation in the motion plan")
        plan = load_plan(args.plan)
        if args.strict:
            plan = replace(plan, strict=True)
    else:
        if not args.prompt:
            raise ArdyError("provide a motion prompt or --plan")
        if args.seed is not None and args.seeds is not None:
            raise ArdyError("choose --seed or --seeds")
        plan = ArdyPlan(
            4.0 if args.duration is None else args.duration,
            (PromptCue(0, args.prompt),),
            seeds=tuple(args.seeds or (42 if args.seed is None else args.seed,)),
            continuation=args.continue_from,
            strict=args.strict,
        )
    if args.source_output and args.source_output.suffix.lower() != ".npz":
        raise ArdyError("source output must use the .npz extension")
    outputs = [_candidate_path(args.output, seed, len(plan.seeds)) for seed in plan.seeds]
    if args.source_output:
        outputs.extend(
            _candidate_path(args.source_output, seed, len(plan.seeds)) for seed in plan.seeds
        )
    if args.report:
        outputs.append(args.report)
    inputs = [value for value in (args.plan, plan.continuation) if value is not None]
    inputs.extend(target.source for target in plan.constraints if hasattr(target, "source"))
    resolved = [path.resolve() for path in outputs]
    if len(set(resolved)) != len(resolved) or set(resolved) & {path.resolve() for path in inputs}:
        raise ArdyError("output paths must be distinct and must not overwrite plan/source inputs")
    options = ArdyOptions(
        model=args.model,
        checkpoints_dir=args.checkpoints_dir,
        device=args.device,
        text_device=args.text_device,
        cfg_scale=2.0 if args.cfg_scale is None else args.cfg_scale,
        constraint_cfg_scale=args.constraint_cfg_scale,
        steps=args.steps,
        postprocess=not args.no_postprocess,
        ground=args.ground,
        motion_memory_mib=args.motion_memory_mib,
    )
    results = ArdyBackend(options).generate_plan(plan)
    documents = [build_vrma(result.clip, result.skeleton) for result in results]
    for document in documents:
        validate_vrma(document).require_valid()
    for result, document in zip(results, documents, strict=True):
        path = _candidate_path(args.output, result.seed, len(results))
        document.write(path)
        _print_written(path, result.clip)
        if args.source_output:
            write_source(
                _candidate_path(args.source_output, result.seed, len(results)),
                result.source,
                model=args.model,
                max_bytes=args.motion_memory_mib * 1024 * 1024,
            )
        if result.deviations:
            failed = [item for item in result.deviations if not item.satisfied]
            print(
                f"Seed {result.seed}: {len(failed)} of {len(result.deviations)} "
                "constraint measurements exceed tolerance"
            )
            for item in failed:
                joint = "" if item.joint is None else f" {item.joint}"
                print(
                    f"  target {item.target} at {item.time:g}s{joint} {item.component}: "
                    f"{item.error:.5g} (tolerance {item.tolerance:.5g})"
                )
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(
                {
                    "coordinateSpace": (
                        "ARDY Core source, before optional export grounding and avatar retargeting"
                    ),
                    "candidates": [
                        {
                            "seed": result.seed,
                            "measurements": [asdict(item) for item in result.deviations],
                        }
                        for result in results
                    ],
                },
                indent=2,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
    return 0


def _curate(args: argparse.Namespace) -> int:
    bones = tuple(
        bone
        for argument in args.bones
        for bone in (part.strip() for part in argument.split(","))
        if bone
    )
    bone_policies = _parse_bone_policies(
        bones,
        gain_arguments=args.bone_rotation_gain,
        projection_arguments=args.bone_rotation_projection,
    )
    hips_translation_policy = HipsTranslationPolicy(
        mode=args.hips_translation,
        return_duration=args.hips_return_duration,
    )
    write_curated_vrma(
        args.input,
        args.output,
        bones=bones,
        duration=args.duration,
        fade_in=args.fade_in,
        fade_out=args.fade_out,
        rotation_gain=args.rotation_gain,
        rotation_projection=args.rotation_projection,
        bone_policies=bone_policies,
        hips_translation_policy=hips_translation_policy,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        name=args.name,
    )
    curated = read_vrma(args.output)
    translation_label = (
        "" if hips_translation_policy.mode == "strip" else ", first-relative hips translation"
    )
    print(
        f"Wrote {args.output} "
        f"({curated.frame_count} frames, {curated.duration:.3f}s, "
        f"{len(bones)} rotation tracks{translation_label})"
    )
    return 0


def _parse_bone_policies(
    selected_bones: tuple[str, ...],
    *,
    gain_arguments: Sequence[str],
    projection_arguments: Sequence[str],
) -> tuple[BoneCurationPolicy, ...]:
    gains = _parse_bone_assignments(gain_arguments, label="bone rotation gain")
    projections = _parse_bone_assignments(
        projection_arguments,
        label="bone rotation projection",
    )
    policy_bones = set(gains) | set(projections)
    ordered_policy_bones = tuple(bone for bone in selected_bones if bone in policy_bones) + tuple(
        sorted(policy_bones - set(selected_bones))
    )
    parsed_gains: dict[str, float] = {}
    for bone, value in gains.items():
        try:
            parsed_gains[bone] = float(value)
        except ValueError as exc:
            raise VrmaError(f"{bone} rotation gain must be a number") from exc
    return tuple(
        BoneCurationPolicy(
            bone=bone,
            rotation_gain=parsed_gains.get(bone),
            rotation_projection=projections.get(bone),
        )
        for bone in ordered_policy_bones
    )


def _parse_bone_assignments(
    arguments: Sequence[str],
    *,
    label: str,
) -> dict[str, str]:
    assignments: dict[str, str] = {}
    for argument in arguments:
        bone, separator, value = argument.partition("=")
        bone = bone.strip()
        value = value.strip()
        if not separator or not bone or not value:
            raise VrmaError(f"{label} must use BONE=VALUE")
        if bone in assignments:
            raise VrmaError(f"{label} repeats {bone}")
        assignments[bone] = value
    return assignments


def _print_validation(label: str, report: object, quiet: bool) -> bool:
    errors = report.errors
    warnings = report.warnings
    if not quiet or errors:
        status = "valid" if not errors else "invalid"
        print(f"{label}: {status}")
        for message in errors:
            print(f"  error: {message}")
        for message in warnings:
            print(f"  warning: {message}")
    return not errors


def _validate(args: argparse.Namespace) -> int:
    all_valid = True
    for path in args.paths:
        if path.suffix.lower() != ".zip":
            report = validate_vrma(path)
            all_valid = _print_validation(str(path), report, args.quiet) and all_valid
            continue
        try:
            with zipfile.ZipFile(path) as archive:
                members = sorted(
                    name for name in archive.namelist() if name.lower().endswith(".vrma")
                )
                if not members:
                    print(f"{path}: invalid\n  error: ZIP contains no .vrma files")
                    all_valid = False
                for member in members:
                    report = validate_vrma(archive.read(member))
                    label = f"{path}!{member}"
                    all_valid = _print_validation(label, report, args.quiet) and all_valid
        except (OSError, zipfile.BadZipFile) as exc:
            print(f"{path}: invalid\n  error: {exc}")
            all_valid = False
    return 0 if all_valid else 1


def _inspect(args: argparse.Namespace) -> int:
    glb = Glb.read(args.path)
    extensions = glb.document.get("extensions", {})
    if isinstance(extensions, dict) and VRMA_EXTENSION in extensions:
        summary = read_vrma(glb).summary()
    else:
        summary = VrmAvatar.from_glb(glb).summary()
    if args.as_json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        for key, value in summary.items():
            if isinstance(value, float):
                value = f"{value:.6g}"
            print(f"{key}: {value}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "generate":
            return _generate(args)
        if args.command == "curate":
            return _curate(args)
        if args.command == "validate":
            return _validate(args)
        if args.command == "inspect":
            return _inspect(args)
    except (VroomError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("cancelled", file=sys.stderr)
        return 130
    parser.error(f"unknown command {args.command!r}")
    return 2

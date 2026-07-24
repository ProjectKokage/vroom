"""Command-line interface for generation, inspection, and validation."""

from __future__ import annotations

import argparse
import json
import os
import sys
import zipfile
from collections.abc import Sequence
from pathlib import Path

from .errors import VrmaError, VroomError
from .glb import Glb
from .hy_motion import HyMotionBackend, HyMotionOptions, find_hy_motion_root
from .motion import load_hy_motion_skeleton
from .vrm import VrmAvatar
from .vrma import VRMA_EXTENSION, build_vrma, read_vrma, validate_vrma


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vroom",
        description="Generate portable VRM Animation (.vrma) files with HY-Motion 1.0.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="generate one VRMA clip from text")
    generate.add_argument("prompt", help="English description of the desired human motion")
    generate.add_argument("-o", "--output", type=Path, required=True, help="output .vrma path")
    generate.add_argument(
        "--model-path",
        type=Path,
        default=Path(
            os.environ.get(
                "HY_MOTION_MODEL_PATH",
                "vendor/HY-Motion-1.0/ckpts/tencent/HY-Motion-1.0-Lite",
            )
        ),
        help="directory containing HY-Motion config.yml and latest.ckpt",
    )
    generate.add_argument(
        "--hy-motion-root",
        type=Path,
        help="HY-Motion submodule root (or set VROOM_HY_MOTION_ROOT)",
    )
    generate.add_argument("--duration", type=float, default=4.0, help="duration in seconds")
    generate.add_argument("--seed", type=int, default=42, help="random seed")
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
    generate.add_argument("--cfg-scale", type=float, default=5.0, help="classifier-free guidance")
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
    hy_root = find_hy_motion_root(args.hy_motion_root)
    options = HyMotionOptions(
        model_path=args.model_path,
        hy_motion_root=hy_root,
        device=args.device,
        text_device=args.text_device,
        text_models=args.text_models,
        cfg_scale=args.cfg_scale,
        validation_steps=args.validation_steps,
        ground=not args.no_ground,
    )
    clip = HyMotionBackend(options).generate(
        args.prompt,
        duration=args.duration,
        seed=args.seed,
    )
    skeleton = load_hy_motion_skeleton(hy_root)
    glb = build_vrma(clip, skeleton)
    report = validate_vrma(glb)
    report.require_valid()
    glb.write(args.output)
    print(
        f"Wrote {args.output} ({clip.frame_count} frames, {clip.duration:.3f}s at {clip.fps:g} fps)"
    )
    return 0


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
        if args.command == "validate":
            return _validate(args)
        if args.command == "inspect":
            return _inspect(args)
    except VroomError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    parser.error(f"unknown command {args.command!r}")
    return 2

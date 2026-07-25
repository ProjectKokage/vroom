"""Lazy, side-effect-conscious integration with the vendored HY-Motion model."""

from __future__ import annotations

import copy
import gc
import importlib
import math
import os
import sys
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from .errors import DeviceError, HyMotionError
from .motion import MotionClip, load_hy_motion_skeleton, motion_from_hy_output


def select_device(requested: str, torch_api: Any) -> str:
    """Resolve an explicit or automatic Torch device without importing Torch."""

    value = requested.strip().lower()
    cuda = getattr(torch_api, "cuda", None)
    cuda_available = bool(cuda is not None and cuda.is_available())
    cuda_count = int(cuda.device_count()) if cuda_available else 0
    backends = getattr(torch_api, "backends", None)
    mps = getattr(backends, "mps", None) if backends is not None else None
    mps_available = bool(mps is not None and mps.is_available())

    if value == "auto":
        if cuda_available and cuda_count:
            return "cuda:0"
        if mps_available:
            return "mps"
        return "cpu"
    if value == "cpu":
        return "cpu"
    if value == "mps":
        if not mps_available:
            raise DeviceError("MPS was requested, but torch.backends.mps.is_available() is false")
        return "mps"
    if value == "cuda":
        value = "cuda:0"
    if value.startswith("cuda:"):
        try:
            index = int(value.split(":", 1)[1])
        except ValueError as exc:
            raise DeviceError(f"invalid CUDA device {requested!r}") from exc
        if not cuda_available:
            raise DeviceError("CUDA was requested, but torch.cuda.is_available() is false")
        if not 0 <= index < cuda_count:
            raise DeviceError(
                f"CUDA device {index} is outside the available range 0..{cuda_count - 1}"
            )
        return f"cuda:{index}"
    raise DeviceError("device must be auto, cpu, mps, cuda, or cuda:N")


def find_hy_motion_root(explicit: str | Path | None = None) -> Path:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(Path(explicit))
    environment = os.environ.get("VROOM_HY_MOTION_ROOT")
    if environment:
        candidates.append(Path(environment))
    candidates.extend(
        [
            Path.cwd() / "vendor" / "HY-Motion-1.0",
            Path(__file__).resolve().parents[2] / "vendor" / "HY-Motion-1.0",
        ]
    )
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if (resolved / "hymotion" / "pipeline" / "motion_diffusion.py").is_file():
            return resolved
    locations = ", ".join(str(candidate) for candidate in candidates)
    raise HyMotionError(f"cannot find the HY-Motion submodule (checked: {locations})")


@contextmanager
def _working_directory(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _module_device(module: Any) -> Any:
    try:
        return next(module.parameters()).device
    except StopIteration as exc:
        raise ValueError("the module has no parameters") from exc


def _patch_upstream_device_helpers() -> tuple[ModuleType, ModuleType]:
    """Patch upstream's CUDA-or-CPU assumption without modifying the submodule."""

    converter = importlib.import_module("hymotion.utils.type_converter")
    converter.get_module_device = _module_device

    motion_diffusion = importlib.import_module("hymotion.pipeline.motion_diffusion")
    motion_diffusion.get_module_device = _module_device
    # VRMA only needs rotations and root translation. Avoid multi-GB mesh LBS.
    motion_diffusion.WoodenMesh = lambda: None

    mmdit = importlib.import_module("hymotion.network.hymotion_mmdit")
    mmdit.get_module_device = _module_device
    loaders = importlib.import_module("hymotion.utils.loaders")
    return motion_diffusion, loaders


def _empty_device_cache(torch_api: Any, device_name: str) -> None:
    if device_name.startswith("cuda"):
        torch_api.cuda.empty_cache()
    elif device_name == "mps" and hasattr(torch_api, "mps"):
        torch_api.mps.empty_cache()


def _use_device_random_generator(configured: Any, device_name: str) -> bool:
    """Keep checkpoint reproducibility on CUDA and force portable CPU noise elsewhere."""

    return bool(configured) and device_name.startswith("cuda")


def _move_text_conditioning(
    text_vector: Any,
    text_context: Any,
    context_length: Any,
    device_name: str,
) -> dict[str, Any]:
    """Move conditioning in the float32 dtype consumed by the motion transformer."""

    return {
        "text_vec_raw": text_vector.float().to(device_name),
        "text_ctxt_raw": text_context.float().to(device_name),
        "text_ctxt_raw_length": context_length.to(device_name),
    }


def _configure_text_models(source: str, hy_root: Path) -> str:
    normalized = source.lower()
    if normalized not in {"auto", "local", "hugging-face"}:
        raise HyMotionError("text model source must be auto, local, or hugging-face")
    local_qwen = hy_root / "ckpts" / "Qwen3-8B"
    local_clip = hy_root / "ckpts" / "clip-vit-large-patch14"
    local_available = local_qwen.is_dir() and local_clip.is_dir()
    if normalized == "auto":
        normalized = "local" if local_available else "hugging-face"
    if normalized == "local" and not local_available:
        raise HyMotionError(
            "local text encoders are missing; expected ckpts/Qwen3-8B and "
            "ckpts/clip-vit-large-patch14 under the HY-Motion submodule"
        )
    os.environ["USE_HF_MODELS"] = "1" if normalized == "hugging-face" else "0"
    # Support repeated library calls that deliberately switch sources after the
    # upstream module has already been imported.
    loaded = sys.modules.get("hymotion.network.text_encoders.text_encoder")
    if loaded is not None:
        qwen_path = "Qwen/Qwen3-8B" if normalized == "hugging-face" else str(local_qwen)
        clip_path = (
            "openai/clip-vit-large-patch14" if normalized == "hugging-face" else str(local_clip)
        )
        loaded.LLM_ENCODER_LAYOUT["qwen3"]["module_path"] = qwen_path
        loaded.SENTENCE_EMB_LAYOUT["clipl"]["module_path"] = clip_path
    return normalized


@dataclass(frozen=True, slots=True)
class HyMotionOptions:
    model_path: Path
    hy_motion_root: Path | None = None
    device: str = "auto"
    text_device: str = "auto"
    text_models: str = "auto"
    cfg_scale: float = 5.0
    validation_steps: int | None = None
    ground: bool = True


class HyMotionBackend:
    """Generate a VRMA-ready clip without upstream export side effects."""

    def __init__(self, options: HyMotionOptions):
        self.options = options

    def generate(self, prompt: str, *, duration: float, seed: int) -> MotionClip:
        return self.generate_variants(
            prompt,
            duration=duration,
            seeds=(seed,),
        )[0]

    def generate_variants(
        self,
        prompt: str,
        *,
        duration: float,
        seeds: Sequence[int],
    ) -> tuple[MotionClip, ...]:
        """Generate bounded same-prompt seed variants with one model load."""

        if not prompt.strip():
            raise HyMotionError("prompt must not be empty")
        if not math.isfinite(duration) or duration <= 0:
            raise HyMotionError("duration must be a positive finite number")
        normalized_seeds = tuple(seeds)
        if not normalized_seeds or len(normalized_seeds) > 16:
            raise HyMotionError("seeds must contain between 1 and 16 values")
        if len(set(normalized_seeds)) != len(normalized_seeds):
            raise HyMotionError("seeds must be unique")
        if any(
            not isinstance(seed, int) or not 0 <= seed <= 2**63 - 1
            for seed in normalized_seeds
        ):
            raise HyMotionError(
                "each seed must be an integer between 0 and 2^63-1"
            )
        if not math.isfinite(self.options.cfg_scale) or self.options.cfg_scale <= 0:
            raise HyMotionError("CFG scale must be a positive finite number")
        if self.options.validation_steps is not None and self.options.validation_steps <= 0:
            raise HyMotionError("validation steps must be positive")

        hy_root = find_hy_motion_root(self.options.hy_motion_root)
        model_path = Path(self.options.model_path).expanduser().resolve()
        config_path = model_path / "config.yml"
        checkpoint_path = model_path / "latest.ckpt"
        stats_dir = hy_root / "stats"
        missing = [
            path
            for path in (
                config_path,
                checkpoint_path,
                stats_dir / "Mean.npy",
                stats_dir / "Std.npy",
            )
            if not path.is_file()
        ]
        if missing:
            formatted = ", ".join(str(path) for path in missing)
            raise HyMotionError(
                f"missing HY-Motion runtime files: {formatted}. "
                "See the model download commands in README.md"
            )

        # MPS fallbacks are read by Torch at runtime; set this before importing it.
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        try:
            torch_api = importlib.import_module("torch")
            yaml = importlib.import_module("yaml")
        except ImportError as exc:
            raise HyMotionError(
                "inference dependencies are not installed; run `uv sync --extra inference`"
            ) from exc

        motion_device_name = select_device(self.options.device, torch_api)
        if self.options.text_device == "auto":
            # Qwen bfloat16 is most reliable on CPU when the motion model uses MPS.
            text_device_name = "cpu" if motion_device_name == "mps" else motion_device_name
        elif self.options.text_device == "same":
            text_device_name = motion_device_name
        else:
            text_device_name = select_device(self.options.text_device, torch_api)

        _configure_text_models(self.options.text_models, hy_root)
        skeleton = load_hy_motion_skeleton(hy_root)
        if str(hy_root) not in sys.path:
            sys.path.insert(0, str(hy_root))

        try:
            with config_path.open("r", encoding="utf-8") as stream:
                config = yaml.safe_load(stream)
        except (OSError, yaml.YAMLError) as exc:
            raise HyMotionError(f"cannot read {config_path}: {exc}") from exc
        required_keys = {
            "train_pipeline",
            "train_pipeline_args",
            "network_module",
            "network_module_args",
        }
        if not isinstance(config, dict) or not required_keys <= config.keys():
            raise HyMotionError(f"{config_path} is not a supported HY-Motion config")
        if (
            not isinstance(config["train_pipeline"], str)
            or not isinstance(config["train_pipeline_args"], dict)
            or not isinstance(config["network_module"], str)
            or not isinstance(config["network_module_args"], dict)
        ):
            raise HyMotionError(f"{config_path} has invalid HY-Motion component settings")

        train_args = copy.deepcopy(config["train_pipeline_args"])
        train_args["mean_std_dir"] = str(stats_dir)

        try:
            with _working_directory(hy_root):
                _, loaders = _patch_upstream_device_helpers()
                pipeline = loaders.load_object(
                    config["train_pipeline"],
                    train_args,
                    network_module=config["network_module"],
                    network_module_args=config["network_module_args"],
                )
                pipeline.load_in_demo(str(checkpoint_path), build_text_encoder=False)
                pipeline.body_model = None
                pipeline.random_generator_on_gpu = _use_device_random_generator(
                    pipeline.random_generator_on_gpu,
                    motion_device_name,
                )
                if self.options.validation_steps is not None:
                    pipeline.validation_steps = self.options.validation_steps
                pipeline.to(torch_api.device(motion_device_name)).eval()

                minimum_duration = 20 / float(pipeline.output_mesh_fps)
                maximum_duration = pipeline.train_frames / float(pipeline.output_mesh_fps)
                if not minimum_duration <= duration <= maximum_duration:
                    raise HyMotionError(
                        f"duration must be between {minimum_duration:.3g} and "
                        f"{maximum_duration:.3g} seconds for this checkpoint"
                    )

                # Encode first, then release the 8B text model before diffusion.
                text_encoder_module = importlib.import_module(
                    "hymotion.network.text_encoders.text_encoder"
                )
                text_encoder_module.get_module_device = _module_device
                encoder = loaders.load_object(
                    pipeline._text_encoder_module,
                    pipeline._text_encoder_cfg,
                )
                encoder.to(torch_api.device(text_device_name)).eval()
                with torch_api.inference_mode():
                    text_vector, text_context, context_length = encoder.encode(
                        [prompt] * len(normalized_seeds)
                    )
                    hidden_state = _move_text_conditioning(
                        text_vector,
                        text_context,
                        context_length,
                        motion_device_name,
                    )
                del encoder
                gc.collect()
                _empty_device_cache(torch_api, text_device_name)

                with torch_api.inference_mode():
                    output = pipeline.generate(
                        prompt,
                        list(normalized_seeds),
                        duration,
                        cfg_scale=self.options.cfg_scale,
                        hidden_state_dict=hidden_state,
                    )
                fps = float(pipeline.output_mesh_fps)
        except HyMotionError:
            raise
        except Exception as exc:
            raise HyMotionError(f"HY-Motion inference failed: {exc}") from exc

        return tuple(
            motion_from_hy_output(
                output,
                skeleton,
                sample=index,
                fps=fps,
                name=f"HY-Motion seed {seed}: {prompt[:48]}",
                ground=self.options.ground,
            )
            for index, seed in enumerate(normalized_seeds)
        )

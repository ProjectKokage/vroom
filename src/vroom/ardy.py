"""Lazy ARDY integration for bounded offline VRMA authoring."""

from __future__ import annotations

import gc
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .ardy_constraints import ConstraintDeviation, build_targets, frame_at, measure_targets
from .ardy_loading import load_local_encoder, load_motion_model
from .ardy_motion import motion_from_ardy_output, skeleton_from_ardy
from .ardy_plan import ArdyPlan, PromptCue
from .ardy_source import DEFAULT_MOTION_MEMORY_MIB, ArdySource, read_source, validate_source
from .errors import ArdyError
from .hy_motion import select_device
from .motion import MotionClip, Skeleton


@dataclass(frozen=True, slots=True)
class ArdyOptions:
    model: str = "core"
    checkpoints_dir: Path | None = None
    device: str = "auto"
    text_device: str = "auto"
    cfg_scale: float = 2.0
    constraint_cfg_scale: float = 2.0
    steps: int | None = None
    postprocess: bool = True
    ground: bool = False
    motion_memory_mib: int = DEFAULT_MOTION_MEMORY_MIB

    def __post_init__(self) -> None:
        for name in ("model", "device", "text_device"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ArdyError(f"{name} must be a non-empty string")
        for name in ("cfg_scale", "constraint_cfg_scale"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ArdyError(f"{name} must be finite and non-negative")
        if self.steps is not None and (type(self.steps) is not int or self.steps < 1):
            raise ArdyError("steps must be a positive integer")
        if type(self.motion_memory_mib) is not int or self.motion_memory_mib < 1:
            raise ArdyError("motion_memory_mib must be a positive integer")
        for name in ("postprocess", "ground"):
            if type(getattr(self, name)) is not bool:
                raise ArdyError(f"{name} must be boolean")
        if self.checkpoints_dir is not None:
            object.__setattr__(self, "checkpoints_dir", Path(self.checkpoints_dir))


@dataclass(frozen=True, slots=True)
class ArdyGeneration:
    clip: MotionClip
    skeleton: Skeleton
    source: ArdySource
    deviations: tuple[ConstraintDeviation, ...]
    seed: int


class ArdyBackend:
    """Load one model per call and reuse it for all requested seed variants."""

    def __init__(self, options: ArdyOptions | None = None):
        self.options = options or ArdyOptions()

    def generate(self, prompt: str, *, duration: float, seed: int) -> MotionClip:
        return self.generate_variants(prompt, duration=duration, seeds=(seed,))[0]

    def generate_variants(
        self, prompt: str, *, duration: float, seeds: Any
    ) -> tuple[MotionClip, ...]:
        plan = ArdyPlan(duration, (PromptCue(0, prompt),), seeds=tuple(seeds))
        return tuple(result.clip for result in self.generate_plan(plan))

    def generate_plan(self, plan: ArdyPlan) -> tuple[ArdyGeneration, ...]:
        if not isinstance(plan, ArdyPlan):
            raise ArdyError("generation requires a validated ArdyPlan")
        # Torch reads this switch when imported; embedding applications that
        # already imported Torch must set it before starting their process.
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        try:
            import torch
            from ardy.model.registry import resolve_model_name
        except (ImportError, OSError) as exc:
            raise ArdyError("install the separate ARDY environment: uv sync --extra ardy") from exc
        device = select_device(self.options.device, torch)
        text_device = self.options.text_device
        if text_device == "auto":
            text_device = "cpu" if device == "mps" else device
        elif text_device == "same":
            text_device = device
        text_device = select_device(text_device, torch)
        model = encoder = None
        try:
            name = resolve_model_name(self.options.model)
            if name not in {"ARDY-Core-RP-20FPS-Horizon40", "ARDY-Core-RP-20FPS-Horizon8"}:
                raise ArdyError("VRMA generation requires an ARDY Core human skeleton model")
            # Construct on CPU and cast before moving: upstream skeleton assets
            # include float64 buffers, which MPS cannot represent.
            model = load_motion_model(name, self.options.checkpoints_dir)
            model.float()
            skeleton_from_ardy(model.skeleton)
            prepared = self._prepare_plan(plan, model)
            encoder = load_local_encoder(text_device)
            embeddings = self._encode_prompts(plan, encoder, text_device)
            # Only embeddings are needed during motion generation. Releasing
            # the 8B encoder keeps both accelerators and CPU memory available.
            del encoder
            encoder = None
            gc.collect()
            self._empty_cache(torch, text_device)
            model.to(device)
            # Skeleton indexing/FK and whole-clip correction run on CPU;
            # diffusion uses only the neural weights on the accelerator.
            model.skeleton.cpu()
            model.device = device
            return self._generate_loaded(plan, model, embeddings, prepared=prepared)
        except ArdyError:
            raise
        except Exception as exc:
            message = str(exc)
            if any(word in message.lower() for word in ("gated", "401", "403", "access token")):
                raise ArdyError(
                    "ARDY text encoding needs Hugging Face login and access to "
                    "Meta-Llama-3-8B-Instruct; run hf auth login and follow "
                    "the model access instructions"
                ) from exc
            raise ArdyError(f"ARDY generation failed ({type(exc).__name__}): {message}") from exc
        finally:
            del model, encoder
            gc.collect()
            for used_device in dict.fromkeys((text_device, device)):
                self._empty_cache(torch, used_device)

    @staticmethod
    def _empty_cache(torch: Any, device: str) -> None:
        # Releasing cached blocks is best effort after object disposal; a
        # failed accelerator must not replace the original error/cancellation.
        try:
            if device.startswith("cuda"):
                with torch.cuda.device(device):
                    torch.cuda.empty_cache()
            elif device == "mps":
                torch.mps.empty_cache()
        except RuntimeError:
            pass

    def _validate_plan_grid(self, plan: ArdyPlan, model: Any) -> int:
        fps = float(model.motion_rep.fps)
        frames = frame_at(plan.duration, fps, "duration") + 1
        if frames < 2:
            raise ArdyError("duration must include at least two motion samples")
        for cue in plan.prompts:
            frame_at(cue.time, fps, "prompt time")
        for target in plan.constraints:
            frame_at(target.time, fps, "target time")
        patch = int(model.num_frames_per_token)
        if len(plan.prompts) > 1 and frame_at(plan.prompts[1].time, fps, "prompt time") < patch:
            raise ArdyError(
                f"the first prompt must last at least {patch / fps:g}s to establish motion history"
            )
        # Reserve for source arrays, decoded poses, conditioning, native
        # features, Python VRMA tracks and validation copies. This budget is
        # explicitly configurable and is separate from model/attention memory.
        bytes_per_frame = 4 * (16 * int(model.motion_rep.motion_rep_dim) + 32 * 27 * 4)
        required = frames * bytes_per_frame * len(plan.seeds)
        if required > self.options.motion_memory_mib * 1024 * 1024:
            raise ArdyError(
                f"motion working data needs approximately {math.ceil(required / 1024**2)} MiB; "
                "increase --motion-memory-mib or shorten the plan"
            )
        steps = self.options.steps
        if steps is not None and steps > int(model.diffusion.num_base_steps):
            raise ArdyError(f"steps must be between 1 and {model.diffusion.num_base_steps}")
        return frames

    @staticmethod
    def _encode_prompts(plan: ArdyPlan, encoder: Any, device: str) -> tuple[Any, ...]:
        # The public encoder silently crops to doc_max_length and launches a
        # worker on every visible GPU. Use its single-device encoding primitive
        # with one prompt at a time, retaining the same tokenizer/pooling path.
        llm = encoder
        result = []
        for cue in plan.prompts:
            ids = llm.tokenizer(cue.text, add_special_tokens=False, truncation=False)["input_ids"]
            if len(ids) > llm.doc_max_length:
                raise ArdyError(
                    f"prompt exceeds the text encoder's {llm.doc_max_length}-token document limit"
                )
            prepared = llm._convert_to_str("", cue.text)
            full = llm.prepare_for_tokenization(prepared).replace("!@#$%^&*()", "")
            if len(llm.tokenizer(full, truncation=False)["input_ids"]) > llm.max_length:
                raise ArdyError("prompt exceeds the text encoder context window")
            result.append(llm._encode([prepared], device=device).float()[:, None])
        return tuple(result)

    def _prepare_plan(self, plan: ArdyPlan, model: Any) -> tuple[int, Any, ArdySource | None]:
        frames = self._validate_plan_grid(plan, model)
        fps = float(model.motion_rep.fps)
        budget = self.options.motion_memory_mib * 1024 * 1024
        targets = build_targets(plan, model, max_bytes=budget)
        prefix = (
            None if plan.continuation is None else read_source(plan.continuation, max_bytes=budget)
        )
        if prefix is not None:
            if prefix.fps != fps:
                raise ArdyError("continuation source fps must match the selected model")
            if not model.num_frames_per_token <= prefix.frame_count < frames:
                raise ArdyError(
                    "continuation source must contain at least one motion token "
                    "and be shorter than total duration"
                )
        return frames, targets, prefix

    def _generate_loaded(
        self,
        plan: ArdyPlan,
        model: Any,
        embeddings: tuple[Any, ...],
        *,
        prepared: tuple[int, Any, ArdySource | None] | None = None,
    ) -> tuple[ArdyGeneration, ...]:
        import torch
        from ardy.postprocess import post_process_motion
        from ardy.tools import seed_everything

        frames, targets, prefix = prepared or self._prepare_plan(plan, model)
        fps = float(model.motion_rep.fps)
        budget = self.options.motion_memory_mib * 1024 * 1024
        steps = self.options.steps or int(model.diffusion.num_base_steps)
        results = []
        for seed in plan.seeds:
            seed_everything(seed)
            with torch.inference_mode():
                output = self._sample(plan, model, embeddings, targets, frames, steps, prefix)
                # Model-decoded arrays cross into the C++ correction engine
                # only after validating shape, finiteness and proper rotations.
                validate_source(
                    output["local_rot_mats"][0].cpu().numpy(),
                    output["root_positions"][0].cpu().numpy(),
                    fps,
                    max_bytes=budget,
                )
                contacts = output["foot_contacts"]
                if (
                    contacts.shape != (1, frames, 4)
                    or not torch.isfinite(contacts).all()
                    or (contacts < 0).any()
                    or (contacts > 1).any()
                ):
                    raise ArdyError("ARDY returned invalid foot contact values")
                if self.options.postprocess:
                    output.update(
                        post_process_motion(
                            output["local_rot_mats"],
                            output["root_positions"],
                            output["foot_contacts"],
                            model.skeleton,
                            constraint_lst=list(targets.correction) or None,
                        )
                    )
                if prefix is not None:
                    # Correction may adjust earlier frames. A continuation
                    # keeps the exact original prefix before measuring targets.
                    output["local_rot_mats"][0, : prefix.frame_count] = torch.tensor(
                        prefix.local_rot_mats
                    )
                    output["root_positions"][0, : prefix.frame_count] = torch.tensor(
                        prefix.root_positions
                    )
                    rots, joints, _ = model.skeleton.fk(
                        output["local_rot_mats"], output["root_positions"]
                    )
                    output.update(global_rot_mats=rots, posed_joints=joints)
                deviations = measure_targets(targets, output, model)
                failed = [value for value in deviations if not value.satisfied]
                if plan.strict and failed:
                    value = failed[0]
                    joint = "" if value.joint is None else f" {value.joint}"
                    raise ArdyError(
                        f"target {value.target} at {value.time:g}s{joint} "
                        f"misses {value.component}: "
                        f"{value.error:.5g} > {value.tolerance:.5g}; no output written"
                    )
                source = validate_source(
                    output["local_rot_mats"][0].cpu().numpy(),
                    output["root_positions"][0].cpu().numpy(),
                    fps,
                    max_bytes=budget,
                )
                clip = motion_from_ardy_output(
                    output,
                    model.skeleton,
                    fps=fps,
                    name=f"ARDY seed {seed}",
                    ground=self.options.ground,
                )
                results.append(
                    ArdyGeneration(
                        clip, skeleton_from_ardy(model.skeleton), source, deviations, seed
                    )
                )
        return tuple(results)

    def _sample(
        self,
        plan: ArdyPlan,
        model: Any,
        embeddings: tuple[Any, ...],
        targets: Any,
        frames: int,
        steps: int,
        prefix: ArdySource | None,
    ) -> dict[str, Any]:
        import torch

        rep, device = model.motion_rep, model.device
        patch, horizon = int(model.num_frames_per_token), int(model.gen_horizon_len)
        # ARDY was trained on windows of at most ten seconds. Reserve one
        # generation horizon and divide the remainder between history/future.
        window = int(10 * rep.fps) // patch * patch
        history_limit = window - horizon
        if history_limit < patch:
            raise ArdyError("ARDY checkpoint has no history space in its trained window")
        pieces = []
        cursor = 0
        if prefix is not None:
            features = rep(
                torch.tensor(prefix.local_rot_mats)[None],
                torch.tensor(prefix.root_positions)[None],
                to_normalize=True,
            )
            pieces.append(features.cpu())
            cursor = prefix.frame_count
        cue_frames = [frame_at(cue.time, rep.fps, "prompt time") for cue in plan.prompts]
        last_constraint = max((goal.frame for goal in targets.goals), default=-1)
        history = None if not pieces else pieces[-1][:, -history_limit:]
        while cursor < frames:
            cue_index = max(index for index, frame in enumerate(cue_frames) if frame <= cursor)
            next_cue = cue_frames[cue_index + 1] if cue_index + 1 < len(cue_frames) else frames
            take = min(horizon, frames - cursor, next_cue - cursor)
            future = min(max(0, last_constraint + 1 - cursor - horizon), history_limit // 2)
            future = math.ceil(future / patch) * patch
            history_count = min(cursor, history_limit - future) // patch * patch
            initial = None if not history_count else history[:, -history_count:].to(device)
            start = cursor - history_count
            window_frames = history_count + horizon + future
            # Restrict conditioning allocations and attention to this window.
            indices, values = {}, {}
            for feature, chunks in targets.indices.items():
                kept_indices, kept_values = [], []
                for item, value in zip(chunks, targets.values[feature], strict=True):
                    frame = int(item.reshape(-1)[0])
                    if cursor <= frame < start + window_frames:
                        shifted = item.clone()
                        if shifted.ndim == 1:
                            shifted -= start
                        else:
                            shifted[:, 0] -= start
                        kept_indices.append(shifted)
                        kept_values.append(value)
                if kept_indices:
                    indices[feature], values[feature] = kept_indices, kept_values
            observed, mask = rep.create_conditions(indices, values, window_frames, True, "cpu")
            samples = model.autoregressive_step(
                num_frames=window_frames,
                num_denoising_steps=steps,
                motion_mask=mask[None].to(device),
                observed_motion=observed[None].to(device),
                cfg_weight=(self.options.cfg_scale, self.options.constraint_cfg_scale),
                text_feat=embeddings[cue_index].to(device),
                text_pad_mask=torch.ones((1, 1), device=device, dtype=torch.bool),
                init_history_sequence=initial,
            )
            if samples.shape != (1, history_count + horizon, rep.motion_rep_dim):
                raise ArdyError("ARDY returned malformed motion frame dimensions")
            chunk = samples[:, history_count : history_count + take].detach().cpu()
            if chunk.shape[1] != take or not torch.isfinite(chunk).all():
                raise ArdyError("ARDY returned malformed or non-finite motion frames")
            pieces.append(chunk)
            history = chunk if history is None else torch.cat((history, chunk), dim=1)
            history = history[:, -history_limit:].clone()
            cursor += take
        motion = torch.cat(pieces, dim=1)
        return rep.inverse(motion, is_normalized=True)

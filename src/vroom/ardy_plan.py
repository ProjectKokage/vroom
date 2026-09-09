"""Bounded, dependency-free authoring plans for ARDY motion generation.

Times are seconds, positions are meters in the ARDY source skeleton's space,
headings are radians, and rotations are XYZW quaternions. Model frame and token
alignment is checked by the runtime after its configuration is loaded.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

from .errors import ArdyError

# Bound decoded authoring data independently of the much larger motion tensors.
MAX_PLAN_BYTES = 1024 * 1024
# A target's Python objects occupy roughly hundreds of bytes before tensors.
MAX_PLAN_ITEMS = MAX_PLAN_BYTES // 256
MAX_PLAN_SEEDS = 16
_FLOAT32_MAX = 3.4028234663852886e38

type Vec2 = tuple[float, float]
type Vec3 = tuple[float, float, float]
type Quat = tuple[float, float, float, float]


def _number(value: Any, label: str, *, minimum: float | None = None) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or abs(value) > _FLOAT32_MAX
        or not math.isfinite(value)
        or (minimum is not None and value < minimum)
    ):
        suffix = "" if minimum is None else f" and at least {minimum:g}"
        raise ArdyError(f"{label} must be a finite float32 number{suffix}")


def _text(value: Any, label: str) -> None:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ArdyError(f"{label} must be non-empty text without NUL characters")
    try:
        length = len(value.encode("utf-8"))
    except UnicodeEncodeError as exc:
        raise ArdyError(f"{label} must be valid UTF-8 text") from exc
    if length > MAX_PLAN_BYTES:
        raise ArdyError(f"{label} exceeds the {MAX_PLAN_BYTES}-byte authoring budget")


def _vector(value: Any, size: int, label: str) -> tuple[float, ...]:
    if not isinstance(value, (tuple, list)) or len(value) != size:
        raise ArdyError(f"{label} must contain {size} numbers")
    for component in value:
        _number(component, label)
    return tuple(float(component) for component in value)


def _path(value: Any, label: str) -> Path:
    if not isinstance(value, (str, Path)):
        raise ArdyError(f"{label} must name an ARDY .npz source")
    _text(str(value), label)
    path = Path(value).expanduser()
    if path.suffix.lower() != ".npz":
        raise ArdyError(f"{label} must name an ARDY .npz source")
    return path


def _tolerances(position: Any, rotation: Any) -> None:
    for value, label in (
        (position, "position tolerance"),
        (rotation, "rotation tolerance"),
    ):
        if value is not None:
            _number(value, label, minimum=0)


@dataclass(frozen=True, slots=True)
class PromptCue:
    time: float
    text: str

    def __post_init__(self) -> None:
        _number(self.time, "prompt time", minimum=0)
        _text(self.text, "prompt")


@dataclass(frozen=True, slots=True)
class RootTarget:
    time: float
    position: Vec2
    heading: float | None = None
    position_tolerance: float | None = None
    rotation_tolerance: float | None = None

    def __post_init__(self) -> None:
        _number(self.time, "root target time", minimum=0)
        object.__setattr__(self, "position", _vector(self.position, 2, "root position XZ"))
        if self.heading is not None:
            _number(self.heading, "root heading")
        _tolerances(self.position_tolerance, self.rotation_tolerance)


@dataclass(frozen=True, slots=True)
class PoseTarget:
    time: float
    source: Path
    source_time: float
    joints: tuple[str, ...] | None = None
    position_tolerance: float | None = None
    rotation_tolerance: float | None = None

    def __post_init__(self) -> None:
        _number(self.time, "pose target time", minimum=0)
        _number(self.source_time, "pose source time", minimum=0)
        object.__setattr__(self, "source", _path(self.source, "pose source"))
        if self.joints is not None:
            if not isinstance(self.joints, (tuple, list)) or not self.joints:
                raise ArdyError("pose joints must be a non-empty list")
            if len(self.joints) > MAX_PLAN_ITEMS:
                raise ArdyError("pose joints exceed the authoring item budget")
            for joint in self.joints:
                _text(joint, "joint name")
            if len(set(self.joints)) != len(self.joints):
                raise ArdyError("pose joints must be unique")
            if sum(len(joint.encode("utf-8")) for joint in self.joints) > MAX_PLAN_BYTES:
                raise ArdyError("pose joints exceed the authoring byte budget")
            object.__setattr__(self, "joints", tuple(self.joints))
        _tolerances(self.position_tolerance, self.rotation_tolerance)


@dataclass(frozen=True, slots=True)
class JointTarget:
    time: float
    joint: str
    root_position: Vec3
    root_heading: float
    position: Vec3 | None = None
    rotation: Quat | None = None
    position_tolerance: float | None = None
    rotation_tolerance: float | None = None

    def __post_init__(self) -> None:
        _number(self.time, "joint target time", minimum=0)
        _text(self.joint, "joint name")
        object.__setattr__(self, "root_position", _vector(self.root_position, 3, "root position"))
        _number(self.root_heading, "root heading")
        if self.position is None and self.rotation is None:
            raise ArdyError("joint target requires position or rotation")
        if self.position is not None:
            object.__setattr__(self, "position", _vector(self.position, 3, "joint position"))
        if self.rotation is not None:
            rotation = _vector(self.rotation, 4, "joint rotation XYZW")
            if abs(math.sqrt(sum(value * value for value in rotation)) - 1) > 1e-3:
                raise ArdyError("joint rotation XYZW must be normalized")
            object.__setattr__(self, "rotation", rotation)
        _tolerances(self.position_tolerance, self.rotation_tolerance)


type ArdyTarget = RootTarget | PoseTarget | JointTarget


@dataclass(frozen=True, slots=True)
class ArdyPlan:
    duration: float
    prompts: tuple[PromptCue, ...]
    seeds: tuple[int, ...] = (42,)
    constraints: tuple[ArdyTarget, ...] = ()
    continuation: Path | None = None
    strict: bool = False

    def __post_init__(self) -> None:
        _number(self.duration, "duration", minimum=0)
        if self.duration == 0:
            raise ArdyError("duration must be positive")
        if not isinstance(self.strict, bool):
            raise ArdyError("strict must be a boolean")
        for name in ("prompts", "seeds", "constraints"):
            value = getattr(self, name)
            if not isinstance(value, (tuple, list)):
                raise ArdyError(f"{name} must be a list")
            if len(value) > MAX_PLAN_ITEMS:
                raise ArdyError(f"{name} exceed the authoring item budget")
            object.__setattr__(self, name, tuple(value))
        if not self.prompts or any(not isinstance(cue, PromptCue) for cue in self.prompts):
            raise ArdyError("prompts must contain PromptCue values")
        if self.prompts[0].time != 0:
            raise ArdyError("the first prompt must start at time 0")
        if any(later.time <= earlier.time for earlier, later in pairwise(self.prompts)):
            raise ArdyError("prompt times must be strictly increasing")
        if not 1 <= len(self.seeds) <= MAX_PLAN_SEEDS or any(
            isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32
            for seed in self.seeds
        ):
            raise ArdyError("seeds must contain 1 to 16 integers between 0 and 2^32-1")
        if len(set(self.seeds)) != len(self.seeds):
            raise ArdyError("seeds must be unique")
        if any(
            not isinstance(target, (RootTarget, PoseTarget, JointTarget))
            for target in self.constraints
        ):
            raise ArdyError("constraints must contain RootTarget, PoseTarget, or JointTarget")
        if any(cue.time > self.duration for cue in (*self.prompts, *self.constraints)):
            raise ArdyError("prompt and target times must not exceed duration")
        if self.continuation is not None:
            object.__setattr__(self, "continuation", _path(self.continuation, "continuation"))
        if len(self.prompts) + len(self.constraints) > MAX_PLAN_ITEMS:
            raise ArdyError("plan exceeds the authoring item budget")
        texts = [cue.text for cue in self.prompts]
        if self.continuation is not None:
            texts.append(str(self.continuation))
        for target in self.constraints:
            if isinstance(target, PoseTarget):
                texts.extend((str(target.source), *(target.joints or ())))
            elif isinstance(target, JointTarget):
                texts.append(target.joint)
        if sum(len(text.encode("utf-8")) for text in texts) > MAX_PLAN_BYTES:
            raise ArdyError("plan text exceeds the authoring byte budget")


def _object(value: Any, allowed: set[str], required: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ArdyError(f"{label} must be an object")
    unknown = value.keys() - allowed
    missing = required - value.keys()
    if unknown or missing:
        raise ArdyError(
            f"{label} has unknown keys {sorted(unknown)} or missing keys {sorted(missing)}"
        )
    return value


def _relative_path(value: Any, base: Path, label: str) -> Path:
    path = _path(value, label)
    return path if path.is_absolute() else base / path


def _target(value: Any, base: Path) -> ArdyTarget:
    common = {"type", "time", "positionTolerance", "rotationTolerance"}
    if not isinstance(value, dict):
        raise ArdyError("constraint must be an object")
    target_type = value.get("type")
    fields = {
        "root": ({"position", "heading"}, {"position"}),
        "pose": ({"source", "sourceTime", "joints"}, {"source", "sourceTime"}),
        "joint": (
            {"joint", "position", "rotation", "rootPosition", "rootHeading"},
            {"joint", "rootPosition", "rootHeading"},
        ),
    }
    if not isinstance(target_type, str) or target_type not in fields:
        raise ArdyError("constraint type must be root, pose, or joint")
    allowed, required = fields[target_type]
    _object(value, common | allowed, {"type", "time"} | required, "constraint")
    kwargs = {
        "time": value["time"],
        "position_tolerance": value.get("positionTolerance"),
        "rotation_tolerance": value.get("rotationTolerance"),
    }
    if target_type == "root":
        return RootTarget(position=value["position"], heading=value.get("heading"), **kwargs)
    if target_type == "pose":
        return PoseTarget(
            source=_relative_path(value["source"], base, "pose source"),
            source_time=value["sourceTime"],
            joints=value.get("joints"),
            **kwargs,
        )
    return JointTarget(
        joint=value["joint"],
        root_position=value["rootPosition"],
        root_heading=value["rootHeading"],
        position=value.get("position"),
        rotation=value.get("rotation"),
        **kwargs,
    )


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ArdyError(f"plan repeats JSON key {key!r}")
        result[key] = value
    return result


def load_plan(path: str | Path) -> ArdyPlan:
    """Read a closed JSON plan; NPZ paths are relative to the plan's directory."""

    path = Path(path).expanduser().resolve()
    try:
        with path.open("rb") as stream:
            payload = stream.read(MAX_PLAN_BYTES + 1)
        if len(payload) > MAX_PLAN_BYTES:
            raise ArdyError(f"plan exceeds the {MAX_PLAN_BYTES}-byte authoring budget")
        data = json.loads(payload, object_pairs_hook=_unique_object)
    except (OSError, ValueError, RecursionError) as exc:
        raise ArdyError(f"cannot read ARDY plan {path}: {exc}") from exc
    data = _object(
        data,
        {"duration", "prompts", "seeds", "constraints", "continuation", "strict"},
        {"duration", "prompts"},
        "plan",
    )
    for name in ("prompts", "constraints"):
        value = data.get(name, [])
        if not isinstance(value, list) or len(value) > MAX_PLAN_ITEMS:
            raise ArdyError(f"{name} must be a list within the authoring item budget")
    prompts = tuple(
        PromptCue(**_object(cue, {"time", "text"}, {"time", "text"}, "prompt"))
        for cue in data["prompts"]
    )
    return ArdyPlan(
        duration=data["duration"],
        prompts=prompts,
        seeds=data.get("seeds", (42,)),
        constraints=tuple(_target(target, path.parent) for target in data.get("constraints", [])),
        continuation=(
            None
            if data.get("continuation") is None
            else _relative_path(data["continuation"], path.parent, "continuation")
        ),
        strict=data.get("strict", False),
    )

"""Translate validated authoring targets into ARDY conditioning and diagnostics."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from .ardy_plan import ArdyPlan, JointTarget, PoseTarget, RootTarget
from .ardy_source import read_source
from .errors import ArdyError


@dataclass(frozen=True, slots=True)
class ConstraintDeviation:
    target: int
    time: float
    component: str
    error: float
    tolerance: float
    joint: str | None = None

    @property
    def satisfied(self) -> bool:
        return self.error <= self.tolerance


@dataclass(frozen=True, slots=True)
class _Goal:
    target: int
    frame: int
    feature: str
    joint: int | None
    value: Any
    tolerance: float


@dataclass(frozen=True, slots=True)
class _Targets:
    indices: dict[str, list[Any]]
    values: dict[str, list[Any]]
    correction: tuple[Any, ...]
    goals: tuple[_Goal, ...]


def frame_at(time: float, fps: float, label: str) -> int:
    value = time * fps
    frame = round(value)
    if not math.isclose(value, frame, abs_tol=1e-6, rel_tol=0):
        raise ArdyError(f"{label} must fall on the model's {1 / fps:g}-second frame grid")
    return frame


def _joint_index(name: str, skeleton: Any) -> int:
    from .ardy_motion import CORE_TO_VRM

    source = {vrm: core for core, vrm in CORE_TO_VRM.items()}.get(name, name)
    try:
        return skeleton.bone_order_names.index(source)
    except ValueError as exc:
        raise ArdyError(f"unknown Core/VRM joint {name!r}") from exc


def build_targets(plan: ArdyPlan, model: Any, *, max_bytes: int) -> _Targets:
    """Build sparse CPU tensors; conflicting targets fail before inference."""
    import numpy as np
    import torch
    from ardy.constraints import (
        FullBodyConstraintSet,
        LeftFootConstraintSet,
        LeftHandConstraintSet,
        RightFootConstraintSet,
        RightHandConstraintSet,
        Root2DConstraintSet,
    )
    from ardy.geometry import quaternion_to_matrix
    from ardy.motion_rep.tools import compute_heading_angle

    skeleton = model.skeleton
    fps = float(model.motion_rep.fps)
    goals: list[_Goal] = []
    correction: list[Any] = []
    entries: dict[tuple[str, int, int | None], Any] = {}

    def put(frame: int, feature: str, value: Any, joint: int | None = None) -> Any:
        value = np.asarray(value, dtype=np.float32)
        key = (feature, frame, joint)
        if key in entries and not np.allclose(entries[key], value, atol=1e-6, rtol=0):
            raise ArdyError(f"conflicting {feature} targets at {frame / fps:g}s")
        entries[key] = value
        return value

    def add(
        target: int,
        frame: int,
        feature: str,
        value: Any,
        tolerance: float,
        joint: int | None = None,
    ) -> None:
        value = put(frame, feature, value, joint)
        goals.append(_Goal(target, frame, feature, joint, value, tolerance))

    def root(
        target: int,
        frame: int,
        position: Any,
        heading: float | None,
        ptol: float,
        rtol: float,
        *,
        height: bool = True,
    ) -> None:
        xz = [position[0], position[2]] if height else position
        add(target, frame, "root_2d", xz, ptol)
        if height:
            add(target, frame, "root_y_pos", [position[1]], ptol)
        if heading is not None:
            add(target, frame, "global_root_heading", [math.cos(heading), math.sin(heading)], rtol)

    for index, target in enumerate(plan.constraints):
        frame = frame_at(target.time, fps, f"target {index} time")
        ptol = 0.05 if target.position_tolerance is None else target.position_tolerance
        rtol = 0.1 if target.rotation_tolerance is None else target.rotation_tolerance
        if isinstance(target, RootTarget):
            root(index, frame, target.position, target.heading, ptol, rtol, height=False)
            correction.append(
                Root2DConstraintSet(
                    skeleton,
                    torch.tensor([frame]),
                    torch.tensor([target.position]),
                    None if target.heading is None else torch.tensor([target.heading]),
                )
            )
        elif isinstance(target, PoseTarget):
            source = read_source(target.source, max_bytes=max_bytes)
            source_frame = frame_at(target.source_time, source.fps, "pose sourceTime")
            if not 0 <= source_frame < source.frame_count:
                raise ArdyError("pose sourceTime is outside the source motion")
            rotations = torch.tensor(source.local_rot_mats[source_frame : source_frame + 1])
            roots = torch.tensor(source.root_positions[source_frame : source_frame + 1])
            global_rots, positions, _ = skeleton.fk(rotations, roots)
            positions = positions.cpu()
            global_rots = global_rots.cpu()
            heading = float(compute_heading_angle(positions, skeleton)[0])
            root(index, frame, roots[0], heading, ptol, rtol)
            joints = (
                list(range(27))
                if target.joints is None
                else [_joint_index(name, skeleton) for name in target.joints]
            )
            # Hips position is required by ARDY whenever world-space positions
            # are constrained, even when only one limb was selected.
            put(frame, "global_joints_positions", roots[0].numpy(), 0)
            for joint in joints:
                add(index, frame, "global_joints_positions", positions[0, joint], ptol, joint)
                add(index, frame, "global_joints_rots", global_rots[0, joint], rtol, joint)
            kwargs = dict(
                skeleton=skeleton,
                frame_indices=torch.tensor([frame]),
                global_joints_positions=positions,
                global_joints_rots=global_rots,
            )
            if target.joints is None:
                correction.append(FullBodyConstraintSet(**kwargs))
            else:
                classes = {
                    "LeftHand": LeftHandConstraintSet,
                    "RightHand": RightHandConstraintSet,
                    "LeftFoot": LeftFootConstraintSet,
                    "RightFoot": RightFootConstraintSet,
                }
                for joint in joints:
                    cls = classes.get(skeleton.bone_order_names[joint])
                    if cls is not None:
                        correction.append(cls(root_2d=None, **kwargs))
        elif isinstance(target, JointTarget):
            joint = _joint_index(target.joint, skeleton)
            root(index, frame, target.root_position, target.root_heading, ptol, rtol)
            if target.position is not None:
                put(frame, "global_joints_positions", target.root_position, 0)
                add(index, frame, "global_joints_positions", target.position, ptol, joint)
            if target.rotation is not None:
                x, y, z, w = target.rotation
                matrix = quaternion_to_matrix(torch.tensor([w, x, y, z]))
                add(index, frame, "global_joints_rots", matrix, rtol, joint)

    indices: dict[str, list[Any]] = defaultdict(list)
    values: dict[str, list[Any]] = defaultdict(list)
    for (feature, frame, joint), value in entries.items():
        indices[feature].append(torch.tensor([frame] if joint is None else [[frame, joint]]))
        values[feature].append(torch.tensor(value).reshape((1, *value.shape)))
    return _Targets(dict(indices), dict(values), tuple(correction), tuple(goals))


def measure_targets(
    targets: _Targets, output: dict[str, Any], model: Any
) -> tuple[ConstraintDeviation, ...]:
    """Measure corrected source motion; avatar retargeting can change positions."""
    import numpy as np
    from ardy.motion_rep.tools import compute_heading_angle

    positions = output["posed_joints"][0].detach().cpu()
    rotations = output["global_rot_mats"][0].detach().cpu().numpy()
    roots = output["root_positions"][0].detach().cpu().numpy()
    headings = compute_heading_angle(positions, model.skeleton).numpy()
    result: list[ConstraintDeviation] = []
    for goal in targets.goals:
        frame, feature = goal.frame, goal.feature
        if feature == "root_2d":
            error = float(np.linalg.norm(roots[frame, [0, 2]] - goal.value))
        elif feature == "root_y_pos":
            error = abs(float(roots[frame, 1] - goal.value[0]))
        elif feature == "global_root_heading":
            expected = math.atan2(float(goal.value[1]), float(goal.value[0]))
            error = abs(math.remainder(float(headings[frame]) - expected, 2 * math.pi))
        elif feature == "global_joints_positions":
            error = float(np.linalg.norm(positions[frame, goal.joint].numpy() - goal.value))
        else:
            relative = rotations[frame, goal.joint] @ goal.value.T
            error = math.acos(float(np.clip((np.trace(relative) - 1) / 2, -1, 1)))
        result.append(
            ConstraintDeviation(
                goal.target,
                frame / model.motion_rep.fps,
                feature,
                error,
                goal.tolerance,
                None if goal.joint is None else model.skeleton.bone_order_names[goal.joint],
            )
        )
    return tuple(result)

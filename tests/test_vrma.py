from __future__ import annotations

import copy
from dataclasses import replace

import pytest

from vroom.errors import VrmaError
from vroom.glb import AccessorReader, Glb
from vroom.motion import MotionClip, Skeleton
from vroom.vrma import build_vrma, read_vrma, validate_vrma, write_vrma


def test_writer_builds_valid_vrma(
    identity_clip: MotionClip,
    hy_skeleton: Skeleton,
) -> None:
    glb = build_vrma(identity_clip, hy_skeleton)
    report = validate_vrma(glb)

    assert report.errors == []
    assert report.warnings == []
    vrma = read_vrma(Glb.from_bytes(glb.to_bytes()))
    assert vrma.frame_count == identity_clip.frame_count
    assert vrma.duration == pytest.approx(identity_clip.duration)
    assert vrma.nominal_fps == pytest.approx(30.0)
    assert len(vrma.humanoid_bones) == 22
    assert len(glb.document["nodes"]) == 22
    assert "leftIndexProximal" not in vrma.humanoid_bones

    animation = glb.document["animations"][0]
    targets = [channel["target"] for channel in animation["channels"]]
    assert targets[0] == {"node": 0, "path": "translation"}
    assert all(target["path"] != "scale" for target in targets)
    assert all(target["node"] == 0 for target in targets if target["path"] == "translation")
    time_accessor = animation["samplers"][0]["input"]
    times = AccessorReader(glb).read(time_accessor)
    assert [value[0] for value in times] == pytest.approx((0.0, 1 / 30, 2 / 30, 3 / 30))


def test_writer_requires_vrma_suffix(
    tmp_path,
    identity_clip: MotionClip,
    hy_skeleton: Skeleton,
) -> None:
    with pytest.raises(VrmaError, match=r"\.vrma"):
        write_vrma(tmp_path / "motion.glb", identity_clip, hy_skeleton)


def test_writer_reports_float32_overflow_as_vrma_error(
    identity_clip: MotionClip,
    hy_skeleton: Skeleton,
) -> None:
    oversized = replace(
        identity_clip,
        hips_translations=((1e100, 0.0, 0.0),) * identity_clip.frame_count,
    )

    with pytest.raises(VrmaError, match="outside the float32 range"):
        build_vrma(oversized, hy_skeleton)


def test_validator_rejects_duplicate_node_path(
    identity_clip: MotionClip,
    hy_skeleton: Skeleton,
) -> None:
    original = build_vrma(identity_clip, hy_skeleton)
    document = copy.deepcopy(original.document)
    document["animations"][0]["channels"].append(
        copy.deepcopy(document["animations"][0]["channels"][0])
    )

    report = validate_vrma(Glb(document, original.binary))

    assert any("duplicates" in error for error in report.errors)


def test_validator_rejects_non_hips_translation(
    identity_clip: MotionClip,
    hy_skeleton: Skeleton,
) -> None:
    original = build_vrma(identity_clip, hy_skeleton)
    document = copy.deepcopy(original.document)
    document["animations"][0]["channels"][0]["target"]["node"] = 1

    report = validate_vrma(Glb(document, original.binary))

    assert any("leftUpperLeg must not contain translation" in error for error in report.errors)


def test_validator_rejects_non_uniform_humanoid_scale(
    identity_clip: MotionClip,
    hy_skeleton: Skeleton,
) -> None:
    original = build_vrma(identity_clip, hy_skeleton)
    document = copy.deepcopy(original.document)
    document["nodes"][0]["scale"] = [1.0, 2.0, 1.0]

    report = validate_vrma(Glb(document, original.binary))

    assert any("equal positive finite" in error for error in report.errors)


@pytest.mark.parametrize("bound", ["min", "max"])
def test_validator_requires_animation_input_bounds(
    bound: str,
    identity_clip: MotionClip,
    hy_skeleton: Skeleton,
) -> None:
    original = build_vrma(identity_clip, hy_skeleton)
    document = copy.deepcopy(original.document)
    time_accessor = document["animations"][0]["samplers"][0]["input"]
    document["accessors"][time_accessor].pop(bound)

    report = validate_vrma(Glb(document, original.binary))

    assert any(f"one-value {bound}" in error for error in report.errors)


def test_validator_rejects_animation_input_bound_mismatch(
    identity_clip: MotionClip,
    hy_skeleton: Skeleton,
) -> None:
    original = build_vrma(identity_clip, hy_skeleton)
    document = copy.deepcopy(original.document)
    time_accessor = document["animations"][0]["samplers"][0]["input"]
    document["accessors"][time_accessor]["max"] = [99.0]

    report = validate_vrma(Glb(document, original.binary))

    assert any("accessor max" in error and "does not match" in error for error in report.errors)


def test_validator_rejects_duplicate_node_children(
    identity_clip: MotionClip,
    hy_skeleton: Skeleton,
) -> None:
    original = build_vrma(identity_clip, hy_skeleton)
    document = copy.deepcopy(original.document)
    document["nodes"][0]["children"].append(document["nodes"][0]["children"][0])

    report = validate_vrma(Glb(document, original.binary))

    assert any("more than once" in error for error in report.errors)


def test_validator_rejects_node_cycles(
    identity_clip: MotionClip,
    hy_skeleton: Skeleton,
) -> None:
    original = build_vrma(identity_clip, hy_skeleton)
    document = copy.deepcopy(original.document)
    document["nodes"][-1]["children"] = [0]

    report = validate_vrma(Glb(document, original.binary))

    assert any("hierarchy contains a cycle" in error for error in report.errors)


@pytest.mark.parametrize(
    "animation",
    [
        {},
        {"samplers": [], "channels": []},
    ],
)
def test_reader_rejects_missing_animation_samplers(
    animation: dict,
    identity_clip: MotionClip,
    hy_skeleton: Skeleton,
) -> None:
    original = build_vrma(identity_clip, hy_skeleton)
    document = copy.deepcopy(original.document)
    document["animations"] = [animation]

    with pytest.raises(VrmaError, match="no samplers"):
        read_vrma(Glb(document, original.binary))

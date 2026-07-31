from __future__ import annotations

import copy
import hashlib
import math
from itertools import pairwise

import pytest

import vroom.curation as curation
from vroom.cli import main
from vroom.curation import (
    CURATION_METADATA_KEY,
    IDENTITY_QUATERNION,
    BoneCurationPolicy,
    HipsTranslationPolicy,
    curate_vrma,
)
from vroom.errors import VrmaError
from vroom.glb import AccessorReader, Glb
from vroom.motion import Bone, MotionClip, Quat, Skeleton
from vroom.vrm import HUMANOID_PARENTS, REQUIRED_HUMANOID_BONES
from vroom.vrma import build_vrma, read_vrma, validate_vrma

_EXTRA_BODY_BONES = {
    "chest",
    "upperChest",
    "neck",
    "leftShoulder",
    "rightShoulder",
}


def _body_skeleton() -> Skeleton:
    names = tuple(
        name
        for name in HUMANOID_PARENTS
        if name in REQUIRED_HUMANOID_BONES or name in _EXTRA_BODY_BONES
    )
    indices: dict[str, int] = {}
    bones: list[Bone] = []
    positions: list[tuple[float, float, float]] = []
    for name in names:
        parent_name = HUMANOID_PARENTS[name]
        while parent_name is not None and parent_name not in indices:
            parent_name = HUMANOID_PARENTS[parent_name]
        parent = None if parent_name is None else indices[parent_name]
        translation = (0.0, 1.0, 0.0) if parent is None else (0.0, 0.1, 0.0)
        parent_position = (0.0, 0.0, 0.0) if parent is None else positions[parent]
        position = tuple(parent_position[axis] + translation[axis] for axis in range(3))
        indices[name] = len(bones)
        bones.append(Bone(name=name, parent=parent, translation=translation))
        positions.append(position)
    return Skeleton(tuple(bones), tuple(positions), source_floor_y=0.0)


def _rotation_z(degrees: float, *, negate: bool = False) -> Quat:
    half_angle = math.radians(degrees) / 2
    result = (0.0, 0.0, math.sin(half_angle), math.cos(half_angle))
    return tuple(-value for value in result) if negate else result


def _rotation_axis(x: float, y: float, z: float, degrees: float) -> Quat:
    length = math.sqrt(x * x + y * y + z * z)
    half_angle = math.radians(degrees) / 2
    scale = math.sin(half_angle) / length
    return (x * scale, y * scale, z * scale, math.cos(half_angle))


def _multiply_rotations(left: Quat, right: Quat) -> Quat:
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    return (
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    )


def _source_glb(sample_times: tuple[float, ...] | None = None) -> Glb:
    skeleton = _body_skeleton()
    frames = 6 if sample_times is None else len(sample_times)
    return build_vrma(
        MotionClip(
            fps=30.0,
            hips_translations=tuple((0.0, 1.0, frame * 0.01) for frame in range(frames)),
            rotations={
                "head": tuple(
                    _rotation_z(20.0 + frame * 20.0, negate=frame == 2) for frame in range(frames)
                ),
                "leftUpperArm": tuple(_rotation_z(-10.0 + frame * 10.0) for frame in range(frames)),
                "rightUpperArm": tuple(_rotation_z(5.0 + frame * 5.0) for frame in range(frames)),
            },
            name="source motion",
            sample_times=sample_times,
        ),
        skeleton,
    )


def _returnable_source_glb() -> Glb:
    skeleton = _body_skeleton()
    translation_z = (0.0, 0.01, 0.025, 0.04, 0.035, 0.03)
    return build_vrma(
        MotionClip(
            fps=30.0,
            hips_translations=tuple((0.0, 1.0, value) for value in translation_z),
            rotations={
                "head": tuple(
                    _rotation_z(degrees) for degrees in (20.0, 40.0, 70.0, 100.0, 90.0, 80.0)
                ),
                "leftUpperLeg": tuple(
                    _rotation_z(degrees) for degrees in (5.0, 15.0, 30.0, 45.0, 40.0, 35.0)
                ),
            },
            name="returnable full-body motion",
        ),
        skeleton,
    )


def _translation_source_glb(translation_z: tuple[float, ...]) -> Glb:
    skeleton = _body_skeleton()
    frames = len(translation_z)
    return build_vrma(
        MotionClip(
            fps=30.0,
            hips_translations=tuple((0.0, 1.0, value) for value in translation_z),
            rotations={
                "head": tuple(
                    _rotation_z(degrees)
                    for degrees in (20.0, 40.0, 70.0, 100.0, 90.0, 80.0)[:frames]
                ),
            },
            name="translation boundary motion",
        ),
        skeleton,
    )


def _rotation_tracks(glb: Glb) -> dict[str, tuple[tuple[int | float, ...], ...]]:
    vrma = read_vrma(glb)
    bone_by_node = {node: bone for bone, node in vrma.humanoid_bones.items()}
    reader = AccessorReader(glb)
    samplers = vrma.animation["samplers"]
    return {
        bone_by_node[channel["target"]["node"]]: reader.read(samplers[channel["sampler"]]["output"])
        for channel in vrma.animation["channels"]
        if channel["target"]["path"] == "rotation"
    }


def _translation_track(glb: Glb) -> tuple[tuple[int | float, ...], ...] | None:
    vrma = read_vrma(glb)
    reader = AccessorReader(glb)
    samplers = vrma.animation["samplers"]
    for channel in vrma.animation["channels"]:
        if channel["target"]["path"] == "translation":
            return reader.read(samplers[channel["sampler"]]["output"])
    return None


def test_curate_selects_relative_rotation_tracks_and_retimes() -> None:
    source = _source_glb()
    source_bytes = source.to_bytes()

    curated = curate_vrma(
        source_bytes,
        bones=("leftUpperArm", "head"),
        duration=2.0,
        fade_in=1.0,
        fade_out=1.0,
        start_frame=0,
        end_frame=4,
        name="greeting",
    )

    report = validate_vrma(curated)
    assert report.errors == []
    assert read_vrma(curated).duration == pytest.approx(2.0)
    assert read_vrma(curated).frame_count == 5
    assert curated.document["animations"][0]["name"] == "greeting"

    channels = curated.document["animations"][0]["channels"]
    assert {channel["target"]["path"] for channel in channels} == {"rotation"}
    tracks = _rotation_tracks(curated)
    assert set(tracks) == {"head", "leftUpperArm"}
    for values in tracks.values():
        assert values[0] == pytest.approx((0.0, 0.0, 0.0, 1.0))
        assert values[-1] == pytest.approx((0.0, 0.0, 0.0, 1.0))
        assert all(
            sum(a * b for a, b in zip(previous, current, strict=True)) >= 0
            for previous, current in pairwise(values)
        )

    # At one second the fade has reached full weight: 60 degrees - 20 degrees.
    assert tracks["head"][2] == pytest.approx(_rotation_z(40.0), abs=1e-6)
    metadata = curated.document["asset"]["extras"][CURATION_METADATA_KEY]
    assert metadata["source"] == {
        "sha256": hashlib.sha256(source_bytes).hexdigest(),
        "byteLength": len(source_bytes),
    }
    assert metadata["version"] == 3
    assert metadata["transform"]["bones"] == ["head", "leftUpperArm"]
    assert metadata["transform"]["startFrame"] == 0
    assert metadata["transform"]["endFrame"] == 4
    assert metadata["transform"]["rotationGain"] == 1.0
    assert metadata["transform"]["rotationProjection"] == "none"
    assert metadata["hipsTranslationPolicy"] == "none"
    assert metadata["mainDurationMicroseconds"] == 2_000_000
    assert metadata["returnDurationMicroseconds"] == 0
    assert metadata["durationMicroseconds"] == 2_000_000
    assert metadata["maximumRootDisplacementMeters"] == 0.0
    assert metadata["maximumRootVelocityMetersPerSecond"] == 0.0
    assert metadata["transform"]["hipsTranslationReturn"] == {"mode": "none"}


def test_curate_is_byte_deterministic() -> None:
    payload = _source_glb().to_bytes()
    arguments = {
        "bones": ("head",),
        "duration": 1.25,
        "fade_in": 0.25,
        "fade_out": 0.25,
        "rotation_gain": 1.75,
        "start_frame": 1,
        "end_frame": 5,
    }

    first = curate_vrma(payload, **arguments).to_bytes()
    second = curate_vrma(payload, **arguments).to_bytes()

    assert first == second


def test_curate_relative_return_is_byte_deterministic() -> None:
    payload = _returnable_source_glb().to_bytes()
    arguments = {
        "bones": ("head", "leftUpperLeg"),
        "duration": 1.0,
        "bone_policies": (BoneCurationPolicy(bone="leftUpperLeg", rotation_gain=0.8),),
        "hips_translation_policy": HipsTranslationPolicy(
            mode="relative-return",
            return_duration=0.5,
        ),
    }

    first = curate_vrma(payload, **arguments).to_bytes()
    second = curate_vrma(payload, **arguments).to_bytes()

    assert first == second


def test_curate_rotation_gain_amplifies_relative_rotations() -> None:
    curated = curate_vrma(
        _source_glb(),
        bones=("head",),
        duration=2.5,
        rotation_gain=2.5,
        fade_out=1.5,
    )

    tracks = _rotation_tracks(curated)
    # Frame 2 is 40 degrees from the first frame; gain 2.5 extrapolates to 100.
    # The fade begins after this sample and gives the amplified track a
    # shortest-path-safe return to exact identity.
    assert tracks["head"][2] == pytest.approx(_rotation_z(100.0), abs=1e-6)
    metadata = curated.document["asset"]["extras"][CURATION_METADATA_KEY]
    assert metadata["transform"]["rotationGain"] == 2.5


def test_curate_uses_inverse_first_times_frame_for_noncommuting_rotations() -> None:
    skeleton = _body_skeleton()
    first = _rotation_axis(1.0, 0.0, 0.0, 70.0)
    relative_y = _rotation_axis(0.0, 1.0, 0.0, 35.0)
    second = _multiply_rotations(first, relative_y)
    source = build_vrma(
        MotionClip(
            fps=30.0,
            hips_translations=(),
            rotations={"head": (first, second, first)},
            name="noncommuting source",
        ),
        skeleton,
    )

    curated = curate_vrma(source, bones=("head",), duration=1.0)

    track = _rotation_tracks(curated)["head"]
    assert track[0] == pytest.approx(IDENTITY_QUATERNION)
    assert track[1] == pytest.approx(relative_y, abs=1e-6)
    assert track[2] == pytest.approx(IDENTITY_QUATERNION)


@pytest.mark.parametrize(
    ("projection", "twist_axis", "swing_axis"),
    [
        ("twist-x", (1.0, 0.0, 0.0), (0.0, 1.0, 1.0)),
        ("twist-y", (0.0, 1.0, 0.0), (1.0, 0.0, 1.0)),
        ("twist-z", (0.0, 0.0, 1.0), (1.0, 1.0, 0.0)),
    ],
)
def test_curate_keeps_only_twist_around_selected_local_axis(
    projection: str,
    twist_axis: tuple[float, float, float],
    swing_axis: tuple[float, float, float],
) -> None:
    skeleton = _body_skeleton()
    identity = IDENTITY_QUATERNION
    first_swing = _rotation_axis(*swing_axis, 30.0)
    second_swing = _rotation_axis(
        -swing_axis[0],
        -swing_axis[1],
        -swing_axis[2],
        25.0,
    )
    first_twist = _rotation_axis(*twist_axis, 40.0)
    second_twist = _rotation_axis(*twist_axis, -35.0)
    source = build_vrma(
        MotionClip(
            fps=30.0,
            hips_translations=(),
            rotations={
                "head": (
                    identity,
                    _multiply_rotations(first_swing, first_twist),
                    _multiply_rotations(second_swing, second_twist),
                    identity,
                ),
            },
            name="mixed head rotation",
        ),
        skeleton,
    )

    curated = curate_vrma(
        source,
        bones=("head",),
        duration=1.0,
        rotation_projection=projection,
    )

    track = _rotation_tracks(curated)["head"]
    assert track[0] == pytest.approx(identity)
    assert track[1] == pytest.approx(first_twist, abs=1e-6)
    assert track[2] == pytest.approx(second_twist, abs=1e-6)
    assert track[3] == pytest.approx(identity)
    metadata = curated.document["asset"]["extras"][CURATION_METADATA_KEY]
    assert metadata["version"] == 3
    assert metadata["transform"]["rotationProjection"] == projection


def test_curate_zero_rotation_gain_produces_neutral_tracks() -> None:
    curated = curate_vrma(
        _source_glb(),
        bones=("head", "leftUpperArm"),
        duration=1.0,
        rotation_gain=0.0,
    )

    for values in _rotation_tracks(curated).values():
        assert values == pytest.approx([IDENTITY_QUATERNION] * len(values))


def test_curate_applies_per_bone_rotation_policies() -> None:
    curated = curate_vrma(
        _source_glb(),
        bones=("head", "leftUpperArm"),
        duration=2.5,
        fade_out=1.5,
        rotation_gain=0.5,
        bone_policies=(
            BoneCurationPolicy(
                bone="head",
                rotation_gain=2.0,
                rotation_projection="twist-z",
            ),
        ),
    )

    tracks = _rotation_tracks(curated)
    # Frame 2 is 40 degrees from the head's first frame and 20 degrees from
    # the arm's first frame. The head override and global arm gain differ.
    assert tracks["head"][2] == pytest.approx(_rotation_z(80.0), abs=1e-6)
    assert tracks["leftUpperArm"][2] == pytest.approx(_rotation_z(10.0), abs=1e-6)
    policies = curated.document["asset"]["extras"][CURATION_METADATA_KEY]["perBonePolicies"]
    assert policies == [
        {
            "bone": "head",
            "rotationGain": 2.0,
            "rotationProjection": "twist-z",
        },
        {
            "bone": "leftUpperArm",
            "rotationGain": 0.5,
            "rotationProjection": "none",
        },
    ]


def test_curate_relative_return_preserves_main_full_body_motion_and_closes() -> None:
    curated = curate_vrma(
        _returnable_source_glb(),
        bones=("head", "leftUpperLeg"),
        duration=1.0,
        fade_in=0.0,
        fade_out=0.0,
        hips_translation_policy=HipsTranslationPolicy(
            mode="relative-return",
            return_duration=0.5,
        ),
    )

    vrma = read_vrma(curated)
    translations = _translation_track(curated)
    assert translations is not None
    tracks = _rotation_tracks(curated)
    rest = (0.0, 1.0, 0.0)

    # A one-sample rest hold precedes every original main sample. Four return
    # intervals end with a second exact-rest sample.
    assert vrma.frame_count == 11
    assert vrma.duration == pytest.approx(1.7)
    for value in translations[:2]:
        assert value == pytest.approx(rest)
    for actual, expected_z in zip(
        translations[1:7],
        (0.0, 0.01, 0.025, 0.04, 0.035, 0.03),
        strict=True,
    ):
        assert actual == pytest.approx((0.0, 1.0, expected_z))
    for value in translations[-2:]:
        assert value == pytest.approx(rest)
    for value in tracks["head"][:2]:
        assert value == pytest.approx(IDENTITY_QUATERNION)
    for actual, expected_degrees in zip(
        tracks["head"][1:7],
        (0.0, 20.0, 50.0, 80.0, 70.0, 60.0),
        strict=True,
    ):
        assert actual == pytest.approx(_rotation_z(expected_degrees), abs=1e-6)
    assert tracks["head"][6] != pytest.approx(IDENTITY_QUATERNION)
    assert tracks["head"][7] != pytest.approx(IDENTITY_QUATERNION)
    for value in tracks["head"][-2:]:
        assert value == pytest.approx(IDENTITY_QUATERNION)

    # The first sampled return segment exactly continues the incoming LINEAR
    # root velocity; the terminal hold makes the endpoint velocity zero.
    incoming = tuple(
        (translations[6][axis] - translations[5][axis]) / (vrma.times[6] - vrma.times[5])
        for axis in range(3)
    )
    first_return = tuple(
        (translations[7][axis] - translations[6][axis]) / (vrma.times[7] - vrma.times[6])
        for axis in range(3)
    )
    assert first_return == pytest.approx(incoming, abs=1e-6)

    metadata = curated.document["asset"]["extras"][CURATION_METADATA_KEY]
    transform = metadata["transform"]
    assert transform["durationSeconds"] == pytest.approx(1.7)
    assert transform["mainDurationSeconds"] == 1.0
    assert transform["endpoints"] == "rest with zero start and end velocity"
    assert metadata["hipsTranslationPolicy"] == "relative-return"
    assert metadata["mainDurationMicroseconds"] == 1_000_000
    assert metadata["returnDurationMicroseconds"] == 500_000
    assert metadata["durationMicroseconds"] == 1_700_000
    assert transform["hipsTranslationReturn"] == {
        "mode": "relative-return",
        "basis": "rest + (frame - first)",
        "mainSamples": "preserved",
        "startHoldSeconds": pytest.approx(0.2),
        "returnDurationSeconds": 0.5,
        "returnIntervals": 4,
        "returnInterpolation": "sampled cubic Hermite",
        "endpointVelocity": "zero",
        "maximumExtraDisplacementRatio": 0.25,
        "minimumExtraDisplacementMeters": 0.05,
        "absoluteMaximumDisplacementMeters": 0.5,
    }
    measured_displacement = max(math.dist(value, rest) for value in translations)
    measured_velocity = max(
        math.dist(translations[index], translations[index - 1])
        / (vrma.times[index] - vrma.times[index - 1])
        for index in range(1, len(translations))
    )
    assert metadata["maximumRootDisplacementMeters"] >= measured_displacement
    assert metadata["maximumRootVelocityMetersPerSecond"] >= measured_velocity
    assert metadata["maximumRootDisplacementMeters"] == pytest.approx(
        measured_displacement,
        rel=1e-6,
    )
    assert metadata["maximumRootVelocityMetersPerSecond"] == pytest.approx(
        measured_velocity,
        rel=1e-6,
    )


def test_curate_relative_return_allows_bounded_continuation_past_main_peak() -> None:
    curated = curate_vrma(
        _source_glb(),
        bones=("head",),
        duration=1.0,
        hips_translation_policy=HipsTranslationPolicy(
            mode="relative-return",
            return_duration=0.5,
        ),
    )

    translations = _translation_track(curated)
    assert translations is not None
    displacements = [math.dist(value, translations[0]) for value in translations]
    assert max(displacements) > 0.05
    assert max(displacements) <= 0.1
    head = _rotation_tracks(curated)["head"]
    angles = [2.0 * math.acos(min(1.0, abs(value[3]))) for value in head]
    assert max(angles[7:]) > max(angles[1:7])
    assert max(angles) <= math.pi


@pytest.mark.parametrize(
    ("translation_z", "message"),
    [
        ((0.0, 0.01, 0.02, 0.03, 0.04, 0.2), "bounded translation continuation"),
        ((0.0, 0.1, 0.2, 0.3, 0.4, 0.51), "absolute 0.5 meter"),
    ],
)
def test_curate_relative_return_rejects_root_motion_beyond_caps(
    translation_z: tuple[float, ...],
    message: str,
) -> None:
    with pytest.raises(VrmaError, match=message):
        curate_vrma(
            _translation_source_glb(translation_z),
            bones=("head",),
            duration=1.0,
            hips_translation_policy=HipsTranslationPolicy(
                mode="relative-return",
                return_duration=0.5,
            ),
        )


def test_curate_preserves_relative_sample_timing() -> None:
    source = _source_glb((0.0, 0.1, 0.4, 1.0, 2.0, 3.0))

    curated = curate_vrma(
        source,
        bones=("head",),
        duration=3.8,
        start_frame=1,
        end_frame=4,
    )

    assert read_vrma(curated).times == pytest.approx((0.0, 0.6, 1.8, 3.8))
    assert read_vrma(curated).duration == pytest.approx(3.8)


def test_curate_keyframe_times_do_not_round_past_requested_duration() -> None:
    curated = curate_vrma(
        _source_glb(),
        bones=("head",),
        duration=1.2,
        fade_in=0.2,
        fade_out=0.3,
    )
    vrma = read_vrma(curated)

    assert vrma.times[-1] <= 1.2
    assert 1.2 - vrma.times[-1] < 1e-6
    assert _rotation_tracks(curated)["head"][-1] == IDENTITY_QUATERNION


def test_curate_rejects_an_identity_endpoint_with_no_shortest_path() -> None:
    skeleton = _body_skeleton()
    source = build_vrma(
        MotionClip(
            fps=30.0,
            hips_translations=(),
            rotations={
                "head": (
                    IDENTITY_QUATERNION,
                    _rotation_z(170.0),
                    _rotation_z(340.0),
                    IDENTITY_QUATERNION,
                )
            },
            name="winding source",
        ),
        skeleton,
    )

    with pytest.raises(VrmaError, match="cannot reach the exact identity endpoint"):
        curate_vrma(source, bones=("head",), duration=1.0)


def test_curate_bounds_source_bytes(monkeypatch) -> None:
    payload = _source_glb().to_bytes()
    monkeypatch.setattr(curation, "MAX_CURATION_SOURCE_BYTES", len(payload) - 1)

    with pytest.raises(VrmaError, match="curation source has"):
        curate_vrma(payload, bones=("head",), duration=1.0)


def test_curate_caps_file_read_before_glb_decode(tmp_path, monkeypatch) -> None:
    source = tmp_path / "source.vrma"
    _source_glb().write(source)
    monkeypatch.setattr(curation, "MAX_CURATION_SOURCE_BYTES", 11)

    with pytest.raises(VrmaError, match="curation source has"):
        curate_vrma(source, bones=("head",), duration=1.0)


def test_curate_bounds_source_nodes(monkeypatch) -> None:
    source = _source_glb()
    node_count = len(source.document["nodes"])
    monkeypatch.setattr(curation, "MAX_CURATION_SOURCE_NODES", node_count - 1)

    with pytest.raises(VrmaError, match="nodes"):
        curate_vrma(source, bones=("head",), duration=1.0)


def test_curate_bounds_source_frames(monkeypatch) -> None:
    monkeypatch.setattr(curation, "MAX_CURATION_SOURCE_FRAMES", 5)

    with pytest.raises(VrmaError, match="6 samples"):
        curate_vrma(_source_glb(), bones=("head",), duration=1.0)


def test_curate_counts_overlapping_accessors_against_global_component_budget(monkeypatch) -> None:
    source = _source_glb()
    document = copy.deepcopy(source.document)
    document["accessors"].append(copy.deepcopy(document["accessors"][0]))
    overlapping = Glb(document, source.binary)
    monkeypatch.setattr(curation, "MAX_CURATION_SOURCE_COMPONENTS", 101)

    with pytest.raises(VrmaError, match="102 numeric components"):
        curate_vrma(overlapping, bones=("head",), duration=1.0)


def test_curate_bounds_source_tracks(monkeypatch) -> None:
    monkeypatch.setattr(curation, "MAX_CURATION_SOURCE_TRACKS", 3)

    with pytest.raises(VrmaError, match="4 animation channels"):
        curate_vrma(_source_glb(), bones=("head",), duration=1.0)


def test_curate_bounds_serialized_output(monkeypatch) -> None:
    monkeypatch.setattr(curation, "MAX_CURATION_OUTPUT_BYTES", 1)

    with pytest.raises(VrmaError, match="curated output has"):
        curate_vrma(_source_glb(), bones=("head",), duration=1.0)


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"bones": (), "duration": 1.0}, "at least one"),
        ({"bones": ("head", "head"), "duration": 1.0}, "must not be repeated"),
        ({"bones": ("head",), "duration": 0.0}, "positive finite"),
        (
            {"bones": ("head",), "duration": 1.0, "rotation_gain": -0.01},
            "rotation gain",
        ),
        (
            {"bones": ("head",), "duration": 1.0, "rotation_gain": 4.01},
            "rotation gain",
        ),
        (
            {"bones": ("head",), "duration": 1.0, "rotation_gain": math.inf},
            "rotation gain",
        ),
        (
            {"bones": ("head",), "duration": 1.0, "rotation_projection": "yaw"},
            "rotation projection",
        ),
        (
            {"bones": ("head",), "duration": 1.0, "rotation_projection": None},
            "rotation projection",
        ),
        (
            {"bones": ("head",), "duration": 1.0, "fade_in": -0.1},
            "non-negative finite",
        ),
        (
            {
                "bones": ("head",),
                "duration": 1.0,
                "fade_in": 0.6,
                "fade_out": 0.6,
            },
            "must not overlap",
        ),
        (
            {"bones": ("head",), "duration": 1.0, "start_frame": 0, "end_frame": 0},
            "at least two",
        ),
        (
            {"bones": ("head",), "duration": 1.0, "end_frame": 99},
            "outside",
        ),
        ({"bones": ("leftHand",), "duration": 1.0}, "no rotation tracks"),
        ({"bones": ("tail",), "duration": 1.0}, "no humanoid bones"),
    ],
)
def test_curate_rejects_invalid_arguments(arguments: dict, message: str) -> None:
    with pytest.raises(VrmaError, match=message):
        curate_vrma(_source_glb(), **arguments)


def test_curate_rejects_unselected_or_repeated_bone_policies() -> None:
    with pytest.raises(VrmaError, match="require selected bones: leftUpperArm"):
        curate_vrma(
            _source_glb(),
            bones=("head",),
            duration=1.0,
            bone_policies=(BoneCurationPolicy(bone="leftUpperArm", rotation_gain=0.5),),
        )

    repeated = (
        BoneCurationPolicy(bone="head", rotation_gain=0.5),
        BoneCurationPolicy(bone="head", rotation_projection="twist-z"),
    )
    with pytest.raises(VrmaError, match="must not repeat"):
        curate_vrma(
            _source_glb(),
            bones=("head",),
            duration=1.0,
            bone_policies=repeated,
        )


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"mode": "keep", "return_duration": 0.0}, "mode"),
        ({"mode": "strip", "return_duration": 0.5}, "requires relative-return"),
        ({"mode": "relative-return", "return_duration": 0.0}, "positive"),
        ({"mode": "relative-return", "return_duration": math.inf}, "finite"),
    ],
)
def test_hips_translation_policy_rejects_invalid_values(
    arguments: dict,
    message: str,
) -> None:
    with pytest.raises(VrmaError, match=message):
        HipsTranslationPolicy(**arguments)


def test_curate_relative_return_requires_hips_translation() -> None:
    skeleton = _body_skeleton()
    source = build_vrma(
        MotionClip(
            fps=30.0,
            hips_translations=(),
            rotations={
                "head": (
                    IDENTITY_QUATERNION,
                    _rotation_z(20.0),
                    _rotation_z(10.0),
                ),
            },
        ),
        skeleton,
    )

    with pytest.raises(VrmaError, match="requires a hips translation track"):
        curate_vrma(
            source,
            bones=("head",),
            duration=1.0,
            hips_translation_policy=HipsTranslationPolicy(
                mode="relative-return",
                return_duration=0.5,
            ),
        )


def test_curate_rejects_non_linear_input() -> None:
    source = _source_glb()
    document = copy.deepcopy(source.document)
    document["animations"][0]["samplers"][0]["interpolation"] = "STEP"

    with pytest.raises(VrmaError, match="sampled LINEAR"):
        curate_vrma(
            Glb(document, source.binary),
            bones=("head",),
            duration=1.0,
        )


def test_curate_cli_writes_and_validates_output(tmp_path, capsys) -> None:
    source = tmp_path / "source.vrma"
    output = tmp_path / "curated.vrma"
    _source_glb().write(source)

    code = main(
        [
            "curate",
            str(source),
            "--output",
            str(output),
            "--bones",
            "head,leftUpperArm",
            "--duration",
            "1.5",
            "--fade-in",
            "0.25",
            "--fade-out",
            "0.25",
            "--rotation-gain",
            "1.5",
            "--rotation-projection",
            "twist-z",
            "--start-frame",
            "1",
            "--end-frame",
            "5",
        ]
    )

    assert code == 0
    assert "2 rotation tracks" in capsys.readouterr().out
    assert output.is_file()
    assert validate_vrma(output).errors == []
    metadata = Glb.read(output).document["asset"]["extras"][CURATION_METADATA_KEY]
    assert metadata["transform"]["rotationGain"] == 1.5
    assert metadata["transform"]["rotationProjection"] == "twist-z"


def test_curate_cli_applies_full_body_policies_and_relative_return(
    tmp_path,
    capsys,
) -> None:
    source = tmp_path / "source.vrma"
    output = tmp_path / "curated.vrma"
    _returnable_source_glb().write(source)

    code = main(
        [
            "curate",
            str(source),
            "--output",
            str(output),
            "--bones",
            "head,leftUpperLeg",
            "--duration",
            "1",
            "--fade-in",
            "0",
            "--fade-out",
            "0",
            "--bone-rotation-gain",
            "leftUpperLeg=0.75",
            "--bone-rotation-projection",
            "head=twist-z",
            "--hips-translation",
            "relative-return",
            "--hips-return-duration",
            "0.5",
        ]
    )

    assert code == 0
    assert "first-relative hips translation" in capsys.readouterr().out
    assert validate_vrma(output).errors == []
    assert _translation_track(Glb.read(output)) is not None
    metadata = Glb.read(output).document["asset"]["extras"][CURATION_METADATA_KEY]
    assert metadata["perBonePolicies"] == [
        {
            "bone": "leftUpperLeg",
            "rotationGain": 0.75,
            "rotationProjection": "none",
        },
        {
            "bone": "head",
            "rotationGain": 1.0,
            "rotationProjection": "twist-z",
        },
    ]
    assert metadata["hipsTranslationPolicy"] == "relative-return"


def test_curate_cli_rejects_malformed_bone_policy(tmp_path, capsys) -> None:
    source = tmp_path / "source.vrma"
    output = tmp_path / "curated.vrma"
    _source_glb().write(source)

    code = main(
        [
            "curate",
            str(source),
            "--output",
            str(output),
            "--bones",
            "head",
            "--duration",
            "1",
            "--fade-in",
            "0",
            "--fade-out",
            "0",
            "--bone-rotation-gain",
            "head=strong",
        ]
    )

    assert code == 2
    assert "head rotation gain must be a number" in capsys.readouterr().err
    assert not output.exists()


def test_curate_cli_reports_invalid_arguments_without_writing(tmp_path, capsys) -> None:
    source = tmp_path / "source.vrma"
    output = tmp_path / "curated.vrma"
    _source_glb().write(source)

    code = main(
        [
            "curate",
            str(source),
            "--output",
            str(output),
            "--bones",
            "head",
            "--duration",
            "1.0",
            "--fade-in",
            "0.75",
            "--fade-out",
            "0.75",
        ]
    )

    assert code == 2
    assert "must not overlap" in capsys.readouterr().err
    assert not output.exists()


def test_curate_cli_reports_float32_unsafe_duration_without_writing(tmp_path, capsys) -> None:
    source = tmp_path / "source.vrma"
    output = tmp_path / "curated.vrma"
    _source_glb().write(source)

    code = main(
        [
            "curate",
            str(source),
            "--output",
            str(output),
            "--bones",
            "head",
            "--duration",
            "3.5e38",
            "--fade-in",
            "0",
            "--fade-out",
            "0",
        ]
    )

    assert code == 2
    assert "must not exceed 3600 seconds" in capsys.readouterr().err
    assert not output.exists()

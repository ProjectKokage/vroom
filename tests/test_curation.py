from __future__ import annotations

import copy
import hashlib
import math
from itertools import pairwise

import pytest

import vroom.curation as curation
from vroom.cli import main
from vroom.curation import CURATION_METADATA_KEY, IDENTITY_QUATERNION, curate_vrma
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
    assert metadata["version"] == 2
    assert metadata["transform"]["bones"] == ["head", "leftUpperArm"]
    assert metadata["transform"]["startFrame"] == 0
    assert metadata["transform"]["endFrame"] == 4
    assert metadata["transform"]["rotationGain"] == 1.0
    assert metadata["transform"]["rotationProjection"] == "none"


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
    assert metadata["version"] == 2
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

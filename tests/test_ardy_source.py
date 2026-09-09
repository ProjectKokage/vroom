from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pytest

import vroom.ardy_source as sources
from vroom.ardy_source import ArdySource, read_source, validate_source, write_source
from vroom.errors import ArdyError

_BUDGET = 128 * 1024


def _arrays(frames: int = 3, dtype=np.float32) -> dict[str, np.ndarray]:
    rotations = np.broadcast_to(np.eye(3, dtype=dtype), (frames, 27, 3, 3)).copy()
    roots = np.zeros((frames, 3), dtype=dtype)
    roots[:, 1] = 1
    roots[:, 2] = np.arange(frames) * 0.05
    return {"local_rot_mats": rotations, "root_positions": roots, "fps": np.asarray(20)}


def _source() -> ArdySource:
    arrays = _arrays()
    return validate_source(
        arrays["local_rot_mats"],
        arrays["root_positions"],
        20,
        max_bytes=_BUDGET,
    )


def _npy(array: np.ndarray) -> bytes:
    stream = io.BytesIO()
    np.lib.format.write_array(stream, array)
    return stream.getvalue()


def _write_zip(path: Path, members: list[tuple[str, bytes]]) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in members:
            archive.writestr(name, data)


def test_source_export_roundtrip_is_readonly_and_contains_no_prompt(tmp_path: Path) -> None:
    source = _source()
    path = write_source(tmp_path / "nested" / "motion.npz", source, model="core", max_bytes=_BUDGET)

    with np.load(path, allow_pickle=False) as saved:
        assert set(saved.files) == {
            "local_rot_mats",
            "root_positions",
            "fps",
            "skeleton",
            "source_version",
            "model",
        }
        assert saved["skeleton"].item() == "core27"
        assert saved["source_version"].item() == 1
        assert saved["model"].item() == "core"
    actual = read_source(path, max_bytes=_BUDGET)
    assert actual.fps == 20
    assert actual.frame_count == 3
    np.testing.assert_array_equal(actual.local_rot_mats, source.local_rot_mats)
    np.testing.assert_array_equal(actual.root_positions, source.root_positions)
    assert actual.local_rot_mats.dtype == actual.root_positions.dtype == np.float32
    assert not actual.local_rot_mats.flags.writeable
    assert not actual.root_positions.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        actual.root_positions[0, 0] = 1


def test_upstream_npz_optional_outputs_do_not_persist_text_on_reexport(tmp_path: Path) -> None:
    path = tmp_path / "upstream.npz"
    arrays = _arrays(dtype=np.float64)
    arrays.update(
        posed_joints=np.zeros((3, 27, 3)),
        global_rot_mats=arrays["local_rot_mats"],
        foot_contacts=np.zeros((3, 4)),
        smooth_root_pos=arrays["root_positions"],
        global_root_heading=np.asarray([[1, 0]] * 3),
        text=np.asarray("A private authoring prompt."),
    )
    np.savez(path, **arrays)

    source = read_source(path, max_bytes=_BUDGET)
    exported = write_source(tmp_path / "export.npz", source, model="core", max_bytes=_BUDGET)

    assert source.local_rot_mats.dtype == np.float32
    with np.load(exported, allow_pickle=False) as saved:
        assert "text" not in saved
    assert b"private authoring prompt" not in exported.read_bytes()


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64, np.dtype(">f4")])
def test_validate_source_copies_and_converts_numeric_arrays(dtype) -> None:
    arrays = _arrays(dtype=dtype)
    source = validate_source(
        arrays["local_rot_mats"],
        arrays["root_positions"],
        np.float32(20),
        max_bytes=_BUDGET,
    )
    arrays["root_positions"][:] = 99
    assert source.root_positions[0, 1] == 1
    assert source.local_rot_mats.dtype == source.root_positions.dtype == np.float32


def test_upstream_fortran_arrays_and_unsigned_fps_are_supported(tmp_path: Path) -> None:
    path = tmp_path / "fortran.npz"
    arrays = _arrays()
    arrays["root_positions"] = np.asfortranarray(arrays["root_positions"])
    arrays["local_rot_mats"] = np.asfortranarray(arrays["local_rot_mats"])
    arrays["fps"] = np.asarray(20, dtype=np.uint32)
    np.savez(path, **arrays)
    source = read_source(path, max_bytes=_BUDGET)
    np.testing.assert_array_equal(source.root_positions, arrays["root_positions"])
    assert source.fps == 20


@pytest.mark.parametrize(
    "change",
    [
        {"fps": np.asarray(True)},
        {"fps": np.asarray("20")},
        {"fps": np.asarray([20])},
        {"fps": np.asarray(0)},
        {"fps": np.asarray(-1)},
        {"fps": np.asarray(np.nan)},
        {"fps": np.asarray(np.inf)},
        {"skeleton": np.asarray("g1")},
        {"skeleton": np.asarray(["core27"])},
        {"skeleton": np.asarray(27)},
        {"source_version": np.asarray(2)},
        {"source_version": np.asarray(True)},
        {"source_version": np.asarray(1.0)},
        {"source_version": np.asarray([1])},
        {"model": np.asarray(1)},
        {"model": np.asarray("")},
        {"model": np.asarray(["core"])},
        {"unknown": np.asarray(1)},
        {"root_positions": np.zeros((2, 3), dtype=np.float32)},
        {"root_positions": np.zeros((3, 3), dtype=np.int32)},
        {"local_rot_mats": np.zeros((3, 22, 3, 3), dtype=np.float32)},
    ],
)
def test_source_rejects_bad_metadata_shapes_and_types(tmp_path: Path, change: dict) -> None:
    path = tmp_path / "bad.npz"
    np.savez(path, **(_arrays() | change))
    with pytest.raises(ArdyError):
        read_source(path, max_bytes=_BUDGET)


@pytest.mark.parametrize("missing", ["local_rot_mats", "root_positions", "fps"])
def test_source_requires_motion_arrays_and_fps(tmp_path: Path, missing: str) -> None:
    arrays = _arrays()
    del arrays[missing]
    path = tmp_path / "missing.npz"
    np.savez(path, **arrays)
    with pytest.raises(ArdyError, match="needs"):
        read_source(path, max_bytes=_BUDGET)


@pytest.mark.parametrize("field", ["local_rot_mats", "text", "model", "posed_joints"])
def test_source_refuses_object_arrays_before_numpy_load(
    tmp_path: Path, monkeypatch, field: str
) -> None:
    path = tmp_path / "object.npz"
    np.savez(path, **(_arrays() | {field: np.asarray({"payload": "not unpickled"}, dtype=object)}))

    def forbidden_load(*args, **kwargs):
        pytest.fail("object array reached NumPy loading")

    monkeypatch.setattr(np, "load", forbidden_load)
    with pytest.raises(ArdyError, match="object"):
        read_source(path, max_bytes=_BUDGET)


def test_source_refuses_duplicate_archive_keys(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.npz"
    members = [(f"{name}.npy", _npy(array)) for name, array in _arrays().items()]
    members.append(members[0])
    with pytest.warns(UserWarning, match="Duplicate name"):
        _write_zip(path, members)
    with pytest.raises(ArdyError, match="duplicate"):
        read_source(path, max_bytes=_BUDGET)


@pytest.mark.parametrize("frames", [10, 100_000_000])
def test_source_refuses_lying_npy_shape_before_allocation(
    tmp_path: Path, monkeypatch, frames: int
) -> None:
    stream = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        stream,
        {"descr": "<f4", "fortran_order": False, "shape": (frames, 27, 3, 3)},
    )
    path = tmp_path / "lying.npz"
    members = [
        (f"{name}.npy", _npy(array))
        for name, array in _arrays().items()
        if name != "local_rot_mats"
    ]
    members.append(("local_rot_mats.npy", stream.getvalue()))
    _write_zip(path, members)

    def forbidden_load(*args, **kwargs):
        pytest.fail("unverified NPY shape reached NumPy allocation")

    monkeypatch.setattr(np, "load", forbidden_load)
    with pytest.raises(ArdyError, match=r"shape|oversized"):
        read_source(path, max_bytes=_BUDGET)


def test_source_refuses_archive_expansion_before_numpy_load(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "compressed.npz"
    np.savez_compressed(path, **_arrays(frames=200))
    budget = path.stat().st_size + 1024

    def forbidden_load(*args, **kwargs):
        pytest.fail("oversized expansion reached NumPy allocation")

    monkeypatch.setattr(np, "load", forbidden_load)
    with pytest.raises(ArdyError, match="expanded"):
        read_source(path, max_bytes=budget)


@pytest.mark.parametrize("failure", ["nan", "infinity", "reflection", "shear", "overflow"])
def test_source_rejects_invalid_motion_values(tmp_path: Path, failure: str) -> None:
    arrays = _arrays(dtype=np.float64)
    if failure == "nan":
        arrays["local_rot_mats"][1, 4, 0, 0] = np.nan
    elif failure == "infinity":
        arrays["root_positions"][1, 0] = np.inf
    elif failure == "reflection":
        arrays["local_rot_mats"][1, 4, 0, 0] = -1
    elif failure == "shear":
        arrays["local_rot_mats"][1, 4, 0, 1] = 0.5
    else:
        arrays["root_positions"][1, 0] = 1e100
    path = tmp_path / "invalid.npz"
    np.savez(path, **arrays)
    with pytest.raises(ArdyError, match=r"non-finite|improper|orthogonal|float32"):
        read_source(path, max_bytes=_BUDGET)


@pytest.mark.parametrize("fps", [True, "20", None, 0, -1, float("nan"), float("inf"), 10**1000])
def test_python_source_validator_rejects_bad_fps(fps) -> None:
    arrays = _arrays()
    with pytest.raises(ArdyError, match="fps"):
        validate_source(arrays["local_rot_mats"], arrays["root_positions"], fps, max_bytes=_BUDGET)


def test_float16_conversion_respects_float32_memory_budget() -> None:
    arrays = _arrays(dtype=np.float16)
    budget = arrays["local_rot_mats"].nbytes + arrays["root_positions"].nbytes
    with pytest.raises(ArdyError, match="memory budget"):
        validate_source(arrays["local_rot_mats"], arrays["root_positions"], 20, max_bytes=budget)


def test_float16_file_expansion_is_checked_before_numpy_load(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "half.npz"
    arrays = _arrays(frames=20, dtype=np.float16)
    np.savez(path, **arrays)
    budget = path.stat().st_size
    monkeypatch.setattr(
        np, "load", lambda *args, **kwargs: pytest.fail("float32 would exceed budget")
    )
    with pytest.raises(ArdyError, match="float32 motion"):
        read_source(path, max_bytes=budget)


@pytest.mark.parametrize("budget", [True, 0, -1, 1.5, None])
def test_direct_source_api_rejects_invalid_budgets(budget) -> None:
    arrays = _arrays()
    with pytest.raises(ArdyError, match="budget"):
        validate_source(arrays["local_rot_mats"], arrays["root_positions"], 20, max_bytes=budget)


def test_source_file_size_is_checked_before_zip_decode(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "large.npz"
    path.write_bytes(b"xx")
    monkeypatch.setattr(
        sources.zipfile,
        "ZipFile",
        lambda *args: pytest.fail("oversized file reached ZIP parsing"),
    )
    with pytest.raises(ArdyError, match="file exceeds"):
        read_source(path, max_bytes=1)


def test_source_refuses_invalid_arrays_before_allocating_them(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "wrong-shape.npz"
    np.savez(path, **(_arrays() | {"local_rot_mats": np.zeros((3, 22, 3, 3))}))
    monkeypatch.setattr(np, "load", lambda *args, **kwargs: pytest.fail("invalid shape was loaded"))
    with pytest.raises(ArdyError, match="local_rot_mats"):
        read_source(path, max_bytes=_BUDGET)


@pytest.mark.parametrize("failure", ["serialize", "replace"])
def test_failed_source_write_preserves_destination_and_removes_temporary_file(
    tmp_path: Path,
    monkeypatch,
    failure: str,
) -> None:
    path = tmp_path / "existing.npz"
    path.write_bytes(b"original")

    def fail(*args, **kwargs):
        raise OSError("injected write failure")

    if failure == "serialize":
        monkeypatch.setattr(np, "savez", fail)
    else:
        monkeypatch.setattr(sources.os, "replace", fail)
    with pytest.raises(ArdyError, match="cannot write"):
        write_source(path, _source(), model="core", max_bytes=_BUDGET)
    assert path.read_bytes() == b"original"
    assert list(tmp_path.iterdir()) == [path]


def test_write_validates_raw_source_before_touching_destination(tmp_path: Path) -> None:
    path = tmp_path / "output.npz"
    arrays = _arrays()
    arrays["local_rot_mats"][0, 0, 0, 0] = -1
    source = ArdySource(arrays["local_rot_mats"], arrays["root_positions"], 20)
    with pytest.raises(ArdyError, match="improper"):
        write_source(path, source, model="core")
    assert not path.exists()


def test_write_archive_budget_prevents_promotion(tmp_path: Path) -> None:
    source = _source()
    budget = source.local_rot_mats.nbytes + source.root_positions.nbytes
    with pytest.raises(ArdyError, match="archive exceeds"):
        write_source(tmp_path / "output.npz", source, model="core", max_bytes=budget)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("payload", [b"not a zip", b"PK\x03\x04", b""])
def test_malformed_archives_fail_cleanly(tmp_path: Path, payload: bytes) -> None:
    path = tmp_path / "bad.npz"
    path.write_bytes(payload)
    with pytest.raises(ArdyError, match="cannot read"):
        read_source(path, max_bytes=_BUDGET)

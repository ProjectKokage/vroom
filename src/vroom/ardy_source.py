"""Bounded, pickle-free ARDY motion references and explicit source exports."""

from __future__ import annotations

import math
import os
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import ArdyError

# The default working-data budget, independent of model weights. Callers can
# increase it explicitly for longer clips; archive expansion shares this budget.
DEFAULT_MOTION_MEMORY_MIB = 256


@dataclass(frozen=True, slots=True)
class ArdySource:
    local_rot_mats: Any  # CPU NumPy [T, 27, 3, 3], never object arrays.
    root_positions: Any  # CPU NumPy [T, 3].
    fps: float

    @property
    def frame_count(self) -> int:
        return len(self.root_positions)


def _numpy() -> Any:
    try:
        import numpy
    except ImportError as exc:
        raise ArdyError("ARDY source files require the ardy extra") from exc
    return numpy


def _require_budget(max_bytes: int) -> None:
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
        raise ArdyError("ARDY source memory budget must be a positive integer")


def validate_source(rotations: Any, roots: Any, fps: float, *, max_bytes: int) -> ArdySource:
    """Validate native Core output before model conditioning or serialization."""
    _require_budget(max_bytes)
    np = _numpy()
    rotations = np.asarray(rotations)
    roots = np.asarray(roots)
    if rotations.ndim != 4 or rotations.shape[1:] != (27, 3, 3):
        raise ArdyError("ARDY Core source local_rot_mats must have shape [frames, 27, 3, 3]")
    if not len(rotations) or roots.shape != (len(rotations), 3):
        raise ArdyError("ARDY source root_positions must match the nonempty rotation frame count")
    if rotations.dtype.kind != "f" or roots.dtype.kind != "f":
        raise ArdyError("ARDY source motion must contain floating-point arrays")
    if max(rotations.nbytes + roots.nbytes, (rotations.size + roots.size) * 4) > max_bytes:
        raise ArdyError("ARDY source exceeds the motion memory budget")
    if (
        isinstance(fps, (bool, np.bool_))
        or not isinstance(fps, (int, float, np.integer, np.floating))
        or not 0 < fps <= np.finfo(np.float32).max
    ):
        raise ArdyError("ARDY source fps must be positive and finite")
    if not np.isfinite(rotations).all() or not np.isfinite(roots).all():
        raise ArdyError("ARDY source contains non-finite motion values")
    if roots.min() < -np.finfo(np.float32).max or roots.max() > np.finfo(np.float32).max:
        raise ArdyError("ARDY source root positions exceed the float32 range")
    if rotations.min() < -1.002 or rotations.max() > 1.002:
        raise ArdyError("ARDY source contains a non-orthogonal rotation matrix")
    # Process each frame separately: validation must not allocate another whole
    # clip of matrix products when the source is already near its byte budget.
    for frame in rotations:
        frame = frame.astype(np.float64)
        if not np.allclose(frame @ frame.swapaxes(-1, -2), np.eye(3), atol=2e-3):
            raise ArdyError("ARDY source contains a non-orthogonal rotation matrix")
        if not np.allclose(np.linalg.det(frame), 1, atol=2e-3):
            raise ArdyError("ARDY source contains an improper rotation matrix")
    rotations = np.array(rotations, dtype=np.float32, copy=True)
    roots = np.array(roots, dtype=np.float32, copy=True)
    rotations.flags.writeable = False
    roots.flags.writeable = False
    return ArdySource(rotations, roots, float(fps))


def read_source(path: str | Path, *, max_bytes: int) -> ArdySource:
    """Read a vroom source export or an upstream Core NPZ without unpickling."""
    _require_budget(max_bytes)
    np = _numpy()
    path = Path(path)
    try:
        with path.open("rb") as source_file:
            if os.fstat(source_file.fileno()).st_size > max_bytes:
                raise ArdyError("ARDY source file exceeds the motion memory budget")
            with zipfile.ZipFile(source_file) as archive:
                entries = archive.infolist()
                names = [entry.filename for entry in entries]
                if len(set(names)) != len(names):
                    raise ArdyError("ARDY source archive contains duplicate entries")
                allowed = {
                    "local_rot_mats.npy",
                    "root_positions.npy",
                    "fps.npy",
                    "posed_joints.npy",
                    "global_rot_mats.npy",
                    "smooth_root_pos.npy",
                    "global_root_heading.npy",
                    "foot_contacts.npy",
                    "text.npy",
                    "skeleton.npy",
                    "source_version.npy",
                    "model.npy",
                }
                if not set(names) <= allowed:
                    raise ArdyError("ARDY source archive contains unknown fields")
                if sum(entry.file_size for entry in entries) > max_bytes:
                    raise ArdyError("expanded ARDY source exceeds the motion memory budget")
                required = {"local_rot_mats.npy", "root_positions.npy", "fps.npy"}
                if not required <= set(names):
                    raise ArdyError("ARDY source needs local_rot_mats, root_positions, and fps")
                # Inspect every header, including unused upstream output fields,
                # before any array is loaded. Header claims must match ZIP bytes.
                headers = {}
                for entry in entries:
                    with archive.open(entry) as stream:
                        version = np.lib.format.read_magic(stream)
                        shape, _, dtype = np.lib.format._read_array_header(stream, version)
                        size = math.prod(shape) * dtype.itemsize
                        if dtype.hasobject or size > max_bytes:
                            raise ArdyError("ARDY source contains an object or oversized array")
                        if size != entry.file_size - stream.tell():
                            raise ArdyError("ARDY source NPY shape does not match its stored bytes")
                        headers[entry.filename] = (shape, dtype)
                shape, dtype = headers["local_rot_mats.npy"]
                if len(shape) != 4 or shape[1:] != (27, 3, 3) or not shape[0] or dtype.kind != "f":
                    raise ArdyError(
                        "ARDY Core source local_rot_mats must be floating [frames,27,3,3]"
                    )
                root_shape, root_dtype = headers["root_positions.npy"]
                if root_shape != (shape[0], 3) or root_dtype.kind != "f":
                    raise ArdyError("ARDY source root_positions must be floating [frames,3]")
                if shape[0] * (27 * 9 + 3) * 4 > max_bytes:
                    raise ArdyError("ARDY source float32 motion exceeds the memory budget")
                for name, kinds in (
                    ("fps.npy", "fiu"),
                    ("skeleton.npy", "U"),
                    ("source_version.npy", "iu"),
                    ("model.npy", "U"),
                ):
                    if name in headers:
                        field_shape, field_dtype = headers[name]
                        if field_shape != () or field_dtype.kind not in kinds:
                            raise ArdyError(f"ARDY source {name[:-4]} has invalid scalar metadata")
            source_file.seek(0)
            with np.load(source_file, allow_pickle=False) as arrays:
                if "skeleton" in arrays and arrays["skeleton"].item() != "core27":
                    raise ArdyError("ARDY source must use the Core27 skeleton")
                if "source_version" in arrays and arrays["source_version"].item() != 1:
                    raise ArdyError("unsupported ARDY source format version")
                if "model" in arrays and not arrays["model"].item().strip():
                    raise ArdyError("ARDY source model must not be empty")
                return validate_source(
                    arrays["local_rot_mats"],
                    arrays["root_positions"],
                    arrays["fps"].item(),
                    max_bytes=max_bytes,
                )
    except ArdyError:
        raise
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        EOFError,
        OverflowError,
        RuntimeError,
        NotImplementedError,
        zipfile.BadZipFile,
    ) as exc:
        raise ArdyError(f"cannot read ARDY source {path}: {exc}") from exc


def write_source(
    path: str | Path,
    source: ArdySource,
    *,
    model: str,
    max_bytes: int = DEFAULT_MOTION_MEMORY_MIB * 1024 * 1024,
) -> Path:
    """Atomically save reusable motion only; prompts are not stored."""
    np = _numpy()
    path = Path(path)
    if path.suffix.lower() != ".npz":
        raise ArdyError("ARDY source output must use the .npz extension")
    if not isinstance(model, str) or not model.strip():
        raise ArdyError("ARDY source model must be non-empty text")
    source = validate_source(
        source.local_rot_mats,
        source.root_positions,
        source.fps,
        max_bytes=max_bytes,
    )
    temporary: str | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as stream:
            temporary = stream.name
            np.savez(
                stream,
                local_rot_mats=source.local_rot_mats,
                root_positions=source.root_positions,
                fps=np.asarray(source.fps),
                skeleton=np.asarray("core27"),
                source_version=np.asarray(1),
                model=np.asarray(model),
            )
            if stream.tell() > max_bytes:
                raise ArdyError("ARDY source archive exceeds the motion memory budget")
        os.replace(temporary, path)
    except (OSError, ValueError, TypeError) as exc:
        raise ArdyError(f"cannot write ARDY source {path}: {exc}") from exc
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)
    return path

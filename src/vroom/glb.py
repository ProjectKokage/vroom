"""Small, dependency-free GLB 2.0 reader and writer.

Only the pieces needed by VRM and VRM Animation are implemented. The module is
deliberately strict about container bounds so malformed animation files fail
with useful messages instead of leaking ``struct.error`` or ``KeyError``.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import GlbError

GLB_MAGIC = b"glTF"
JSON_CHUNK = b"JSON"
BIN_CHUNK = b"BIN\x00"

_COMPONENT_FORMATS = {
    5120: "b",  # BYTE
    5121: "B",  # UNSIGNED_BYTE
    5122: "h",  # SHORT
    5123: "H",  # UNSIGNED_SHORT
    5125: "I",  # UNSIGNED_INT
    5126: "f",  # FLOAT
}
_TYPE_WIDTHS = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
    "MAT2": 4,
    "MAT3": 9,
    "MAT4": 16,
}
_NORMALIZED_COMPONENTS = {
    5120: lambda value: max(float(value) / 127.0, -1.0),
    5121: lambda value: float(value) / 255.0,
    5122: lambda value: max(float(value) / 32767.0, -1.0),
    5123: lambda value: float(value) / 65535.0,
}


def _pad(data: bytes, fill: bytes) -> bytes:
    return data + fill * (-len(data) % 4)


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value}")


@dataclass(frozen=True, slots=True)
class Glb:
    """A parsed binary glTF document."""

    document: dict[str, Any]
    binary: bytes = b""

    @classmethod
    def read(cls, path: str | Path) -> Glb:
        source = Path(path)
        try:
            return cls.from_bytes(source.read_bytes())
        except OSError as exc:
            raise GlbError(f"cannot read {source}: {exc}") from exc

    @classmethod
    def from_bytes(cls, data: bytes) -> Glb:
        if len(data) < 12:
            raise GlbError("file is shorter than the 12-byte GLB header")

        magic, version, declared_length = struct.unpack_from("<4sII", data)
        if magic != GLB_MAGIC:
            raise GlbError("not a GLB file (missing glTF magic)")
        if version != 2:
            raise GlbError(f"unsupported GLB version {version}; expected 2")
        if declared_length != len(data):
            raise GlbError(f"GLB header declares {declared_length} bytes, but file has {len(data)}")

        chunks: list[tuple[bytes, bytes]] = []
        offset = 12
        while offset < len(data):
            if offset + 8 > len(data):
                raise GlbError("truncated GLB chunk header")
            chunk_length, chunk_type = struct.unpack_from("<I4s", data, offset)
            offset += 8
            if chunk_length % 4:
                raise GlbError(f"{chunk_type!r} chunk length is not aligned to 4 bytes")
            end = offset + chunk_length
            if end > len(data):
                raise GlbError(f"{chunk_type!r} chunk extends past the end of the file")
            chunks.append((chunk_type, data[offset:end]))
            offset = end

        if not chunks or chunks[0][0] != JSON_CHUNK:
            raise GlbError("the first GLB chunk must be JSON")
        if sum(chunk_type == JSON_CHUNK for chunk_type, _ in chunks) != 1:
            raise GlbError("a GLB file must contain exactly one JSON chunk")
        if sum(chunk_type == BIN_CHUNK for chunk_type, _ in chunks) > 1:
            raise GlbError("multiple BIN chunks are not supported")

        try:
            decoded = chunks[0][1].rstrip(b" \t\r\n\x00").decode("utf-8")
            document = json.loads(decoded, parse_constant=_reject_json_constant)
        except ValueError as exc:
            raise GlbError(f"invalid GLB JSON chunk: {exc}") from exc
        if not isinstance(document, dict):
            raise GlbError("the glTF root must be a JSON object")

        binary = next((chunk for kind, chunk in chunks if kind == BIN_CHUNK), b"")
        buffers = document.get("buffers", [])
        if buffers and isinstance(buffers, list) and isinstance(buffers[0], dict):
            byte_length = buffers[0].get("byteLength")
            if isinstance(byte_length, int):
                if byte_length > len(binary):
                    raise GlbError(
                        f"buffer declares {byte_length} bytes, but BIN chunk has {len(binary)}"
                    )
                binary = binary[:byte_length]
        return cls(document=document, binary=binary)

    def to_bytes(self) -> bytes:
        try:
            raw_json = json.dumps(
                self.document,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise GlbError(f"document cannot be encoded as JSON: {exc}") from exc

        json_chunk = _pad(raw_json, b" ")
        chunks = [struct.pack("<I4s", len(json_chunk), JSON_CHUNK) + json_chunk]
        if self.binary or self.document.get("buffers"):
            binary_chunk = _pad(self.binary, b"\x00")
            chunks.append(struct.pack("<I4s", len(binary_chunk), BIN_CHUNK) + binary_chunk)

        payload = b"".join(chunks)
        return struct.pack("<4sII", GLB_MAGIC, 2, 12 + len(payload)) + payload

    def write(self, path: str | Path) -> Path:
        target = Path(path)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(self.to_bytes())
        except OSError as exc:
            raise GlbError(f"cannot write {target}: {exc}") from exc
        return target


class AccessorReader:
    """Decode tightly packed or interleaved glTF accessors from a GLB BIN chunk."""

    def __init__(self, glb: Glb):
        self._document = glb.document
        self._binary = glb.binary

    def read(self, accessor_index: int) -> tuple[tuple[int | float, ...], ...]:
        accessors = self._document.get("accessors", [])
        views = self._document.get("bufferViews", [])
        if not isinstance(accessors, list):
            raise GlbError("glTF accessors must be an array")
        if not isinstance(views, list):
            raise GlbError("glTF bufferViews must be an array")
        if (
            not isinstance(accessor_index, int)
            or isinstance(accessor_index, bool)
            or accessor_index < 0
        ):
            raise GlbError(f"invalid accessor index {accessor_index!r}")
        try:
            accessor = accessors[accessor_index]
        except (IndexError, TypeError) as exc:
            raise GlbError(f"invalid accessor index {accessor_index}") from exc
        if not isinstance(accessor, dict):
            raise GlbError(f"accessor {accessor_index} is not an object")
        if "sparse" in accessor:
            raise GlbError("sparse accessors are not supported")

        view_index = accessor.get("bufferView")
        if not isinstance(view_index, int) or isinstance(view_index, bool) or view_index < 0:
            raise GlbError(f"accessor {accessor_index} has no bufferView")
        try:
            view = views[view_index]
        except (IndexError, TypeError) as exc:
            raise GlbError(
                f"accessor {accessor_index} has invalid bufferView {view_index}"
            ) from exc
        if not isinstance(view, dict) or view.get("buffer", 0) != 0:
            raise GlbError("only buffer 0 in the GLB BIN chunk is supported")

        component_type = accessor.get("componentType")
        accessor_type = accessor.get("type")
        count = accessor.get("count")
        if component_type not in _COMPONENT_FORMATS:
            raise GlbError(f"unsupported accessor component type {component_type}")
        if accessor_type not in _TYPE_WIDTHS:
            raise GlbError(f"unsupported accessor type {accessor_type!r}")
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise GlbError(f"accessor {accessor_index} has invalid count {count!r}")

        width = _TYPE_WIDTHS[accessor_type]
        format_string = "<" + _COMPONENT_FORMATS[component_type] * width
        item_size = struct.calcsize(format_string)
        stride = view.get("byteStride", item_size)
        if not isinstance(stride, int) or isinstance(stride, bool) or stride < item_size:
            raise GlbError(f"bufferView {view_index} has invalid byteStride {stride!r}")

        view_offset = view.get("byteOffset", 0)
        accessor_offset = accessor.get("byteOffset", 0)
        view_length = view.get("byteLength")
        if not all(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0
            for value in (view_offset, accessor_offset)
        ):
            raise GlbError("buffer offsets must be non-negative integers")
        if not isinstance(view_length, int) or isinstance(view_length, bool) or view_length < 0:
            raise GlbError(f"bufferView {view_index} has invalid byteLength")

        start = view_offset + accessor_offset
        required = 0 if count == 0 else (count - 1) * stride + item_size
        if accessor_offset + required > view_length or start + required > len(self._binary):
            raise GlbError(f"accessor {accessor_index} exceeds its bufferView")

        values = tuple(
            struct.unpack_from(format_string, self._binary, start + row * stride)
            for row in range(count)
        )
        if accessor.get("normalized") is True:
            converter = _NORMALIZED_COMPONENTS.get(component_type)
            if converter is None:
                raise GlbError(
                    f"accessor {accessor_index} cannot normalize component type {component_type}"
                )
            return tuple(tuple(converter(component) for component in value) for value in values)
        return values

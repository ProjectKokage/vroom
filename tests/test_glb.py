from __future__ import annotations

import math
import struct

import pytest

from vroom.errors import GlbError
from vroom.glb import AccessorReader, Glb


def test_glb_round_trip_and_padding() -> None:
    document = {
        "asset": {"version": "2.0"},
        "buffers": [{"byteLength": 5}],
    }
    encoded = Glb(document, b"abcde").to_bytes()

    assert len(encoded) % 4 == 0
    assert struct.unpack_from("<4sII", encoded) == (b"glTF", 2, len(encoded))
    decoded = Glb.from_bytes(encoded)
    assert decoded.document == document
    assert decoded.binary == b"abcde"


def test_glb_rejects_wrong_declared_length() -> None:
    encoded = bytearray(Glb({"asset": {"version": "2.0"}}).to_bytes())
    struct.pack_into("<I", encoded, 8, len(encoded) + 4)

    with pytest.raises(GlbError, match="declares"):
        Glb.from_bytes(bytes(encoded))


def test_glb_rejects_non_finite_json_when_writing() -> None:
    with pytest.raises(GlbError, match="encoded as JSON"):
        Glb({"asset": {"version": "2.0"}, "extras": {"bad": math.nan}}).to_bytes()


def test_glb_rejects_non_finite_json_when_reading() -> None:
    raw_json = b'{"asset":{"version":"2.0"},"extras":{"bad":NaN}}'
    json_chunk = raw_json + b" " * (-len(raw_json) % 4)
    payload = struct.pack("<I4s", len(json_chunk), b"JSON") + json_chunk
    encoded = struct.pack("<4sII", b"glTF", 2, 12 + len(payload)) + payload

    with pytest.raises(GlbError, match="non-finite JSON number"):
        Glb.from_bytes(encoded)


def test_accessor_reader_supports_byte_stride() -> None:
    binary = struct.pack("<3f f 3f f", 1, 2, 3, 99, 4, 5, 6, 99)
    document = {
        "asset": {"version": "2.0"},
        "buffers": [{"byteLength": len(binary)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(binary), "byteStride": 16}
        ],
        "accessors": [{"bufferView": 0, "componentType": 5126, "count": 2, "type": "VEC3"}],
    }

    assert AccessorReader(Glb(document, binary)).read(0) == (
        (1.0, 2.0, 3.0),
        (4.0, 5.0, 6.0),
    )


def test_accessor_reader_decodes_normalized_signed_bytes() -> None:
    binary = struct.pack("<4b", 0, 0, 0, 127)
    document = {
        "asset": {"version": "2.0"},
        "buffers": [{"byteLength": len(binary)}],
        "bufferViews": [{"buffer": 0, "byteLength": len(binary)}],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": 5120,
                "normalized": True,
                "count": 1,
                "type": "VEC4",
            }
        ],
    }

    assert AccessorReader(Glb(document, binary)).read(0) == ((0.0, 0.0, 0.0, 1.0),)


@pytest.mark.parametrize("index", [-1, True])
def test_accessor_reader_rejects_negative_and_boolean_indices(index: int) -> None:
    with pytest.raises(GlbError, match="invalid accessor index"):
        AccessorReader(Glb({"accessors": [], "bufferViews": []})).read(index)


@pytest.mark.parametrize("field", ["accessors", "bufferViews"])
def test_accessor_reader_rejects_non_array_collections(field: str) -> None:
    document = {
        "asset": {"version": "2.0"},
        "bufferViews": [{"buffer": 0, "byteLength": 4}],
        "accessors": [{"bufferView": 0, "componentType": 5126, "count": 1, "type": "SCALAR"}],
    }
    document[field] = {}

    with pytest.raises(GlbError, match=rf"glTF {field} must be an array"):
        AccessorReader(Glb(document, struct.pack("<f", 0.0))).read(0)

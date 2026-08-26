"""Safe, versioned Arrow nodes for portable NumPy-facing values.

This module is deliberately limited to the value layer.  Typed result and
component-state envelopes are added by the artifact layer in a later Sprint
74 task.  The wire format is an Arrow IPC file containing one binary row per
NumPy array and a canonical JSON manifest in schema metadata.  Public callers
receive NumPy/domain objects; PyArrow objects never cross this API boundary.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from io import BytesIO
from typing import cast

import pyarrow as pa  # pyright: ignore[reportMissingTypeStubs]

from latent_anything._portable_contract import (
    PortableLimits as PortableLimits,
)
from latent_anything._portable_contract import (
    PortableNodeError as PortableNodeError,
)
from latent_anything._portable_contract import (
    canonical_json,
    checked_shape,
)
from latent_anything._portable_nodes import PortableDecoder, PortableEncoder

_SCHEMA_VERSION = "portable-node-v1"
_MANIFEST_KEY = b"latent-anything-manifest"
_VERSION_KEY = b"latent-anything-schema-version"
_ARROW_SCHEMA = pa.schema(
    [
        pa.field("array_id", pa.int64(), nullable=False),
        pa.field("dtype", pa.string(), nullable=False),
        pa.field("shape", pa.list_(pa.int64()), nullable=False),
        pa.field("payload", pa.binary(), nullable=False),
    ]
)


def encode_portable(value: object, *, limits: PortableLimits | None = None) -> bytes:
    """Encode a bounded value into a versioned Arrow IPC byte string."""

    encoder = PortableEncoder(limits or PortableLimits())
    manifest = encoder.value(value)
    manifest_json = canonical_json(manifest)
    arrays = encoder.arrays
    table = pa.Table.from_arrays(
        [
            pa.array(range(len(arrays)), type=pa.int64()),
            pa.array([dtype for dtype, _, _ in arrays], type=pa.string()),
            pa.array([list(shape) for _, shape, _ in arrays], type=pa.list_(pa.int64())),
            pa.array([payload for _, _, payload in arrays], type=pa.binary()),
        ],
        schema=_ARROW_SCHEMA,
    ).replace_schema_metadata(
        {_VERSION_KEY: _SCHEMA_VERSION.encode("ascii"), _MANIFEST_KEY: manifest_json.encode("utf-8")}
    )
    output = BytesIO()
    with pa.ipc.new_file(output, table.schema) as writer:
        writer.write_table(table)
    return output.getvalue()


def decode_portable(payload: object, *, limits: PortableLimits | None = None) -> object:
    """Decode a versioned Arrow IPC byte string into NumPy/domain values."""

    if not isinstance(payload, bytes):
        raise TypeError("portable payload must be bytes")
    raw_payload = cast(bytes, payload)
    selected_limits = limits or PortableLimits()
    if len(raw_payload) > selected_limits.max_input_bytes:
        raise PortableNodeError("portable input exceeds maximum configured size")
    try:
        reader = pa.ipc.open_file(pa.BufferReader(raw_payload))
    except (pa.ArrowException, ValueError) as exc:
        raise PortableNodeError(f"portable Arrow payload cannot be read: {exc}") from exc
    schema = reader.schema
    if schema.remove_metadata() != _ARROW_SCHEMA:
        raise PortableNodeError("portable Arrow schema does not match portable-node-v1")
    metadata = schema.metadata or {}
    if metadata.get(_VERSION_KEY) != _SCHEMA_VERSION.encode("ascii"):
        raise PortableNodeError("unsupported portable schema version")
    manifest_bytes = metadata.get(_MANIFEST_KEY)
    if manifest_bytes is None:
        raise PortableNodeError("portable manifest is missing")
    if len(manifest_bytes) > selected_limits.max_manifest_bytes:
        raise PortableNodeError("portable manifest exceeds maximum configured size")
    try:
        manifest = json.loads(manifest_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PortableNodeError("portable manifest is not valid JSON") from exc
    if reader.num_record_batches > selected_limits.max_record_batches:
        raise PortableNodeError("portable Arrow payload exceeds record-batch limit")
    arrays: list[tuple[str, tuple[int, ...], bytes]] = []
    total_rows = 0
    try:
        for batch_index in range(reader.num_record_batches):
            batch = reader.get_batch(batch_index)
            total_rows += batch.num_rows
            if total_rows > selected_limits.max_array_rows:
                raise PortableNodeError("portable Arrow payload exceeds array-row limit")
            for row in range(batch.num_rows):
                array_id = batch.column("array_id")[row].as_py()
                dtype = batch.column("dtype")[row].as_py()
                shape = batch.column("shape")[row].as_py()
                raw_array_payload = batch.column("payload")[row].as_py()
                if (
                    isinstance(array_id, bool)
                    or not isinstance(array_id, int)
                    or array_id != len(arrays)
                    or not isinstance(dtype, str)
                    or not isinstance(shape, list)
                    or not isinstance(raw_array_payload, bytes)
                ):
                    raise PortableNodeError("portable array table contains malformed values")
                arrays.append((dtype, checked_shape(cast(Sequence[object], shape), selected_limits), raw_array_payload))
    except PortableNodeError:
        raise
    except (pa.ArrowException, IndexError, TypeError, ValueError) as exc:
        raise PortableNodeError(f"portable Arrow array table cannot be read: {exc}") from exc
    return PortableDecoder(arrays, selected_limits).value(manifest)


__all__ = ["PortableLimits", "PortableNodeError", "decode_portable", "encode_portable"]

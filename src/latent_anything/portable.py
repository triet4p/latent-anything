"""Safe, versioned Arrow nodes for portable NumPy-facing values.

This module is deliberately limited to the value layer.  Typed result and
component-state envelopes are added by the artifact layer in a later Sprint
74 task.  The wire format is an Arrow IPC file containing one binary row per
NumPy array and a canonical JSON manifest in schema metadata.  Public callers
receive NumPy/domain objects; PyArrow objects never cross this API boundary.
"""

from __future__ import annotations

import base64
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from io import BytesIO
from typing import Any, cast

import numpy as np
import pyarrow as pa  # pyright: ignore[reportMissingTypeStubs]

from latent_anything.covariance import CovarianceState
from latent_anything.latent_space import LatentSpace
from latent_anything.latent_value import LatentValue
from latent_anything.trajectory import Trajectory

_SCHEMA_VERSION = "portable-node-v1"
_DEFAULT_MAX_INPUT_BYTES = 768 * 1024 * 1024
_DEFAULT_MAX_MANIFEST_BYTES = 1 * 1024 * 1024
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


class PortableNodeError(ValueError):
    """Raised when a portable node is unsupported, malformed, or unsafe."""


@dataclass(frozen=True)
class PortableLimits:
    """Resource limits applied while encoding and decoding portable nodes."""

    max_depth: int = 32
    max_nodes: int = 10_000
    max_array_bytes: int = 256 * 1024 * 1024
    max_total_array_bytes: int = 512 * 1024 * 1024
    max_shape_dimension: int = 10_000_000
    max_shape_rank: int = 64
    max_record_batches: int = 128
    max_array_rows: int = 10_000
    max_input_bytes: int = _DEFAULT_MAX_INPUT_BYTES
    max_manifest_bytes: int = _DEFAULT_MAX_MANIFEST_BYTES

    def __post_init__(self) -> None:
        if self.max_depth < 1 or self.max_nodes < 1:
            raise ValueError("portable limits must allow at least one positive depth and node")
        if self.max_array_bytes < 0 or self.max_total_array_bytes < 0:
            raise ValueError("portable byte limits must be non-negative")
        if (
            self.max_shape_dimension < 1
            or self.max_shape_rank < 1
            or self.max_record_batches < 1
            or self.max_array_rows < 1
            or self.max_input_bytes < 1
            or self.max_manifest_bytes < 1
        ):
            raise ValueError("portable structural and input limits must be positive")


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise PortableNodeError(f"portable metadata is not canonical JSON: {exc}") from exc


def _checked_shape(shape: Sequence[object], limits: PortableLimits) -> tuple[int, ...]:
    if len(shape) > limits.max_shape_rank:
        raise PortableNodeError("array rank exceeds portable structural limit")
    result: list[int] = []
    elements = 1
    for raw in shape:
        if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0 or raw > limits.max_shape_dimension:
            raise PortableNodeError(f"invalid or oversized array dimension {raw!r}")
        result.append(raw)
        elements *= raw
        if elements > limits.max_total_array_bytes:
            raise PortableNodeError("array shape exceeds portable allocation guard")
    return tuple(result)


def _expected_array_bytes(shape: tuple[int, ...], dtype: np.dtype[Any]) -> int:
    elements = 1
    for dimension in shape:
        elements *= dimension
    return elements * dtype.itemsize


class _Encoder:
    def __init__(self, limits: PortableLimits) -> None:
        self.limits = limits
        self.arrays: list[tuple[str, tuple[int, ...], bytes]] = []
        self.total_array_bytes = 0
        self.nodes = 0
        self.active: set[int] = set()

    def _visit(self, depth: int) -> None:
        if depth > self.limits.max_depth:
            raise PortableNodeError("portable value exceeds maximum nesting depth")
        self.nodes += 1
        if self.nodes > self.limits.max_nodes:
            raise PortableNodeError("portable value exceeds maximum node count")

    def _enter(self, value: object) -> int:
        identity = id(value)
        if identity in self.active:
            raise PortableNodeError("cyclic portable values are not supported")
        self.active.add(identity)
        return identity

    def _leave(self, identity: int) -> None:
        self.active.remove(identity)

    def _array(self, value: np.ndarray) -> dict[str, object]:
        if value.dtype.hasobject:
            raise PortableNodeError("object-dtype NumPy arrays are not portable")
        contiguous = np.ascontiguousarray(value)
        payload = contiguous.tobytes(order="C")
        if len(payload) > self.limits.max_array_bytes:
            raise PortableNodeError("array exceeds maximum per-array byte limit")
        self.total_array_bytes += len(payload)
        if self.total_array_bytes > self.limits.max_total_array_bytes:
            raise PortableNodeError("arrays exceed maximum total byte limit")
        array_id = len(self.arrays)
        shape = _checked_shape(value.shape, self.limits)
        self.arrays.append((value.dtype.str, shape, payload))
        return {"kind": "ndarray", "array_id": array_id}

    def value(self, value: object, depth: int = 0) -> object:
        self._visit(depth)
        if value is None or isinstance(value, str | bool | int):
            return value
        if isinstance(value, float):
            if not np.isfinite(value):
                raise PortableNodeError("non-finite metadata floats are not portable")
            return value
        if isinstance(value, np.ndarray):
            return self._array(value)
        if isinstance(value, np.generic):
            return self.value(value.item(), depth + 1)
        if isinstance(value, bytes):
            return {"kind": "bytes", "value": base64.b64encode(value).decode("ascii")}
        if isinstance(value, LatentValue):
            identity = self._enter(value)
            try:
                return {
                    "kind": "latent_value",
                    "data": self.value(value.to_numpy(), depth + 1),
                    "space": self.value(value.space, depth + 1),
                    "metadata": self.value(dict(value.metadata), depth + 1),
                }
            finally:
                self._leave(identity)
        if isinstance(value, Trajectory):
            identity = self._enter(value)
            try:
                return {
                    "kind": "trajectory",
                    "data": self.value(value.to_numpy(), depth + 1),
                    "metadata": self.value(dict(value.metadata), depth + 1),
                }
            finally:
                self._leave(identity)
        if isinstance(value, LatentSpace):
            identity = self._enter(value)
            try:
                space_state = cast(Any, value)
                payload: dict[str, object] = {
                    "kind": "latent_space",
                    "dim": value.dim,
                    "geometry": value.geometry,
                    "source_model": value.source_model,
                    "metadata": self.value(value.metadata, depth + 1),
                    "n_gaussians": value.n_gaussians,
                    "position_dim": space_state._position_dim,
                    "scale_dim": space_state._scale_dim,
                    "color_dim": space_state._color_dim,
                    "codebook_size": value.codebook_size,
                }
                if value.covariance is not None:
                    payload["covariance"] = self.value(value.covariance, depth + 1)
                return payload
            finally:
                self._leave(identity)
        if isinstance(value, CovarianceState):
            identity = self._enter(value)
            try:
                return {
                    "kind": "covariance_state",
                    "mean": self.value(value.mean, depth + 1),
                    "covariance": self.value(value.covariance, depth + 1),
                    "n_samples": value.n_samples,
                    "source_representation_identity": value.source_representation_identity,
                    "reg_coef": value.reg_coef,
                    "provenance": self.value(dict(value.provenance), depth + 1),
                }
            finally:
                self._leave(identity)
        if isinstance(value, Mapping):
            identity = self._enter(value)
            try:
                result: dict[str, object] = {}
                for key in sorted(value, key=lambda item: str(item)):
                    if not isinstance(key, str):
                        raise PortableNodeError("portable mapping keys must be strings")
                    result[key] = self.value(value[key], depth + 1)
                return {"kind": "mapping", "items": result}
            finally:
                self._leave(identity)
        if isinstance(value, tuple):
            identity = self._enter(value)
            try:
                return {"kind": "tuple", "items": [self.value(item, depth + 1) for item in value]}
            finally:
                self._leave(identity)
        if isinstance(value, list):
            identity = self._enter(value)
            try:
                return {"kind": "list", "items": [self.value(item, depth + 1) for item in value]}
            finally:
                self._leave(identity)
        if isinstance(value, frozenset):
            identity = self._enter(value)
            try:
                items = [self.value(item, depth + 1) for item in value]
                items.sort(key=_canonical_json)
                return {"kind": "frozenset", "items": items}
            finally:
                self._leave(identity)
        raise PortableNodeError(f"unsupported portable value type: {type(value).__name__}")


def encode_portable(value: object, *, limits: PortableLimits | None = None) -> bytes:
    """Encode a bounded value into a versioned Arrow IPC byte string."""

    encoder = _Encoder(limits or PortableLimits())
    manifest = encoder.value(value)
    manifest_json = _canonical_json(manifest)
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


class _Decoder:
    def __init__(self, arrays: list[tuple[str, tuple[int, ...], bytes]], limits: PortableLimits) -> None:
        self.arrays = arrays
        self.limits = limits
        self.nodes = 0
        self.total_array_bytes = 0

    def value(self, value: object, depth: int = 0) -> object:
        if depth > self.limits.max_depth:
            raise PortableNodeError("portable value exceeds maximum nesting depth")
        self.nodes += 1
        if self.nodes > self.limits.max_nodes:
            raise PortableNodeError("portable value exceeds maximum node count")
        if value is None or isinstance(value, str | bool | int | float):
            return value
        if not isinstance(value, dict):
            raise PortableNodeError("portable manifest node must be a JSON object")
        kind = value.get("kind")
        if kind == "ndarray":
            raw_id = value.get("array_id")
            if isinstance(raw_id, bool) or not isinstance(raw_id, int) or raw_id < 0 or raw_id >= len(self.arrays):
                raise PortableNodeError("portable array reference is invalid")
            dtype_text, shape, payload = self.arrays[raw_id]
            if len(payload) > self.limits.max_array_bytes:
                raise PortableNodeError("array exceeds maximum per-array byte limit")
            self.total_array_bytes += len(payload)
            if self.total_array_bytes > self.limits.max_total_array_bytes:
                raise PortableNodeError("arrays exceed maximum total byte limit")
            try:
                dtype = np.dtype(dtype_text)
            except (TypeError, ValueError) as exc:
                raise PortableNodeError("portable array dtype is invalid") from exc
            if dtype.hasobject:
                raise PortableNodeError("object-dtype NumPy arrays are not portable")
            expected = _expected_array_bytes(shape, dtype)
            if expected > self.limits.max_total_array_bytes:
                raise PortableNodeError("array shape exceeds portable allocation guard")
            if expected != len(payload):
                raise PortableNodeError("portable array payload length does not match dtype and shape")
            return np.frombuffer(payload, dtype=dtype).reshape(shape).copy()
        if kind == "bytes":
            encoded = value.get("value")
            if not isinstance(encoded, str):
                raise PortableNodeError("portable bytes node is malformed")
            try:
                return base64.b64decode(encoded.encode("ascii"), validate=True)
            except (ValueError, UnicodeEncodeError) as exc:
                raise PortableNodeError("portable bytes node is not valid base64") from exc
        if kind == "mapping":
            items = value.get("items")
            if not isinstance(items, dict):
                raise PortableNodeError("portable mapping node is malformed")
            return {key: self.value(item, depth + 1) for key, item in items.items()}
        if kind in {"list", "tuple", "frozenset"}:
            items = value.get("items")
            if not isinstance(items, list):
                raise PortableNodeError("portable sequence node is malformed")
            decoded = [self.value(item, depth + 1) for item in items]
            if kind == "tuple":
                return tuple(decoded)
            if kind == "frozenset":
                return frozenset(decoded)
            return decoded
        if kind == "latent_space":
            metadata = self.value(value.get("metadata"), depth + 1)
            if not isinstance(metadata, dict):
                raise PortableNodeError("latent space metadata is malformed")
            kwargs = {
                "dim": value.get("dim"),
                "geometry": value.get("geometry"),
                "source_model": value.get("source_model"),
                "metadata": metadata,
                "n_gaussians": value.get("n_gaussians"),
                "position_dim": value.get("position_dim"),
                "scale_dim": value.get("scale_dim"),
                "color_dim": value.get("color_dim"),
                "codebook_size": value.get("codebook_size"),
            }
            covariance = value.get("covariance")
            if covariance is not None:
                decoded_covariance = self.value(covariance, depth + 1)
                if not isinstance(decoded_covariance, CovarianceState):
                    raise PortableNodeError("latent space covariance is malformed")
                kwargs["covariance"] = decoded_covariance
            try:
                return LatentSpace(**cast(Any, kwargs))
            except (TypeError, ValueError) as exc:
                raise PortableNodeError(f"latent space is invalid: {exc}") from exc
        if kind == "covariance_state":
            mean = self.value(value.get("mean"), depth + 1)
            covariance = self.value(value.get("covariance"), depth + 1)
            provenance = self.value(value.get("provenance"), depth + 1)
            if (
                not isinstance(mean, np.ndarray)
                or not isinstance(covariance, np.ndarray)
                or not isinstance(provenance, dict)
            ):
                raise PortableNodeError("covariance state is malformed")
            try:
                return CovarianceState(
                    mean,
                    covariance,
                    cast(int, value.get("n_samples")),
                    cast(str, value.get("source_representation_identity")),
                    cast(float, value.get("reg_coef")),
                    provenance,
                )
            except (TypeError, ValueError) as exc:
                raise PortableNodeError(f"covariance state is invalid: {exc}") from exc
        if kind == "latent_value":
            data = self.value(value.get("data"), depth + 1)
            space = self.value(value.get("space"), depth + 1)
            metadata = self.value(value.get("metadata"), depth + 1)
            if not isinstance(data, np.ndarray) or not isinstance(space, LatentSpace) or not isinstance(metadata, dict):
                raise PortableNodeError("latent value is malformed")
            try:
                return LatentValue(data, space, metadata)
            except (TypeError, ValueError) as exc:
                raise PortableNodeError(f"latent value is invalid: {exc}") from exc
        if kind == "trajectory":
            data = self.value(value.get("data"), depth + 1)
            metadata = self.value(value.get("metadata"), depth + 1)
            if not isinstance(data, np.ndarray) or not isinstance(metadata, dict):
                raise PortableNodeError("trajectory is malformed")
            try:
                return Trajectory(data, metadata=metadata)
            except (TypeError, ValueError) as exc:
                raise PortableNodeError(f"trajectory is invalid: {exc}") from exc
        raise PortableNodeError(f"unsupported portable manifest kind: {kind!r}")


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
                arrays.append(
                    (dtype, _checked_shape(cast(Sequence[object], shape), selected_limits), raw_array_payload)
                )
    except PortableNodeError:
        raise
    except (pa.ArrowException, IndexError, TypeError, ValueError) as exc:
        raise PortableNodeError(f"portable Arrow array table cannot be read: {exc}") from exc
    return _Decoder(arrays, selected_limits).value(manifest)


__all__ = ["PortableLimits", "PortableNodeError", "decode_portable", "encode_portable"]

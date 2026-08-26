"""Internal limits and validation helpers for the portable node codec."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

_DEFAULT_MAX_INPUT_BYTES = 768 * 1024 * 1024
_DEFAULT_MAX_MANIFEST_BYTES = 1 * 1024 * 1024


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


def canonical_json(value: object) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise PortableNodeError(f"portable metadata is not canonical JSON: {exc}") from exc


def checked_shape(shape: Sequence[object], limits: PortableLimits) -> tuple[int, ...]:
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


def expected_array_bytes(shape: tuple[int, ...], dtype: np.dtype[Any]) -> int:
    elements = 1
    for dimension in shape:
        elements *= dimension
    return elements * dtype.itemsize


# Keep the public facade's historical class identity/repr while the internal
# contract module owns the shared definitions used by both codec directions.
PortableNodeError.__module__ = "latent_anything.portable"
PortableLimits.__module__ = "latent_anything.portable"

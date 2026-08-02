"""Immutable latent values associated with one concrete :class:`LatentSpace`."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from types import MappingProxyType
from typing import Any, cast

import numpy as np

from latent_anything.latent_space import LatentSpace
from latent_anything.trajectory import Trajectory


def _freeze_metadata(value: Any) -> Any:
    """Recursively copy metadata into immutable containers."""
    if isinstance(value, Mapping):
        mapping = cast(Mapping[str, Any], value)
        return MappingProxyType({key: _freeze_metadata(item) for key, item in mapping.items()})
    if isinstance(value, list | tuple):
        sequence = cast(list[Any] | tuple[Any, ...], value)
        return tuple(_freeze_metadata(item) for item in sequence)
    if isinstance(value, set | frozenset):
        values = cast(set[Any] | frozenset[Any], value)
        return frozenset(_freeze_metadata(item) for item in values)
    if isinstance(value, np.ndarray):
        copied = cast(np.ndarray[Any, Any], value).copy()
        copied.setflags(write=False)
        return copied
    return deepcopy(value)


def coordinate_identity(space: LatentSpace, metadata: Mapping[str, Any]) -> str:
    """Return the canonical coordinate-system identity of one latent value.

    The identity is a ``"::"``-joined string built from, in order:
    ``source_representation_identity`` (declared in space or value metadata),
    the space's ``source_model``, and a revision token (``model_version``,
    ``revision``, ``source_model_revision``, or ``checkpoint`` from metadata).
    Two values are arithmetically compatible only when their identities match;
    an empty identity means the coordinate system is not declared.
    """
    tokens: list[str] = []
    for key in ("source_representation_identity", "coordinate_system"):
        value = space.metadata.get(key) or metadata.get(key)
        if isinstance(value, str) and value.strip():
            tokens.append(value.strip())
    if space.source_model.strip():
        tokens.append(space.source_model.strip())
    for key in ("model_version", "revision", "source_model_revision", "checkpoint"):
        value = metadata.get(key) or space.metadata.get(key)
        if isinstance(value, str) and value.strip():
            tokens.append(value.strip())
    return "::".join(tokens)


def assert_arithmetic_compatible(a: LatentValue, b: LatentValue) -> None:
    """Reject latent arithmetic across unrelated coordinate systems.

    Raises ``ValueError`` unless both values are provably in the same
    coordinate system: same geometry, same point shape, same stored shape, and
    a matching, declared coordinate-system identity. This prevents silently
    returning plausible-looking arrays from vectors that live in different
    spaces (different models, layers, or checkpoints).
    """
    if a.space.geometry != b.space.geometry:
        msg = f"latent arithmetic requires the same geometry, got {a.space.geometry!r} and {b.space.geometry!r}"
        raise ValueError(msg)
    if a.item_shape != b.item_shape:
        msg = f"latent arithmetic requires the same point shape, got {a.item_shape} and {b.item_shape}"
        raise ValueError(msg)
    if a.shape != b.shape:
        msg = "latent arithmetic requires the same stored shape, got {a.shape} and {b.shape}"
        raise ValueError(msg)
    identity_a = a.identity
    identity_b = b.identity
    if not identity_a or not identity_b:
        msg = (
            "latent arithmetic requires a declared coordinate-system identity; "
            "set source_model on the LatentSpace or source_representation_identity "
            "(model_version/revision) in metadata"
        )
        raise ValueError(msg)
    if identity_a != identity_b:
        msg = f"latent arithmetic across unrelated coordinate systems is not allowed: {identity_a!r} != {identity_b!r}"
        raise ValueError(msg)


def _arithmetic_metadata(
    base: Mapping[str, Any],
    *,
    op: str,
    operand_identity: str,
    coefficients: tuple[float, ...] | None = None,
) -> Mapping[str, Any]:
    """Extend *base* metadata with the arithmetic operation and a provenance chain."""
    provenance: list[Any] = list(cast(Any, base.get("provenance", ())))
    provenance.append(
        {
            "operation": "latent_arithmetic",
            "op": op,
            "operand_identity": operand_identity,
            "coefficients": list(coefficients) if coefficients is not None else None,
        }
    )
    operation: dict[str, Any] = {
        "kind": "latent_arithmetic",
        "op": op,
        "operand_identity": operand_identity,
    }
    if coefficients is not None:
        operation["coefficients"] = list(coefficients)
    return {
        **dict(base),
        "operation": operation,
        "provenance": provenance,
    }


class LatentValue:
    """An immutable single latent state or batch of states in a ``LatentSpace``.

    Flat values use ``(dim,)``, ``(batch, dim)``, or leading structured axes
    such as ``(batch, sequence, dim)``. Structured values use their space's
    point shape, for example ``(n_gaussians, param_dim)``. Input ownership is
    never retained and every conversion returns a copy.
    """

    def __init__(self, data: np.ndarray, space: LatentSpace, metadata: Mapping[str, Any] | None = None) -> None:
        point_ndim = len(space.shape)
        if data.ndim < point_ndim:
            msg = f"Expected at least {point_ndim}D point shape {space.shape}, got {data.ndim}D"
            raise ValueError(msg)
        if tuple(data.shape[-point_ndim:]) != space.shape:
            msg = f"Expected trailing point shape {space.shape}, got {data.shape}"
            raise ValueError(msg)
        if data.ndim > point_ndim and data.shape[0] < 1:
            raise ValueError("A latent batch must contain at least one value")
        points = data.reshape((-1, *space.shape)) if data.ndim > point_ndim else data[np.newaxis, ...]
        for point in points:
            space.validate_point(point)
        stored = data.copy()
        stored.setflags(write=False)
        self._data = stored
        self._space = deepcopy(space)
        self._metadata = _freeze_metadata(dict(metadata) if metadata is not None else {})

    @property
    def space(self) -> LatentSpace:
        """Return the explicitly associated latent space."""

        return deepcopy(self._space)

    @property
    def metadata(self) -> Mapping[str, Any]:
        """Return an immutable defensive snapshot of value metadata."""

        return cast(Mapping[str, Any], _freeze_metadata(self._metadata))

    @property
    def identity(self) -> str:
        """Return the canonical coordinate-system identity of this value.

        Built from the associated space's ``source_representation_identity``,
        ``source_model``, and revision metadata (see :func:`coordinate_identity`).
        An empty string means the coordinate system is not declared.
        """

        return coordinate_identity(self._space, self._metadata)

    @property
    def is_batch(self) -> bool:
        """Whether this value contains a first-axis batch."""

        return self._data.ndim > len(self._space.shape)

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """Return all leading axes before the latent point shape."""

        return self._data.shape[: -len(self._space.shape)] if self.is_batch else ()

    @property
    def batch_size(self) -> int | None:
        """Return the batch size, or ``None`` for one state."""

        return int(self._data.shape[0]) if self.is_batch else None

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the full stored array shape."""

        return self._data.shape

    @property
    def item_shape(self) -> tuple[int, ...]:
        """Return the shape of one state in the associated space."""

        return self._space.shape

    def to_numpy(self) -> np.ndarray:
        """Return a writable defensive copy of the stored NumPy value."""

        return self._data.copy()

    def __len__(self) -> int:
        """Return batch size; a single value has length one."""

        return self.batch_size if self.batch_size is not None else 1

    def __getitem__(self, key: int | slice) -> LatentValue:
        """Select values from a batch while preserving the associated space."""

        if not self.is_batch:
            raise TypeError("Only batched LatentValue instances support indexing")
        selected = self._data[key]
        if isinstance(key, slice) and selected.shape[0] < 1:
            raise ValueError("A LatentValue slice must contain at least one value")
        return LatentValue(selected, self._space, self._metadata)

    @classmethod
    def from_trajectory(cls, trajectory: Trajectory, space: LatentSpace | None = None) -> LatentValue:
        """Create a flat latent batch from a beta ``Trajectory`` instance."""

        target_space = space if space is not None else LatentSpace(dim=trajectory.dim)
        return cls(trajectory.to_numpy(), target_space)

    def to_trajectory(self) -> Trajectory:
        """Convert a flat value or batch to the compatible beta ``Trajectory``."""

        if self._space.geometry == "gaussian_set" or self._data.ndim > 2:
            raise ValueError("Structured values cannot be converted to a flat Trajectory")
        data = self._data if self.is_batch else self._data[np.newaxis, :]
        return Trajectory(data)

    # ── Latent arithmetic ─────────────────────────────────────────────
    #
    # Arithmetic is only meaningful when both values are provably in the same
    # coordinate system (same geometry, shape, and declared identity). Every
    # operation returns a new immutable ``LatentValue`` whose metadata records
    # the operation and grows a provenance chain.

    def _apply_binary(self, other: LatentValue, *, op: str) -> LatentValue:
        assert_arithmetic_compatible(self, other)
        if op == "add":
            result = self._data + other._data
            coefficients = (1.0, 1.0)
        else:
            result = self._data - other._data
            coefficients = (1.0, -1.0)
        metadata = _arithmetic_metadata(
            self._metadata,
            op=op,
            operand_identity=other.identity,
            coefficients=coefficients,
        )
        return LatentValue(result, self._space, metadata)

    def add(self, other: LatentValue) -> LatentValue:
        """Return a new value ``self + other`` (analogy arithmetic building block).

        Both values must share geometry, point shape, stored shape, and a
        declared, matching coordinate-system identity.
        """
        return self._apply_binary(other, op="add")

    def subtract(self, other: LatentValue) -> LatentValue:
        """Return a new value ``self - other`` (analogy arithmetic building block).

        Both values must share geometry, point shape, stored shape, and a
        declared, matching coordinate-system identity.
        """
        return self._apply_binary(other, op="subtract")

    def add_scaled(self, other: LatentValue, coefficient: float) -> LatentValue:
        """Return ``self + coefficient * other`` (steering-style addition).

        Both values must share geometry, point shape, stored shape, and a
        declared, matching coordinate-system identity.
        """
        assert_arithmetic_compatible(self, other)
        result = self._data + coefficient * other._data
        metadata = _arithmetic_metadata(
            self._metadata,
            op="add_scaled",
            operand_identity=other.identity,
            coefficients=(1.0, float(coefficient)),
        )
        return LatentValue(result, self._space, metadata)

    def scale(self, coefficient: float) -> LatentValue:
        """Return ``coefficient * self`` with a scaled-arith operation record."""
        if not np.isfinite(coefficient):
            msg = f"scale coefficient must be finite, got {coefficient!r}"
            raise ValueError(msg)
        result = coefficient * self._data
        metadata = _arithmetic_metadata(
            self._metadata,
            op="scale",
            operand_identity=self.identity,
            coefficients=(float(coefficient),),
        )
        return LatentValue(result, self._space, metadata)

    def __add__(self, other: object) -> LatentValue:
        """Elementwise addition with coordinate-system compatibility checks."""
        if not isinstance(other, LatentValue):
            return NotImplemented
        return self.add(other)

    def __sub__(self, other: object) -> LatentValue:
        """Elementwise subtraction with coordinate-system compatibility checks."""
        if not isinstance(other, LatentValue):
            return NotImplemented
        return self.subtract(other)

    def __repr__(self) -> str:
        return f"LatentValue(shape={self.shape}, geometry={self._space.geometry!r}, is_batch={self.is_batch})"

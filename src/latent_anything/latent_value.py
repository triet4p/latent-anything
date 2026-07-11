"""Immutable latent values associated with one concrete :class:`LatentSpace`."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from types import MappingProxyType
from typing import Any

import numpy as np

from latent_anything.latent_space import LatentSpace
from latent_anything.trajectory import Trajectory


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
        self._space = space
        self._metadata = MappingProxyType(deepcopy(dict(metadata)) if metadata is not None else {})

    @property
    def space(self) -> LatentSpace:
        """Return the explicitly associated latent space."""

        return self._space

    @property
    def metadata(self) -> Mapping[str, Any]:
        """Return immutable value metadata."""

        return self._metadata

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

    def __repr__(self) -> str:
        return f"LatentValue(shape={self.shape}, geometry={self._space.geometry!r}, is_batch={self.is_batch})"

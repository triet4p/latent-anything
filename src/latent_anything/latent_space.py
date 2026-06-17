"""A concrete LatentSpace class representing a Euclidean flat vector space."""

from __future__ import annotations

from typing import Any

import numpy as np


class LatentSpace:
    """Represents a latent space with Euclidean flat geometry.

    This is a concrete, hardcoded implementation supporting only
    Euclidean flat vector spaces. It will be generalized to an
    interface once instance #3 of a different philosophy appears
    (Rule of Three, see INCREMENTAL.md §4a).

    Parameters
    ----------
    dim : int
        Dimensionality of the latent space.
    source_model : str, optional
        Name or identifier of the source model.
    metadata : dict, optional
        Additional metadata about the latent space.
    """

    geometry: str = "euclidean"

    def __init__(
        self,
        dim: int,
        source_model: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if dim < 1:
            msg = f"dim must be >= 1, got {dim}"
            raise ValueError(msg)
        self.dim = dim
        self.source_model = source_model
        self.metadata = dict(metadata) if metadata is not None else {}

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the shape of a single latent point in this space."""
        return (self.dim,)

    def validate_point(self, point: np.ndarray) -> None:
        """Validate that a point is compatible with this space.

        Parameters
        ----------
        point : np.ndarray
            Point to validate.

        Raises
        ------
        TypeError
            If point is not a numpy array.
        ValueError
            If point shape does not match (dim,).
        """
        if point.shape != (self.dim,):
            msg = f"Expected shape ({self.dim},), got {point.shape}"
            raise ValueError(msg)

    def __repr__(self) -> str:
        return f"LatentSpace(dim={self.dim}, geometry={self.geometry!r}, source_model={self.source_model!r})"

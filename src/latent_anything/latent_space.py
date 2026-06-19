"""A concrete geometry-aware latent space with Euclidean and spherical support."""

from __future__ import annotations

from typing import Any

import numpy as np


class LatentSpace:
    """Represents a latent space with concrete geometry-aware operations.

    This is a concrete, hardcoded implementation supporting two
    validated geometry cases so far: Euclidean flat vectors
    (``"euclidean"``) and unit-norm spherical vectors
    (``"unit_norm"``). It will be generalized to an interface once
    instance #3 of a different philosophy appears (Rule of Three,
    see INCREMENTAL.md §4a).

    Parameters
    ----------
    dim : int
        Dimensionality of the latent space.
    geometry : str, optional
        Geometry hint for dispatch. Supported values are
        ``"euclidean"`` and ``"unit_norm"``.
    source_model : str, optional
        Name or identifier of the source model.
    metadata : dict, optional
        Additional metadata about the latent space.
    """

    _GEOMETRIES: frozenset[str] = frozenset({"euclidean", "unit_norm"})

    def __init__(
        self,
        dim: int,
        geometry: str = "euclidean",
        source_model: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if dim < 1:
            msg = f"dim must be >= 1, got {dim}"
            raise ValueError(msg)
        if geometry not in self._GEOMETRIES:
            msg = f"Unknown geometry {geometry!r}, expected one of {sorted(self._GEOMETRIES)}"
            raise ValueError(msg)
        self.geometry = geometry
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
            If point shape does not match (dim,), or for ``unit_norm``
            geometry if ``||point|| != 1``.
        """
        if point.shape != (self.dim,):
            msg = f"Expected shape ({self.dim},), got {point.shape}"
            raise ValueError(msg)
        if self.geometry == "unit_norm":
            norm = np.linalg.norm(point)
            if abs(norm - 1.0) > 1e-10:
                msg = f"unit_norm requires ||point|| = 1, got {norm}"
                raise ValueError(msg)

    def distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute distance between two points in this space.

        Dispatches on ``self.geometry``:

        - ``euclidean``: Euclidean distance ``||a - b||``.
        - ``unit_norm``: Angular distance ``arccos(clip(a·b, -1, 1))``.

        Parameters
        ----------
        a, b : np.ndarray
            1-D arrays of shape ``(dim,)``.

        Returns
        -------
        float
            Distance value.
        """
        if self.geometry == "euclidean":
            return float(np.linalg.norm(a - b))
        # unit_norm: angular distance
        cos_angle = np.clip(np.dot(a, b), -1.0, 1.0)
        return float(np.arccos(cos_angle))

    def interpolate(self, a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
        """Interpolate between two points in this space.

        Dispatches on ``self.geometry``:

        - ``euclidean``: Linear interpolation ``(1-t)*a + t*b`` (lerp).
        - ``unit_norm``: Spherical linear interpolation (slerp) along the
          geodesic on the unit sphere. Edge case ``ω ≈ 0`` returns ``a``.

        Parameters
        ----------
        a, b : np.ndarray
            1-D arrays of shape ``(dim,)``.
        t : float
            Interpolation parameter in ``[0, 1]``.

        Returns
        -------
        np.ndarray
            Interpolated point.
        """
        if self.geometry == "euclidean":
            return (1.0 - t) * a + t * b
        # unit_norm: slerp
        cos_omega = np.clip(np.dot(a, b), -1.0, 1.0)
        omega = np.arccos(cos_omega)
        sin_omega = np.sin(omega)
        # Edge case: ω ≈ 0 or ω ≈ π → sin(ω) ≈ 0, slerp degenerates
        if sin_omega < 1e-10:
            return (1.0 - t) * a + t * b
        return np.sin((1.0 - t) * omega) / sin_omega * a + np.sin(t * omega) / sin_omega * b

    def normalize(self, point: np.ndarray) -> np.ndarray:
        """Normalize a point to satisfy this space's geometry constraint.

        - ``euclidean``: Returns a copy unchanged.
        - ``unit_norm``: Projects onto the unit sphere (``point / ||point||``).

        Parameters
        ----------
        point : np.ndarray
            1-D array of shape ``(dim,)``.

        Returns
        -------
        np.ndarray
            Normalized point.

        Raises
        ------
        ValueError
            If geometry is ``unit_norm`` and the point is a zero vector.
        """
        if self.geometry == "euclidean":
            return point.copy()
        norm = np.linalg.norm(point)
        if norm < 1e-15:
            msg = "Cannot normalize zero vector on sphere"
            raise ValueError(msg)
        return point / norm

    def __repr__(self) -> str:
        return f"LatentSpace(dim={self.dim}, geometry={self.geometry!r}, source_model={self.source_model!r})"

"""A concrete geometry-aware latent space with Euclidean, spherical,
and Gaussian-set support."""

from __future__ import annotations

from typing import Any

import numpy as np

from latent_anything.geometry import (
    discrete_distance,
    discrete_interpolate,
    gaussian_distance,
    gaussian_interpolate,
    validate_discrete_code,
    validate_gaussian_set,
)

# Internal: slice layout of a Gaussian parameter vector.
# Default: position(3) + scale(3) + opacity(1) + color(3) = 10 columns.
_GAUSSIAN_PARAM_NAMES = ("position", "scale", "opacity", "color")


def _build_gaussian_layout(
    position_dim: int, scale_dim: int, opacity_dim: int, color_dim: int
) -> dict[str, tuple[int, int]]:
    """Build a param-name → (start_col, length) mapping."""
    offset = 0
    layout: dict[str, tuple[int, int]] = {}
    for name, dim in zip(
        _GAUSSIAN_PARAM_NAMES,
        (position_dim, scale_dim, opacity_dim, color_dim),
        strict=True,
    ):
        layout[name] = (offset, dim)
        offset += dim
    return layout


class LatentSpace:
    """Represents a latent space with concrete geometry-aware operations.

    This is a concrete implementation supporting three validated geometry
    cases: Euclidean flat vectors (``"euclidean"``), unit-norm spherical
    vectors (``"unit_norm"``), and fixed-size Gaussian-set structured
    points (``"gaussian_set"``). It will be generalized to an interface
    once instance #4 of a different philosophy appears (Rule of Three,
    see INCREMENTAL.md §4a).

    Parameters
    ----------
    dim : int
        Dimensionality of the latent space for flat geometries. For
        ``gaussian_set`` this corresponds to ``param_dim`` (computed
        from ``position_dim + scale_dim + 1 + color_dim``) and is
        validated against that sum.
    geometry : str, optional
        Geometry hint for dispatch. Supported values are
        ``"euclidean"``, ``"unit_norm"``, and ``"gaussian_set"``.
    source_model : str, optional
        Name or identifier of the source model.
    metadata : dict, optional
        Additional metadata about the latent space. For
        ``gaussian_set`` geometry, a ``"gaussian_set_param_layout"``
        key is auto-populated.
    n_gaussians : int | None, optional
        Number of Gaussians in the set. Required for
        ``geometry="gaussian_set"``, ignored otherwise.
    position_dim : int, optional
        Spatial dimensions per Gaussian (default 3).
    scale_dim : int, optional
        Scale dimensions per Gaussian (default 3).
    color_dim : int, optional
        Color channels per Gaussian (default 3).
    """

    _GEOMETRIES: frozenset[str] = frozenset({"euclidean", "unit_norm", "gaussian_set", "discrete_code"})

    def __init__(
        self,
        dim: int,
        geometry: str = "euclidean",
        source_model: str = "",
        metadata: dict[str, Any] | None = None,
        n_gaussians: int | None = None,
        position_dim: int = 3,
        scale_dim: int = 3,
        color_dim: int = 3,
        codebook_size: int | None = None,
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

        # Gaussian-set specific fields
        self._n_gaussians: int | None = None
        self._position_dim: int = position_dim
        self._scale_dim: int = scale_dim
        self._color_dim: int = color_dim
        self._opacity_dim: int = 1
        self._codebook_size: int | None = None

        if geometry == "gaussian_set":
            if n_gaussians is None or n_gaussians < 1:
                msg = f"gaussian_set requires n_gaussians >= 1, got {n_gaussians!r}"
                raise ValueError(msg)
            self._n_gaussians = n_gaussians
            # Validate dim matches computed param_dim
            expected_pdim = position_dim + scale_dim + 1 + color_dim
            if dim != expected_pdim:
                msg = (
                    f"dim={dim} does not match computed param_dim"
                    f" (position_dim + scale_dim + 1 + color_dim = {expected_pdim})"
                    f" for gaussian_set geometry"
                )
                raise ValueError(msg)
            # Populate metadata with param layout
            layout = _build_gaussian_layout(position_dim, scale_dim, 1, color_dim)
            self.metadata.setdefault("gaussian_set_param_layout", layout)
        elif geometry == "discrete_code":
            if codebook_size is None or codebook_size < 2:
                raise ValueError(f"discrete_code requires codebook_size >= 2, got {codebook_size!r}")
            self._codebook_size = codebook_size
            self.metadata.setdefault("codebook_size", codebook_size)
            self.metadata.setdefault("interpolation", "unsupported")

    @property
    def n_gaussians(self) -> int | None:
        """Number of Gaussians in the set (``None`` for non-set geometries)."""
        return self._n_gaussians

    @property
    def codebook_size(self) -> int | None:
        """Declared categorical codebook size for ``discrete_code`` geometry."""

        return self._codebook_size

    @property
    def param_dim(self) -> int:
        """Total parameter dimensionality per Gaussian (only meaningful for
        ``gaussian_set``)."""
        return self._position_dim + self._scale_dim + self._opacity_dim + self._color_dim

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the shape of a single latent point in this space.

        For flat geometries (``euclidean``, ``unit_norm``) returns
        ``(dim,)``. For ``gaussian_set`` returns
        ``(n_gaussians, param_dim)``.
        """
        if self.geometry == "gaussian_set":
            return (self._n_gaussians, self.param_dim)  # type: ignore[arg-type]
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
            If point shape does not match expected shape, or for
            ``unit_norm`` geometry if ``||point|| != 1``, or for
            ``gaussian_set`` geometry if numeric constraints are
            violated.
        """
        expected_shape = self.shape
        if point.shape != expected_shape:
            msg = f"Expected shape {expected_shape}, got {point.shape}"
            raise ValueError(msg)
        if self.geometry == "unit_norm":
            norm = np.linalg.norm(point)
            if abs(norm - 1.0) > 1e-10:
                msg = f"unit_norm requires ||point|| = 1, got {norm}"
                raise ValueError(msg)
        elif self.geometry == "gaussian_set":
            validate_gaussian_set(
                point, position_dim=self._position_dim, scale_dim=self._scale_dim, color_dim=self._color_dim
            )
        elif self.geometry == "discrete_code":
            validate_discrete_code(point, codebook_size=self._codebook_size)  # type: ignore[arg-type]

    # ── Backwards-compatible Gaussian helpers ──────────────────────────

    def _validate_gaussian_set_point(self, point: np.ndarray) -> None:
        """Validate numeric constraints for a Gaussian-set point."""
        validate_gaussian_set(
            point, position_dim=self._position_dim, scale_dim=self._scale_dim, color_dim=self._color_dim
        )

    def _gaussian_set_sort_indices(self, point: np.ndarray) -> np.ndarray:
        """Return lexicographic sort indices by position columns."""
        pdim = self._position_dim
        return np.lexsort(point[:, :pdim].T)

    def _gaussian_set_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Permutation-aware distance for fixed-size Gaussian sets.

        Sorts both sets by position (lexicographic), then computes the
        Frobenius norm of the difference between corresponding Gaussians.
        """
        return gaussian_distance(a, b, position_dim=self._position_dim)

    def _gaussian_set_interpolate(self, a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
        """Interpolate between two Gaussian-set points.

        Sorts both sets by position for correspondence, then:
        - position → lerp
        - scale → lerp in log-space (guarantees positivity)
        - opacity → lerp + clamp to [0, 1]
        - color → lerp + clamp to [0, 1]
        """
        return gaussian_interpolate(
            a, b, t, position_dim=self._position_dim, scale_dim=self._scale_dim, color_dim=self._color_dim
        )

    # ── Public API ────────────────────────────────────────────────────

    def distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute distance between two points in this space.

        Dispatches on ``self.geometry``:

        - ``euclidean``: Euclidean distance ``||a - b||``.
        - ``unit_norm``: Angular distance ``arccos(clip(a·b, -1, 1))``.
        - ``gaussian_set``: Permutation-aware distance — sorts sets
          by position then computes Frobenius norm of the difference.

        Parameters
        ----------
        a, b : np.ndarray
            Points in this space. For flat geometries: 1-D arrays of
            shape ``(dim,)``. For ``gaussian_set``: 2-D arrays of
            shape ``(n_gaussians, param_dim)``.

        Returns
        -------
        float
            Distance value.
        """
        if self.geometry == "euclidean":
            return float(np.linalg.norm(a - b))
        elif self.geometry == "unit_norm":
            cos_angle = np.clip(np.dot(a, b), -1.0, 1.0)
            return float(np.arccos(cos_angle))
        elif self.geometry == "gaussian_set":
            return self._gaussian_set_distance(a, b)
        elif self.geometry == "discrete_code":
            return discrete_distance(a, b)
        else:
            msg = f"No distance implementation for geometry {self.geometry!r}"
            raise ValueError(msg)

    def interpolate(self, a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
        """Interpolate between two points in this space.

        Dispatches on ``self.geometry``:

        - ``euclidean``: Linear interpolation ``(1-t)*a + t*b`` (lerp).
        - ``unit_norm``: Spherical linear interpolation (slerp) along the
          geodesic on the unit sphere. Edge case ``sin(ω) ≈ 0`` falls
          back to lerp.
        - ``gaussian_set``: Sorts by position for correspondence, then
          interpolates each parameter channel with appropriate constraints
          (log-space for scale, clamp for opacity/color).

        Parameters
        ----------
        a, b : np.ndarray
            Points in this space. For flat geometries: 1-D arrays of
            shape ``(dim,)``. For ``gaussian_set``: 2-D arrays of
            shape ``(n_gaussians, param_dim)``.
        t : float
            Interpolation parameter in ``[0, 1]``.

        Returns
        -------
        np.ndarray
            Interpolated point.
        """
        if self.geometry == "euclidean":
            return (1.0 - t) * a + t * b
        elif self.geometry == "unit_norm":
            cos_omega = np.clip(np.dot(a, b), -1.0, 1.0)
            omega = np.arccos(cos_omega)
            sin_omega = np.sin(omega)
            if sin_omega < 1e-10:
                return (1.0 - t) * a + t * b
            return np.sin((1.0 - t) * omega) / sin_omega * a + np.sin(t * omega) / sin_omega * b
        elif self.geometry == "gaussian_set":
            return self._gaussian_set_interpolate(a, b, t)
        elif self.geometry == "discrete_code":
            return discrete_interpolate(a, b, t)
        else:
            msg = f"No interpolate implementation for geometry {self.geometry!r}"
            raise ValueError(msg)

    def normalize(self, point: np.ndarray) -> np.ndarray:
        """Normalize a point to satisfy this space's geometry constraint.

        - ``euclidean``: Returns a copy unchanged.
        - ``unit_norm``: Projects onto the unit sphere (``point / ||point||``).
        - ``gaussian_set``: Returns a copy unchanged (constraints are
          enforced via ``validate_point`` and interpolation clamping).

        Parameters
        ----------
        point : np.ndarray
            Point to normalize.

        Returns
        -------
        np.ndarray
            Normalized point.

        Raises
        ------
        ValueError
            If geometry is ``unit_norm`` and the point is a zero vector.
        """
        if self.geometry == "euclidean" or self.geometry == "gaussian_set":
            return point.copy()
        elif self.geometry == "discrete_code":
            validate_discrete_code(point, codebook_size=self._codebook_size)  # type: ignore[arg-type]
            return point.copy()
        elif self.geometry == "unit_norm":
            norm = np.linalg.norm(point)
            if norm < 1e-15:
                msg = "Cannot normalize zero vector on sphere"
                raise ValueError(msg)
            return point / norm
        else:
            msg = f"No normalize implementation for geometry {self.geometry!r}"
            raise ValueError(msg)

    def __repr__(self) -> str:
        if self.geometry == "gaussian_set":
            return (
                f"LatentSpace(dim={self.dim}, geometry={self.geometry!r},"
                f" n_gaussians={self._n_gaussians},"
                f" source_model={self.source_model!r})"
            )
        return f"LatentSpace(dim={self.dim}, geometry={self.geometry!r}, source_model={self.source_model!r})"

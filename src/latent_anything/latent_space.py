"""A concrete geometry-aware latent space with validated latent geometries."""

from __future__ import annotations

from typing import Any

import numpy as np

from latent_anything.covariance import CovarianceConfig, CovarianceState, fit_covariance_state
from latent_anything.gaussian_3d import GAUSSIAN_3D_PARAM_DIM, validate_gaussian_3d
from latent_anything.geometry import (
    covariance_interpolate,
    discrete_distance,
    discrete_interpolate,
    gaussian_distance,
    gaussian_interpolate,
    mahalanobis_distance,
    unwhiten_point,
    validate_covariance,
    validate_discrete_code,
    validate_gaussian_set,
    whiten_point,
)
from latent_anything.pose import SE3, SO3

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

    This is a concrete implementation supporting validated geometry
    cases: Euclidean flat vectors (``"euclidean"``), unit-norm spherical
    vectors (``"unit_norm"``), and fixed-size Gaussian-set structured
    points (``"gaussian_set"``), and categorical code vectors
    (``"discrete_code"``), SO(3) matrices (``"so3"``), and SE(3) homogeneous
    matrices (``"se3"``). The geometry-specific cases use focused algorithms,
    not a speculative geometry hierarchy.

    Parameters
    ----------
    dim : int
        Dimensionality of the latent space for flat geometries. For
        ``gaussian_set`` this corresponds to ``param_dim`` (computed
        from ``position_dim + scale_dim + 1 + color_dim``) and is
        validated against that sum.
    geometry : str, optional
        Geometry hint for dispatch. Supported values are
        ``"euclidean"``, ``"unit_norm"``, ``"gaussian_set"``,
        ``"discrete_code"``, and ``"anisotropic"``.
    source_model : str, optional
        Name or identifier of the source model.
    metadata : dict, optional
        Additional metadata about the latent space. For
        ``gaussian_set`` geometry, a ``"gaussian_set_param_layout"``
        key is auto-populated.
    covariance : CovarianceState | None, optional
        Fitted covariance geometry for ``geometry="anisotropic"``. May be
        attached at construction or fitted later via :meth:`fit_covariance`.
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

    _GEOMETRIES: frozenset[str] = frozenset(
        {"euclidean", "unit_norm", "gaussian_set", "gaussian_3d", "discrete_code", "anisotropic", "so3", "se3"}
    )

    def __init__(
        self,
        dim: int,
        geometry: str = "euclidean",
        source_model: str = "",
        metadata: dict[str, Any] | None = None,
        covariance: CovarianceState | None = None,
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
        if geometry == "so3" and dim != 9:
            raise ValueError("so3 LatentSpace uses flattened 3x3 rotation matrices and requires dim=9")
        if geometry == "se3" and dim != 16:
            raise ValueError("se3 LatentSpace uses flattened 4x4 pose matrices and requires dim=16")

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
        # Anisotropic covariance geometry (None until fitted or attached)
        self._covariance: CovarianceState | None = None

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
        elif geometry == "gaussian_3d":
            if n_gaussians is None or n_gaussians < 1:
                raise ValueError(f"gaussian_3d requires n_gaussians >= 1, got {n_gaussians!r}")
            if dim != GAUSSIAN_3D_PARAM_DIM:
                raise ValueError(f"gaussian_3d requires dim={GAUSSIAN_3D_PARAM_DIM}, got {dim}")
            self._n_gaussians = n_gaussians
            self.metadata.setdefault("representation", "3d_gaussian_splat")
        elif geometry == "discrete_code":
            if codebook_size is None or codebook_size < 2:
                raise ValueError(f"discrete_code requires codebook_size >= 2, got {codebook_size!r}")
            self._codebook_size = codebook_size
            self.metadata.setdefault("codebook_size", codebook_size)
            self.metadata.setdefault("interpolation", "unsupported")
        elif geometry == "anisotropic":
            if covariance is not None:
                self._attach_covariance(covariance)
        elif geometry in {"so3", "se3"}:
            self.metadata.setdefault("representation", "rotation_matrix" if geometry == "so3" else "homogeneous_matrix")
            self.metadata.setdefault("angle_unit", "rad")
            self.metadata.setdefault("position_unit", "m")

    @property
    def n_gaussians(self) -> int | None:
        """Number of Gaussians in the set (``None`` for non-set geometries)."""
        return self._n_gaussians

    @property
    def codebook_size(self) -> int | None:
        """Declared categorical codebook size for ``discrete_code`` geometry."""

        return self._codebook_size

    @property
    def covariance(self) -> CovarianceState | None:
        """Return the fitted covariance geometry, or ``None`` if not attached.

        Only meaningful for ``geometry="anisotropic"``. The returned value is
        the immutable ``CovarianceState``; cross-space reuse is a caller error.
        """

        return self._covariance

    def fit_covariance(
        self,
        data: np.ndarray,
        *,
        source_representation_identity: str,
        config: CovarianceConfig | None = None,
        provenance: dict[str, Any] | None = None,
    ) -> LatentSpace:
        """Fit an anisotropic covariance metric on *data* and attach it to this space.

        Requires ``geometry="anisotropic"``. Fitting is bound to the supplied
        ``source_representation_identity`` (dataset/model/layer), so a fitted
        metric cannot be silently reused for a different representation.

        Parameters
        ----------
        data : np.ndarray
            2D array of shape ``(n_samples, dim)`` with ``n_samples > dim``.
        source_representation_identity : str
            Identity the fitted metric is valid for.
        config : CovarianceConfig | None
            Fitting configuration; defaults to ``CovarianceConfig()``.
        provenance : dict[str, Any] | None
            Free-form provenance attached to the fitted state.

        Returns
        -------
        LatentSpace
            This space with the covariance attached (mutates in place).
        """
        if self.geometry != "anisotropic":
            msg = f"fit_covariance requires geometry='anisotropic', got {self.geometry!r}"
            raise ValueError(msg)
        state = fit_covariance_state(
            data,
            source_representation_identity=source_representation_identity,
            config=config,
            provenance=provenance,
        )
        self._attach_covariance(state)
        return self

    def _attach_covariance(self, state: CovarianceState) -> None:
        """Validate and attach a fitted covariance, keeping metadata in sync."""
        if state.mean.shape != (self.dim,):
            msg = f"covariance mean shape {state.mean.shape} does not match dim {self.dim}"
            raise ValueError(msg)
        validate_covariance(state.covariance, dim=self.dim)
        self._covariance = state
        self.metadata["covariance_fitted"] = True
        self.metadata["covariance_source_representation_identity"] = state.source_representation_identity
        self.metadata["covariance_provenance"] = dict(state.provenance)
        self.metadata["interpolation"] = "metric-geodesic"

    @property
    def param_dim(self) -> int:
        """Total parameter dimensionality per Gaussian (only meaningful for
        ``gaussian_set``)."""
        if self.geometry == "gaussian_3d":
            return GAUSSIAN_3D_PARAM_DIM
        return self._position_dim + self._scale_dim + self._opacity_dim + self._color_dim

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the shape of a single latent point in this space.

        For flat geometries (``euclidean``, ``unit_norm``) returns
        ``(dim,)``. For ``gaussian_set`` returns
        ``(n_gaussians, param_dim)``.
        """
        if self.geometry in {"gaussian_set", "gaussian_3d"}:
            return (self._n_gaussians, self.param_dim)  # type: ignore[arg-type]
        if self.geometry == "so3":
            return (3, 3)
        if self.geometry == "se3":
            return (4, 4)
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
        if self.geometry == "so3":
            SO3(point)
        elif self.geometry == "se3":
            SE3.from_matrix(point)
        elif self.geometry == "anisotropic":
            if not np.isfinite(point).all():
                raise ValueError("anisotropic requires finite points")
        elif self.geometry == "unit_norm":
            norm = np.linalg.norm(point)
            if abs(norm - 1.0) > 1e-10:
                msg = f"unit_norm requires ||point|| = 1, got {norm}"
                raise ValueError(msg)
        elif self.geometry == "gaussian_3d":
            validate_gaussian_3d(point)
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
        - ``anisotropic``: Mahalanobis distance ``sqrt((a-b)^T C^{-1} (a-b))``
          using the fitted covariance (requires :meth:`fit_covariance`).
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
        self.validate_point(a)
        self.validate_point(b)
        if self.geometry == "so3":
            return SO3(a).distance(SO3(b))
        elif self.geometry == "se3":
            return SE3.from_matrix(a).distance(SE3.from_matrix(b))
        elif self.geometry == "euclidean":
            return float(np.linalg.norm(a - b))
        elif self.geometry == "anisotropic":
            covariance = self._require_covariance()
            return mahalanobis_distance(a, b, covariance.covariance)
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
        - ``anisotropic``: Constant-metric geodesic interpolation computed in
          the whitened frame (requires a fitted covariance). For a constant
          covariance this coincides with the affine coordinate lerp; it is
          never applied silently — see :meth:`fit_covariance`.
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
        self.validate_point(a)
        self.validate_point(b)
        if not 0.0 <= t <= 1.0:
            raise ValueError("t must be in [0, 1]")
        if self.geometry == "so3":
            return SO3(a).interpolate(SO3(b), t).matrix
        elif self.geometry == "se3":
            return SE3.from_matrix(a).interpolate(SE3.from_matrix(b), t).matrix
        elif self.geometry == "euclidean":
            return (1.0 - t) * a + t * b
        elif self.geometry == "anisotropic":
            covariance = self._require_covariance()
            return covariance_interpolate(a, b, t, mean=covariance.mean, covariance=covariance.covariance)
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
        if (
            self.geometry in {"euclidean", "gaussian_set", "gaussian_3d", "so3", "se3"}
            or self.geometry == "anisotropic"
        ):
            self.validate_point(point)
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

    def _require_covariance(self) -> CovarianceState:
        """Return the fitted covariance or raise a clear fitted-state error."""
        if self._covariance is None:
            msg = (
                "anisotropic space has no fitted covariance; call fit_covariance() "
                "with a source_representation_identity before metric operations"
            )
            raise ValueError(msg)
        return self._covariance

    def whiten(self, point: np.ndarray) -> np.ndarray:
        """Whiten *point* under this space's metric (``z = C^{-1/2}(x - mean)``).

        Only valid for ``geometry="anisotropic"`` with a fitted covariance.
        """
        covariance = self._require_covariance()
        self.validate_point(point)
        return whiten_point(point, covariance.mean, covariance.covariance)

    def unwhiten(self, point: np.ndarray) -> np.ndarray:
        """Invert :meth:`whiten` (``x = mean + C^{1/2} z``).

        Only valid for ``geometry="anisotropic"`` with a fitted covariance.
        """
        covariance = self._require_covariance()
        if point.shape != (self.dim,):
            msg = f"Expected shape {(self.dim,)}, got {point.shape}"
            raise ValueError(msg)
        return unwhiten_point(point, covariance.mean, covariance.covariance)

    def __repr__(self) -> str:
        if self.geometry == "gaussian_set":
            return (
                f"LatentSpace(dim={self.dim}, geometry={self.geometry!r},"
                f" n_gaussians={self._n_gaussians},"
                f" source_model={self.source_model!r})"
            )
        if self.geometry == "anisotropic":
            fitted = self._covariance is not None
            return (
                f"LatentSpace(dim={self.dim}, geometry={self.geometry!r},"
                f" covariance_fitted={fitted},"
                f" source_model={self.source_model!r})"
            )
        return f"LatentSpace(dim={self.dim}, geometry={self.geometry!r}, source_model={self.source_model!r})"

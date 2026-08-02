"""Focused algorithms for concrete latent geometries.

This module is deliberately function-based: four running geometry cases prove
the extraction boundary, but not an abstract geometry hierarchy. It also holds
the pure covariance algorithms (validation, regularization, Mahalanobis
distance, whitening, inverse whitening, and metric interpolation) that the
``anisotropic`` geometry consumes through the ``LatentSpace`` facade.
"""

from __future__ import annotations

import numpy as np


def validate_gaussian_set(point: np.ndarray, *, position_dim: int, scale_dim: int, color_dim: int) -> None:
    """Validate positive Gaussian scales plus bounded opacity and colour."""

    scale_end = position_dim + scale_dim
    if np.any(point[:, position_dim:scale_end] <= 0):
        raise ValueError("gaussian_set requires scale components > 0")
    if np.any((point[:, scale_end] < 0) | (point[:, scale_end] > 1)):
        raise ValueError("gaussian_set requires opacity in [0, 1]")
    colors = point[:, scale_end + 1 : scale_end + 1 + color_dim]
    if np.any((colors < 0) | (colors > 1)):
        raise ValueError("gaussian_set requires color channels in [0, 1]")


def _gaussian_order(point: np.ndarray, position_dim: int) -> np.ndarray:
    return np.lexsort(point[:, :position_dim].T)


def gaussian_distance(a: np.ndarray, b: np.ndarray, *, position_dim: int) -> float:
    """Return a permutation-aware Frobenius distance between Gaussian sets."""

    return float(np.linalg.norm(a[_gaussian_order(a, position_dim)] - b[_gaussian_order(b, position_dim)]))


def gaussian_interpolate(
    a: np.ndarray, b: np.ndarray, t: float, *, position_dim: int, scale_dim: int, color_dim: int
) -> np.ndarray:
    """Interpolate matched Gaussian parameters while preserving constraints."""

    a_sorted = a[_gaussian_order(a, position_dim)]
    b_sorted = b[_gaussian_order(b, position_dim)]
    result = np.empty_like(a_sorted)
    scale_end = position_dim + scale_dim
    result[:, :position_dim] = (1 - t) * a_sorted[:, :position_dim] + t * b_sorted[:, :position_dim]
    result[:, position_dim:scale_end] = np.exp(
        (1 - t) * np.log(np.maximum(a_sorted[:, position_dim:scale_end], 1e-10))
        + t * np.log(np.maximum(b_sorted[:, position_dim:scale_end], 1e-10))
    )
    result[:, scale_end] = np.clip((1 - t) * a_sorted[:, scale_end] + t * b_sorted[:, scale_end], 0, 1)
    result[:, scale_end + 1 : scale_end + 1 + color_dim] = np.clip(
        (1 - t) * a_sorted[:, scale_end + 1 : scale_end + 1 + color_dim]
        + t * b_sorted[:, scale_end + 1 : scale_end + 1 + color_dim],
        0,
        1,
    )
    return result


def validate_discrete_code(point: np.ndarray, *, codebook_size: int) -> None:
    """Validate integer categorical codes in the declared codebook range."""

    if not np.issubdtype(point.dtype, np.integer):
        raise TypeError("discrete_code requires an integer NumPy dtype")
    if np.any(point < 0) or np.any(point >= codebook_size):
        raise ValueError(f"discrete_code requires codes in [0, {codebook_size - 1}]")


def discrete_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Return normalized Hamming distance for equal-size code vectors."""

    return float(np.count_nonzero(a != b) / a.size)


def discrete_interpolate(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """Reject continuous interpolation between categorical code assignments."""

    del a, b, t
    raise ValueError("discrete_code has no continuous interpolation; choose a codebook-aware operation")


# ── Anisotropic covariance geometry ────────────────────────────────────
#
# A constant anisotropic metric is a positive-definite covariance matrix C.
# All algorithms here are pure functions; fitting/ownership/provenance live
# in the ``CovarianceState`` value under ``covariance.py``, and ``LatentSpace``
# dispatches on ``geometry == "anisotropic"`` through this module.


def validate_covariance(covariance: np.ndarray, *, dim: int) -> None:
    """Validate that *covariance* is a symmetric positive-definite matrix.

    Checks shape ``(dim, dim)``, finiteness, symmetry, and positive
    eigenvalues. Raises ``ValueError`` with a specific message otherwise.
    """
    matrix = np.asarray(covariance, dtype=np.float64)
    if matrix.shape != (dim, dim):
        msg = f"anisotropic requires covariance of shape ({dim}, {dim}), got {matrix.shape}"
        raise ValueError(msg)
    if not np.isfinite(matrix).all():
        raise ValueError("anisotropic requires finite covariance entries")
    if not np.allclose(matrix, matrix.T, atol=1e-10):
        raise ValueError("anisotropic requires a symmetric covariance")
    eigenvalues = np.linalg.eigvalsh(matrix)
    if np.any(eigenvalues <= 0):
        raise ValueError("anisotropic requires a positive-definite covariance")


def regularize_covariance(covariance: np.ndarray, *, reg_coef: float) -> np.ndarray:
    """Return ``covariance + reg_coef * I`` re-symmetrized and validated.

    Diagonal loading guarantees positive definiteness even for a
    near-singular empirical estimate, and the result is re-symmetrized to
    absorb floating-point asymmetry before being validated.
    """
    if reg_coef <= 0:
        msg = f"reg_coef must be > 0, got {reg_coef}"
        raise ValueError(msg)
    matrix = np.asarray(covariance, dtype=np.float64)
    dim = matrix.shape[0]
    if matrix.ndim != 2 or matrix.shape != (dim, dim):
        msg = f"regularize_covariance expects a square matrix, got shape {matrix.shape}"
        raise ValueError(msg)
    loaded = matrix + reg_coef * np.eye(dim)
    symmetrized = 0.5 * (loaded + loaded.T)
    validate_covariance(symmetrized, dim=dim)
    return symmetrized


def mahalanobis_distance(a: np.ndarray, b: np.ndarray, covariance: np.ndarray) -> float:
    """Return the Mahalanobis distance ``sqrt((a-b)^T C^{-1} (a-b))``.

    ``covariance`` must be positive definite. The inverse is solved rather
    than inverted so the computation stays numerically stable.
    """
    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    solved = np.linalg.solve(covariance, diff)
    return float(np.sqrt(np.dot(diff, solved)))


def whiten_point(point: np.ndarray, mean: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    """Whiten one centered point: ``z = C^{-1/2} (x - mean)``.

    After whitening the point has identity covariance under the metric,
    so Euclidean geometry in the whitened frame equals the metric geometry.
    """
    centered = np.asarray(point, dtype=np.float64) - np.asarray(mean, dtype=np.float64)
    return np.linalg.solve(np.linalg.cholesky(covariance), centered)


def unwhiten_point(whitened: np.ndarray, mean: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    """Invert :func:`whiten_point`: ``x = mean + C^{1/2} z``."""
    lower = np.linalg.cholesky(covariance)
    return np.asarray(mean, dtype=np.float64) + lower @ np.asarray(whitened, dtype=np.float64)


def covariance_interpolate(
    a: np.ndarray, b: np.ndarray, t: float, *, mean: np.ndarray, covariance: np.ndarray
) -> np.ndarray:
    """Interpolate along the constant-metric geodesic in the whitened frame.

    Semantics decision (Sprint 48)
    -----------------------------
    A **constant** covariance defines a flat metric, so the geodesic between
    two points is the affine segment ``(1-t)a + t b``. Because whitening is an
    affine map, this whitened-frame interpolation is *numerically identical* to
    the raw-coordinate lerp. The implementation is deliberately routed through
    the declared metric rather than silently applying Euclidean lerp so that:

    - the metric must be fitted and declared before interpolation (no silent
      Euclidean fallback on an unfitted space),
    - endpoints are validated against the space,
    - a future position-dependent (pullback/density-aware) metric — Sprint 50 —
      can replace this function without changing the ``LatentSpace`` API.

    Euclidean lerp is therefore not *wrong* here; it is simply no longer the
    silent default. Callers get the same result, but the space now documents
    the metric it interpolates under.
    """
    z_a = whiten_point(a, mean, covariance)
    z_b = whiten_point(b, mean, covariance)
    z_t = (1.0 - t) * z_a + t * z_b
    return unwhiten_point(z_t, mean, covariance)


def fit_covariance(data: np.ndarray, *, reg_coef: float) -> tuple[np.ndarray, np.ndarray]:
    """Fit an empirical mean and regularized covariance from a 2D sample.

    Returns ``(mean, covariance)``. Uses unbiased sample covariance
    (``ddof=1``) plus diagonal loading so the result is positive definite.
    Requires more samples than dimensions for a full-rank estimate.
    """
    values = np.asarray(data, dtype=np.float64)
    if values.ndim != 2:
        msg = f"fit_covariance expects 2D data, got {values.ndim}D"
        raise ValueError(msg)
    if values.shape[0] <= values.shape[1]:
        msg = f"anisotropic fitting requires more samples than dimensions, got {values.shape[0]} <= {values.shape[1]}"
        raise ValueError(msg)
    if values.shape[0] < 2:
        raise ValueError("fit_covariance requires at least 2 samples")
    if not np.isfinite(values).all():
        raise ValueError("fit_covariance requires finite data")
    mean = np.mean(values, axis=0)
    centered = values - mean
    covariance = (centered.T @ centered) / (values.shape[0] - 1)
    return mean, regularize_covariance(covariance, reg_coef=reg_coef)

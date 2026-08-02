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


# ── Orthonormal subspace projection ─────────────────────────────────
#
# Subspace projection decomposes a point into ``z = P z + (I - P) z`` where
# ``P = U U^T`` is the orthogonal projection onto the span of the orthonormal
# columns of ``U``. The concept component ``P z`` keeps the semantic directions
# in the subspace while the residual ``(I - P) z`` keeps everything else. All
# algorithms here are pure functions; ownership/provenance/identity binding live
# in the ``OrthonormalSubspace`` value under ``projection.py``.


def validate_orthonormal_basis(basis: np.ndarray, *, dim: int) -> None:
    """Validate that *basis* has ``dim`` rows and orthonormal columns.

    Checks shape ``(dim, n_basis)`` with ``1 <= n_basis < dim``, finiteness,
    and ``U^T U = I``. A proper subspace (not the full space) is required so
    the residual is never the zero vector for every input.
    """
    matrix = np.asarray(basis, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != dim:
        msg = f"orthonormal basis requires shape ({dim}, n_basis), got {matrix.shape}"
        raise ValueError(msg)
    if not np.isfinite(matrix).all():
        raise ValueError("orthonormal basis requires finite entries")
    n_basis = matrix.shape[1]
    if n_basis < 1 or n_basis >= dim:
        msg = f"orthonormal basis requires 1 <= n_basis < dim={dim}, got {n_basis}"
        raise ValueError(msg)
    gram = matrix.T @ matrix
    if not np.allclose(gram, np.eye(n_basis), atol=1e-8):
        raise ValueError("orthonormal basis requires U^T U = I")


def orthonormalize_directions(directions: np.ndarray) -> np.ndarray:
    """Return an orthonormal basis spanning the column space of *directions*.

    ``directions`` is a 2D array whose columns are candidate directions (for
    example stacked probe coefficients or concept directions). The result is
    a ``(dim, rank)`` orthonormal basis computed by QR, dropping the columns
    that correspond to numerically zero pivots.
    """
    matrix = np.asarray(directions, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
        msg = f"orthonormalize_directions expects a non-empty 2D array, got {matrix.shape}"
        raise ValueError(msg)
    if not np.isfinite(matrix).all():
        raise ValueError("orthonormalize_directions requires finite directions")
    q, _ = np.linalg.qr(matrix, mode="reduced")
    rank = int(np.linalg.matrix_rank(matrix, tol=1e-9))
    if rank < 1:
        raise ValueError("directions span a zero-dimensional subspace")
    basis = q[:, :rank]
    for index in range(basis.shape[1]):
        column = basis[:, index]
        pivot = int(np.argmax(np.abs(column)))
        if column[pivot] < 0:
            basis[:, index] = -column
    return basis


def project_point(point: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Return the orthogonal projection ``P z = U (U^T z)`` onto ``span(U)``.

    ``basis`` must be a validated ``(dim, n_basis)`` orthonormal basis. This is
    the closest point to ``z`` in the subspace under the Euclidean metric.
    Accepts a single ``(dim,)`` point or a ``(..., dim)`` batch.
    """
    matrix = np.asarray(basis, dtype=np.float64)
    value = np.asarray(point, dtype=np.float64)
    if value.ndim == 1:
        return matrix @ (matrix.T @ value)
    return value @ matrix @ matrix.T


def remove_point(point: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Return the residual ``(I - P) z = z - U (U^T z)``.

    This is the projection onto the orthogonal complement of ``span(U)`` and
    removes every component along the concept directions. Accepts a single
    ``(dim,)`` point or a ``(..., dim)`` batch.
    """
    value = np.asarray(point, dtype=np.float64)
    return value - project_point(value, basis)


def concept_coverage(point: np.ndarray, basis: np.ndarray) -> float:
    """Return the fraction of ``||z||^2`` lying inside the subspace.

    ``||P z||^2 / ||z||^2`` is the squared cosine between ``z`` and the
    subspace and lies in ``[0, 1]``. The zero vector is treated as fully
    covered (no structure outside the subspace).
    """
    value = np.asarray(point, dtype=np.float64)
    norm_squared = float(np.dot(value, value))
    if norm_squared < 1e-300:
        return 1.0
    projected = project_point(value, basis)
    return float(np.dot(projected, projected) / norm_squared)


def subspace_alignment(a: np.ndarray, b: np.ndarray) -> float:
    """Return the mean squared cosine between two orthonormal subspaces.

    Computes the singular values of ``U_a^T U_b`` (the principal angles) and
    returns their mean square. ``1.0`` means the two bases span the same
    subspace and ``0.0`` means they are orthogonal. This is a symmetric
    comparison used to show that different basis families are *not*
    interchangeable.
    """
    matrix_a = np.asarray(a, dtype=np.float64)
    matrix_b = np.asarray(b, dtype=np.float64)
    if matrix_a.ndim != 2 or matrix_b.ndim != 2 or matrix_a.shape[0] != matrix_b.shape[0]:
        msg = f"subspace_alignment expects two bases with matching rows, got {matrix_a.shape} and {matrix_b.shape}"
        raise ValueError(msg)
    singular = np.linalg.svd(matrix_a.T @ matrix_b, compute_uv=False)
    return float(np.mean(np.square(singular)))

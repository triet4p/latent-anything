"""Focused algorithms for concrete latent geometries.

This module is deliberately function-based: four running geometry cases prove
the extraction boundary, but not an abstract geometry hierarchy.
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

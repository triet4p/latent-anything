"""Geometry-aware manipulation of fixed-schema 3D Gaussian latents."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

from latent_anything.pose import SE3, SO3

_POSITION = slice(0, 3)
_ROTATION = slice(3, 7)
_SCALE = slice(7, 10)
_OPACITY = 10
_COLOR = slice(11, 14)


def _check_latent(latent: np.ndarray) -> np.ndarray:
    value = np.asarray(latent, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 14:
        raise ValueError("3D Gaussian latent must have shape (n, 14)")
    if not np.isfinite(value).all() or np.any(value[:, _SCALE] <= 0):
        raise ValueError("3D Gaussian latent must be finite with positive scales")
    if np.any(np.linalg.norm(value[:, _ROTATION], axis=1) < 1e-8):
        raise ValueError("3D Gaussian rotations require non-zero quaternions")
    if np.any((value[:, _OPACITY] < 0) | (value[:, _OPACITY] > 1)):
        raise ValueError("3D Gaussian opacity must be in [0, 1]")
    if np.any((value[:, _COLOR] < 0) | (value[:, _COLOR] > 1)):
        raise ValueError("3D Gaussian colors must be in [0, 1]")
    return value.copy()


def _quaternion_from_rotation(rotation: SO3) -> np.ndarray:
    """Convert a rotation matrix to the adapter's normalized ``xyzw`` order."""
    matrix = rotation.matrix
    trace = float(np.trace(matrix))
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2
        w, x, y, z = (
            0.25 * s,
            (matrix[2, 1] - matrix[1, 2]) / s,
            (matrix[0, 2] - matrix[2, 0]) / s,
            (matrix[1, 0] - matrix[0, 1]) / s,
        )
    else:
        diagonal = np.diag(matrix)
        axis = int(np.argmax(diagonal))
        if axis == 0:
            s = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2
            x, y, z, w = (
                0.25 * s,
                (matrix[0, 1] + matrix[1, 0]) / s,
                (matrix[0, 2] + matrix[2, 0]) / s,
                (matrix[2, 1] - matrix[1, 2]) / s,
            )
        elif axis == 1:
            s = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2
            x, y, z, w = (
                (matrix[0, 1] + matrix[1, 0]) / s,
                0.25 * s,
                (matrix[1, 2] + matrix[2, 1]) / s,
                (matrix[0, 2] - matrix[2, 0]) / s,
            )
        else:
            s = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2
            x, y, z, w = (
                (matrix[0, 2] + matrix[2, 0]) / s,
                (matrix[1, 2] + matrix[2, 1]) / s,
                0.25 * s,
                (matrix[1, 0] - matrix[0, 1]) / s,
            )
    return np.array([x, y, z, w], dtype=np.float64)


def rigid_transform(latent: np.ndarray, transform: SE3, indices: Iterable[int] | None = None) -> np.ndarray:
    """Apply a validated rigid world transform to selected positions and rotations."""
    value = _check_latent(latent)
    result = value.copy()
    selected = np.arange(len(value)) if indices is None else np.asarray(list(indices), dtype=int)
    if selected.size and (selected.min() < 0 or selected.max() >= len(value)):
        raise IndexError("Gaussian index out of range")
    result[selected, _POSITION] = (transform.rotation.matrix @ value[selected, _POSITION].T).T + transform.translation
    for index in selected:
        quaternion = value[index, _ROTATION]
        result[index, _ROTATION] = _quaternion_from_rotation(
            transform.rotation.compose(SO3.from_quaternion(quaternion, order="xyzw"))
        )
    return result


def edit_opacity(
    latent: np.ndarray, indices: Iterable[int], *, value: float | None = None, multiplier: float | None = None
) -> np.ndarray:
    """Set or scale selected opacity values, with an explicit [0, 1] bound."""
    if (value is None) == (multiplier is None):
        raise ValueError("provide exactly one of value or multiplier")
    result = _check_latent(latent)
    selected = np.asarray(list(indices), dtype=int)
    if selected.size and (selected.min() < 0 or selected.max() >= len(result)):
        raise IndexError("Gaussian index out of range")
    if value is not None:
        opacity = float(value)
    else:
        assert multiplier is not None
        opacity = result[selected, _OPACITY] * multiplier
    result[selected, _OPACITY] = opacity
    if np.any((result[:, _OPACITY] < 0) | (result[:, _OPACITY] > 1)):
        raise ValueError("edited opacity must remain in [0, 1]")
    return result


def edit_color(latent: np.ndarray, indices: Iterable[int], color: Sequence[float], *, mode: str = "set") -> np.ndarray:
    """Set or add RGB values for selected Gaussians, bounded to [0, 1]."""
    result = _check_latent(latent)
    rgb = np.asarray(color, dtype=np.float64)
    if rgb.shape != (3,) or not np.isfinite(rgb).all():
        raise ValueError("color must be a finite RGB vector with shape (3,)")
    if mode not in {"set", "add"}:
        raise ValueError("mode must be 'set' or 'add'")
    selected = np.asarray(list(indices), dtype=int)
    result[selected, _COLOR] = rgb if mode == "set" else result[selected, _COLOR] + rgb
    if np.any((result[:, _COLOR] < 0) | (result[:, _COLOR] > 1)):
        raise ValueError("edited color must remain in [0, 1]")
    return result


def remove_gaussians(latent: np.ndarray, indices: Iterable[int]) -> np.ndarray:
    """Remove selected rows while retaining a valid non-empty Gaussian set."""
    value = _check_latent(latent)
    selected = set(int(index) for index in indices)
    if not selected or any(index < 0 or index >= len(value) for index in selected):
        raise IndexError("Gaussian index out of range")
    if len(selected) >= len(value):
        raise ValueError("cannot remove every Gaussian")
    return np.delete(value, sorted(selected), axis=0)


def merge_gaussians(latent: np.ndarray, groups: Sequence[Sequence[int]]) -> np.ndarray:
    """Merge each index group by opacity-weighted attributes and log-scale mean."""
    value = _check_latent(latent)
    consumed: set[int] = set()
    rows: list[np.ndarray] = []
    for group in groups:
        indices = tuple(dict.fromkeys(int(index) for index in group))
        if (
            len(indices) < 2
            or any(index < 0 or index >= len(value) for index in indices)
            or consumed.intersection(indices)
        ):
            raise ValueError("merge groups must contain two or more distinct, disjoint valid indices")
        consumed.update(indices)
        weights = value[list(indices), _OPACITY]
        total = float(weights.sum())
        weights = np.full(len(indices), 1.0 / len(indices)) if total <= 1e-12 else weights / total
        merged = np.zeros(14, dtype=np.float64)
        merged[_POSITION] = weights @ value[list(indices), _POSITION]
        merged[_ROTATION] = _quaternion_from_rotation(
            SO3.from_quaternion(weights @ value[list(indices), _ROTATION], order="xyzw")
        )
        merged[_SCALE] = np.exp(weights @ np.log(value[list(indices), _SCALE]))
        merged[_OPACITY] = min(1.0, float(weights @ value[list(indices), _OPACITY]))
        merged[_COLOR] = np.clip(weights @ value[list(indices), _COLOR], 0.0, 1.0)
        rows.append(merged)
    keep = [index for index in range(len(value)) if index not in consumed]
    return np.vstack([value[keep], *rows])


def naive_parameter_arithmetic(latent: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """Intentionally invalid baseline: add values without geometry constraints."""
    value = np.asarray(latent, dtype=np.float64)
    offset = np.asarray(delta, dtype=np.float64)
    if value.shape != offset.shape:
        raise ValueError("latent and delta must have matching shapes")
    return value + offset

"""Private validation and pure numerical helpers for reward/value evidence."""

from __future__ import annotations

import numpy as np


def finite_array(value: np.ndarray, *, name: str) -> np.ndarray:
    if not np.issubdtype(value.dtype, np.number):
        raise TypeError(f"{name} must have a numeric dtype")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values")
    return np.asarray(value, dtype=np.float64)


def matrix(value: np.ndarray, *, name: str, width: int | None = None) -> np.ndarray:
    result = finite_array(value, name=name)
    if result.ndim != 2 or (width is not None and result.shape[1] != width):
        suffix = f", {width}" if width is not None else ""
        raise ValueError(f"{name} must have shape (n{suffix}), got {result.shape}")
    return result


def vector(value: np.ndarray, *, name: str, length: int | None = None) -> np.ndarray:
    result = finite_array(value, name=name)
    if result.ndim != 1 or (length is not None and result.shape[0] != length):
        raise ValueError(f"{name} must have shape ({length or 'n'},), got {result.shape}")
    return result


def bool_vector(value: np.ndarray | None, *, name: str, length: int, default: bool) -> np.ndarray:
    if value is None:
        return np.full(length, default, dtype=bool)
    if value.ndim != 1 or value.shape[0] != length:
        raise ValueError(f"{name} must have shape ({length},), got {getattr(value, 'shape', None)}")
    return np.asarray(value, dtype=bool).copy()


def freeze_float_array(value: np.ndarray) -> np.ndarray:
    copied = np.asarray(value, dtype=np.float64).copy()
    copied.setflags(write=False)
    return copied


def freeze_bool_array(value: np.ndarray) -> np.ndarray:
    copied = np.asarray(value, dtype=bool).copy()
    copied.setflags(write=False)
    return copied


def validate_discount(discount: float) -> float:
    if isinstance(discount, bool) or not np.isfinite(discount) or not 0.0 <= discount < 1.0:
        raise ValueError(f"discount must be finite and in [0, 1), got {discount}")
    return float(discount)


def discounted_returns(
    rewards: np.ndarray,
    *,
    discount: float,
    masks: np.ndarray | None = None,
    terminals: np.ndarray | None = None,
) -> np.ndarray:
    """Compute masked, terminal-aware discounted returns."""

    gamma = validate_discount(discount)
    values = finite_array(rewards, name="rewards")
    if values.ndim not in {1, 2}:
        raise ValueError(f"rewards must be 1D or 2D, got {values.shape}")
    was_vector = values.ndim == 1
    episodes = values[None, :] if was_vector else values
    episode_count, horizon = episodes.shape
    if masks is None:
        valid = np.ones_like(episodes, dtype=bool)
    else:
        mask_values = np.asarray(masks)
        if mask_values.shape != values.shape:
            raise ValueError(f"masks must have shape {values.shape}, got {mask_values.shape}")
        valid = mask_values[None, :] if was_vector else mask_values
        valid = np.asarray(valid, dtype=bool)
    if terminals is None:
        terminal_values = np.zeros_like(episodes, dtype=bool)
    else:
        terminal_array = np.asarray(terminals)
        if terminal_array.shape != values.shape:
            raise ValueError(f"terminals must have shape {values.shape}, got {terminal_array.shape}")
        terminal_values = terminal_array[None, :] if was_vector else terminal_array
        terminal_values = np.asarray(terminal_values, dtype=bool)

    result = np.zeros_like(episodes, dtype=np.float64)
    for episode in range(episode_count):
        running = 0.0
        for step in range(horizon - 1, -1, -1):
            if not valid[episode, step]:
                running = 0.0
                continue
            if terminal_values[episode, step] or step == horizon - 1 or not valid[episode, step + 1]:
                running = float(episodes[episode, step])
            else:
                running = float(episodes[episode, step]) + gamma * running
            result[episode, step] = running
    return result[0] if was_vector else result


def bellman_residuals(
    rewards: np.ndarray,
    values: np.ndarray,
    next_values: np.ndarray,
    masks: np.ndarray,
    terminals: np.ndarray,
    *,
    discount: float,
) -> np.ndarray:
    continuation = masks & ~terminals
    if len(continuation) > 1:
        continuation[:-1] &= masks[1:]
    continuation[-1] = False
    residuals = values - (rewards + discount * continuation * next_values)
    residuals[~masks] = 0.0
    return residuals


def bellman_residuals_batch(
    rewards: np.ndarray,
    values: np.ndarray,
    next_values: np.ndarray,
    masks: np.ndarray,
    terminals: np.ndarray,
    *,
    discount: float,
) -> np.ndarray:
    continuation = masks & ~terminals
    if continuation.shape[1] > 1:
        continuation[:, :-1] &= masks[:, 1:]
    continuation[:, -1] = False
    residuals = values - (rewards + discount * continuation * next_values)
    residuals[~masks] = 0.0
    return residuals


def summary_metrics(
    predicted: np.ndarray, target: np.ndarray, masks: np.ndarray
) -> tuple[float, float, float, float, float]:
    valid = masks.astype(bool)
    if not np.any(valid):
        raise ValueError("at least one valid transition is required")
    difference = predicted[valid] - target[valid]
    return (
        float(np.sqrt(np.mean(np.square(difference)))),
        float(np.mean(np.abs(difference))),
        float(np.mean(difference)),
        float(np.mean(predicted[valid])),
        float(np.mean(target[valid])),
    )

"""Private integer-token, action, and padding validation helpers."""

from __future__ import annotations

import numpy as np


def finite_array(value: object, *, name: str) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a numpy array")
    if not np.issubdtype(value.dtype, np.number) or not np.isfinite(value).all():
        raise ValueError(f"{name} must contain finite numeric values")
    return value


def validate_sequence_tokens(
    tokens: np.ndarray,
    *,
    tokens_per_frame: int,
    pad_token_id: int,
    vocab_size: int,
    allow_padding: bool,
) -> np.ndarray:
    values = finite_array(tokens, name="token_sequences")
    if values.ndim != 3 or values.shape[2] != tokens_per_frame:
        raise ValueError(f"token_sequences must have shape (episodes, time, {tokens_per_frame})")
    if not np.issubdtype(values.dtype, np.integer):
        raise TypeError("token_sequences must preserve integer token IDs")
    if allow_padding:
        if np.any(values < 0) or np.any(values > pad_token_id):
            raise ValueError("token IDs must be within the codebook or padding token")
    else:
        validate_tokens(
            values.reshape(-1, tokens_per_frame),
            name="token_sequences",
            tokens_per_frame=tokens_per_frame,
            vocab_size=vocab_size,
            allow_single=False,
        )
    if values.shape[1] < 2:
        raise ValueError("token sequences need at least an initial and next frame")
    return values.astype(np.int64, copy=False)


def validate_tokens(
    tokens: np.ndarray,
    *,
    name: str,
    tokens_per_frame: int,
    vocab_size: int,
    allow_single: bool = False,
    allow_sequence: bool = False,
) -> np.ndarray:
    values = finite_array(tokens, name=name)
    expected_ndim = 3 if allow_sequence else 2
    valid_single = allow_single and values.ndim in {1, 2}
    if (values.ndim != expected_ndim and not valid_single) or values.shape[-1] != tokens_per_frame:
        raise ValueError(f"{name} must end with token shape ({tokens_per_frame},)")
    if not np.issubdtype(values.dtype, np.integer):
        if not np.all(values == np.floor(values)):
            raise TypeError(f"{name} must contain integer token IDs")
        values = values.astype(np.int64)
    if np.any(values < 0) or np.any(values >= vocab_size):
        raise ValueError(f"{name} contains invalid token IDs; expected [0, {vocab_size - 1}]")
    if allow_single and values.ndim == 1:
        values = values[None, :]
    return values.astype(np.int64, copy=False)


def validate_actions(actions: np.ndarray, *, batch_size: int, action_dim: int) -> np.ndarray:
    values = finite_array(actions, name="action")
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2 or values.shape != (batch_size, action_dim):
        raise ValueError(f"action must have shape ({batch_size}, {action_dim})")
    return values.astype(np.float64, copy=False)


def validate_training_actions(
    actions: np.ndarray,
    sequence_mask: np.ndarray | None,
    sequence_shape: tuple[int, int],
    *,
    action_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    values = finite_array(actions, name="actions")
    expected_shape = (sequence_shape[0], sequence_shape[1] - 1, action_dim)
    if values.ndim != 3 or values.shape != expected_shape:
        raise ValueError(f"actions must have shape {expected_shape}")
    if sequence_mask is None:
        mask = np.ones(values.shape[:2], dtype=bool)
    else:
        mask_values = finite_array(sequence_mask, name="sequence_mask")
        if mask_values.shape != values.shape[:2] or not np.all(np.isin(mask_values, [0, 1])):
            raise ValueError("sequence_mask must be a binary array matching the transition batch")
        mask = mask_values.astype(bool)
    return values.astype(np.float64, copy=False), mask

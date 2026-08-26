"""Private teacher-forced and free-running token metric aggregation."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
from torch import nn

from latent_anything._tokenized_dynamics import AutoregressiveDynamics
from latent_anything._tokenized_training import sample_next_tokens


def teacher_forced_metrics(
    dynamics: AutoregressiveDynamics,
    tokens: np.ndarray,
    actions: np.ndarray,
    mask: np.ndarray,
    *,
    vocab_size: int,
) -> tuple[float, float, float, int]:
    current = tokens[:, :-1][mask.astype(bool)]
    target = tokens[:, 1:][mask.astype(bool)]
    flat_actions = actions[mask.astype(bool)]
    current_tensor = torch.from_numpy(current.astype(np.int64))  # pyright: ignore[reportUnknownMemberType]
    target_tensor = torch.from_numpy(target.astype(np.int64))  # pyright: ignore[reportUnknownMemberType]
    action_tensor = torch.from_numpy(flat_actions.astype(np.float32))  # pyright: ignore[reportUnknownMemberType]
    with torch.no_grad():
        hidden = dynamics.encode_context(current_tensor, action_tensor)
        logits = dynamics.decode_teacher_forced(hidden, action_tensor, target_tensor)
        loss = nn.functional.cross_entropy(logits.reshape(-1, vocab_size), target_tensor.reshape(-1))
        accuracy = (logits.argmax(dim=-1) == target_tensor).float().mean()
    cross_entropy = float(loss)  # pyright: ignore[reportUnknownArgumentType]
    return cross_entropy, float(np.exp(cross_entropy)), float(accuracy), int(target.size)  # pyright: ignore[reportUnknownArgumentType]


def free_running_metrics(
    dynamics: AutoregressiveDynamics,
    tokens: np.ndarray,
    actions: np.ndarray,
    mask: np.ndarray,
    *,
    vocab_size: int,
    tokens_per_frame: int,
    pad_token_id: int,
    decode: Callable[[np.ndarray], np.ndarray] | None,
    decoded_targets: np.ndarray | None,
    task_proxy: Callable[[np.ndarray], np.ndarray] | None,
    failure_threshold: float,
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...] | None, tuple[float, ...] | None, int | None]:
    if not 0.0 < failure_threshold <= 1.0:
        raise ValueError("failure_threshold must be in (0, 1]")
    del pad_token_id  # Padding has already been removed from valid transitions.
    horizon = actions.shape[1]
    errors: list[float] = []
    exact: list[float] = []
    decoded_errors: list[float] | None = [] if decoded_targets is not None else None
    task_accuracy: list[float] | None = [] if task_proxy is not None and decoded_targets is not None else None
    current = tokens[:, 0].copy()
    for index in range(horizon):
        valid = mask[:, index].astype(bool)
        target = tokens[:, index + 1]
        result = sample_next_tokens(
            dynamics,
            current,
            actions[:, index],
            vocab_size=vocab_size,
            tokens_per_frame=tokens_per_frame,
            sampling="greedy",
            temperature=1.0,
            top_k=None,
            seed=None,
        )
        predicted = result.tokens
        if not np.any(valid):
            errors.append(float("nan"))
            exact.append(float("nan"))
        else:
            errors.append(float(np.mean(predicted[valid] != target[valid])))
            exact.append(float(np.mean(np.all(predicted[valid] == target[valid], axis=1))))
            if decoded_errors is not None and decoded_targets is not None and decode is not None:
                decoded = decode(predicted[valid])
                decoded_errors.append(float(np.mean(np.square(decoded - decoded_targets[valid, index + 1]))))
                if task_accuracy is not None and task_proxy is not None:
                    predicted_labels = np.asarray(task_proxy(decoded))
                    target_labels = np.asarray(task_proxy(decoded_targets[valid, index + 1]))
                    if predicted_labels.shape != target_labels.shape:
                        raise ValueError("task_proxy must return matching label shapes")
                    task_accuracy.append(float(np.mean(predicted_labels == target_labels)))
        current = predicted
    failure_horizon = next(
        (index + 1 for index, value in enumerate(errors) if np.isfinite(value) and value > failure_threshold),
        None,
    )
    return (
        tuple(errors),
        tuple(exact),
        None if decoded_errors is None else tuple(decoded_errors),
        None if task_accuracy is None else tuple(task_accuracy),
        failure_horizon,
    )

"""Small model-bound helpers for the L04 direct logit-lens lane."""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from scripts._m14_l04_tcav_runtime import RealExecutionError, parameter_digest, resolve_target_token


def seed_everything(seed: int, torch: Any) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def summarize_generation(
    result: Any,
    rows: Sequence[Mapping[str, Any]],
    target_ids: Mapping[str, int],
    other_ids: Mapping[str, int],
) -> tuple[list[dict[str, Any]], float, float]:
    """Extract layerwise target probabilities without retaining prompt text."""
    hidden_states = result.hidden_states
    lens_results = result.lens_results
    if len(hidden_states) != 13 or len(lens_results) != 13:
        raise ValueError("GPT-2 direct lens requires exactly 13 native hidden states and lens results")
    layers = [int(state.layer) for state in hidden_states]
    lens_layers = [int(lens.layer) for lens in lens_results]
    if layers != list(range(13)) or lens_layers != layers:
        raise ValueError("native hidden-state/lens layer mapping is not 0..12")
    logits = np.asarray(result.logits)
    terminal_lens = np.asarray(lens_results[-1].logits)
    terminal_prob = np.asarray(lens_results[-1].probabilities)
    if logits.shape != terminal_lens.shape or terminal_prob.shape != logits.shape:
        raise ValueError("terminal direct lens and forward logits shapes do not match")
    abs_error = np.abs(terminal_lens - logits)
    relative_error = abs_error / np.maximum(np.abs(logits), 1e-12)
    records: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        target_text = str(row["target_text"])
        other_text = " false" if target_text == " true" else " true"
        position = int(np.flatnonzero(np.asarray(result.attention_mask)[index]).max())
        target_token = int(target_ids[target_text])
        other_token = int(other_ids[other_text])
        target_values: list[float] = []
        other_values: list[float] = []
        for lens in lens_results:
            probabilities = np.asarray(lens.probabilities)
            target_values.append(float(probabilities[index, position, target_token]))
            other_values.append(float(probabilities[index, position, other_token]))
        margin = target_values[-1] - other_values[-1]
        records.append(
            {
                "row_id": str(row["row_id"]),
                "group_id": str(row["group_id"]),
                "split": str(row["split"]),
                "causal_pair_id": str(row["causal_pair_id"]),
                "target_position": position,
                "target_token_id": target_token,
                "other_token_id": other_token,
                "target_probabilities": target_values,
                "other_probabilities": other_values,
                "target_margin": float(margin),
                "finite": bool(
                    np.isfinite(target_values).all()
                    and np.isfinite(other_values).all()
                    and np.isfinite(abs_error[index]).all()
                ),
            }
        )
    return records, float(np.max(abs_error)), float(np.max(relative_error))


def validate_targets(tokenizer: Any) -> tuple[dict[str, int], dict[str, str]]:
    resolved = {text: resolve_target_token(tokenizer, text) for text in (" true", " false")}
    if any(token_string != text for text, (_token_id, token_string) in resolved.items()):
        raise ValueError("direct lens target token decode does not exactly match frozen target text")
    ids = {text: int(token_id) for text, (token_id, _token_string) in resolved.items()}
    strings = {text: token_string for text, (_token_id, token_string) in resolved.items()}
    if ids[" true"] == ids[" false"]:
        raise ValueError("direct lens target tokens must be distinct")
    return ids, strings


__all__ = [
    "RealExecutionError",
    "parameter_digest",
    "seed_everything",
    "summarize_generation",
    "validate_targets",
]

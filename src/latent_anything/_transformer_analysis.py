"""Private logit-lens and token-rank calculations for the transformer facade."""

from __future__ import annotations

from typing import Any

import numpy as np


def apply_logit_lens(model: Any, hidden_state: Any, *, apply_final_norm: bool = True) -> np.ndarray:
    """Project a hidden state through the model's final norm and LM head.

    Native decoder-only ``output_hidden_states`` tuples may already contain
    the model's final normalized state.  Callers must set ``apply_final_norm``
    to ``False`` for that terminal native state to avoid applying the final
    normalization twice.
    """
    import torch

    with torch.no_grad():
        if apply_final_norm and hasattr(model, "transformer") and hasattr(model.transformer, "ln_f"):
            normalized = model.transformer.ln_f(hidden_state)
        else:
            normalized = hidden_state
        logits = model.lm_head(normalized)
    logits_np = logits.detach().cpu().numpy().copy()
    logits_np.setflags(write=False)
    return logits_np


def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    exp_logits = np.exp(shifted)
    return exp_logits / np.sum(exp_logits, axis=axis, keepdims=True)


def compute_top_tokens(
    probabilities: np.ndarray,
    top_k: int,
) -> list[list[list[tuple[int, float]]]]:
    batch_size, seq_len, _ = probabilities.shape
    top_tokens: list[list[list[tuple[int, float]]]] = []
    for batch_index in range(batch_size):
        batch_tokens: list[list[tuple[int, float]]] = []
        for sequence_index in range(seq_len):
            position_probabilities = probabilities[batch_index, sequence_index]
            top_indices = np.argsort(position_probabilities)[::-1][:top_k]
            batch_tokens.append([(int(index), float(position_probabilities[index])) for index in top_indices])
        top_tokens.append(batch_tokens)
    return top_tokens


def compute_token_rank_trajectories(
    lens_results: list[Any],
    seq_len: int,
    tokenizer: Any,
) -> tuple[Any, ...]:
    if not lens_results or seq_len < 1:
        return ()
    final_probabilities = lens_results[-1].probabilities
    trajectories: list[Any] = []
    for position in range(min(seq_len, 10)):
        top_token_id = int(np.argmax(final_probabilities[0, position]))
        top_token_str = tokenizer.decode([top_token_id])
        ranks: list[int] = []
        probabilities: list[float] = []
        layers: list[int] = []
        for result in lens_results:
            position_probabilities = result.probabilities[0, position]
            sorted_indices = np.argsort(position_probabilities)[::-1]
            matches = np.where(sorted_indices == top_token_id)[0]
            ranks.append(int(matches[0]) + 1 if len(matches) > 0 else len(sorted_indices))
            probabilities.append(float(position_probabilities[top_token_id]))
            layers.append(result.layer)
        trajectories.append((top_token_id, top_token_str, ranks, probabilities, layers))
    return tuple(trajectories)

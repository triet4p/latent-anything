"""Memory-bounded hidden-state extraction for the L03 real-model lane."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Any

import numpy as np


def _as_mask(value: Any) -> np.ndarray:
    """Convert torch or NumPy attention masks at the integration boundary."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=bool)


def _pool(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    denominator = np.maximum(mask.sum(axis=1, keepdims=True), 1)
    return (np.asarray(values, dtype=np.float64) * mask[:, :, None]).sum(axis=1) / denominator


def extract_batched(
    integration: Any,
    prompts: Sequence[str],
    *,
    layers: Sequence[int],
    max_length: int,
    batch_size: int,
    request_factory: Any,
) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    """Extract pooled hidden states in bounded batches, preserving order.

    The integration remains the owner of tokenization and the real forward
    pass.  This helper copies only pooled requested layers before releasing
    the complete generation result (including logits and lens arrays).
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    output: dict[int, list[np.ndarray]] = {int(layer): [] for layer in layers}
    lengths: list[int] = []
    batch_count = 0
    for start in range(0, len(prompts), batch_size):
        batch = tuple(prompts[start : start + batch_size])
        tokenized = integration.tokenize(batch, max_length=max_length, return_tensors="pt")
        attention = _as_mask(tokenized["attention_mask"])
        batch_lengths = attention.sum(axis=1).astype(int).tolist()
        if any(length >= max_length for length in batch_lengths):
            raise ValueError("a prompt reaches max_length and may have been truncated")
        lengths.extend(batch_lengths)
        result = integration.generate(
            request_factory(
                batch,
                max_length=max_length,
                layers=tuple(int(layer) for layer in layers),
            )
        )
        result_mask = _as_mask(result.attention_mask)
        by_layer = {int(state.layer): state.values for state in result.hidden_states}
        for layer in layers:
            if int(layer) not in by_layer:
                raise ValueError(f"requested hidden layer {layer} was not returned")
            output[int(layer)].append(_pool(by_layer[int(layer)], result_mask))
        del by_layer, result, tokenized
        batch_count += 1
    if not lengths:
        raise ValueError("cannot extract hidden states from an empty prompt sequence")
    return (
        {layer: np.concatenate(chunks, axis=0) for layer, chunks in output.items()},
        {
            "inference_batch_size": batch_size,
            "inference_batch_count": batch_count,
            "token_length_distribution": dict(sorted((str(k), v) for k, v in Counter(lengths).items())),
            "token_length_min": min(lengths),
            "token_length_max": max(lengths),
            "truncation_checked": True,
        },
    )

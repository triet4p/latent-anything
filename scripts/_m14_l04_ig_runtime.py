"""Model/resource seams for the M14 L04 Integrated Gradients handler."""

from __future__ import annotations

import hashlib
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from latent_anything.integrated_gradients import compute_integrated_gradients
from latent_anything.tcav import TransformerLogitTarget


@dataclass(frozen=True)
class Row:
    row_id: str
    group_id: str
    split: str
    input_ids: np.ndarray
    attention_mask: np.ndarray


class RealExecutionError(RuntimeError):
    """An execution failure carrying observed resource facts."""

    def __init__(self, message: str, resources: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.resources = dict(resources)


def seed_everything(seed: int, torch: Any) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parameter_digest(model: Any) -> str:
    digest = hashlib.sha256()
    for parameter in model.parameters():
        digest.update(parameter.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def target_attribution(
    model: Any,
    row: Row,
    *,
    target_token: int,
    other_token: int,
    steps: int,
    baseline: str,
    seed: int,
    source_model_version: str,
    batch_ids: np.ndarray,
    batch_mask: np.ndarray,
    batch_index: int,
) -> tuple[np.ndarray, float, float]:
    from latent_anything.integrated_gradients import IntegratedGradientsConfig

    del seed
    position = int(np.flatnonzero(row.attention_mask).max())
    target = TransformerLogitTarget(token_id=target_token, position=position, batch_index=batch_index)
    other = TransformerLogitTarget(token_id=other_token, position=position, batch_index=batch_index)
    target_result = compute_integrated_gradients(
        model,
        batch_ids,
        batch_mask,
        target,
        config=IntegratedGradientsConfig(
            target_layer=6,
            activation_position=position,
            activation_batch_index=batch_index,
            n_steps=steps,
            baseline=baseline,  # type: ignore[arg-type]
        ),
        source_model_version=source_model_version,
    )
    other_result = compute_integrated_gradients(
        model,
        batch_ids,
        batch_mask,
        other,
        config=IntegratedGradientsConfig(
            target_layer=6,
            activation_position=position,
            activation_batch_index=batch_index,
            n_steps=steps,
            baseline=baseline,  # type: ignore[arg-type]
        ),
        source_model_version=source_model_version,
    )
    attribution = np.asarray(target_result.attributions - other_result.attributions, dtype=np.float64)
    delta = float(target_result.completeness_delta - other_result.completeness_delta)
    return attribution, delta, float(attribution.sum() - delta)


def read_rows(integration: Any, rows: Sequence[Mapping[str, Any]], max_length: int) -> list[Row]:
    prompts = tuple(str(row["prompt"]) for row in rows)
    encoded = integration.tokenize(prompts, max_length=max_length, return_tensors="pt")
    input_ids = encoded["input_ids"].detach().cpu().numpy()
    attention_mask = encoded["attention_mask"].detach().cpu().numpy()
    result: list[Row] = []
    for index, row in enumerate(rows):
        mask = np.asarray(attention_mask[index], dtype=np.int64)
        if not np.any(mask):
            raise ValueError(f"row {row['row_id']!r} has no non-padding token")
        result.append(
            Row(
                str(row["row_id"]),
                str(row["group_id"]),
                str(row["split"]),
                np.asarray(input_ids[index], dtype=np.int64),
                mask,
            )
        )
    return result


__all__ = ["RealExecutionError", "Row", "parameter_digest", "read_rows", "seed_everything", "target_attribution"]

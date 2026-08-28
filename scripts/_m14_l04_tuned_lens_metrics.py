"""Pure metrics for the M14 L04 tuned-lens calibration protocol."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from scripts._m14_l04_ig_metrics import bootstrap


def row_token_kl(teacher_logits: Any, translated_logits: Any, valid_mask: Any) -> np.ndarray:
    """Return one mean token KL value per row, excluding padding positions."""
    import torch

    teacher = teacher_logits.float()
    translated = translated_logits.float()
    mask = valid_mask.to(dtype=torch.bool)
    if teacher.shape != translated.shape or teacher.ndim != 3 or mask.shape != teacher.shape[:2]:
        raise ValueError("tuned-lens logits and mask shapes are incompatible")
    teacher_logp = torch.log_softmax(teacher, dim=-1)
    translated_logp = torch.log_softmax(translated, dim=-1)
    token_kl = torch.sum(torch.exp(teacher_logp) * (teacher_logp - translated_logp), dim=-1)
    counts = mask.sum(dim=1)
    if bool(torch.any(counts <= 0)):
        raise ValueError("each tuned-lens row must contain a non-padding token")
    return (token_kl * mask).sum(dim=1).div(counts).detach().cpu().numpy()


def macro_improvement(
    direct_by_layer: Mapping[int, Sequence[float]], tuned_by_layer: Mapping[int, Sequence[float]]
) -> np.ndarray:
    """Average direct-minus-tuned KL per row across native layers 0..11."""
    layers = tuple(range(12))
    if tuple(sorted(direct_by_layer)) != layers or tuple(sorted(tuned_by_layer)) != layers:
        raise ValueError("tuned-lens macro metric requires exactly fitted native layers 0..11")
    direct = np.asarray([direct_by_layer[layer] for layer in layers], dtype=np.float64)
    tuned = np.asarray([tuned_by_layer[layer] for layer in layers], dtype=np.float64)
    if direct.ndim != 2 or tuned.shape != direct.shape or not np.isfinite(direct).all() or not np.isfinite(tuned).all():
        raise ValueError("tuned-lens layer metrics must be finite row vectors")
    return np.mean(direct - tuned, axis=0)


def improvement_metric(values: Sequence[float], *, seed: int, threshold: float) -> dict[str, Any]:
    """Serialize the strict row-level macro improvement gate."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) == 0 or not np.isfinite(array).all():
        raise ValueError("tuned-lens improvements must be finite and non-empty")
    interval = bootstrap(array.tolist(), seed, replicates=2000, statistic="mean")
    point = float(np.mean(array))
    return {
        "point_estimate": point,
        "confidence_interval_95": interval,
        "units": "nats",
        "aggregation_unit": "independent validation row",
        "statistic": "mean",
        "threshold": float(threshold),
        "comparator": ">",
        "pass": bool(point > threshold),
    }


__all__ = ["improvement_metric", "macro_improvement", "row_token_kl"]

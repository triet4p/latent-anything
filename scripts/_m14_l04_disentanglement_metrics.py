"""Pure metrics and frozen controls for M14 L04.8.

This module deliberately has no model or tokenizer dependency.  It owns the
causal-group aggregation, deterministic shuffle mapping, and bootstrap
semantics so the runtime and validator share one small, auditable contract.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from scripts._m14_l04_ig_metrics import bootstrap

FACTORS = ("animal_cat", "tone_positive")
SEEDS = (17, 29, 41, 53, 67)
BOOTSTRAP_REPLICATES = 2000
POINT_THRESHOLD = 0.1
CI_LOWER_THRESHOLD = 0.05
TRAIN_GROUPS = tuple(f"g{index:02d}" for index in range(1, 9))
HOLDOUT_GROUPS = tuple(f"g{index:02d}" for index in range(9, 13))


def _finite_vector(values: Sequence[float], *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not array.size or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a non-empty finite vector")
    return array


def brier_quality(probabilities: Sequence[float], labels: Sequence[int]) -> float:
    """Return bounded row quality ``1 - (p-y)^2`` averaged by rows."""
    p = _finite_vector(probabilities, name="probabilities")
    y = _finite_vector(labels, name="labels")
    if p.size != y.size or np.any((p < 0.0) | (p > 1.0)) or np.any((y != 0.0) & (y != 1.0)):
        raise ValueError("probabilities and binary labels are incompatible")
    return float(np.mean(1.0 - np.square(p - y)))


def group_factor_quality(
    rows: Sequence[Mapping[str, Any]], probabilities: Sequence[float], labels: Mapping[str, Sequence[int]]
) -> dict[str, dict[str, float]]:
    """Aggregate two rows inside each causal group, then retain each factor."""
    if len(rows) != len(probabilities):
        raise ValueError("row/probability lengths differ")
    grouped: dict[str, dict[str, list[float]]] = {}
    for index, row in enumerate(rows):
        group = str(row["group_id"])
        grouped.setdefault(group, {factor: [] for factor in FACTORS})
        for factor in FACTORS:
            factor_labels = labels.get(factor)
            if factor_labels is None or len(factor_labels) != len(rows):
                raise ValueError(f"labels missing or mis-sized for {factor}")
            grouped[group][factor].append(brier_quality([float(probabilities[index])], [int(factor_labels[index])]))
    return {
        group: {factor: float(np.mean(values)) for factor, values in by_factor.items()}
        for group, by_factor in sorted(grouped.items())
    }


def macro_group_quality(group_factor: Mapping[str, Mapping[str, float]]) -> dict[str, float]:
    """Average the two rows per factor, then macro-average the two factors."""
    result: dict[str, float] = {}
    for group, values in sorted(group_factor.items()):
        if set(values) != set(FACTORS):
            raise ValueError(f"group {group!r} does not contain exactly the frozen factors")
        result[group] = float(np.mean([float(values[factor]) for factor in FACTORS]))
    return result


def deterministic_group_derangement(groups: Sequence[str], seed: int) -> list[dict[str, Any]]:
    """Map each group to another group and mark the mandatory slot reversal."""
    names = sorted(str(group) for group in groups)
    if len(names) < 2 or len(set(names)) != len(names):
        raise ValueError("a derangement requires at least two distinct groups")
    order = np.random.default_rng(int(seed)).permutation(len(names)).tolist()
    mapping = []
    for offset, source_index in enumerate(order):
        target_index = order[(offset + 1) % len(order)]
        mapping.append(
            {"source_group": names[source_index], "target_group": names[target_index], "slot_reversal": True}
        )
    return sorted(mapping, key=lambda item: item["source_group"])


def mapping_digest(mapping: Sequence[Mapping[str, Any]]) -> str:
    payload = json.dumps(list(mapping), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def shuffled_labels(
    rows: Sequence[Mapping[str, Any]], labels: Mapping[str, Sequence[int]], mapping: Sequence[Mapping[str, Any]]
) -> dict[str, list[int]]:
    """Apply group-block derangement plus reversal without touching holdout rows."""
    source_to_target = {str(item["source_group"]): str(item["target_group"]) for item in mapping}
    group_rows: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        group_rows.setdefault(str(row["group_id"]), []).append(index)
    if any(len(indices) != 2 for indices in group_rows.values()):
        raise ValueError("frozen shuffle requires exactly two rows per causal group")
    result = {factor: [int(value) for value in values] for factor, values in labels.items()}
    for source, indices in group_rows.items():
        target_indices = group_rows.get(source_to_target.get(source, ""))
        if target_indices is None:
            raise ValueError("shuffle mapping is not a bijection over train groups")
        for factor in FACTORS:
            values = labels[factor]
            result[factor][indices[0]] = int(values[target_indices[1]])
            result[factor][indices[1]] = int(values[target_indices[0]])
    return result


def fixture_row_summary(row: Mapping[str, Any]) -> dict[str, Any]:
    """Return the exact fixture linkage without retaining prompt/text bytes."""
    result = {
        "row_id": str(row["row_id"]),
        "group_id": str(row["group_id"]),
        "causal_pair_id": str(row["causal_pair_id"]),
        "condition": str(row["condition"]),
        "split": str(row["split"]),
        "target_text_sha256": hashlib.sha256(str(row["target_text"]).encode("utf-8")).hexdigest(),
        "prompt_sha256": hashlib.sha256(str(row["prompt"]).encode("utf-8")).hexdigest(),
        "task_sha256": hashlib.sha256(str(row["task"]).encode("utf-8")).hexdigest(),
        "factor_labels": {factor: int(row["factor_labels"][factor]) for factor in FACTORS},
    }
    return result


def metric(
    values: Sequence[float],
    *,
    seed: int,
    point_threshold: float = POINT_THRESHOLD,
    ci_lower_threshold: float = CI_LOWER_THRESHOLD,
) -> dict[str, Any]:
    array = _finite_vector(values, name="metric values")
    interval = bootstrap(array.tolist(), int(seed), replicates=BOOTSTRAP_REPLICATES)
    point = float(np.mean(array))
    return {
        "point_estimate": point,
        "confidence_interval_95": interval,
        "units": "dimensionless Brier quality gain",
        "aggregation_unit": "independent causal group",
        "statistic": "mean",
        "threshold": float(point_threshold),
        "ci_lower_threshold": float(ci_lower_threshold),
        "comparator": ">",
        "pass": bool(point > float(point_threshold) and interval[0] > float(ci_lower_threshold)),
    }


__all__ = [
    "BOOTSTRAP_REPLICATES",
    "CI_LOWER_THRESHOLD",
    "FACTORS",
    "HOLDOUT_GROUPS",
    "POINT_THRESHOLD",
    "SEEDS",
    "TRAIN_GROUPS",
    "brier_quality",
    "bootstrap",
    "deterministic_group_derangement",
    "group_factor_quality",
    "fixture_row_summary",
    "macro_group_quality",
    "mapping_digest",
    "metric",
    "shuffled_labels",
]

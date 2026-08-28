"""Small metric helpers for the M14 L04 Integrated Gradients lane."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

VALID_STATISTICS = {"mean", "median"}
VALID_COMPARATORS = {"<=", "<", ">=", ">"}


def _validated_values(values: Sequence[float]) -> np.ndarray:
    if len(values) == 0:
        raise ValueError("metric bootstrap requires at least one value")
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not np.isfinite(array).all():
        raise ValueError("metric values must be a finite one-dimensional sequence")
    return array


def _validated_statistic(statistic: str) -> str:
    if statistic not in VALID_STATISTICS:
        raise ValueError(f"unsupported statistic {statistic!r}")
    return statistic


def _validated_threshold(threshold: float) -> float:
    if isinstance(threshold, bool) or not np.isfinite(float(threshold)):
        raise ValueError("metric threshold must be a finite number")
    return float(threshold)


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return 0.0 if denominator == 0.0 else float(np.dot(left, right) / denominator)


def bootstrap(values: Sequence[float], seed: int, replicates: int = 2000, *, statistic: str = "mean") -> list[float]:
    array = _validated_values(values)
    statistic = _validated_statistic(statistic)
    if type(replicates) is not int or replicates <= 0:
        raise ValueError("bootstrap replicates must be a positive integer")
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(array), size=(replicates, len(array)))
    estimates = np.median(array[samples], axis=1) if statistic == "median" else array[samples].mean(axis=1)
    return [float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))]


def metric(
    values: Sequence[float],
    *,
    seed: int,
    threshold: float,
    comparator: str,
    units: str = "dimensionless",
    statistic: str = "mean",
) -> dict[str, Any]:
    array = _validated_values(values)
    statistic = _validated_statistic(statistic)
    threshold = _validated_threshold(threshold)
    if comparator not in VALID_COMPARATORS:
        raise ValueError(f"unsupported comparator {comparator!r}")
    point = float(np.median(array) if statistic == "median" else np.mean(array))
    if comparator == "<=":
        passed = point <= threshold
    elif comparator == "<":
        passed = point < threshold
    elif comparator == ">=":
        passed = point >= threshold
    else:
        passed = point > threshold
    return {
        "point_estimate": point,
        "confidence_interval_95": bootstrap(values, seed, statistic=statistic),
        "units": units,
        "aggregation_unit": "independent causal group",
        "statistic": statistic,
        "threshold": threshold,
        "comparator": comparator,
        "pass": bool(passed),
    }


def group_means(rows: Sequence[dict[str, Any]], key: str) -> list[float]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        grouped.setdefault(str(row["group_id"]), []).append(float(row[key]))
    return [float(np.mean(values)) for _, values in sorted(grouped.items())]


__all__ = ["bootstrap", "cosine", "group_means", "metric"]

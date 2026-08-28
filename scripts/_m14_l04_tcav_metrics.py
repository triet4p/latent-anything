"""Small, fail-closed statistics for the M14 L04 TCAV lane."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def finite_values(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError("TCAV metric values must be a non-empty finite one-dimensional sequence")
    return array


def group_means(rows: Sequence[Mapping[str, Any]], key: str) -> list[float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if "group_id" not in row or key not in row:
            raise ValueError("TCAV group rows require group_id and the requested metric")
        grouped[str(row["group_id"])].append(float(row[key]))
    if not grouped:
        raise ValueError("TCAV group aggregation requires at least one group")
    return [float(np.mean(values)) for _, values in sorted(grouped.items())]


def bootstrap(values: Sequence[float], seed: int, replicates: int = 2000) -> list[float]:
    array = finite_values(values)
    if type(replicates) is not int or replicates <= 0:
        raise ValueError("bootstrap replicates must be a positive integer")
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(array), size=(replicates, len(array)))
    estimates = array[samples].mean(axis=1)
    return [float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))]


def wilson_lower(successes: int, total: int, z: float = 1.959963984540054) -> float:
    if type(successes) is not int or type(total) is not int or total <= 0 or not 0 <= successes <= total:
        raise ValueError("Wilson interval requires integer successes and a positive total")
    n = float(total)
    p = float(successes) / n
    denominator = 1.0 + z * z / n
    centre = p + z * z / (2.0 * n)
    spread = z * float(np.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)))
    return float((centre - spread) / denominator)


def metric(
    values: Sequence[float],
    *,
    seed: int,
    threshold: float,
    comparator: str,
    statistic: str = "mean",
    units: str = "dimensionless",
    aggregation_unit: str = "independent causal group",
) -> dict[str, Any]:
    array = finite_values(values)
    if statistic not in {"mean", "median"}:
        raise ValueError(f"unsupported statistic {statistic!r}")
    if comparator not in {"<=", "<", ">=", ">"}:
        raise ValueError(f"unsupported comparator {comparator!r}")
    if not np.isfinite(float(threshold)):
        raise ValueError("metric threshold must be finite")
    point = float(np.median(array) if statistic == "median" else np.mean(array))
    passed = {
        "<=": point <= threshold,
        "<": point < threshold,
        ">=": point >= threshold,
        ">": point > threshold,
    }[comparator]
    interval = bootstrap(array.tolist(), seed)
    return {
        "point_estimate": point,
        "confidence_interval_95": interval,
        "units": units,
        "aggregation_unit": aggregation_unit,
        "statistic": statistic,
        "threshold": float(threshold),
        "comparator": comparator,
        "pass": bool(passed),
    }


def corrected_empirical_p(observed: float, null_values: Sequence[float]) -> float:
    if not np.isfinite(float(observed)):
        raise ValueError("observed TCAV statistic must be finite")
    null = finite_values(null_values)
    return float((1.0 + np.count_nonzero(null >= float(observed))) / (1.0 + len(null)))


__all__ = ["bootstrap", "corrected_empirical_p", "finite_values", "group_means", "metric", "wilson_lower"]

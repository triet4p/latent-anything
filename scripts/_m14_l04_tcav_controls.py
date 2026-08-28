"""Control and verdict assembly for the M14 L04 TCAV handler."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from scripts._m14_l04_tcav_metrics import metric


def assemble_controls(
    *,
    shuffled_scores: Sequence[float],
    random_scores: Sequence[float],
    matched_scores: Sequence[float],
    off_target_scores: Sequence[float],
    zero_differences: Sequence[float],
    seed: int,
    sensitivity_reference: float,
) -> dict[str, Any]:
    """Build the five controls against the emitted primary sensitivity.

    Null/control scores are the same TCAV sensitivity statistic as the real
    score.  Their threshold is therefore a paired, data-dependent reference;
    callers and validators record/link it explicitly rather than presenting it
    as a frozen plan threshold.
    """
    reference = {"metric": "primary_sensitivity", "value": float(sensitivity_reference)}
    shuffled = metric(shuffled_scores, seed=seed, threshold=sensitivity_reference, comparator="<")
    random = metric(random_scores, seed=seed + 1, threshold=sensitivity_reference, comparator="<")
    matched = metric(matched_scores, seed=seed + 2, threshold=sensitivity_reference, comparator="<")
    off_target = metric(off_target_scores, seed=seed + 3, threshold=sensitivity_reference, comparator="<")
    zero = metric(zero_differences, seed=seed + 4, threshold=1e-6, comparator="<=")
    return {
        "shuffled_concept_labels": {
            "metrics": {"tcav_sensitivity": shuffled},
            "reference": reference,
            "pass": bool(shuffled["pass"]),
        },
        "random_concept_directions": {
            "metrics": {"tcav_sensitivity": random},
            "reference": reference,
            "pass": bool(random["pass"]),
        },
        "matched_norm_null": {
            "metrics": {"tcav_sensitivity": matched},
            "reference": reference,
            "pass": bool(matched["pass"]),
        },
        "off_target_target_token": {
            "metrics": {"tcav_sensitivity": off_target},
            "reference": reference,
            "pass": bool(off_target["pass"]),
        },
        "zero_strength_identity": {
            "metrics": {"absolute_margin_difference": zero},
            "pass": bool(zero["pass"]),
        },
    }


__all__ = ["assemble_controls"]

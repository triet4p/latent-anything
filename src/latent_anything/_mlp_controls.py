"""Pure nonlinear control result types and classification helpers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NonlinearControls:
    """Nonlinear control baselines and memorization test results."""

    shuffled_label_accuracy: float
    memorization_ratio: float
    passed_memorization_test: bool
    chance_accuracy: float


@dataclass(frozen=True)
class ProbeComparison:
    """Side-by-side comparison of linear and nonlinear probe results."""

    linear_accuracy: float
    nonlinear_accuracy: float
    gap: float
    classification: str
    linear_ci95: float
    nonlinear_ci95: float


def classify_probe_comparison(
    *,
    linear_accuracy: float,
    nonlinear_accuracy: float,
    linear_ci95: float,
    memorization_prone: bool,
    accuracy_threshold: float,
    gap_threshold: float,
) -> ProbeComparison:
    """Assemble the stable qualitative linear/nonlinear comparison result."""
    gap = nonlinear_accuracy - linear_accuracy
    above_chance_linear = linear_accuracy >= accuracy_threshold
    above_chance_nonlinear = nonlinear_accuracy >= accuracy_threshold
    meaningful_gap = abs(gap) >= gap_threshold

    if memorization_prone:
        classification = "memorization-prone"
    elif not above_chance_linear and not above_chance_nonlinear:
        classification = "unsupported"
    elif above_chance_linear and above_chance_nonlinear and meaningful_gap and gap > 0:
        classification = "nonlinear-only"
    elif above_chance_linear and not above_chance_nonlinear:
        classification = "linear-only"
    elif above_chance_linear and above_chance_nonlinear and not meaningful_gap:
        classification = "both"
    elif not above_chance_linear and above_chance_nonlinear:
        classification = "nonlinear-only"
    else:
        classification = "both"

    return ProbeComparison(
        linear_accuracy=linear_accuracy,
        nonlinear_accuracy=nonlinear_accuracy,
        gap=gap,
        classification=classification,
        linear_ci95=linear_ci95,
        nonlinear_ci95=0.0,
    )

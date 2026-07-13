"""Compact, control-aware evaluation for latent explanation claims."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class InterventionEffect:
    """Observed effect of a direction, including its unwanted side effects."""

    target_factor_change: float
    off_target_change: float
    decode_degradation: float


@dataclass(frozen=True)
class ExplanationEvaluation:
    """Typed result for fidelity, predictability, controls, and interventions."""

    reconstruction_mse: float
    factor_predictability: float
    shuffled_label_predictability: float
    input_feature_predictability: float
    stability_mean: float
    stability_ci95: float
    locality: float
    intervention_effect: InterventionEffect
    accepts_explanation: bool

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible evidence without exposing mutable state."""
        return asdict(self)


def probe_accuracy(features: np.ndarray, labels: np.ndarray, *, random_state: int) -> float:
    """Fit one held-out linear probe; labels must have at least two classes."""
    if features.ndim != 2 or labels.ndim != 1 or len(features) != len(labels):
        raise ValueError("features must be 2D and labels a matching 1D array")
    if len(np.unique(labels)) < 2:
        raise ValueError("labels must contain at least two classes")
    rng = np.random.default_rng(random_state)
    train_indices: list[int] = []
    test_indices: list[int] = []
    classes = np.unique(labels)
    for label in classes:
        indices = rng.permutation(np.flatnonzero(labels == label))
        split = max(1, int(len(indices) * 0.7))
        train_indices.extend(indices[:split].tolist())
        test_indices.extend(indices[split:].tolist())
    train_x, train_y = features[train_indices], labels[train_indices]
    test_x, test_y = features[test_indices], labels[test_indices]
    centroids = np.stack([train_x[train_y == label].mean(axis=0) for label in classes])
    distances = np.sum((test_x[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
    predicted = np.asarray(classes[np.argmin(distances, axis=1)])
    matches = np.asarray(predicted == test_y, dtype=np.float64)
    return float(matches.mean())


def intervention_effect(
    before: np.ndarray,
    after: np.ndarray,
    target_score_before: np.ndarray,
    target_score_after: np.ndarray,
    off_target_score_before: np.ndarray,
    off_target_score_after: np.ndarray,
    decoded_before: np.ndarray,
    decoded_after: np.ndarray,
) -> InterventionEffect:
    """Measure a direction without conflating target movement and degradation."""
    if before.shape != after.shape or decoded_before.shape != decoded_after.shape:
        raise ValueError("before/after tensors must have matching shapes")
    return InterventionEffect(
        target_factor_change=float(np.mean(target_score_after - target_score_before)),
        off_target_change=float(np.mean(np.abs(off_target_score_after - off_target_score_before))),
        decode_degradation=float(np.mean((decoded_after - decoded_before) ** 2)),
    )


def evaluate_explanation(
    latents: np.ndarray,
    labels: np.ndarray,
    input_features: np.ndarray,
    originals: np.ndarray,
    reconstructions: np.ndarray,
    seed_probe_scores: list[float],
    effect: InterventionEffect,
    *,
    random_state: int = 0,
    min_probe_gain: float = 0.05,
    max_off_target_change: float = 0.1,
    max_reconstruction_mse: float = 0.1,
    max_decode_degradation: float = 0.1,
    max_stability_ci95: float = 0.05,
) -> ExplanationEvaluation:
    """Evaluate a claim against shuffled-label/input baselines and thresholds."""
    if not seed_probe_scores:
        raise ValueError("seed_probe_scores must not be empty")
    predictability = probe_accuracy(latents, labels, random_state=random_state)
    shuffled = np.random.default_rng(random_state).permutation(labels)
    shuffled_score = probe_accuracy(latents, shuffled, random_state=random_state)
    input_score = probe_accuracy(input_features, labels, random_state=random_state)
    score_array = np.asarray(seed_probe_scores, dtype=float)
    ci95 = float(1.96 * score_array.std(ddof=0) / np.sqrt(len(score_array)))
    locality = float(max(0.0, effect.target_factor_change - effect.off_target_change))
    reconstruction_mse = float(np.mean((reconstructions - originals) ** 2))
    values_to_check = (
        predictability,
        shuffled_score,
        input_score,
        reconstruction_mse,
        ci95,
        effect.target_factor_change,
        effect.off_target_change,
        effect.decode_degradation,
    )
    accepted = all(np.isfinite(values_to_check)) and (
        predictability >= shuffled_score + min_probe_gain
        and predictability >= input_score
        and effect.target_factor_change > 0
        and effect.off_target_change <= max_off_target_change
        and reconstruction_mse <= max_reconstruction_mse
        and effect.decode_degradation <= max_decode_degradation
        and ci95 <= max_stability_ci95
    )
    return ExplanationEvaluation(
        reconstruction_mse=reconstruction_mse,
        factor_predictability=predictability,
        shuffled_label_predictability=shuffled_score,
        input_feature_predictability=input_score,
        stability_mean=float(score_array.mean()),
        stability_ci95=ci95,
        locality=locality,
        intervention_effect=effect,
        accepts_explanation=accepted,
    )

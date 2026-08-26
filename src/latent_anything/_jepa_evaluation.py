"""Private JEPA health and metric aggregation helpers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class JEPALatentHealthValues:
    mean_variance: float
    min_variance: float
    max_variance: float
    covariance_condition: float
    effective_rank: float
    participation_ratio: float
    collapsed_fraction: float
    collapse_score: float
    n_samples: int
    latent_dim: int


@dataclass(frozen=True, slots=True)
class JEPAPredictionEvaluation:
    mse: float
    rmse: float
    mean_error: float
    collapsed_baseline_mse: float
    improvement_over_collapsed: float
    target_health: JEPALatentHealthValues
    n_samples: int


@dataclass(frozen=True, slots=True)
class JEPARolloutEvaluation:
    errors_by_horizon: tuple[float, ...]
    mean_error: float
    final_error: float
    horizon_drift: float
    error_growth_ratio: float
    n_episodes: int
    stable: bool


def compute_latent_health(latents: np.ndarray, *, collapse_variance_threshold: float = 1e-5) -> JEPALatentHealthValues:
    """Compute covariance/effective-rank diagnostics without a decoder."""

    if latents.ndim != 2 or latents.shape[0] < 1:
        raise ValueError("latents must be a non-empty two-dimensional array")
    variances = np.var(latents, axis=0)
    if latents.shape[0] < 2:
        covariance = np.zeros((latents.shape[1], latents.shape[1]), dtype=np.float64)
    else:
        covariance = np.asarray(np.cov(latents, rowvar=False), dtype=np.float64)
        if covariance.ndim == 0:
            covariance = covariance.reshape(1, 1)
    eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 0.0)
    total = float(np.sum(eigenvalues))
    squared = float(np.sum(np.square(eigenvalues)))
    effective_rank = total * total / squared if squared > 0.0 else 0.0
    positive = eigenvalues[eigenvalues > collapse_variance_threshold]
    condition = float(np.max(positive) / np.min(positive)) if positive.size else float("inf")
    collapsed_fraction = float(np.mean(variances <= collapse_variance_threshold))
    collapse_score = 1.0 - effective_rank / latents.shape[1]
    return JEPALatentHealthValues(
        mean_variance=float(np.mean(variances)),
        min_variance=float(np.min(variances)),
        max_variance=float(np.max(variances)),
        covariance_condition=condition,
        effective_rank=float(effective_rank),
        participation_ratio=float(effective_rank),
        collapsed_fraction=collapsed_fraction,
        collapse_score=float(np.clip(collapse_score, 0.0, 1.0)),
        n_samples=int(latents.shape[0]),
        latent_dim=int(latents.shape[1]),
    )


def aggregate_prediction_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    *,
    variance_floor: float,
) -> JEPAPredictionEvaluation:
    errors = predictions - targets
    mse = float(np.mean(np.square(errors)))
    baseline = np.mean(targets, axis=0)
    baseline_mse = float(np.mean(np.square(baseline[None, :] - targets)))
    improvement = 0.0 if baseline_mse <= variance_floor else 1.0 - mse / baseline_mse
    return JEPAPredictionEvaluation(
        mse=mse,
        rmse=float(np.sqrt(mse)),
        mean_error=float(np.mean(np.linalg.norm(errors, axis=1))),
        collapsed_baseline_mse=baseline_mse,
        improvement_over_collapsed=float(improvement),
        target_health=compute_latent_health(targets),
        n_samples=int(targets.shape[0]),
    )


def aggregate_rollout_metrics(
    targets: np.ndarray,
    mask: np.ndarray,
    predictions: Sequence[np.ndarray],
    *,
    variance_floor: float,
    stability_norm_limit: float,
) -> JEPARolloutEvaluation:
    errors: list[list[float]] = []
    max_norm = 0.0
    for episode, prediction in enumerate(predictions):
        length = int(np.sum(mask[episode]))
        if length == 0:
            continue
        prediction_values = prediction[1:]
        target = targets[episode, 1 : length + 1]
        max_norm = max(max_norm, float(np.max(np.linalg.norm(prediction_values, axis=1))))
        errors.append([float(np.linalg.norm(value)) for value in target - prediction_values])
    errors_by_horizon = tuple(
        float(np.mean([row[index] for row in errors if index < len(row)]))
        for index in range(mask.shape[1])
        if any(index < len(row) for row in errors)
    )
    first = errors_by_horizon[0] if errors_by_horizon else 0.0
    final = errors_by_horizon[-1] if errors_by_horizon else 0.0
    ratio = final / first if first > variance_floor else 0.0
    return JEPARolloutEvaluation(
        errors_by_horizon=errors_by_horizon,
        mean_error=float(np.mean(errors_by_horizon)) if errors_by_horizon else 0.0,
        final_error=final,
        horizon_drift=final - first,
        error_growth_ratio=float(ratio),
        n_episodes=len(errors),
        stable=bool(np.isfinite(max_norm) and max_norm <= stability_norm_limit),
    )

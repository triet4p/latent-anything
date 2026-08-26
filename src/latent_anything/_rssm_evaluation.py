"""Private RSSM metric aggregation helpers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from latent_anything.rssm import RSSMPrediction


@dataclass(frozen=True, slots=True)
class RSSMOneStepEvaluation:
    mse: float
    rmse: float
    negative_log_likelihood: float
    kl_divergence: float
    coverage: float
    mean_error: float
    n_samples: int


@dataclass(frozen=True, slots=True)
class RSSMRolloutEvaluation:
    errors_by_horizon: tuple[float, ...]
    kl_by_horizon: tuple[float, ...]
    coverage_by_horizon: tuple[float, ...]
    mean_error: float
    final_error: float
    mean_kl: float
    mean_coverage: float
    stable: bool


def aggregate_one_step_metrics(
    predictions: Sequence[Sequence[RSSMPrediction]],
    next_states: np.ndarray,
    mask: np.ndarray,
    *,
    interval_level: float,
    posterior_scale_factor: float,
) -> RSSMOneStepEvaluation:
    """Aggregate masked teacher-forced predictions without owning model state."""

    targets = next_states[:, 1:, :][mask]
    selected = [
        prediction for row, row_mask in zip(predictions, mask) for prediction, valid in zip(row, row_mask) if valid
    ]
    means = np.asarray([prediction.mean for prediction in selected])
    errors = means - targets
    lower = np.asarray([prediction.interval(interval_level)[0] for prediction in selected])
    upper = np.asarray([prediction.interval(interval_level)[1] for prediction in selected])
    nll = float(-np.mean([prediction.log_prob(target) for prediction, target in zip(selected, targets)]))
    kl = float(
        np.mean(
            [
                prediction.kl_to_observation(target, posterior_scale_factor=posterior_scale_factor)
                for prediction, target in zip(selected, targets)
            ]
        )
    )
    coverage = float(np.mean((targets >= lower) & (targets <= upper)))
    mse = float(np.mean(np.square(errors)))
    return RSSMOneStepEvaluation(
        mse=mse,
        rmse=float(np.sqrt(mse)),
        negative_log_likelihood=nll,
        kl_divergence=kl,
        coverage=coverage,
        mean_error=float(np.mean(np.linalg.norm(errors, axis=1))),
        n_samples=len(selected),
    )


def aggregate_rollout_metrics(
    targets: np.ndarray,
    mask: np.ndarray,
    means: Sequence[np.ndarray],
    scales: Sequence[np.ndarray],
    *,
    variance_floor: float,
    stability_norm_limit: float,
) -> RSSMRolloutEvaluation:
    """Aggregate masked open-loop paths produced by the recurrent runtime."""

    errors: list[list[float]] = []
    kls: list[list[float]] = []
    coverages: list[list[float]] = []
    predicted_norms: list[float] = []
    for episode, (mean, rollout_scale) in enumerate(zip(means, scales)):
        length = int(np.sum(mask[episode]))
        if length == 0:
            continue
        scale = np.maximum(rollout_scale[1:], np.sqrt(variance_floor))
        target = targets[episode, 1 : length + 1]
        differences = target - mean[1:]
        errors.append([float(np.linalg.norm(value)) for value in differences])
        predicted_norms.append(float(np.max(np.linalg.norm(mean[1:], axis=1))))
        lower = mean[1:] - 1.959963984540054 * scale
        upper = mean[1:] + 1.959963984540054 * scale
        coverages.append(
            [
                float(np.mean((target[index] >= lower[index]) & (target[index] <= upper[index])))
                for index in range(length)
            ]
        )
        kls.append([float(0.5 * np.sum(np.square(differences[index] / scale[index]))) for index in range(length)])
    horizon = targets.shape[1] - 1
    errors_by_horizon = tuple(
        float(np.mean([row[index] for row in errors if index < len(row)]))
        for index in range(horizon)
        if any(index < len(row) for row in errors)
    )
    kl_by_horizon = tuple(
        float(np.mean([row[index] for row in kls if index < len(row)])) for index in range(len(errors_by_horizon))
    )
    coverage_by_horizon = tuple(
        float(np.mean([row[index] for row in coverages if index < len(row)])) for index in range(len(errors_by_horizon))
    )
    max_state_norm = max(predicted_norms, default=0.0)
    return RSSMRolloutEvaluation(
        errors_by_horizon=errors_by_horizon,
        kl_by_horizon=kl_by_horizon,
        coverage_by_horizon=coverage_by_horizon,
        mean_error=float(np.mean(errors_by_horizon)) if errors_by_horizon else 0.0,
        final_error=errors_by_horizon[-1] if errors_by_horizon else 0.0,
        mean_kl=float(np.mean(kl_by_horizon)) if kl_by_horizon else 0.0,
        mean_coverage=float(np.mean(coverage_by_horizon)) if coverage_by_horizon else 1.0,
        stable=bool(np.isfinite(max_state_norm) and max_state_norm <= stability_norm_limit),
    )

"""Held-out multi-view metrics for 3D Gaussian interventions."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from latent_anything.adapters.gaussian_3d_renderer import GaussianCamera


@dataclass(frozen=True)
class MultiViewEvaluation:
    """Quantitative evidence for one Gaussian intervention."""

    target_position_change: float
    off_target_drift: float
    multi_view_image_consistency: float
    render_quality_degradation: float
    per_view_effect: tuple[float, ...]


def evaluate_multiview(
    baseline: np.ndarray,
    edited: np.ndarray,
    *,
    target_indices: Sequence[int],
    cameras: Sequence[GaussianCamera],
    render: Callable[[np.ndarray, GaussianCamera], np.ndarray],
) -> MultiViewEvaluation:
    """Render both states from held-out cameras and calculate bounded metrics."""
    if baseline.shape != edited.shape or baseline.ndim != 2 or baseline.shape[1] != 14:
        raise ValueError("baseline and edited latents must have matching shape (n, 14)")
    target = np.asarray(tuple(target_indices), dtype=int)
    mask = np.ones(len(baseline), dtype=bool)
    mask[target] = False
    displacement = np.linalg.norm(edited[:, :3] - baseline[:, :3], axis=1)
    target_change = float(np.mean(np.asarray(displacement[target], dtype=np.float64))) if target.size else 0.0
    off_target = float(np.max(displacement[mask])) if np.any(mask) else 0.0
    effects: list[float] = []
    degradations: list[float] = []
    for camera in cameras:
        before = np.asarray(render(baseline, camera), dtype=np.float64)
        after = np.asarray(render(edited, camera), dtype=np.float64)
        if before.shape != after.shape:
            raise ValueError("render outputs must have matching shapes")
        difference = float(np.mean(np.abs(after - before)))
        effects.append(difference)
        degradations.append(float(np.mean((after - before) ** 2)))
    if not effects:
        raise ValueError("at least one held-out camera is required")
    effect_array = np.asarray(effects, dtype=np.float64)
    consistency = (
        1.0
        if float(effect_array.mean()) <= 1e-12
        else float(np.clip(1.0 - effect_array.std() / effect_array.mean(), 0.0, 1.0))
    )
    return MultiViewEvaluation(target_change, off_target, consistency, float(np.mean(degradations)), tuple(effects))

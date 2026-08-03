"""Generate quantitative Sprint 53 temporal-analysis evidence."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from latent_anything import (
    LatentSpace,
    SegmentationConfig,
    SmoothingConfig,
    Trajectory,
    detect_change_points,
    evaluate_boundaries,
    smooth_trajectory,
    smoothing_distortion,
)


def build_proxy_policy_trajectory(seed: int = 53) -> tuple[Trajectory, tuple[int, ...]]:
    """Build a demonstration-like trajectory with proxy phase annotations."""
    rng = np.random.default_rng(seed)
    hold_a = 0.02 * rng.normal(size=(15, 2))
    slow = np.column_stack([np.linspace(0.0, 1.0, 20), np.zeros(20)])
    fast = np.column_stack([1.0 + np.linspace(0.0, 4.0, 20), np.zeros(20)])
    hold_b = np.column_stack([np.full(15, 5.0), np.zeros(15)]) + 0.02 * rng.normal(size=(15, 2))
    values = np.concatenate([hold_a, slow, fast, hold_b])
    metadata = {"source": "synthetic_policy_proxy", "phase_labels": "proxy_annotations"}
    return Trajectory(values, metadata=metadata), (15, 35, 55)


def main() -> None:
    trajectory, truth = build_proxy_policy_trajectory()
    space = LatentSpace(dim=2, geometry="euclidean", source_model="proxy_policy")
    smoothed = smooth_trajectory(trajectory, space, config=SmoothingConfig(window=5))
    segmentation = detect_change_points(
        smoothed.trajectory,
        space,
        config=SegmentationConfig(sensitivity=1.0, min_segment_length=5, context=4),
    )
    boundary_metrics = evaluate_boundaries(segmentation.boundaries, truth, tolerance=2)
    report = {
        "representation": "2D latent policy trajectory with proxy phase annotations",
        "n_steps": len(trajectory),
        "truth_boundaries": list(truth),
        "predicted_boundaries": list(segmentation.boundaries),
        "boundary_metrics_tolerance_2": {
            "precision": boundary_metrics.precision,
            "recall": boundary_metrics.recall,
            "f1": boundary_metrics.f1,
            "mean_tolerance": boundary_metrics.mean_tolerance,
        },
        "smoothing": {
            "config": dict(smoothed.provenance),
            "mean_distortion": smoothing_distortion(trajectory, smoothed.trajectory, space),
            "maximum_distortion": smoothed.maximum_distortion,
        },
        "segmentation": {
            "threshold": segmentation.threshold,
            "scores": list(segmentation.scores),
            "confidence": list(segmentation.confidence),
            "provenance": dict(segmentation.provenance),
        },
    }
    output = Path("artifacts")
    output.mkdir(exist_ok=True)
    (output / "trajectory_temporal_benchmark.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

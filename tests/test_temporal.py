"""Synthetic and quantitative tests for trajectory temporal analysis."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from latent_anything import (
    LatentSpace,
    SegmentationConfig,
    SmoothingConfig,
    Trajectory,
    detect_change_points,
    evaluate_boundaries,
    smooth_trajectory,
)


def _space(dim: int = 1) -> LatentSpace:
    return LatentSpace(dim=dim, geometry="euclidean", source_model="synthetic")


def test_smoothing_removes_noise_and_preserves_metadata() -> None:
    clean = np.linspace(0.0, 1.0, 41)[:, None]
    noisy = clean.copy()
    noisy[::2] += 0.18
    trajectory = Trajectory(noisy, metadata={"episode": "demo", "phase": "annotated"})
    result = smooth_trajectory(trajectory, _space(), config=SmoothingConfig(window=5))
    assert result.trajectory.metadata["episode"] == "demo"
    assert np.std(np.diff(result.trajectory.to_numpy()[:, 0])) < np.std(np.diff(noisy[:, 0]))
    assert result.mean_distortion > 0
    with pytest.raises(TypeError):
        result.trajectory.metadata["episode"] = "changed"  # type: ignore[index]


def test_smoothing_uses_geometry_interpolation_on_unit_sphere() -> None:
    space = LatentSpace(dim=2, geometry="unit_norm", source_model="synthetic")
    angles = np.linspace(0.0, np.pi / 2, 15)
    values = np.column_stack([np.cos(angles), np.sin(angles)])
    result = smooth_trajectory(Trajectory(values), space, config=SmoothingConfig(window=3))
    assert_allclose(np.linalg.norm(result.trajectory.to_numpy(), axis=1), 1.0, atol=1e-7)


def test_change_points_recover_two_phase_velocity_boundary() -> None:
    first = np.column_stack([np.arange(0, 10, dtype=float), np.zeros(10)])
    second = np.column_stack([np.full(10, 9.0) + np.arange(10, dtype=float) * 3.0, np.zeros(10)])
    trajectory = Trajectory(np.concatenate([first, second]), metadata={"episode": "synthetic"})
    result = detect_change_points(
        trajectory,
        _space(2),
        config=SegmentationConfig(sensitivity=1.0, min_segment_length=3, context=3),
    )
    assert result.boundaries == (10,)
    assert result.source_metadata["episode"] == "synthetic"
    assert result.confidence == (1.0,)


def test_change_points_reject_short_segments_and_no_change() -> None:
    constant = Trajectory(np.arange(30, dtype=float)[:, None])
    result = detect_change_points(constant, _space(), config=SegmentationConfig(sensitivity=3.0))
    assert result.boundaries == ()
    short = np.concatenate([np.arange(10.0), np.array([10.2, 10.4]), np.arange(11.0, 21.0)])[..., None]
    result_short = detect_change_points(
        Trajectory(short), _space(), config=SegmentationConfig(sensitivity=0.0, min_segment_length=4)
    )
    assert all(segment.length >= 4 for segment in result_short.segments)


def test_boundary_metrics_are_tolerance_aware() -> None:
    metrics = evaluate_boundaries([9, 20], [10, 30], tolerance=1)
    assert metrics.precision == 0.5
    assert metrics.recall == 0.5
    assert metrics.f1 == 0.5
    assert metrics.mean_tolerance == 1.0

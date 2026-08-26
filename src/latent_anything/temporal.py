"""Smoothing, segmentation, and quantitative evaluation for latent trajectories."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from latent_anything.latent_space import LatentSpace
from latent_anything.trajectory import Trajectory


@dataclass(frozen=True)
class SmoothingConfig:
    """Configuration for a centered, geometry-aware moving-average smoother."""

    window: int = 5
    weighting: str = "uniform"

    def __post_init__(self) -> None:
        if self.window < 1 or self.window % 2 == 0:
            raise ValueError("window must be a positive odd integer")
        if self.weighting not in {"uniform", "triangular"}:
            raise ValueError("weighting must be 'uniform' or 'triangular'")


@dataclass(frozen=True)
class SmoothedTrajectory:
    """A smoothed trajectory plus reproducibility and distortion metadata."""

    trajectory: Trajectory
    provenance: Mapping[str, Any]
    mean_distortion: float
    maximum_distortion: float


@dataclass(frozen=True)
class SegmentationConfig:
    """Robust velocity-change detector configuration."""

    sensitivity: float = 3.0
    min_segment_length: int = 3
    context: int = 3
    threshold: float | None = None

    def __post_init__(self) -> None:
        if self.sensitivity < 0:
            raise ValueError("sensitivity must be >= 0")
        if self.min_segment_length < 1:
            raise ValueError("min_segment_length must be >= 1")
        if self.context < 1:
            raise ValueError("context must be >= 1")
        if self.threshold is not None and self.threshold < 0:
            raise ValueError("threshold must be >= 0")


@dataclass(frozen=True)
class Segment:
    """Half-open trajectory segment boundaries."""

    start: int
    stop: int

    @property
    def length(self) -> int:
        """Return the number of frames in this half-open segment."""
        return self.stop - self.start


@dataclass(frozen=True)
class ChangePointResult:
    """Typed segmentation result with scores and hyperparameter provenance."""

    boundaries: tuple[int, ...]
    segments: tuple[Segment, ...]
    scores: tuple[float, ...]
    confidence: tuple[float, ...]
    velocity: tuple[float, ...]
    threshold: float
    geometry: str
    source_metadata: Mapping[str, Any]
    provenance: Mapping[str, Any]


@dataclass(frozen=True)
class BoundaryMetrics:
    """Tolerance-aware precision/recall for change-point boundaries."""

    precision: float
    recall: float
    f1: float
    mean_tolerance: float
    matched: int
    predicted: int
    truth: int


def _weights(width: int, weighting: str) -> np.ndarray:
    if weighting == "uniform":
        return np.ones(width, dtype=np.float64)
    distances = np.abs(np.arange(width, dtype=np.float64) - (width - 1) / 2)
    return (distances.max() + 1.0 - distances).astype(np.float64)


def _weighted_point(points: np.ndarray, weights: np.ndarray, space: LatentSpace) -> np.ndarray:
    if space.geometry == "euclidean":
        return np.average(points, axis=0, weights=weights)
    result = points[0].copy()
    total = float(weights[0])
    for point, weight in zip(points[1:], weights[1:], strict=True):
        next_total = total + float(weight)
        result = space.interpolate(result, point, float(weight) / next_total)
        total = next_total
    return result


def smooth_trajectory(
    trajectory: Trajectory,
    space: LatentSpace,
    *,
    config: SmoothingConfig | None = None,
) -> SmoothedTrajectory:
    """Smooth a trajectory while preserving geometry and immutable metadata."""
    options = config or SmoothingConfig()
    values = trajectory.to_numpy()
    radius = options.window // 2
    weights = _weights(options.window, options.weighting)
    smoothed = np.empty_like(values, dtype=np.float64)
    for index in range(len(trajectory)):
        indices = np.clip(np.arange(index - radius, index + radius + 1), 0, len(trajectory) - 1)
        smoothed[index] = _weighted_point(values[indices], weights, space)
    distances = np.asarray([space.distance(before, after) for before, after in zip(values, smoothed, strict=True)])
    result = Trajectory(smoothed, metadata=trajectory.metadata)
    provenance: dict[str, Any] = {
        "method": "centered_geometry_aware_moving_average",
        "window": options.window,
        "weighting": options.weighting,
        "geometry": space.geometry,
        "source_metadata": dict(trajectory.metadata),
    }
    return SmoothedTrajectory(result, provenance, float(np.mean(distances)), float(np.max(distances)))


def _velocity(values: np.ndarray, space: LatentSpace) -> np.ndarray:
    return np.asarray([space.distance(a, b) for a, b in zip(values[:-1], values[1:], strict=True)], dtype=np.float64)


def detect_change_points(
    trajectory: Trajectory,
    space: LatentSpace,
    *,
    config: SegmentationConfig | None = None,
) -> ChangePointResult:
    """Detect phase boundaries from robust local changes in latent velocity."""
    options = config or SegmentationConfig()
    values = trajectory.to_numpy()
    velocity = _velocity(values, space)
    scores = np.zeros(max(len(trajectory) - 1, 0), dtype=np.float64)
    for boundary in range(1, len(trajectory) - 1):
        left_start = max(0, boundary - options.context)
        right_stop = min(len(velocity), boundary + options.context)
        left = velocity[left_start:boundary]
        right = velocity[boundary:right_stop]
        if len(left) and len(right):
            scores[boundary] = abs(float(np.mean(left)) - float(np.mean(right)))
    nonzero = scores[scores > 0]
    median = float(np.median(nonzero)) if len(nonzero) else 0.0
    mad = float(np.median(np.abs(nonzero - median))) if len(nonzero) else 0.0
    robust_scale = max(1.4826 * mad, np.finfo(np.float64).eps)
    threshold = options.threshold if options.threshold is not None else median + options.sensitivity * robust_scale
    candidates = [
        index
        for index, score in enumerate(scores)
        if score > threshold
        and index >= options.min_segment_length
        and len(trajectory) - index >= options.min_segment_length
    ]
    selected: list[int] = []
    for candidate in sorted(candidates, key=lambda index: (-scores[index], index)):
        if all(abs(candidate - prior) >= options.min_segment_length for prior in selected):
            selected.append(candidate)
    boundaries = tuple(sorted(selected))
    edges = (0, *boundaries, len(trajectory))
    segments = tuple(Segment(start, stop) for start, stop in zip(edges[:-1], edges[1:], strict=True))
    selected_scores = tuple(float(scores[index]) for index in boundaries)
    max_score = max(selected_scores, default=0.0)
    confidence = tuple(score / max_score if max_score else 0.0 for score in selected_scores)
    provenance: dict[str, Any] = {
        "signal": "latent_velocity",
        "sensitivity": options.sensitivity,
        "min_segment_length": options.min_segment_length,
        "context": options.context,
        "threshold": options.threshold,
        "robust_center": median,
        "robust_scale": robust_scale,
    }
    return ChangePointResult(
        boundaries=boundaries,
        segments=segments,
        scores=selected_scores,
        confidence=confidence,
        velocity=tuple(float(value) for value in velocity),
        threshold=float(threshold),
        geometry=space.geometry,
        source_metadata=trajectory.metadata,
        provenance=provenance,
    )


def evaluate_boundaries(
    predicted: tuple[int, ...] | list[int],
    truth: tuple[int, ...] | list[int],
    *,
    tolerance: int = 1,
) -> BoundaryMetrics:
    """Match predicted boundaries to truth once each within an index tolerance."""
    if tolerance < 0:
        raise ValueError("tolerance must be >= 0")
    remaining = list(truth)
    errors: list[int] = []
    for candidate in sorted(predicted):
        matches = [
            (abs(candidate - target), index)
            for index, target in enumerate(remaining)
            if abs(candidate - target) <= tolerance
        ]
        if matches:
            error, index = min(matches)
            errors.append(error)
            remaining.pop(index)
    matched = len(errors)
    precision = matched / len(predicted) if predicted else 1.0 if not truth else 0.0
    recall = matched / len(truth) if truth else 1.0 if not predicted else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    mean_tolerance = float(np.mean(errors)) if errors else float("inf")
    return BoundaryMetrics(precision, recall, f1, mean_tolerance, matched, len(predicted), len(truth))


def smoothing_distortion(original: Trajectory, smoothed: Trajectory, space: LatentSpace) -> float:
    """Return mean geometry distance between original and smoothed points."""
    if original.shape != smoothed.shape:
        raise ValueError("original and smoothed trajectories must have the same shape")
    return float(np.mean([space.distance(a, b) for a, b in zip(original.to_numpy(), smoothed.to_numpy(), strict=True)]))

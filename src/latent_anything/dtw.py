"""Geometry-aware dynamic time warping for latent trajectories."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from latent_anything.latent_space import LatentSpace
from latent_anything.trajectory import Trajectory

TrajectoryInput = Trajectory | np.ndarray | Sequence[np.ndarray]


@dataclass(frozen=True)
class DTWConfig:
    """Constraints and normalization policy for a DTW comparison."""

    window: int | None = None
    max_step_distance: float | None = None
    normalization: str = "path_length"
    max_cells: int = 2_000_000
    return_cost_matrix: bool = False

    def __post_init__(self) -> None:
        if self.window is not None and self.window < 0:
            raise ValueError("window must be >= 0")
        if self.max_step_distance is not None and self.max_step_distance < 0:
            raise ValueError("max_step_distance must be >= 0")
        if self.normalization not in {"none", "path_length", "reference_length", "query_length"}:
            raise ValueError(
                "normalization must be one of 'none', 'path_length', 'reference_length', or 'query_length'"
            )
        if self.max_cells < 1:
            raise ValueError("max_cells must be >= 1")


@dataclass(frozen=True)
class DTWCostSummary:
    """Compact summary of the accumulated DTW cost matrix."""

    shape: tuple[int, int]
    finite_cells: int
    minimum: float
    maximum: float
    terminal_cost: float


@dataclass(frozen=True)
class DTWResult:
    """Typed output of a geometry-aware DTW comparison."""

    distance: float
    raw_distance: float
    path: tuple[tuple[int, int], ...]
    point_costs: tuple[float, ...]
    cost_summary: DTWCostSummary
    normalization: str
    geometry: str
    provenance: dict[str, Any]
    cost_matrix: np.ndarray | None = None


def _as_points(value: TrajectoryInput, *, name: str) -> np.ndarray:
    points = value.to_numpy() if isinstance(value, Trajectory) else np.asarray(value, dtype=np.float64)
    if points.ndim != 2:
        raise ValueError(f"{name} must be a 2D array of points")
    if points.shape[1] < 1:
        raise ValueError(f"{name} must have at least one feature")
    if not np.isfinite(points).all():
        raise ValueError(f"{name} must contain only finite values")
    return points


def _normalization_denominator(policy: str, path_length: int, n_query: int, n_reference: int) -> float:
    if policy == "none":
        return 1.0
    if policy == "path_length":
        return float(path_length)
    if policy == "query_length":
        return float(n_query)
    return float(n_reference)


def compute_dtw(
    query: TrajectoryInput,
    reference: TrajectoryInput,
    space: LatentSpace,
    *,
    config: DTWConfig | None = None,
) -> DTWResult:
    """Compute an optimal DTW alignment using ``space.distance`` point costs.

    The exact traceback requires one predecessor cell per admissible matrix
    cell. ``DTWConfig.max_cells`` is therefore an explicit memory guard.
    Costs are tie-broken deterministically in diagonal, vertical, horizontal
    order, which prefers progress in both sequences when alternatives tie.
    """
    options = config or DTWConfig()
    query_points = _as_points(query, name="query")
    reference_points = _as_points(reference, name="reference")
    expected_shape = space.shape
    expected_columns = space.dim if len(expected_shape) == 1 else int(np.prod(expected_shape))
    if query_points.shape[1] != expected_columns:
        raise ValueError(f"query point dimension {query_points.shape[1]} does not match {expected_columns}")
    if reference_points.shape[1] != query_points.shape[1]:
        raise ValueError("query and reference points must have the same dimension")
    n_query, n_reference = query_points.shape[0], reference_points.shape[0]
    if n_query == 0 or n_reference == 0:
        raise ValueError("DTW requires non-empty query and reference sequences")
    if n_query * n_reference > options.max_cells:
        raise ValueError(f"DTW cost matrix has {n_query * n_reference} cells; max_cells={options.max_cells}")

    point_costs = np.full((n_query, n_reference), np.inf, dtype=np.float64)
    for i in range(n_query):
        for j in range(n_reference):
            if options.window is not None and abs(i - j) > options.window:
                continue
            left = query_points[i]
            right = reference_points[j]
            if len(expected_shape) > 1:
                left = left.reshape(expected_shape)
                right = right.reshape(expected_shape)
            cost = space.distance(left, right)
            if options.max_step_distance is None or cost <= options.max_step_distance:
                point_costs[i, j] = cost

    accumulated = np.full_like(point_costs, np.inf)
    predecessor = np.full((n_query, n_reference), -1, dtype=np.int8)
    for i in range(n_query):
        for j in range(n_reference):
            cost = point_costs[i, j]
            if not np.isfinite(cost):
                continue
            if i == 0 and j == 0:
                accumulated[i, j] = cost
                continue
            candidates: list[tuple[float, int, int, int]] = []
            if i > 0 and j > 0:
                candidates.append((accumulated[i - 1, j - 1], 0, i - 1, j - 1))
            if i > 0:
                candidates.append((accumulated[i - 1, j], 1, i - 1, j))
            if j > 0:
                candidates.append((accumulated[i, j - 1], 2, i, j - 1))
            best = min(candidates, key=lambda item: (item[0], item[1]))
            if np.isfinite(best[0]):
                accumulated[i, j] = cost + best[0]
                predecessor[i, j] = best[1]
    terminal = float(accumulated[-1, -1])
    if not np.isfinite(terminal):
        raise ValueError("DTW constraints leave no valid alignment path")

    path_reversed: list[tuple[int, int]] = []
    i, j = n_query - 1, n_reference - 1
    while True:
        path_reversed.append((i, j))
        if i == 0 and j == 0:
            break
        move = predecessor[i, j]
        if move == 0:
            i, j = i - 1, j - 1
        elif move == 1:
            i -= 1
        elif move == 2:
            j -= 1
        else:
            raise RuntimeError("DTW traceback encountered an invalid predecessor")
    path = tuple(reversed(path_reversed))
    aligned_costs = tuple(float(point_costs[i, j]) for i, j in path)
    denominator = _normalization_denominator(options.normalization, len(path), n_query, n_reference)
    summary = DTWCostSummary(
        shape=point_costs.shape,
        finite_cells=int(np.isfinite(accumulated).sum()),
        minimum=float(np.nanmin(accumulated)),
        maximum=float(np.nanmax(accumulated[np.isfinite(accumulated)])),
        terminal_cost=terminal,
    )
    provenance: dict[str, Any] = {
        "query_length": n_query,
        "reference_length": n_reference,
        "window": options.window,
        "max_step_distance": options.max_step_distance,
        "space_dim": space.dim,
        "source_model": space.source_model,
        "space_metadata": dict(space.metadata),
    }
    return DTWResult(
        distance=terminal / denominator,
        raw_distance=terminal,
        path=path,
        point_costs=aligned_costs,
        cost_summary=summary,
        normalization=options.normalization,
        geometry=space.geometry,
        provenance=provenance,
        cost_matrix=accumulated.copy() if options.return_cost_matrix else None,
    )


def indexwise_distance(query: TrajectoryInput, reference: TrajectoryInput, space: LatentSpace) -> float:
    """Return mean index-wise point distance for equal-length sequences."""
    query_points = _as_points(query, name="query")
    reference_points = _as_points(reference, name="reference")
    if query_points.shape != reference_points.shape:
        raise ValueError("index-wise distance requires equal-shaped sequences")
    if len(query_points) == 0:
        raise ValueError("index-wise distance requires non-empty sequences")
    return float(np.mean([space.distance(a, b) for a, b in zip(query_points, reference_points, strict=True)]))

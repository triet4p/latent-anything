"""Analytic tests for geometry-aware dynamic time warping."""

from __future__ import annotations

import numpy as np
import pytest

from latent_anything import DTWConfig, LatentSpace, Trajectory, compute_dtw, indexwise_distance


def _space(dim: int = 1) -> LatentSpace:
    return LatentSpace(dim=dim, geometry="euclidean", source_model="analytic")


def test_identical_sequences_have_zero_cost_and_diagonal_path() -> None:
    values = np.arange(4, dtype=np.float64).reshape(-1, 1)
    result = compute_dtw(Trajectory(values), Trajectory(values), _space())
    assert result.distance == pytest.approx(0.0)
    assert result.path == tuple((i, i) for i in range(4))
    assert result.cost_summary.shape == (4, 4)


def test_shifted_sequence_uses_geometry_cost() -> None:
    result = compute_dtw(np.array([[0.0], [1.0]]), np.array([[1.0], [2.0]]), _space())
    assert result.raw_distance == pytest.approx(2.0)
    assert result.distance == pytest.approx(1.0)
    assert result.geometry == "euclidean"


def test_unequal_length_sequence_aligns_repeated_point() -> None:
    result = compute_dtw(np.array([[0.0], [1.0], [1.0], [2.0]]), np.array([[0.0], [1.0], [2.0]]), _space())
    assert result.distance == pytest.approx(0.0)
    assert len(result.path) == 4


def test_window_and_step_constraints_are_enforced() -> None:
    with pytest.raises(ValueError, match="no valid alignment"):
        compute_dtw(np.array([[0.0], [1.0]]), np.array([[0.0], [1.0], [2.0]]), _space(), config=DTWConfig(window=0))
    with pytest.raises(ValueError, match="max_step_distance"):
        DTWConfig(max_step_distance=-1)


def test_empty_and_invalid_space_inputs_raise() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        compute_dtw(np.empty((0, 1)), np.ones((1, 1)), _space())
    with pytest.raises(ValueError, match="dimension"):
        compute_dtw(np.ones((1, 2)), np.ones((1, 2)), _space())


def test_ties_are_deterministic_and_cost_matrix_is_optional() -> None:
    query = np.array([[0.0], [1.0]])
    reference = np.array([[0.0], [0.0], [1.0]])
    config = DTWConfig(return_cost_matrix=True)
    first = compute_dtw(query, reference, _space(), config=config)
    second = compute_dtw(query, reference, _space(), config=config)
    assert first.path == second.path
    assert first.cost_matrix is not None
    assert first.cost_matrix.shape == (2, 3)


def test_stretched_sequence_beats_indexwise_baseline() -> None:
    query = np.array([[0.0], [1.0], [2.0]])
    reference = np.array([[0.0], [0.0], [1.0], [2.0]])
    result = compute_dtw(query, reference, _space())
    with pytest.raises(ValueError):
        indexwise_distance(query, reference, _space())
    assert result.distance == pytest.approx(0.0)


def test_memory_guard_rejects_oversized_exact_traceback() -> None:
    with pytest.raises(ValueError, match="max_cells"):
        compute_dtw(np.zeros((3, 1)), np.zeros((3, 1)), _space(), config=DTWConfig(max_cells=8))


def test_euclidean_vectorized_point_costs_preserve_window_and_traceback() -> None:
    query = np.array([[0.0, 0.0], [1.0, 0.5], [2.0, 1.0]])
    reference = np.array([[0.0, 0.0], [0.1, 0.1], [1.0, 0.5], [2.0, 1.0]])
    result = compute_dtw(query, reference, _space(dim=2), config=DTWConfig(window=2, return_cost_matrix=True))
    assert result.cost_matrix is not None
    assert np.isinf(result.cost_matrix[0, 3])
    assert result.path[0] == (0, 0)
    assert result.path[-1] == (2, 3)
    assert result.distance == pytest.approx(0.03535533905932738)

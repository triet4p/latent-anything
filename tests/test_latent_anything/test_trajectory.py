"""Tests for the Trajectory class.

Uses hypothesis for property-based tests on immutability and slicing.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from numpy.testing import assert_array_equal

from latent_anything import Trajectory

# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

# Small integers to keep tests fast
_n_strat = st.integers(min_value=1, max_value=20)
_d_strat = st.integers(min_value=1, max_value=16)

# A 2D array of reasonable size
_arrays = _n_strat.flatmap(
    lambda n: _d_strat.flatmap(lambda d: st.just(np.random.default_rng(42).random((n, d)).astype(np.float64)))
)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestTrajectoryInit:
    def test_from_2d_array(self) -> None:
        data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        traj = Trajectory(data)
        assert len(traj) == 3
        assert traj.dim == 2
        assert traj.shape == (3, 2)

    def test_single_point_valid(self) -> None:
        data = np.array([[0.5, 0.3, 0.1]])
        traj = Trajectory(data)
        assert len(traj) == 1
        assert traj.shape == (1, 3)

    def test_non_array_raises(self) -> None:
        with pytest.raises((TypeError, AttributeError)):
            Trajectory([[1.0, 2.0], [3.0, 4.0]])  # type: ignore[arg-type]

    def test_1d_array_raises(self) -> None:
        with pytest.raises(ValueError, match="Expected 2D array"):
            Trajectory(np.array([1.0, 2.0, 3.0]))

    def test_3d_array_raises(self) -> None:
        with pytest.raises(ValueError, match="Expected 2D array"):
            Trajectory(np.ones((2, 3, 4)))

    def test_empty_array_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one point"):
            Trajectory(np.empty((0, 4)))


# ---------------------------------------------------------------------------
# Immutability (property-based with hypothesis)
# ---------------------------------------------------------------------------


class TestTrajectoryImmutable:
    @given(data=_arrays)
    def test_to_numpy_returns_copy(self, data: np.ndarray) -> None:
        traj = Trajectory(data)
        result = traj.to_numpy()
        # Verify it's equal to the original
        assert_array_equal(result, data)
        # Verify it's a different buffer (copy, not view)
        result[0, 0] = -999.0
        assert_array_equal(traj.to_numpy(), data)

    @given(data=_arrays)
    def test_original_data_mutation_does_not_affect_trajectory(
        self,
        data: np.ndarray,
    ) -> None:
        original_copy = data.copy()
        traj = Trajectory(data)
        # Mutate the original reference
        data[:] = -1.0
        assert_array_equal(traj.to_numpy(), original_copy)


# ---------------------------------------------------------------------------
# Indexing and slicing
# ---------------------------------------------------------------------------


class TestTrajectoryIndexing:
    @given(data=_arrays)
    def test_integer_index_returns_single_point_trajectory(self, data: np.ndarray) -> None:
        traj = Trajectory(data)
        assume(len(traj) >= 2)
        idx = len(traj) // 2
        subset = traj[idx]
        assert isinstance(subset, Trajectory)
        assert subset.shape == (1, data.shape[1])
        assert_array_equal(subset.to_numpy(), data[idx : idx + 1])

    @given(data=_arrays)
    def test_slice_returns_new_trajectory(self, data: np.ndarray) -> None:
        traj = Trajectory(data)
        assume(len(traj) >= 3)
        subset = traj[1:3]
        assert isinstance(subset, Trajectory)
        assert subset.shape == (min(2, len(traj) - 1), data.shape[1])
        assert_array_equal(subset.to_numpy(), data[1:3])

    @given(data=_arrays)
    def test_full_slice_is_independent_copy(self, data: np.ndarray) -> None:
        traj = Trajectory(data)
        full = traj[:]
        assert_array_equal(full.to_numpy(), data)
        # Mutate the returned data
        mut = full.to_numpy()
        mut[:] = 0.0
        # Original should be untouched
        assert_array_equal(traj.to_numpy(), data)

    def test_index_out_of_range_raises(self) -> None:
        traj = Trajectory(np.array([[1.0], [2.0]]))
        with pytest.raises(IndexError):
            _ = traj[5]


# ---------------------------------------------------------------------------
# len, dim, shape
# ---------------------------------------------------------------------------


class TestTrajectoryProperties:
    @given(data=_arrays)
    def test_len_matches_n_points(self, data: np.ndarray) -> None:
        traj = Trajectory(data)
        assert len(traj) == data.shape[0]

    @given(data=_arrays)
    def test_dim_matches_n_features(self, data: np.ndarray) -> None:
        traj = Trajectory(data)
        assert traj.dim == data.shape[1]

    @given(data=_arrays)
    def test_shape_matches_data(self, data: np.ndarray) -> None:
        traj = Trajectory(data)
        assert traj.shape == data.shape


# ---------------------------------------------------------------------------
# Repr
# ---------------------------------------------------------------------------


class TestTrajectoryRepr:
    def test_repr(self) -> None:
        traj = Trajectory(np.ones((5, 3)))
        r = repr(traj)
        assert "Trajectory" in r
        assert "n_points=5" in r
        assert "dim=3" in r

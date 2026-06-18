"""Tests for the Lerp class — B-Method #1, stateless interpolation.

Target: ~14–16 tests covering:
- Construction with/without LatentSpace
- ``__call__`` produces correct interpolation (t=0 → a, t=1 → b, t=0.5 → midpoint)
- Geometry dispatch (slerp stays on sphere, lerp doesn't)
- ``between`` produces correct Trajectory shape
- ``blend_sequence`` produces densified trajectory
- Error cases (mismatched dims, mismatched trajectory lengths)
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from latent_anything import LatentSpace, Trajectory  # noqa: I001
from latent_anything.methods import Lerp

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestLerpInit:
    def test_default_construction(self) -> None:
        lerp = Lerp()
        assert lerp.space is None

    def test_with_euclidean_space(self) -> None:
        space = LatentSpace(dim=8)
        lerp = Lerp(space=space)
        assert lerp.space is space
        assert lerp.space is not None
        assert lerp.space.geometry == "euclidean"

    def test_with_spherical_space(self) -> None:
        space = LatentSpace(dim=8, geometry="unit_norm")
        lerp = Lerp(space=space)
        assert lerp.space is space
        assert lerp.space is not None
        assert lerp.space.geometry == "unit_norm"


# ---------------------------------------------------------------------------
# __call__ — single-point interpolation
# ---------------------------------------------------------------------------


class TestLerpCall:
    def test_t0_returns_a(self) -> None:
        lerp = Lerp()
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([7.0, 8.0, 9.0])
        result = lerp(a, b, 0.0)
        assert_array_almost_equal(result, a)

    def test_t1_returns_b(self) -> None:
        lerp = Lerp()
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([7.0, 8.0, 9.0])
        result = lerp(a, b, 1.0)
        assert_array_almost_equal(result, b)

    def test_t05_midpoint(self) -> None:
        lerp = Lerp()
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([10.0, 20.0, 30.0])
        result = lerp(a, b, 0.5)
        assert_array_almost_equal(result, [5.0, 10.0, 15.0])

    def test_does_not_mutate_inputs(self) -> None:
        lerp = Lerp()
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        a_copy = a.copy()
        b_copy = b.copy()
        _ = lerp(a, b, 0.3)
        assert_array_almost_equal(a, a_copy)
        assert_array_almost_equal(b, b_copy)

    def test_returns_new_array(self) -> None:
        lerp = Lerp()
        a = np.array([1.0, 2.0])
        b = np.array([3.0, 4.0])
        result = lerp(a, b, 0.5)
        assert result is not a
        assert result is not b

    def test_mismatched_shapes_raises(self) -> None:
        lerp = Lerp()
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0])
        with pytest.raises(ValueError, match="same shape"):
            lerp(a, b, 0.5)

    def test_different_dims_raises(self) -> None:
        lerp = Lerp()
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 2.0])
        with pytest.raises(ValueError, match="same shape"):
            lerp(a, b, 0.5)


# ---------------------------------------------------------------------------
# Geometry dispatch: Euclidean vs spherical
# ---------------------------------------------------------------------------


class TestLerpGeometryDispatch:
    def test_euclidean_lerp_default(self) -> None:
        """No space → pure Euclidean (1-t)*a + t*b."""
        lerp = Lerp()
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        for t in [0.25, 0.5, 0.75]:
            pt = lerp(a, b, t)
            # Euclidean lerp departs from sphere norm
            assert np.linalg.norm(pt) < 1.0

    def test_spherical_slerp_stays_on_sphere(self) -> None:
        """With unit_norm space → slerp, path stays on sphere."""
        space = LatentSpace(dim=3, geometry="unit_norm")
        lerp = Lerp(space=space)
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        for t in [0.25, 0.5, 0.75]:
            pt = lerp(a, b, t)
            assert abs(np.linalg.norm(pt) - 1.0) < 1e-10

    def test_spherical_slerp_midpoint(self) -> None:
        """Midpoint of orthogonal unit vectors at 45°."""
        space = LatentSpace(dim=3, geometry="unit_norm")
        lerp = Lerp(space=space)
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        mid = lerp(a, b, 0.5)
        expected = np.array([np.sqrt(2) / 2, np.sqrt(2) / 2, 0.0])
        assert_array_almost_equal(mid, expected)

    def test_slerp_same_vector(self) -> None:
        """Interpolating a vector with itself returns itself."""
        space = LatentSpace(dim=3, geometry="unit_norm")
        lerp = Lerp(space=space)
        a = np.array([0.6, 0.8, 0.0])
        result = lerp(a, a, 0.5)
        assert_array_almost_equal(result, a)

    def test_space_used_consistently(self) -> None:
        """Lerp with Euclidean space should NOT dispatch slerp."""
        euc_space = LatentSpace(dim=3, geometry="euclidean")
        lerp_euc = Lerp(space=euc_space)
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        pt = lerp_euc(a, b, 0.5)
        # Euclidean space → Euclidean lerp, norm < 1
        assert np.linalg.norm(pt) < 1.0


# ---------------------------------------------------------------------------
# between — pointwise trajectory interpolation
# ---------------------------------------------------------------------------


class TestLerpBetween:
    def test_between_produces_correct_shape(self) -> None:
        lerp = Lerp()
        data_a = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        data_b = np.array([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]])
        traj_a = Trajectory(data_a)
        traj_b = Trajectory(data_b)
        result = lerp.between(traj_a, traj_b, 0.5)
        assert len(result) == 3
        assert result.dim == 2
        assert result.shape == (3, 2)

    def test_between_t0(self) -> None:
        lerp = Lerp()
        data_a = np.array([[1.0, 2.0], [3.0, 4.0]])
        data_b = np.array([[5.0, 6.0], [7.0, 8.0]])
        traj_a = Trajectory(data_a)
        traj_b = Trajectory(data_b)
        result = lerp.between(traj_a, traj_b, 0.0)
        assert_array_almost_equal(result.to_numpy(), data_a)

    def test_between_t1(self) -> None:
        lerp = Lerp()
        data_a = np.array([[1.0, 2.0], [3.0, 4.0]])
        data_b = np.array([[5.0, 6.0], [7.0, 8.0]])
        traj_a = Trajectory(data_a)
        traj_b = Trajectory(data_b)
        result = lerp.between(traj_a, traj_b, 1.0)
        assert_array_almost_equal(result.to_numpy(), data_b)

    def test_between_mismatched_lengths_raises(self) -> None:
        lerp = Lerp()
        data_a = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        data_b = np.array([[7.0, 8.0], [9.0, 10.0]])
        traj_a = Trajectory(data_a)
        traj_b = Trajectory(data_b)
        with pytest.raises(ValueError, match="same length"):
            lerp.between(traj_a, traj_b, 0.5)

    def test_between_mismatched_dims_raises(self) -> None:
        lerp = Lerp()
        data_a = np.array([[1.0, 2.0], [3.0, 4.0]])
        data_b = np.array([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])
        traj_a = Trajectory(data_a)
        traj_b = Trajectory(data_b)
        with pytest.raises(ValueError, match="same dim"):
            lerp.between(traj_a, traj_b, 0.5)

    def test_between_returns_new_trajectory(self) -> None:
        lerp = Lerp()
        data_a = np.array([[1.0, 2.0], [3.0, 4.0]])
        data_b = np.array([[5.0, 6.0], [7.0, 8.0]])
        traj_a = Trajectory(data_a)
        traj_b = Trajectory(data_b)
        result = lerp.between(traj_a, traj_b, 0.5)
        assert result is not traj_a
        assert result is not traj_b


# ---------------------------------------------------------------------------
# blend_sequence — trajectory densification
# ---------------------------------------------------------------------------


class TestLerpBlendSequence:
    def test_blend_sequence_default_steps(self) -> None:
        """n_steps=2 doubles points between each original pair."""
        lerp = Lerp()
        data = np.array([[0.0, 0.0], [10.0, 10.0]])
        traj = Trajectory(data)
        dense = lerp.blend_sequence(traj, n_steps=2)
        # [p0, p0.5, p1] for 2 steps with 2 points → 3 points
        assert len(dense) == 3
        assert dense.dim == 2
        # p0.5 is the midpoint
        assert_array_almost_equal(dense.to_numpy()[1], [5.0, 5.0])

    def test_blend_sequence_three_points(self) -> None:
        """3 points, n_steps=2 → 5 points total."""
        lerp = Lerp()
        data = np.array([[0.0, 0.0], [10.0, 10.0], [20.0, 20.0]])
        traj = Trajectory(data)
        dense = lerp.blend_sequence(traj, n_steps=2)
        # [p0, p0.5, p1, p1.5, p2] → 5 points
        assert len(dense) == 5

    def test_blend_sequence_preserves_endpoints(self) -> None:
        lerp = Lerp()
        data = np.array([[0.0, 0.0], [10.0, 10.0], [20.0, 20.0]])
        traj = Trajectory(data)
        dense = lerp.blend_sequence(traj, n_steps=3)
        assert_array_almost_equal(dense.to_numpy()[0], data[0])
        assert_array_almost_equal(dense.to_numpy()[-1], data[-1])

    def test_blend_sequence_n_steps_1(self) -> None:
        """n_steps=1 → no densification (same as original)."""
        lerp = Lerp()
        data = np.array([[0.0, 0.0], [5.0, 5.0], [10.0, 10.0]])
        traj = Trajectory(data)
        dense = lerp.blend_sequence(traj, n_steps=1)
        assert len(dense) == 3
        assert_array_almost_equal(dense.to_numpy(), data)

    def test_blend_sequence_n_steps_0_raises(self) -> None:
        lerp = Lerp()
        data = np.array([[0.0, 0.0], [10.0, 10.0]])
        traj = Trajectory(data)
        with pytest.raises(ValueError, match="n_steps must be >= 1"):
            lerp.blend_sequence(traj, n_steps=0)

    def test_blend_sequence_returns_new_trajectory(self) -> None:
        lerp = Lerp()
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        traj = Trajectory(data)
        result = lerp.blend_sequence(traj, n_steps=2)
        assert result is not traj

    def test_blend_sequence_single_point(self) -> None:
        """Single-point trajectory → same single point."""
        lerp = Lerp()
        data = np.array([[5.0, 5.0]])
        traj = Trajectory(data)
        dense = lerp.blend_sequence(traj, n_steps=2)
        assert len(dense) == 1
        assert_array_almost_equal(dense.to_numpy(), data)

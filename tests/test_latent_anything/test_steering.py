"""Tests for the SteeringVector class — B-Method #2, stateful steering.

Target: ~18–20 tests covering:
- Construction with/without LatentSpace
- ``fit`` learns correct direction from contrast pairs
- ``direction`` property works after fit, raises before fit
- ``__call__`` moves points along learned direction
- Edge cases: zero strength, negative strength, wrong dim, empty arrays
- ``apply_trajectory`` preserves shape and returns new Trajectory
- Spherical normalization with unit_norm space
- Input non-mutation
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from latent_anything import LatentSpace, Trajectory
from latent_anything.methods import SteeringVector

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture
def contrast_8d(
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Simple 8D contrast: positives around [+1, +1, 0...], negatives around [-1, -1, 0...]."""
    center_pos = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    center_neg = np.array([-1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    positives = center_pos + rng.normal(scale=0.3, size=(20, 8))
    negatives = center_neg + rng.normal(scale=0.3, size=(20, 8))
    return positives, negatives


@pytest.fixture
def fitted_sv_8d(contrast_8d: tuple[np.ndarray, np.ndarray]) -> SteeringVector:
    """Pre-fitted SteeringVector on 8D contrast data."""
    positives, negatives = contrast_8d
    sv = SteeringVector()
    sv.fit(positives, negatives)
    return sv


@pytest.fixture
def spherical_contrast(
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Unit-norm 3D contrast: positives around [1,0,0], negatives around [-1,0,0]."""
    pos_center = np.array([1.0, 0.0, 0.0])
    neg_center = np.array([-1.0, 0.0, 0.0])

    def _around(center: np.ndarray, n: int) -> np.ndarray:
        pts = center + rng.normal(scale=0.3, size=(n, 3))
        return pts / np.linalg.norm(pts, axis=1, keepdims=True)

    return _around(pos_center, 20), _around(neg_center, 20)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestSteeringConstruction:
    def test_default_construction(self) -> None:
        sv = SteeringVector()
        assert sv.space is None
        assert not sv.is_fitted

    def test_construction_with_euclidean_space(self) -> None:
        space = LatentSpace(dim=8)
        sv = SteeringVector(space=space)
        assert sv.space is space
        assert sv.space is not None
        assert sv.space.geometry == "euclidean"

    def test_construction_with_spherical_space(self) -> None:
        space = LatentSpace(dim=3, geometry="unit_norm")
        sv = SteeringVector(space=space)
        assert sv.space is space
        assert sv.space is not None
        assert sv.space.geometry == "unit_norm"


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------


class TestSteeringFit:
    def test_fit_learns_direction(self, contrast_8d: tuple[np.ndarray, np.ndarray]) -> None:
        positives, negatives = contrast_8d
        sv = SteeringVector()
        sv.fit(positives, negatives)
        assert sv.is_fitted
        # Direction should be unit norm
        assert abs(np.linalg.norm(sv.direction) - 1.0) < 1e-10

    def test_fit_direction_dot_with_true_direction(self, contrast_8d: tuple[np.ndarray, np.ndarray]) -> None:
        """Learned direction should point from neg region toward pos region."""
        positives, negatives = contrast_8d
        sv = SteeringVector()
        sv.fit(positives, negatives)
        # True direction is approximately [+1, +1, 0...] - [-1, -1, 0...] = [2, 2, 0...]
        # Normalised: [1/sqrt(2), 1/sqrt(2), 0...]
        true_dir = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) / np.sqrt(2)
        dot = np.dot(sv.direction, true_dir)
        assert dot > 0.9, f"Direction dot product {dot:.4f} < 0.9"

    def test_fit_mismatched_dims_raises(self) -> None:
        sv = SteeringVector()
        pos = np.zeros((10, 8))
        neg = np.zeros((10, 4))
        with pytest.raises(ValueError, match="Dimension mismatch"):
            sv.fit(pos, neg)

    def test_fit_empty_positives_raises(self) -> None:
        sv = SteeringVector()
        pos = np.empty((0, 8))
        neg = np.zeros((10, 8))
        with pytest.raises(ValueError, match="empty"):
            sv.fit(pos, neg)

    def test_fit_empty_negatives_raises(self) -> None:
        sv = SteeringVector()
        pos = np.zeros((10, 8))
        neg = np.empty((0, 8))
        with pytest.raises(ValueError, match="empty"):
            sv.fit(pos, neg)

    def test_fit_non_2d_positives_raises(self) -> None:
        sv = SteeringVector()
        pos = np.zeros(8)  # 1D
        neg = np.zeros((10, 8))
        with pytest.raises(ValueError, match="2D"):
            sv.fit(pos, neg)

    def test_fit_identical_means_raises(self) -> None:
        sv = SteeringVector()
        pos = np.ones((10, 4)) * 5.0
        neg = np.ones((10, 4)) * 5.0  # Same means
        with pytest.raises(ValueError, match="zero|identical"):
            sv.fit(pos, neg)

    def test_is_fitted_flag(self, contrast_8d: tuple[np.ndarray, np.ndarray]) -> None:
        positives, negatives = contrast_8d
        sv = SteeringVector()
        assert not sv.is_fitted
        sv.fit(positives, negatives)
        assert sv.is_fitted


# ---------------------------------------------------------------------------
# direction property
# ---------------------------------------------------------------------------


class TestSteeringDirection:
    def test_direction_before_fit_raises(self) -> None:
        sv = SteeringVector()
        with pytest.raises(RuntimeError, match="not fitted"):
            _ = sv.direction

    def test_direction_is_unit_norm(self, fitted_sv_8d: SteeringVector) -> None:
        assert abs(np.linalg.norm(fitted_sv_8d.direction) - 1.0) < 1e-10

    def test_direction_returns_copy(self, fitted_sv_8d: SteeringVector) -> None:
        d1 = fitted_sv_8d.direction
        d2 = fitted_sv_8d.direction
        # Each call returns a new array
        assert d1 is not d2
        assert_array_almost_equal(d1, d2)


# ---------------------------------------------------------------------------
# __call__
# ---------------------------------------------------------------------------


class TestSteeringCall:
    def test_call_before_fit_raises(self) -> None:
        sv = SteeringVector()
        latent = np.zeros(8)
        with pytest.raises(RuntimeError, match="not fitted"):
            sv(latent, 1.0)

    def test_call_moves_in_direction(self, fitted_sv_8d: SteeringVector) -> None:
        latent = np.zeros(8)
        result = fitted_sv_8d(latent, 1.0)
        # Should move along learned direction
        expected_dir = fitted_sv_8d.direction
        assert_array_almost_equal(result, expected_dir)

    def test_call_zero_strength_returns_copy(self, fitted_sv_8d: SteeringVector) -> None:
        latent = np.array([1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        result = fitted_sv_8d(latent, 0.0)
        assert_array_almost_equal(result, latent)
        assert result is not latent  # not the same object

    def test_call_negative_strength_reverses(self, fitted_sv_8d: SteeringVector) -> None:
        latent = np.zeros(8)
        pos = fitted_sv_8d(latent, 1.0)
        neg = fitted_sv_8d(latent, -1.0)
        # Positive and negative should be opposite
        assert_array_almost_equal(pos, -neg)

    def test_call_preserves_input(self, fitted_sv_8d: SteeringVector) -> None:
        latent = np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        latent_copy = latent.copy()
        _ = fitted_sv_8d(latent, 0.5)
        assert_array_almost_equal(latent, latent_copy)

    def test_call_wrong_dim_raises(self, fitted_sv_8d: SteeringVector) -> None:
        latent = np.zeros(4)  # wrong dim
        with pytest.raises(ValueError, match="shape.*does not match"):
            fitted_sv_8d(latent, 1.0)

    def test_call_returns_new_array(self, fitted_sv_8d: SteeringVector) -> None:
        latent = np.zeros(8)
        result = fitted_sv_8d(latent, 0.5)
        assert result is not latent

    def test_call_strength_scales_linearly(self, fitted_sv_8d: SteeringVector) -> None:
        latent = np.zeros(8)
        r1 = fitted_sv_8d(latent, 1.0)
        r2 = fitted_sv_8d(latent, 2.0)
        # Double the strength should double the shift
        assert_array_almost_equal(r2, r1 * 2.0)

    def test_call_with_float32_input(self, fitted_sv_8d: SteeringVector) -> None:
        """Steering works with float32 input (non-float64)."""
        latent = np.zeros(8, dtype=np.float32)
        result = fitted_sv_8d(latent, 1.0)
        assert result.dtype == np.float64  # promoted by arithmetic
        assert result.shape == (8,)


# ---------------------------------------------------------------------------
# apply_trajectory
# ---------------------------------------------------------------------------


class TestSteeringApplyTrajectory:
    def test_apply_trajectory_returns_new(self, fitted_sv_8d: SteeringVector) -> None:
        data = np.zeros((5, 8))
        traj = Trajectory(data)
        result = fitted_sv_8d.apply_trajectory(traj, 1.0)
        assert isinstance(result, Trajectory)
        assert result is not traj

    def test_apply_trajectory_preserves_shape(self, fitted_sv_8d: SteeringVector) -> None:
        data = np.zeros((5, 8))
        traj = Trajectory(data)
        result = fitted_sv_8d.apply_trajectory(traj, 1.0)
        assert result.shape == (5, 8)

    def test_apply_trajectory_moves_all_points(self, fitted_sv_8d: SteeringVector) -> None:
        data = np.zeros((5, 8))
        traj = Trajectory(data)
        result = fitted_sv_8d.apply_trajectory(traj, 1.0)
        expected_dir = fitted_sv_8d.direction
        for row in result.to_numpy():
            assert_array_almost_equal(row, expected_dir)

    def test_apply_trajectory_before_fit_raises(self) -> None:
        sv = SteeringVector()
        traj = Trajectory(np.zeros((3, 8)))
        with pytest.raises(RuntimeError, match="not fitted"):
            sv.apply_trajectory(traj, 1.0)

    def test_apply_trajectory_zero_strength(self, fitted_sv_8d: SteeringVector) -> None:
        data = np.random.default_rng(42).normal(size=(4, 8))
        traj = Trajectory(data)
        result = fitted_sv_8d.apply_trajectory(traj, 0.0)
        assert_array_almost_equal(result.to_numpy(), data)


# ---------------------------------------------------------------------------
# Spherical (geometry-aware) normalization
# ---------------------------------------------------------------------------


class TestSteeringSpherical:
    def test_spherical_normalization(self, spherical_contrast: tuple[np.ndarray, np.ndarray]) -> None:
        positives, negatives = spherical_contrast
        space = LatentSpace(dim=3, geometry="unit_norm")
        sv = SteeringVector(space=space)
        sv.fit(positives, negatives)
        latent = np.array([1.0, 0.0, 0.0])
        steered = sv(latent, 1.0)
        # Steered point should stay on sphere
        assert abs(np.linalg.norm(steered) - 1.0) < 1e-10

    def test_spherical_steering_at_multiple_strengths(self, spherical_contrast: tuple[np.ndarray, np.ndarray]) -> None:
        positives, negatives = spherical_contrast
        space = LatentSpace(dim=3, geometry="unit_norm")
        sv = SteeringVector(space=space)
        sv.fit(positives, negatives)
        latent = np.array([1.0, 0.0, 0.0])
        for s in [0.0, 0.5, 1.0, 2.0]:
            steered = sv(latent, s)
            assert abs(np.linalg.norm(steered) - 1.0) < 1e-10, (
                f"Norm {np.linalg.norm(steered):.6f} != 1 at strength {s}"
            )

    def test_spherical_euclidean_space_no_normalization(
        self, spherical_contrast: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """With Euclidean space (or None), steering leaves the sphere."""
        positives, negatives = spherical_contrast
        sv = SteeringVector(space=None)
        sv.fit(positives, negatives)
        latent = np.array([1.0, 0.0, 0.0])
        steered = sv(latent, 1.0)
        # Euclidean steering should break unit norm
        assert abs(np.linalg.norm(steered) - 1.0) > 1e-6


# ---------------------------------------------------------------------------
# No torch leakage
# ---------------------------------------------------------------------------


class TestSteeringNoTorch:
    def test_no_torch_in_steering_module(self) -> None:
        """Verify the steering module does not import torch."""
        import importlib
        import sys

        # Ensure no torch is loaded from the steering module
        mod_name = "latent_anything.methods.steering"
        if mod_name in sys.modules:
            del sys.modules[mod_name]
        mod = importlib.import_module(mod_name)
        source = mod.__file__ or ""
        with open(source) as f:
            content = f.read()
        assert "torch" not in content, "steering.py should not reference torch"

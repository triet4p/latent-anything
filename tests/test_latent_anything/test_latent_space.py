"""Tests for the LatentSpace class."""

from __future__ import annotations

import numpy as np
import pytest

from latent_anything import LatentSpace


class TestLatentSpaceInit:
    """Construction and basic attributes."""

    def test_default_construction(self) -> None:
        space = LatentSpace(dim=8)
        assert space.dim == 8
        assert space.geometry == "euclidean"
        assert space.source_model == ""
        assert space.metadata == {}

    def test_with_source_model(self) -> None:
        space = LatentSpace(dim=64, source_model="test-vae")
        assert space.source_model == "test-vae"

    def test_with_metadata(self) -> None:
        meta = {"layers": ["fc1", "fc2"], "type": "conv"}
        space = LatentSpace(dim=16, metadata=meta)
        assert space.metadata == meta
        # Ensure we keep our own copy
        meta["extra"] = True  # type: ignore[assignment]
        assert "extra" not in space.metadata

    def test_shape_property(self) -> None:
        space = LatentSpace(dim=128)
        assert space.shape == (128,)

    def test_negative_dim_raises(self) -> None:
        with pytest.raises(ValueError, match="dim must be >= 1"):
            LatentSpace(dim=0)

    def test_negative_dim_raises_negative(self) -> None:
        with pytest.raises(ValueError, match="dim must be >= 1"):
            LatentSpace(dim=-5)


class TestLatentSpaceValidatePoint:
    """Point validation."""

    def test_valid_point(self) -> None:
        space = LatentSpace(dim=4)
        point = np.array([1.0, 2.0, 3.0, 4.0])
        # Should not raise
        space.validate_point(point)

    def test_wrong_shape_raises(self) -> None:
        space = LatentSpace(dim=4)
        with pytest.raises(ValueError, match="Expected shape \\(4,\\)"):
            space.validate_point(np.array([1.0, 2.0, 3.0]))

    def test_non_array_raises(self) -> None:
        space = LatentSpace(dim=4)
        with pytest.raises((TypeError, AttributeError)):
            space.validate_point([1.0, 2.0, 3.0, 4.0])  # type: ignore[arg-type]

    def test_2d_array_raises(self) -> None:
        space = LatentSpace(dim=4)
        with pytest.raises(ValueError, match="Expected shape \\(4,\\)"):
            space.validate_point(np.ones((2, 4)))


class TestLatentSpaceRepr:
    """Representation."""

    def test_repr(self) -> None:
        space = LatentSpace(dim=32, source_model="vae")
        r = repr(space)
        assert "LatentSpace" in r
        assert "dim=32" in r
        assert "euclidean" in r
        assert "vae" in r

    def test_repr_unit_norm(self) -> None:
        space = LatentSpace(dim=3, geometry="unit_norm")
        r = repr(space)
        assert "unit_norm" in r


class TestLatentSpaceGeometry:
    """Geometry parameter handling."""

    def test_default_is_euclidean(self) -> None:
        space = LatentSpace(dim=8)
        assert space.geometry == "euclidean"

    def test_explicit_euclidean(self) -> None:
        space = LatentSpace(dim=8, geometry="euclidean")
        assert space.geometry == "euclidean"

    def test_unit_norm(self) -> None:
        space = LatentSpace(dim=8, geometry="unit_norm")
        assert space.geometry == "unit_norm"

    def test_invalid_geometry_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown geometry"):
            LatentSpace(dim=8, geometry="spherical")

    def test_invalid_geometry_empty_str(self) -> None:
        with pytest.raises(ValueError, match="Unknown geometry"):
            LatentSpace(dim=8, geometry="")

    def test_geometry_is_instance_level(self) -> None:
        eucl = LatentSpace(dim=4, geometry="euclidean")
        sphere = LatentSpace(dim=4, geometry="unit_norm")
        assert eucl.geometry == "euclidean"
        assert sphere.geometry == "unit_norm"


class TestLatentSpaceValidatePointUnitNorm:
    """Point validation for unit_norm geometry."""

    def test_valid_unit_vector(self) -> None:
        space = LatentSpace(dim=3, geometry="unit_norm")
        point = np.array([1.0, 0.0, 0.0])
        space.validate_point(point)

    def test_valid_unit_vector_another(self) -> None:
        space = LatentSpace(dim=3, geometry="unit_norm")
        # A non-trivial unit vector
        v = np.array([0.6, 0.8, 0.0])
        space.validate_point(v)

    def test_non_unit_vector_raises(self) -> None:
        space = LatentSpace(dim=3, geometry="unit_norm")
        with pytest.raises(ValueError, match="unit_norm requires"):
            space.validate_point(np.array([1.0, 2.0, 3.0]))

    def test_zero_vector_raises(self) -> None:
        space = LatentSpace(dim=3, geometry="unit_norm")
        with pytest.raises(ValueError, match="unit_norm requires"):
            space.validate_point(np.zeros(3))

    def test_shape_check_still_applies(self) -> None:
        space = LatentSpace(dim=4, geometry="unit_norm")
        with pytest.raises(ValueError, match="Expected shape \\(4,\\)"):
            space.validate_point(np.array([1.0, 0.0, 0.0]))

    def test_euclidean_still_allows_any_vector(self) -> None:
        space = LatentSpace(dim=3)
        space.validate_point(np.array([1.0, 2.0, 3.0]))
        space.validate_point(np.zeros(3))
        space.validate_point(np.array([-5.0, 100.0, 0.001]))


class TestLatentSpaceDistance:
    """Distance method."""

    def test_euclidean_distance(self) -> None:
        space = LatentSpace(dim=3)
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([3.0, 4.0, 0.0])
        assert space.distance(a, b) == pytest.approx(5.0)

    def test_euclidean_distance_zero(self) -> None:
        space = LatentSpace(dim=3)
        a = np.array([1.0, 2.0, 3.0])
        assert space.distance(a, a) == pytest.approx(0.0)

    def test_unit_norm_angular_distance(self) -> None:
        space = LatentSpace(dim=3, geometry="unit_norm")
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        # orthogonal vectors should have angular distance pi/2
        assert space.distance(a, b) == pytest.approx(np.pi / 2)

    def test_unit_norm_same_vector(self) -> None:
        space = LatentSpace(dim=3, geometry="unit_norm")
        a = np.array([1.0, 0.0, 0.0])
        assert space.distance(a, a) == pytest.approx(0.0)

    def test_unit_norm_opposite_vector(self) -> None:
        space = LatentSpace(dim=3, geometry="unit_norm")
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([-1.0, 0.0, 0.0])
        assert space.distance(a, b) == pytest.approx(np.pi)

    def test_unit_norm_45_degrees(self) -> None:
        space = LatentSpace(dim=2, geometry="unit_norm")
        a = np.array([1.0, 0.0])
        b = np.array([np.sqrt(2) / 2, np.sqrt(2) / 2])
        assert space.distance(a, b) == pytest.approx(np.pi / 4)

    def test_unit_norm_returns_float(self) -> None:
        space = LatentSpace(dim=3, geometry="unit_norm")
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        result = space.distance(a, b)
        assert isinstance(result, float)


class TestLatentSpaceInterpolate:
    """Interpolate method (lerp / slerp)."""

    def test_euclidean_lerp(self) -> None:
        space = LatentSpace(dim=3)
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([10.0, 20.0, 30.0])
        mid = space.interpolate(a, b, 0.5)
        np.testing.assert_array_almost_equal(mid, [5.0, 10.0, 15.0])

    def test_euclidean_lerp_t0(self) -> None:
        space = LatentSpace(dim=3)
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([7.0, 8.0, 9.0])
        result = space.interpolate(a, b, 0.0)
        np.testing.assert_array_almost_equal(result, a)

    def test_euclidean_lerp_t1(self) -> None:
        space = LatentSpace(dim=3)
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([7.0, 8.0, 9.0])
        result = space.interpolate(a, b, 1.0)
        np.testing.assert_array_almost_equal(result, b)

    def test_unit_norm_slerp_midpoint(self) -> None:
        space = LatentSpace(dim=3, geometry="unit_norm")
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        mid = space.interpolate(a, b, 0.5)
        # Midpoint of two orthogonal unit vectors should be at 45°
        expected = np.array([np.sqrt(2) / 2, np.sqrt(2) / 2, 0.0])
        np.testing.assert_array_almost_equal(mid, expected)
        # Must still be a unit vector
        assert abs(np.linalg.norm(mid) - 1.0) < 1e-10

    def test_unit_norm_slerp_t0(self) -> None:
        space = LatentSpace(dim=3, geometry="unit_norm")
        a = np.array([0.6, 0.8, 0.0])
        b = np.array([0.0, 0.0, 1.0])
        result = space.interpolate(a, b, 0.0)
        np.testing.assert_array_almost_equal(result, a)

    def test_unit_norm_slerp_t1(self) -> None:
        space = LatentSpace(dim=3, geometry="unit_norm")
        a = np.array([0.6, 0.8, 0.0])
        b = np.array([0.0, 0.0, 1.0])
        result = space.interpolate(a, b, 1.0)
        np.testing.assert_array_almost_equal(result, b)

    def test_unit_norm_slerp_returns_unit_vector(self) -> None:
        space = LatentSpace(dim=5, geometry="unit_norm")
        rng = np.random.default_rng(42)
        a = rng.normal(size=5)
        a = a / np.linalg.norm(a)
        b = rng.normal(size=5)
        b = b / np.linalg.norm(b)
        for t in [0.2, 0.5, 0.8]:
            result = space.interpolate(a, b, t)
            assert abs(np.linalg.norm(result) - 1.0) < 1e-10

    def test_unit_norm_slerp_same_vector(self) -> None:
        space = LatentSpace(dim=3, geometry="unit_norm")
        a = np.array([0.6, 0.8, 0.0])
        result = space.interpolate(a, a, 0.5)
        np.testing.assert_array_almost_equal(result, a)

    def test_unit_norm_slerp_opposite_vectors(self) -> None:
        """Slerp with angle ≈ π falls back to lerp (sin(ω) ≈ 0)."""
        space = LatentSpace(dim=3, geometry="unit_norm")
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([-1.0, 0.0, 0.0])
        result = space.interpolate(a, b, 0.5)
        # At ω=π, slerp degenerates; lerp gives (a+b)/2 ≈ 0
        np.testing.assert_array_almost_equal(result, np.zeros(3))

    def test_slerp_stays_on_sphere_where_lerp_departs(self) -> None:
        """Slerp stays on sphere; lerp of same points goes inside."""
        space_unit = LatentSpace(dim=3, geometry="unit_norm")
        space_eucl = LatentSpace(dim=3)
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        for t in [0.25, 0.5, 0.75]:
            slerp_pt = space_unit.interpolate(a, b, t)
            lerp_pt = space_eucl.interpolate(a, b, t)
            # Slerp stays on sphere
            assert abs(np.linalg.norm(slerp_pt) - 1.0) < 1e-10
            # Lerp departs from sphere (except at t=0,1)
            assert np.linalg.norm(lerp_pt) < 0.999


class TestLatentSpaceNormalize:
    """Normalize method."""

    def test_euclidean_returns_copy(self) -> None:
        space = LatentSpace(dim=3)
        point = np.array([1.0, 2.0, 3.0])
        result = space.normalize(point)
        np.testing.assert_array_equal(result, point)
        assert result is not point  # Should be a copy

    def test_unit_norm_normalizes(self) -> None:
        space = LatentSpace(dim=3, geometry="unit_norm")
        point = np.array([3.0, 0.0, 0.0])
        result = space.normalize(point)
        expected = np.array([1.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_unit_norm_arbitrary_vector(self) -> None:
        space = LatentSpace(dim=3, geometry="unit_norm")
        point = np.array([1.0, 2.0, 3.0])
        result = space.normalize(point)
        assert abs(np.linalg.norm(result) - 1.0) < 1e-10
        # Direction should be preserved
        expected_dir = point / np.linalg.norm(point)
        np.testing.assert_array_almost_equal(result, expected_dir)

    def test_unit_norm_zero_vector_raises(self) -> None:
        space = LatentSpace(dim=3, geometry="unit_norm")
        with pytest.raises(ValueError, match="Cannot normalize zero vector"):
            space.normalize(np.zeros(3))

    def test_unit_norm_returns_copy(self) -> None:
        space = LatentSpace(dim=3, geometry="unit_norm")
        point = np.array([1.0, 2.0, 3.0])
        result = space.normalize(point)
        assert result is not point  # Should be a copy


class TestLatentSpaceGaussianSetInit:
    """Construction and basic attributes for gaussian_set geometry."""

    def test_gaussian_set_default_params(self) -> None:
        space = LatentSpace(dim=10, geometry="gaussian_set", n_gaussians=100)
        assert space.geometry == "gaussian_set"
        assert space.n_gaussians == 100
        assert space.dim == 10
        assert space.param_dim == 10
        assert space.shape == (100, 10)

    def test_gaussian_set_custom_params(self) -> None:
        space = LatentSpace(
            dim=8,
            geometry="gaussian_set",
            n_gaussians=50,
            position_dim=2,
            scale_dim=2,
            color_dim=3,
        )
        assert space.n_gaussians == 50
        assert space.dim == 8
        assert space.param_dim == 8  # 2 + 2 + 1 + 3
        assert space.shape == (50, 8)

    def test_gaussian_set_requires_n_gaussians(self) -> None:
        with pytest.raises(ValueError, match="gaussian_set requires n_gaussians >= 1"):
            LatentSpace(dim=10, geometry="gaussian_set")

    def test_gaussian_set_zero_gaussians_raises(self) -> None:
        with pytest.raises(ValueError, match="gaussian_set requires n_gaussians >= 1"):
            LatentSpace(dim=10, geometry="gaussian_set", n_gaussians=0)

    def test_gaussian_set_dim_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="dim=.* does not match computed param_dim"):
            LatentSpace(dim=20, geometry="gaussian_set", n_gaussians=100)

    def test_gaussian_set_metadata_has_param_layout(self) -> None:
        space = LatentSpace(dim=10, geometry="gaussian_set", n_gaussians=100)
        assert "gaussian_set_param_layout" in space.metadata
        layout = space.metadata["gaussian_set_param_layout"]
        assert layout["position"] == (0, 3)
        assert layout["scale"] == (3, 3)
        assert layout["opacity"] == (6, 1)
        assert layout["color"] == (7, 3)

    def test_gaussian_set_with_source_model(self) -> None:
        space = LatentSpace(dim=10, geometry="gaussian_set", n_gaussians=100, source_model="3dgs")
        assert space.source_model == "3dgs"

    def test_gaussian_set_repr(self) -> None:
        space = LatentSpace(dim=10, geometry="gaussian_set", n_gaussians=100, source_model="3dgs")
        r = repr(space)
        assert "gaussian_set" in r
        assert "n_gaussians=100" in r
        assert "3dgs" in r


class TestLatentSpaceGaussianSetValidatePoint:
    """Point validation for gaussian_set geometry."""

    @staticmethod
    def _make_valid_point(n_gaussians: int = 10) -> np.ndarray:
        rng = np.random.default_rng(42)
        point = np.zeros((n_gaussians, 10))
        point[:, :3] = rng.normal(size=(n_gaussians, 3))
        point[:, 3:6] = np.exp(rng.normal(size=(n_gaussians, 3)))
        point[:, 6] = rng.uniform(0.0, 1.0, size=n_gaussians)
        point[:, 7:10] = rng.uniform(0.0, 1.0, size=(n_gaussians, 3))
        return point

    def test_valid_gaussian_set_point(self) -> None:
        space = LatentSpace(dim=10, geometry="gaussian_set", n_gaussians=10)
        point = self._make_valid_point()
        space.validate_point(point)

    def test_wrong_n_gaussians_raises(self) -> None:
        space = LatentSpace(dim=10, geometry="gaussian_set", n_gaussians=10)
        point = self._make_valid_point(n_gaussians=5)
        with pytest.raises(ValueError, match="Expected shape"):
            space.validate_point(point)

    def test_wrong_param_dim_raises(self) -> None:
        space = LatentSpace(dim=10, geometry="gaussian_set", n_gaussians=10)
        point = np.ones((10, 8))
        with pytest.raises(ValueError, match="Expected shape"):
            space.validate_point(point)

    def test_negative_scale_raises(self) -> None:
        space = LatentSpace(dim=10, geometry="gaussian_set", n_gaussians=10)
        point = self._make_valid_point()
        point[0, 3] = -1.0
        with pytest.raises(ValueError, match="scale components > 0"):
            space.validate_point(point)

    def test_zero_scale_raises(self) -> None:
        space = LatentSpace(dim=10, geometry="gaussian_set", n_gaussians=10)
        point = self._make_valid_point()
        point[0, 3] = 0.0
        with pytest.raises(ValueError, match="scale components > 0"):
            space.validate_point(point)

    def test_opacity_below_zero_raises(self) -> None:
        space = LatentSpace(dim=10, geometry="gaussian_set", n_gaussians=10)
        point = self._make_valid_point()
        point[0, 6] = -0.1
        with pytest.raises(ValueError, match="opacity in \\[0, 1\\]"):
            space.validate_point(point)

    def test_opacity_above_one_raises(self) -> None:
        space = LatentSpace(dim=10, geometry="gaussian_set", n_gaussians=10)
        point = self._make_valid_point()
        point[0, 6] = 1.1
        with pytest.raises(ValueError, match="opacity in \\[0, 1\\]"):
            space.validate_point(point)

    def test_color_below_zero_raises(self) -> None:
        space = LatentSpace(dim=10, geometry="gaussian_set", n_gaussians=10)
        point = self._make_valid_point()
        point[0, 7] = -0.1
        with pytest.raises(ValueError, match="color channels in \\[0, 1\\]"):
            space.validate_point(point)

    def test_color_above_one_raises(self) -> None:
        space = LatentSpace(dim=10, geometry="gaussian_set", n_gaussians=10)
        point = self._make_valid_point()
        point[0, 7] = 1.1
        with pytest.raises(ValueError, match="color channels in \\[0, 1\\]"):
            space.validate_point(point)


class TestLatentSpaceGaussianSetDistance:
    """Permutation-aware distance for gaussian_set geometry."""

    @staticmethod
    def _make_point(n_gaussians: int = 10, seed: int = 42) -> np.ndarray:
        rng = np.random.default_rng(seed)
        point = np.zeros((n_gaussians, 10))
        point[:, :3] = rng.normal(size=(n_gaussians, 3))
        point[:, 3:6] = np.exp(rng.normal(size=(n_gaussians, 3)))
        point[:, 6] = rng.uniform(0.0, 1.0, size=n_gaussians)
        point[:, 7:10] = rng.uniform(0.0, 1.0, size=(n_gaussians, 3))
        return point

    def test_self_distance_zero(self) -> None:
        space = LatentSpace(dim=10, geometry="gaussian_set", n_gaussians=10)
        point = self._make_point()
        assert space.distance(point, point) == pytest.approx(0.0)

    def test_distance_returns_float(self) -> None:
        space = LatentSpace(dim=10, geometry="gaussian_set", n_gaussians=10)
        a = self._make_point(seed=42)
        b = self._make_point(seed=99)
        result = space.distance(a, b)
        assert isinstance(result, float)
        assert result > 0

    def test_distance_permutation_invariant(self) -> None:
        """Distance should be invariant to Gaussian permutation."""
        space = LatentSpace(dim=10, geometry="gaussian_set", n_gaussians=10)
        a = self._make_point(seed=42)
        b = self._make_point(seed=99)
        rng = np.random.default_rng(123)
        perm = rng.permutation(10)
        a_shuffled = a[perm].copy()
        d1 = space.distance(a, b)
        d2 = space.distance(a_shuffled, b)
        assert d1 == pytest.approx(d2)

    def test_distance_larger_diff_larger_value(self) -> None:
        space = LatentSpace(dim=10, geometry="gaussian_set", n_gaussians=10)
        a = self._make_point(seed=42)
        rng = np.random.default_rng(1)
        b_small = a + 0.01 * rng.normal(size=(10, 10))
        b_big = a + 10.0 * rng.normal(size=(10, 10))
        assert space.distance(a, b_small) < space.distance(a, b_big)


class TestLatentSpaceGaussianSetInterpolate:
    """Interpolation for gaussian_set geometry."""

    @staticmethod
    def _make_point(n_gaussians: int = 10, seed: int = 42) -> np.ndarray:
        rng = np.random.default_rng(seed)
        point = np.zeros((n_gaussians, 10))
        point[:, :3] = rng.normal(size=(n_gaussians, 3))
        point[:, 3:6] = np.exp(rng.normal(size=(n_gaussians, 3)))
        point[:, 6] = rng.uniform(0.0, 1.0, size=n_gaussians)
        point[:, 7:10] = rng.uniform(0.0, 1.0, size=(n_gaussians, 3))
        return point

    def test_interpolate_t0_returns_a(self) -> None:
        space = LatentSpace(dim=10, geometry="gaussian_set", n_gaussians=10)
        a = self._make_point(seed=42)
        b = self._make_point(seed=99)
        result = space.interpolate(a, b, 0.0)
        # Interpolation sorts by position for permutation invariance,
        # so compare against sorted(a) not a itself
        a_idx = space._gaussian_set_sort_indices(a)
        np.testing.assert_array_almost_equal(result, a[a_idx])

    def test_interpolate_t1_returns_b(self) -> None:
        space = LatentSpace(dim=10, geometry="gaussian_set", n_gaussians=10)
        a = self._make_point(seed=42)
        b = self._make_point(seed=99)
        result = space.interpolate(a, b, 1.0)
        b_idx = space._gaussian_set_sort_indices(b)
        np.testing.assert_array_almost_equal(result, b[b_idx])

    def test_interpolate_t05_midpoint_shape(self) -> None:
        space = LatentSpace(dim=10, geometry="gaussian_set", n_gaussians=10)
        a = self._make_point(seed=42)
        b = self._make_point(seed=99)
        result = space.interpolate(a, b, 0.5)
        assert result.shape == (10, 10)

    def test_interpolate_returns_valid_point(self) -> None:
        """Interpolated result should pass validate_point."""
        space = LatentSpace(dim=10, geometry="gaussian_set", n_gaussians=10)
        a = self._make_point(seed=42)
        b = self._make_point(seed=99)
        for t in [0.25, 0.5, 0.75]:
            result = space.interpolate(a, b, t)
            space.validate_point(result)

    def test_interpolate_permutation_invariant(self) -> None:
        """Interpolation result should be invariant to input ordering."""
        space = LatentSpace(dim=10, geometry="gaussian_set", n_gaussians=10)
        a = self._make_point(seed=42)
        b = self._make_point(seed=99)
        rng = np.random.default_rng(123)
        perm = rng.permutation(10)
        a_shuffled = a[perm].copy()
        b_shuffled = b[perm].copy()
        result_orig = space.interpolate(a, b, 0.5)
        result_shuffled = space.interpolate(a_shuffled, b_shuffled, 0.5)
        np.testing.assert_array_almost_equal(result_orig, result_shuffled)

    def test_interpolate_scale_positivity(self) -> None:
        """Scale should remain > 0 after interpolation."""
        space = LatentSpace(dim=10, geometry="gaussian_set", n_gaussians=10)
        a = self._make_point(seed=42)
        b = self._make_point(seed=99)
        for t in [0.1, 0.5, 0.9]:
            result = space.interpolate(a, b, t)
            assert np.all(result[:, 3:6] > 0)

    def test_interpolate_opacity_clamped(self) -> None:
        """Opacity should stay in [0, 1] after interpolation."""
        space = LatentSpace(dim=10, geometry="gaussian_set", n_gaussians=10)
        a = self._make_point(seed=42)
        b = self._make_point(seed=99)
        for t in [0.0, 0.3, 0.7, 1.0]:
            result = space.interpolate(a, b, t)
            assert np.all(result[:, 6] >= 0) and np.all(result[:, 6] <= 1)

    def test_interpolate_color_clamped(self) -> None:
        """Color should stay in [0, 1] after interpolation."""
        space = LatentSpace(dim=10, geometry="gaussian_set", n_gaussians=10)
        a = self._make_point(seed=42)
        b = self._make_point(seed=99)
        for t in [0.0, 0.3, 0.7, 1.0]:
            result = space.interpolate(a, b, t)
            assert np.all(result[:, 7:10] >= 0) and np.all(result[:, 7:10] <= 1)


class TestLatentSpaceGaussianSetNormalize:
    """Normalize for gaussian_set — returns copy unchanged."""

    def test_gaussian_set_returns_copy(self) -> None:
        space = LatentSpace(dim=10, geometry="gaussian_set", n_gaussians=10)
        point = np.ones((10, 10))
        result = space.normalize(point)
        np.testing.assert_array_equal(result, point)
        assert result is not point


class TestLatentSpaceGaussianSetBackwardCompat:
    """Flat-vector geometries must continue to work unchanged."""

    def test_euclidean_still_works(self) -> None:
        space = LatentSpace(dim=8)
        assert space.shape == (8,)
        assert space.n_gaussians is None

    def test_unit_norm_still_works(self) -> None:
        space = LatentSpace(dim=3, geometry="unit_norm")
        assert space.shape == (3,)
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        assert space.distance(a, b) == pytest.approx(np.pi / 2)

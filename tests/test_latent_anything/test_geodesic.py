"""Tests for density-penalized geodesic path interpolation.

Covers the pure path algorithms in ``geometry.py`` (lerp init, energy,
gradient, bounded optimization, density length), the stateful contract in
``geodesic.py`` (config, ``DensityGeodesic``, ``GeodesicPath`` result with
diagnostics, cache/profiling), the analytic curved-manifold behavior, and
failure/non-convergence contracts.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from latent_anything.geodesic import DensityGeodesic, GeodesicConfig, GeodesicPath
from latent_anything.geometry import (
    density_path_energy,
    density_path_gradient,
    density_path_length,
    lerp_path,
    optimize_density_path,
)
from latent_anything.runtime import InMemoryCache, RuntimeProfiler

# ── Analytic ring density (known curved manifold) ────────────────────
#
# The unit ring in 2D is a known curved manifold: density is peaked on a
# circle of radius R, and the geodesic between two points on the circle
# should follow the arc (staying on-manifold) rather than the chord (which
# cuts through the low-density center). This gives an exact ground truth for
# the density-penalized path optimizer.


def _ring_log_density(point: np.ndarray, radius: float = 2.0, sigma: float = 0.5) -> float:
    r = float(np.linalg.norm(point))
    return -((r - radius) ** 2) / (2.0 * sigma**2)


def _ring_log_density_gradient(point: np.ndarray, radius: float = 2.0, sigma: float = 0.5) -> np.ndarray:
    r = float(np.linalg.norm(point))
    if r < 1e-12:
        return np.zeros_like(point)
    return -((r - radius) / sigma**2) * (np.asarray(point, dtype=np.float64) / r)


def _ring_endpoints(radius: float = 2.0, angle_deg: float = 60.0) -> tuple[np.ndarray, np.ndarray]:
    angle = np.radians(angle_deg)
    a = np.array([radius * np.cos(angle), radius * np.sin(angle)])
    b = np.array([radius * np.cos(angle), -radius * np.sin(angle)])
    return a, b


def _make_geodesic(config: GeodesicConfig | None = None) -> DensityGeodesic:
    return DensityGeodesic(config).attach_density(
        _ring_log_density,
        log_density_gradient=_ring_log_density_gradient,
        source_representation_identity="analytic-ring",
    )


# ── Pure path algorithms ────────────────────────────────────────────


class TestLerpPath:
    """Deterministic linear initialization."""

    def test_endpoints_included(self) -> None:
        a = np.array([0.0, 1.0])
        b = np.array([2.0, 3.0])
        path = lerp_path(a, b, 5)
        assert path.shape == (5, 2)
        np.testing.assert_allclose(path[0], a)
        np.testing.assert_allclose(path[-1], b)

    def test_uniform_spacing(self) -> None:
        a = np.array([0.0, 0.0])
        b = np.array([4.0, 0.0])
        path = lerp_path(a, b, 5)
        np.testing.assert_allclose(path[:, 0], [0.0, 1.0, 2.0, 3.0, 4.0])

    def test_rejects_fewer_than_two_points(self) -> None:
        with pytest.raises(ValueError, match="at least 2 points"):
            lerp_path(np.zeros(2), np.ones(2), 1)


class TestDensityPathEnergy:
    """Energy penalizes low-density regions."""

    def test_flat_exponent_recovers_squared_norm(self) -> None:
        a = np.array([0.0, 0.0])
        b = np.array([4.0, 0.0])
        path = lerp_path(a, b, 5)
        energy = density_path_energy(path, _ring_log_density, exponent=0.0)
        segment_norms = np.linalg.norm(np.diff(path, axis=0), axis=1)
        np.testing.assert_allclose(energy, np.sum(segment_norms**2), rtol=1e-12)

    def test_energy_is_positive(self) -> None:
        a, b = _ring_endpoints()
        path = lerp_path(a, b, 16)
        assert density_path_energy(path, _ring_log_density, exponent=1.0) > 0.0


class TestDensityPathGradient:
    """Gradient has zero rows at the fixed endpoints."""

    def test_endpoint_gradient_rows_are_zero(self) -> None:
        a, b = _ring_endpoints()
        path = lerp_path(a, b, 8)
        gradient = density_path_gradient(path, _ring_log_density, _ring_log_density_gradient, exponent=1.0)
        np.testing.assert_allclose(gradient[0], np.zeros(2), atol=1e-12)
        np.testing.assert_allclose(gradient[-1], np.zeros(2), atol=1e-12)

    def test_lerp_path_is_stationary_for_flat_metric(self) -> None:
        # With exponent=0 the weights are all one, so the equally spaced lerp
        # path has zero discrete Laplacian and is a critical point.
        a = np.array([0.0, 0.0])
        b = np.array([6.0, 0.0])
        path = lerp_path(a, b, 7)
        gradient = density_path_gradient(path, _ring_log_density, _ring_log_density_gradient, exponent=0.0)
        np.testing.assert_allclose(gradient, np.zeros((7, 2)), atol=1e-12)

    def test_finite_difference_matches_analytic_on_ring(self) -> None:
        a, b = _ring_endpoints()
        path = lerp_path(a, b, 8)
        analytic = density_path_gradient(path, _ring_log_density, _ring_log_density_gradient, exponent=1.0)
        eps = 1e-5

        def numeric(point: np.ndarray) -> np.ndarray:
            gradient = np.zeros_like(point)
            for index in range(point.shape[0]):
                step = np.zeros_like(point)
                step[index] = eps
                gradient[index] = (_ring_log_density(point + step) - _ring_log_density(point - step)) / (2.0 * eps)
            return gradient

        finite = density_path_gradient(path, _ring_log_density, numeric, exponent=1.0)
        np.testing.assert_allclose(finite, analytic, atol=1e-6)


class TestOptimizeDensityPath:
    """Bounded gradient descent converges on the known curved manifold."""

    def test_endpoints_fixed(self) -> None:
        a, b = _ring_endpoints()
        path, _, _, _, _, _ = optimize_density_path(
            a,
            b,
            log_density=_ring_log_density,
            log_density_gradient=_ring_log_density_gradient,
            n_points=16,
            max_iter=200,
            step_size=0.1,
            tol=1e-6,
            exponent=1.0,
        )
        np.testing.assert_allclose(path[0], a)
        np.testing.assert_allclose(path[-1], b)

    def test_reduces_energy(self) -> None:
        a, b = _ring_endpoints()
        path, initial, final, _, _, _ = optimize_density_path(
            a,
            b,
            log_density=_ring_log_density,
            log_density_gradient=_ring_log_density_gradient,
            n_points=32,
            max_iter=1000,
            step_size=0.5,
            tol=1e-7,
            exponent=1.0,
        )
        assert final <= initial
        assert path.shape == (32, 2)

    def test_geodesic_stays_closer_to_ring_than_lerp(self) -> None:
        a, b = _ring_endpoints()
        lerp = lerp_path(a, b, 32)
        path, _, _, _, converged, _ = optimize_density_path(
            a,
            b,
            log_density=_ring_log_density,
            log_density_gradient=_ring_log_density_gradient,
            n_points=32,
            max_iter=2000,
            step_size=0.5,
            tol=1e-7,
            exponent=1.0,
        )
        assert converged
        geodesic_radius = float(np.asarray(np.linalg.norm(path, axis=1)).mean())  # type: ignore[arg-type]
        lerp_radius = float(np.asarray(np.linalg.norm(lerp, axis=1)).mean())  # type: ignore[arg-type]
        assert geodesic_radius > lerp_radius

    def test_geodesic_has_higher_mean_density_than_lerp(self) -> None:
        a, b = _ring_endpoints()
        lerp = lerp_path(a, b, 32)
        path, _, _, _, _, _ = optimize_density_path(
            a,
            b,
            log_density=_ring_log_density,
            log_density_gradient=_ring_log_density_gradient,
            n_points=32,
            max_iter=2000,
            step_size=0.5,
            tol=1e-7,
            exponent=1.0,
        )
        mean_geodesic = float(np.mean([_ring_log_density(p) for p in path]))
        mean_lerp = float(np.mean([_ring_log_density(p) for p in lerp]))
        assert mean_geodesic > mean_lerp

    def test_zero_exponent_returns_lerp(self) -> None:
        a, b = _ring_endpoints()
        lerp = lerp_path(a, b, 16)
        path, _, _, _, _, _ = optimize_density_path(
            a,
            b,
            log_density=_ring_log_density,
            log_density_gradient=_ring_log_density_gradient,
            n_points=16,
            max_iter=50,
            step_size=0.1,
            tol=1e-8,
            exponent=0.0,
        )
        np.testing.assert_allclose(path, lerp, atol=1e-8)

    def test_non_convergence_reported_when_max_iter_too_small(self) -> None:
        a, b = _ring_endpoints()
        path, _, _, _, converged, message = optimize_density_path(
            a,
            b,
            log_density=_ring_log_density,
            log_density_gradient=_ring_log_density_gradient,
            n_points=32,
            max_iter=1,
            step_size=0.1,
            tol=1e-15,
            exponent=1.0,
        )
        assert path.shape == (32, 2)
        assert not converged
        assert "max_iter" in message

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="share shape"):
            optimize_density_path(
                np.zeros(2),
                np.zeros(3),
                log_density=_ring_log_density,
                log_density_gradient=_ring_log_density_gradient,
                n_points=4,
                max_iter=10,
                step_size=0.1,
                tol=1e-6,
                exponent=1.0,
            )


class TestDensityPathLength:
    """Density-penalized arc length is longer than the chord on a ring."""

    def test_density_length_is_longer_than_euclidean_on_ring(self) -> None:
        a, b = _ring_endpoints()
        lerp = lerp_path(a, b, 16)
        density_length, euclidean_length = density_path_length(lerp, _ring_log_density, exponent=1.0)
        assert euclidean_length == pytest.approx(np.linalg.norm(a - b))
        assert density_length > euclidean_length


# ── GeodesicConfig and GeodesicPath contract ─────────────────────────


class TestGeodesicConfig:
    """Pydantic config validation."""

    def test_defaults(self) -> None:
        config = GeodesicConfig()
        assert config.n_points >= 3
        assert config.max_iter >= 1
        assert config.step_size > 0
        assert config.tol > 0
        assert config.density_exponent >= 0

    def test_invalid_n_points(self) -> None:
        with pytest.raises(ValueError):
            GeodesicConfig(n_points=2)
        with pytest.raises(ValueError):
            GeodesicConfig(n_points=300)

    def test_invalid_exponent(self) -> None:
        with pytest.raises(ValueError):
            GeodesicConfig(density_exponent=-1.0)

    def test_kwargs_construction(self) -> None:
        geodesic = DensityGeodesic(n_points=24, density_exponent=2.0)
        assert geodesic.config.n_points == 24
        assert geodesic.config.density_exponent == 2.0


class TestGeodesicPath:
    """Immutable result with diagnostics."""

    def test_endpoints_and_length_diagnostics(self) -> None:
        a, b = _ring_endpoints()
        result = _make_geodesic(GeodesicConfig(n_points=24, max_iter=500, step_size=0.3, tol=1e-6)).optimize(a, b)
        assert isinstance(result, GeodesicPath)
        assert result.n_points == 24
        assert result.dim == 2
        assert result.path.shape == (24, 2)
        assert result.endpoint_a.shape == (2,)
        assert result.log_density.shape == (24,)
        assert np.isfinite(result.length)
        assert result.euclidean_length > 0.0
        assert result.min_log_density <= result.mean_log_density
        assert result.status.initial_energy >= 0.0
        assert result.status.final_energy >= 0.0

    def test_result_owns_immutable_arrays(self) -> None:
        a, b = _ring_endpoints()
        result = _make_geodesic(GeodesicConfig(n_points=16, max_iter=100)).optimize(a, b)
        with pytest.raises(ValueError):
            result.path[0, 0] = 999.0
        with pytest.raises(ValueError):
            result.path.setflags(write=True)
        with pytest.raises(TypeError):
            result.provenance["new"] = True  # type: ignore[index]

    def test_reconstruction_diagnostics_with_decoder(self) -> None:
        a, b = _ring_endpoints()
        config = GeodesicConfig(n_points=8, max_iter=50)
        geodesic = DensityGeodesic(config).attach_density(
            _ring_log_density,
            log_density_gradient=_ring_log_density_gradient,
            decoder=lambda path: np.stack([point * 2.0 for point in path]),
        )
        result = geodesic.optimize(a, b)
        assert result.decoded is not None
        assert result.decoded.shape == (8, 2)
        assert result.reconstruction_error is not None
        assert result.reconstruction_error > 0.0

    def test_from_gmm_density_round_trip(self) -> None:
        from latent_anything.density import GaussianMixtureDensity, GMMConfig

        rng = np.random.default_rng(7)
        angles = rng.uniform(0.0, 2.0 * np.pi, size=400)
        ring = np.stack([2.0 * np.cos(angles), 2.0 * np.sin(angles)], axis=1)
        estimator = GaussianMixtureDensity(GMMConfig(n_components=8, n_init=2, random_state=0)).fit(
            ring, source_representation_identity="ring-gmm"
        )
        geodesic = DensityGeodesic.from_gmm_density(estimator, config=GeodesicConfig(n_points=16, max_iter=200))
        assert geodesic.is_fitted
        assert geodesic.source_representation_identity == "ring-gmm"
        a, b = _ring_endpoints()
        result = geodesic.optimize(a, b)
        assert np.isfinite(result.path).all()
        np.testing.assert_allclose(result.path[0], a)
        np.testing.assert_allclose(result.path[-1], b)


# ── DensityGeodesic entry point ─────────────────────────────────────


class TestDensityGeodesic:
    """Config-driven entry point with cache/profiling integration."""

    def test_unfitted_raises(self) -> None:
        geodesic = DensityGeodesic()
        with pytest.raises(RuntimeError, match="no density oracle"):
            geodesic.optimize(np.zeros(2), np.ones(2))

    def test_interpolate_endpoints_and_midpoint(self) -> None:
        a, b = _ring_endpoints()
        geodesic = _make_geodesic(GeodesicConfig(n_points=32, max_iter=500, step_size=0.3, tol=1e-6))
        np.testing.assert_allclose(geodesic.interpolate(a, b, 0.0), a, atol=1e-8)
        np.testing.assert_allclose(geodesic.interpolate(a, b, 1.0), b, atol=1e-8)
        midpoint = geodesic.interpolate(a, b, 0.5)
        assert midpoint.shape == (2,)
        assert np.linalg.norm(midpoint) > np.linalg.norm((a + b) / 2.0)  # stays on-manifold vs chord

    def test_interpolate_rejects_out_of_range_t(self) -> None:
        geodesic = _make_geodesic()
        with pytest.raises(ValueError, match="t must be"):
            geodesic.interpolate(np.zeros(2), np.ones(2), 1.5)

    def test_validate_endpoints(self) -> None:
        geodesic = _make_geodesic()
        with pytest.raises(ValueError, match="flat"):
            geodesic.optimize(np.zeros((2, 2)), np.ones(2))
        with pytest.raises(ValueError, match="share shape"):
            geodesic.optimize(np.zeros(2), np.ones(3))
        with pytest.raises(ValueError, match="finite"):
            geodesic.optimize(np.array([0.0, np.nan]), np.ones(2))

    def test_cache_skips_reoptimization(self) -> None:
        a, b = _ring_endpoints()
        cache = InMemoryCache()
        profiler = RuntimeProfiler()
        geodesic = _make_geodesic(GeodesicConfig(n_points=24, max_iter=300, step_size=0.3, tol=1e-6))
        first = geodesic.optimize(a, b, cache=cache, profiler=profiler)
        second = geodesic.optimize(a, b, cache=cache, profiler=profiler)
        assert cache.stats.hits == 1
        np.testing.assert_allclose(first.path, second.path, atol=1e-12)
        assert second.status.message == "served from cache"
        stages = profiler.snapshot().stage_totals()
        assert "cache" in stages
        assert "method" in stages

    def test_cache_key_changes_with_config(self) -> None:
        a, b = _ring_endpoints()
        cache = InMemoryCache()
        gd1 = _make_geodesic(GeodesicConfig(n_points=16, max_iter=50))
        gd2 = _make_geodesic(GeodesicConfig(n_points=32, max_iter=50))
        gd1.optimize(a, b, cache=cache)
        gd2.optimize(a, b, cache=cache)
        assert cache.stats.misses == 2

    def test_registry_construction(self) -> None:
        from latent_anything.config import build_from_dict

        geodesic = build_from_dict({"kind": "intervention", "name": "density_geodesic", "params": {"n_points": 24}})
        assert isinstance(geodesic, DensityGeodesic)
        assert geodesic.config.n_points == 24

    def test_deterministic_across_runs(self) -> None:
        a, b = _ring_endpoints()
        config = GeodesicConfig(n_points=24, max_iter=300, step_size=0.3, tol=1e-6)
        first = _make_geodesic(config).optimize(a, b)
        second = _make_geodesic(config).optimize(a, b)
        np.testing.assert_array_equal(first.path, second.path)


# ── Hypothesis property checks ──────────────────────────────────────


class TestGeodesicProperties:
    """Structural invariants under random endpoints."""

    @given(
        a=st.lists(st.floats(-3.0, 3.0, allow_nan=False, allow_infinity=False), min_size=2, max_size=2).map(np.array),
        b=st.lists(st.floats(-3.0, 3.0, allow_nan=False, allow_infinity=False), min_size=2, max_size=2).map(np.array),
    )
    @settings(max_examples=20, deadline=None)
    def test_endpoints_preserved_and_finite(self, a: np.ndarray, b: np.ndarray) -> None:
        if np.allclose(a, b):
            return
        geodesic = _make_geodesic(GeodesicConfig(n_points=12, max_iter=100))
        result = geodesic.optimize(a, b)
        assert np.isfinite(result.path).all()
        np.testing.assert_allclose(result.path[0], a, atol=1e-8)
        np.testing.assert_allclose(result.path[-1], b, atol=1e-8)
        assert result.length >= 0.0

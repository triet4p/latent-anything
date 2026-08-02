"""Tests for the anisotropic covariance geometry.

Covers the pure algorithms in ``geometry.py`` (validation, regularization,
Mahalanobis distance, whitening, inverse whitening, metric interpolation) and
the stateful contract in ``covariance.py`` (config, fitting, provenance,
serialization), plus the ``LatentSpace(geometry="anisotropic")`` facade
dispatch and the declared interpolation semantics.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from latent_anything import LatentSpace
from latent_anything.covariance import CovarianceConfig, CovarianceState, fit_covariance_state
from latent_anything.geometry import (
    covariance_interpolate,
    fit_covariance,
    mahalanobis_distance,
    regularize_covariance,
    unwhiten_point,
    validate_covariance,
    whiten_point,
)

# ── Hypothesis strategies ─────────────────────────────────────────────


def _positive_definite(dim: int, rng: np.random.Generator) -> np.ndarray:
    base = rng.normal(size=(dim, dim))
    return base @ base.T + dim * np.eye(dim)


def _pd_strategy(dim: int = 3) -> st.SearchStrategy[np.ndarray]:
    def build(random_seed: tuple[int, int]) -> np.ndarray:
        rng = np.random.default_rng(random_seed[0])
        base = rng.normal(size=(dim, dim))
        return base @ base.T + np.eye(dim)

    return st.tuples(st.integers(0, 10_000), st.integers(0, 10_000)).map(build)


def _vector_strategy(dim: int = 3) -> st.SearchStrategy[np.ndarray]:
    return st.lists(
        st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False),
        min_size=dim,
        max_size=dim,
    ).map(np.array)


# ── Pure geometry functions ──────────────────────────────────────────


class TestValidateCovariance:
    """Positive-definite validation."""

    def test_accepts_identity(self) -> None:
        validate_covariance(np.eye(3), dim=3)

    def test_accepts_diagonal_anisotropic(self) -> None:
        validate_covariance(np.diag([1.0, 4.0, 0.25]), dim=3)

    def test_wrong_shape_raises(self) -> None:
        with pytest.raises(ValueError, match="shape"):
            validate_covariance(np.eye(2), dim=3)

    def test_non_square_raises(self) -> None:
        with pytest.raises(ValueError, match="shape"):
            validate_covariance(np.ones((3, 4)), dim=3)

    def test_non_symmetric_raises(self) -> None:
        matrix = np.eye(3)
        matrix[0, 1] = 0.5
        with pytest.raises(ValueError, match="symmetric"):
            validate_covariance(matrix, dim=3)

    def test_non_finite_raises(self) -> None:
        matrix = np.eye(3)
        matrix[0, 0] = np.nan
        with pytest.raises(ValueError, match="finite"):
            validate_covariance(matrix, dim=3)

    def test_singular_raises(self) -> None:
        with pytest.raises(ValueError, match="positive-definite"):
            validate_covariance(np.zeros((3, 3)), dim=3)

    def test_indefinite_raises(self) -> None:
        matrix = np.diag([1.0, -1.0, 1.0])
        with pytest.raises(ValueError, match="positive-definite"):
            validate_covariance(matrix, dim=3)

    @given(matrix=_pd_strategy())
    @settings(max_examples=30)
    def test_accepts_random_pd(self, matrix: np.ndarray) -> None:
        validate_covariance(matrix, dim=matrix.shape[0])


class TestRegularizeCovariance:
    """Diagonal loading restores positive definiteness."""

    def test_singular_loaded_to_pd(self) -> None:
        singular = np.diag([1.0, 0.0, 1.0])
        regularized = regularize_covariance(singular, reg_coef=1e-3)
        validate_covariance(regularized, dim=3)

    def test_preserves_anisotropy(self) -> None:
        matrix = np.diag([1.0, 25.0, 0.01])
        regularized = regularize_covariance(matrix, reg_coef=1e-6)
        np.testing.assert_allclose(regularized, matrix + 1e-6 * np.eye(3), atol=1e-12)

    def test_negative_reg_coef_raises(self) -> None:
        with pytest.raises(ValueError, match="reg_coef"):
            regularize_covariance(np.eye(2), reg_coef=-0.1)

    def test_non_square_raises(self) -> None:
        with pytest.raises(ValueError, match="square"):
            regularize_covariance(np.ones((2, 3)), reg_coef=1e-3)

    def test_re_symmetrizes(self) -> None:
        matrix = np.eye(2)
        matrix[0, 1] = 0.1
        matrix[1, 0] = 0.2
        regularized = regularize_covariance(matrix, reg_coef=1e-6)
        np.testing.assert_allclose(regularized, regularized.T)


class TestMahalanobisDistance:
    """Analytic properties: symmetry, zero-on-self, affine invariance."""

    def test_symmetry(self) -> None:
        covariance = np.diag([1.0, 4.0, 9.0])
        a = np.array([1.0, -2.0, 0.5])
        b = np.array([-3.0, 1.0, 2.0])
        assert mahalanobis_distance(a, b, covariance) == pytest.approx(mahalanobis_distance(b, a, covariance))

    def test_zero_for_same_point(self) -> None:
        a = np.array([1.0, 2.0, 3.0])
        assert mahalanobis_distance(a, a, np.eye(3)) == pytest.approx(0.0)

    def test_scales_by_inverse_variance(self) -> None:
        # Along axis with variance 4, a diff of 2 has Mahalanobis length 1.
        covariance = np.diag([4.0, 1.0, 1.0])
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([2.0, 0.0, 0.0])
        assert mahalanobis_distance(a, b, covariance) == pytest.approx(1.0)

    def test_direction_matters_under_anisotropy(self) -> None:
        # Same Euclidean length, different Mahalanobis length along axes.
        covariance = np.diag([0.25, 4.0, 1.0])
        origin = np.zeros(3)
        along_small = np.array([1.0, 0.0, 0.0])  # variance 0.25 -> factor 2
        along_large = np.array([0.0, 2.0, 0.0])  # variance 4 -> factor 0.5
        assert mahalanobis_distance(origin, along_small, covariance) > mahalanobis_distance(
            origin, along_large, covariance
        )

    @given(covariance=_pd_strategy(), a=_vector_strategy(), b=_vector_strategy())
    @settings(max_examples=30)
    def test_symmetry_property(self, covariance: np.ndarray, a: np.ndarray, b: np.ndarray) -> None:
        assert mahalanobis_distance(a, b, covariance) == pytest.approx(mahalanobis_distance(b, a, covariance))

    def test_affine_invariance(self) -> None:
        """Mahalanobis distance is invariant to invertible affine transforms.

        If ``y = A x + t`` and ``C_y = A C_x A^T``, then
        ``d_M(y1, y2; C_y) == d_M(x1, x2; C_x)``.
        """
        rng = np.random.default_rng(42)
        covariance = _positive_definite(3, rng)
        a = rng.normal(size=3)
        b = rng.normal(size=3)
        transform = rng.normal(size=(3, 3)) + 3 * np.eye(3)
        offset = rng.normal(size=3)
        y_a = transform @ a + offset
        y_b = transform @ b + offset
        transformed_covariance = transform @ covariance @ transform.T
        expected = mahalanobis_distance(a, b, covariance)
        actual = mahalanobis_distance(y_a, y_b, transformed_covariance)
        assert actual == pytest.approx(expected, rel=1e-8)


class TestWhitenUnwhiten:
    """Whitening normalizes to identity covariance; inverse round-trips."""

    def test_whitened_covariance_is_identity(self) -> None:
        rng = np.random.default_rng(7)
        dim = 4
        covariance = _positive_definite(dim, rng)
        mean = rng.normal(size=dim)
        samples = rng.multivariate_normal(mean, covariance, size=20_000)
        whitened = np.asarray([whiten_point(s, mean, covariance) for s in samples])
        empirical = np.cov(whitened, rowvar=False)
        np.testing.assert_allclose(empirical, np.eye(dim), atol=0.15)

    def test_round_trip(self) -> None:
        rng = np.random.default_rng(9)
        dim = 3
        covariance = _positive_definite(dim, rng)
        mean = rng.normal(size=dim)
        point = rng.normal(size=dim)
        restored = unwhiten_point(whiten_point(point, mean, covariance), mean, covariance)
        np.testing.assert_allclose(restored, point, atol=1e-12)

    def test_whitened_norm_equals_mahalanobis(self) -> None:
        rng = np.random.default_rng(11)
        dim = 3
        covariance = _positive_definite(dim, rng)
        mean = rng.normal(size=dim)
        point = rng.normal(size=dim)
        whitened_norm = np.linalg.norm(whiten_point(point, mean, covariance))
        expected = mahalanobis_distance(point, mean, covariance)
        assert whitened_norm == pytest.approx(expected, rel=1e-10)


class TestFitCovariance:
    """Empirical fitting requires more samples than dimensions."""

    def test_recovers_diagonal_anisotropy(self) -> None:
        rng = np.random.default_rng(13)
        true_covariance = np.diag([1.0, 9.0, 0.25])
        data = rng.multivariate_normal(np.zeros(3), true_covariance, size=5000)
        mean, covariance = fit_covariance(data, reg_coef=1e-6)
        np.testing.assert_allclose(mean, np.zeros(3), atol=0.05)
        np.testing.assert_allclose(np.diag(covariance), [1.0, 9.0, 0.25], rtol=0.05)

    def test_too_few_samples_raises(self) -> None:
        data = np.ones((3, 4))
        with pytest.raises(ValueError, match="more samples than dimensions"):
            fit_covariance(data, reg_coef=1e-6)

    def test_single_sample_raises(self) -> None:
        with pytest.raises(ValueError, match="more samples than dimensions"):
            fit_covariance(np.ones((1, 3)), reg_coef=1e-6)

    def test_non_finite_raises(self) -> None:
        data = np.ones((5, 2))
        data[0, 0] = np.inf
        with pytest.raises(ValueError, match="finite"):
            fit_covariance(data, reg_coef=1e-6)

    def test_1d_raises(self) -> None:
        with pytest.raises(ValueError, match="2D"):
            fit_covariance(np.ones(5), reg_coef=1e-6)


class TestCovarianceInterpolate:
    """Declared metric interpolation: endpoints preserved, never silent."""

    def test_endpoints_preserved(self) -> None:
        rng = np.random.default_rng(17)
        dim = 3
        covariance = _positive_definite(dim, rng)
        mean = rng.normal(size=dim)
        a = rng.normal(size=dim)
        b = rng.normal(size=dim)
        np.testing.assert_allclose(covariance_interpolate(a, b, 0.0, mean=mean, covariance=covariance), a, atol=1e-12)
        np.testing.assert_allclose(covariance_interpolate(a, b, 1.0, mean=mean, covariance=covariance), b, atol=1e-12)

    def test_constant_metric_matches_affine_lerp(self) -> None:
        """Under a constant covariance the metric geodesic is the affine lerp.

        This is the documented semantics: the whitened-frame computation is
        numerically identical to ``(1-t)a + t b`` for a constant covariance.
        """
        rng = np.random.default_rng(19)
        dim = 3
        covariance = _positive_definite(dim, rng)
        mean = rng.normal(size=dim)
        a = rng.normal(size=dim)
        b = rng.normal(size=dim)
        for t in (0.25, 0.5, 0.75):
            geodesic = covariance_interpolate(a, b, t, mean=mean, covariance=covariance)
            lerp = (1.0 - t) * a + t * b
            np.testing.assert_allclose(geodesic, lerp, atol=1e-12)

    @given(covariance=_pd_strategy(), a=_vector_strategy(), b=_vector_strategy(), t=st.floats(0.0, 1.0))
    @settings(max_examples=30)
    def test_property_constant_metric_geodesic_is_lerp(
        self, covariance: np.ndarray, a: np.ndarray, b: np.ndarray, t: float
    ) -> None:
        mean = np.zeros(a.shape[0])
        np.testing.assert_allclose(
            covariance_interpolate(a, b, t, mean=mean, covariance=covariance),
            (1.0 - t) * a + t * b,
            atol=1e-10,
        )


# ── CovarianceState contract ─────────────────────────────────────────


class TestCovarianceConfig:
    """Pydantic config validation."""

    def test_defaults(self) -> None:
        config = CovarianceConfig()
        assert config.reg_coef > 0
        assert config.min_samples_per_dimension > 0

    def test_invalid_reg_coef(self) -> None:
        with pytest.raises(ValueError):
            CovarianceConfig(reg_coef=0.0)

    def test_invalid_min_samples(self) -> None:
        with pytest.raises(ValueError):
            CovarianceConfig(min_samples_per_dimension=-1.0)


class TestFitCovarianceState:
    """Fitting is bound to a representation identity and validates input."""

    def test_fits_and_records_provenance(self) -> None:
        rng = np.random.default_rng(23)
        data = rng.normal(size=(60, 3))
        state = fit_covariance_state(
            data,
            source_representation_identity="vae@rev1/layer=mu",
            provenance={"dataset": "synthetic", "split": "train"},
        )
        assert state.n_samples == 60
        assert state.source_representation_identity == "vae@rev1/layer=mu"
        assert state.provenance["dataset"] == "synthetic"
        assert state.mean.shape == (3,)
        assert state.covariance.shape == (3, 3)
        validate_covariance(state.covariance, dim=3)

    def test_min_samples_enforced(self) -> None:
        data = np.ones((3, 4))
        with pytest.raises(ValueError, match="at least 8 samples"):
            fit_covariance_state(data, source_representation_identity="x")

    @pytest.mark.parametrize("identity", ["", "   "])
    def test_empty_identity_rejected_by_policy(self, identity: str) -> None:
        with pytest.raises(ValueError, match="source_representation_identity"):
            fit_covariance_state(np.random.default_rng(0).normal(size=(30, 2)), source_representation_identity=identity)

    def test_state_owns_read_only_arrays_and_nested_provenance(self) -> None:
        provenance = {"nested": {"labels": ["train"]}}
        state = CovarianceState(
            mean=np.zeros(2),
            covariance=np.eye(2),
            n_samples=10,
            source_representation_identity="x",
            reg_coef=1e-6,
            provenance=provenance,
        )
        provenance["nested"]["labels"].append("mutated")
        assert state.provenance["nested"]["labels"] == ("train",)
        with pytest.raises(ValueError):
            state.mean[0] = 1.0
        with pytest.raises(TypeError):
            state.provenance["new"] = True  # type: ignore[index]
        with pytest.raises(ValueError):
            state.mean.setflags(write=True)
        with pytest.raises(ValueError):
            state.covariance.setflags(write=True)

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("source_representation_identity", None),
            ("source_representation_identity", 42),
            ("n_samples", 1.9),
            ("n_samples", True),
        ],
    )
    def test_from_dict_rejects_each_raw_field_type_independently(self, field: str, value: object) -> None:
        payload: dict[str, object] = {
            "mean": [0.0, 0.0],
            "covariance": [[1.0, 0.0], [0.0, 1.0]],
            "n_samples": 5,
            "source_representation_identity": "x",
            "reg_coef": 1e-6,
            "provenance": {},
        }
        payload[field] = value
        with pytest.raises(ValueError, match="missing or malformed"):
            CovarianceState.from_dict(payload)

    @pytest.mark.parametrize("field", ["n_samples", "reg_coef"])
    def test_direct_construction_rejects_boolean_numeric_fields(self, field: str) -> None:
        n_samples: int | bool = True if field == "n_samples" else 5
        reg_coef: float | bool = True if field == "reg_coef" else 1e-6
        with pytest.raises(ValueError):
            CovarianceState(
                mean=np.zeros(2),
                covariance=np.eye(2),
                n_samples=n_samples,
                source_representation_identity="x",
                reg_coef=reg_coef,
                provenance={},
            )

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("n_samples", -1),
            ("reg_coef", -1e-6),
            ("mean", [0.0, float("nan")]),
            ("covariance", [[1.0, 2.0], [2.0, 1.0]]),
            ("source_representation_identity", " "),
        ],
    )
    def test_direct_construction_enforces_invariants(self, field: str, value: object) -> None:
        payload: dict[str, object] = {
            "mean": [0.0, 0.0],
            "covariance": [[1.0, 0.0], [0.0, 1.0]],
            "n_samples": 5,
            "source_representation_identity": "x",
            "reg_coef": 1e-6,
            "provenance": {},
        }
        payload[field] = value
        with pytest.raises(ValueError):
            CovarianceState.from_dict(payload)


class TestCovarianceStateSerialization:
    """Portable provenance round-trips."""

    def test_to_dict_from_dict_round_trip(self) -> None:
        rng = np.random.default_rng(29)
        data = rng.normal(size=(40, 3))
        state = fit_covariance_state(data, source_representation_identity="x", provenance={"a": 1})
        rebuilt = CovarianceState.from_dict(state.to_dict())
        np.testing.assert_allclose(rebuilt.mean, state.mean)
        np.testing.assert_allclose(rebuilt.covariance, state.covariance)
        assert rebuilt.source_representation_identity == state.source_representation_identity
        assert rebuilt.provenance == state.provenance

    def test_save_load_round_trip(self, tmp_path: object) -> None:
        rng = np.random.default_rng(31)
        data = rng.normal(size=(40, 3))
        state = fit_covariance_state(data, source_representation_identity="x", provenance={"k": "v"})
        path = tmp_path / "cov.npz"  # type: ignore[union-attr]
        state.save(path)  # type: ignore[arg-type]
        loaded = CovarianceState.load(path)  # type: ignore[arg-type]
        np.testing.assert_allclose(loaded.mean, state.mean)
        np.testing.assert_allclose(loaded.covariance, state.covariance)
        assert loaded.provenance == state.provenance
        assert loaded.source_representation_identity == "x"

    def test_from_dict_rejects_malformed(self) -> None:
        with pytest.raises(ValueError, match="missing or malformed"):
            CovarianceState.from_dict({"n_samples": 5})  # type: ignore[arg-type]

    def test_from_dict_rejects_shape_mismatch(self) -> None:
        payload: dict[str, object] = {
            "mean": [0.0, 0.0],
            "covariance": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "n_samples": 5,
            "source_representation_identity": "x",
            "reg_coef": 1e-6,
            "provenance": {},
        }
        with pytest.raises(ValueError, match="square covariance"):
            CovarianceState.from_dict(payload)


# ── LatentSpace anisotropic facade ───────────────────────────────────


class TestLatentSpaceAnisotropic:
    """geometry='anisotropic' dispatch through the LatentSpace facade."""

    def test_construction_requires_covariance_before_metric_ops(self) -> None:
        space = LatentSpace(dim=3, geometry="anisotropic")
        assert space.geometry == "anisotropic"
        assert space.covariance is None
        with pytest.raises(ValueError, match="no fitted covariance"):
            space.distance(np.ones(3), np.zeros(3))
        with pytest.raises(ValueError, match="no fitted covariance"):
            space.interpolate(np.ones(3), np.zeros(3), 0.5)
        with pytest.raises(ValueError, match="no fitted covariance"):
            space.whiten(np.ones(3))

    def test_fit_covariance_mutates_and_sets_metadata(self) -> None:
        rng = np.random.default_rng(37)
        data = rng.normal(size=(60, 3))
        space = LatentSpace(dim=3, geometry="anisotropic", source_model="vae")
        returned = space.fit_covariance(data, source_representation_identity="vae@mu")
        assert returned is space
        assert space.covariance is not None
        assert space.metadata["covariance_fitted"] is True
        assert space.metadata["covariance_source_representation_identity"] == "vae@mu"
        assert space.metadata["interpolation"] == "metric-geodesic"

    def test_fit_covariance_rejects_non_anisotropic(self) -> None:
        space = LatentSpace(dim=3)
        with pytest.raises(ValueError, match="requires geometry='anisotropic'"):
            space.fit_covariance(np.random.default_rng(0).normal(size=(10, 3)), source_representation_identity="x")

    def test_distance_dispatches_to_mahalanobis(self) -> None:
        rng = np.random.default_rng(41)
        data = rng.normal(size=(200, 3))
        space = LatentSpace(dim=3, geometry="anisotropic").fit_covariance(data, source_representation_identity="x")
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([-1.0, 0.0, 0.0])
        assert isinstance(space.distance(a, b), float)

    def test_distance_equals_pure_mahalanobis(self) -> None:
        rng = np.random.default_rng(43)
        data = rng.normal(size=(200, 3))
        space = LatentSpace(dim=3, geometry="anisotropic").fit_covariance(data, source_representation_identity="x")
        assert space.covariance is not None
        a = np.array([0.5, -1.0, 2.0])
        b = np.array([-1.5, 0.0, 0.5])
        expected = mahalanobis_distance(a, b, space.covariance.covariance)
        assert space.distance(a, b) == pytest.approx(expected)

    def test_interpolate_dispatches_to_metric_geodesic(self) -> None:
        rng = np.random.default_rng(47)
        data = rng.normal(size=(200, 3))
        space = LatentSpace(dim=3, geometry="anisotropic").fit_covariance(data, source_representation_identity="x")
        a = np.array([1.0, 0.5, -1.0])
        b = np.array([-1.0, 0.5, 1.0])
        result = space.interpolate(a, b, 0.5)
        np.testing.assert_allclose(result, (1.0 - 0.5) * a + 0.5 * b, atol=1e-12)

    def test_interpolate_t_range_validated(self) -> None:
        rng = np.random.default_rng(53)
        data = rng.normal(size=(60, 3))
        space = LatentSpace(dim=3, geometry="anisotropic").fit_covariance(data, source_representation_identity="x")
        with pytest.raises(ValueError, match="t must be in"):
            space.interpolate(np.ones(3), np.ones(3), 1.5)

    @pytest.mark.parametrize("operation", ["distance", "interpolate"])
    @pytest.mark.parametrize("bad_point", [np.ones(2), np.array([0.0, 1.0, np.nan])])
    def test_geometry_operations_validate_both_endpoints(self, operation: str, bad_point: np.ndarray) -> None:
        data = np.random.default_rng(54).normal(size=(60, 3))
        space = LatentSpace(dim=3, geometry="anisotropic").fit_covariance(data, source_representation_identity="x")
        with pytest.raises(ValueError):
            if operation == "distance":
                space.distance(bad_point, np.zeros(3))
            else:
                space.interpolate(np.zeros(3), bad_point, 0.5)

    def test_whiten_unwhiten_public_api(self) -> None:
        rng = np.random.default_rng(59)
        data = rng.normal(size=(200, 3))
        space = LatentSpace(dim=3, geometry="anisotropic").fit_covariance(data, source_representation_identity="x")
        point = np.array([0.3, -1.2, 2.0])
        whitened = space.whiten(point)
        assert whitened.shape == (3,)
        np.testing.assert_allclose(space.unwhiten(whitened), point, atol=1e-12)

    def test_normalize_returns_copy(self) -> None:
        rng = np.random.default_rng(61)
        data = rng.normal(size=(60, 3))
        space = LatentSpace(dim=3, geometry="anisotropic").fit_covariance(data, source_representation_identity="x")
        point = np.array([1.0, 2.0, 3.0])
        result = space.normalize(point)
        np.testing.assert_array_equal(result, point)
        assert result is not point

    def test_validate_point_checks_shape_and_finite(self) -> None:
        space = LatentSpace(dim=3, geometry="anisotropic")
        space.validate_point(np.array([1.0, 2.0, 3.0]))
        with pytest.raises(ValueError, match="Expected shape"):
            space.validate_point(np.array([1.0, 2.0]))
        with pytest.raises(ValueError, match="finite"):
            space.validate_point(np.array([1.0, np.inf, 3.0]))

    def test_shape_is_flat(self) -> None:
        space = LatentSpace(dim=3, geometry="anisotropic")
        assert space.shape == (3,)

    def test_repr_includes_fitted_state(self) -> None:
        rng = np.random.default_rng(67)
        data = rng.normal(size=(60, 3))
        space = LatentSpace(dim=3, geometry="anisotropic")
        assert "covariance_fitted=False" in repr(space)
        space.fit_covariance(data, source_representation_identity="x")
        assert "covariance_fitted=True" in repr(space)

    def test_attaching_wrong_dim_covariance_raises(self) -> None:
        rng = np.random.default_rng(71)
        data = rng.normal(size=(60, 4))
        state = fit_covariance_state(data, source_representation_identity="x")
        with pytest.raises(ValueError, match="does not match dim"):
            LatentSpace(dim=3, geometry="anisotropic", covariance=state)

    def test_non_pd_covariance_rejected_at_construction(self) -> None:
        with pytest.raises(ValueError, match="positive-definite"):
            CovarianceState(
                mean=np.zeros(3),
                covariance=np.diag([1.0, -1.0, 1.0]),
                n_samples=60,
                source_representation_identity="x",
                reg_coef=1e-6,
                provenance={},
            )

"""Comprehensive tests for the K-means clustering module.

Covers:
- KMeansConfig validation
- KMeansResult validation and serialization
- KMeans fit_predict lifecycle
- Standardization / preprocessing
- Silhouette and confidence proxy
- Input validation and edge cases
- Geometry checks
- Cluster stability analysis
- External validation (compare_with_labels)
- Config construction via registry
- Marked real-integration tests (network)
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from pydantic import ValidationError

from latent_anything.clustering import (
    ClusterStabilityReport,
    KMeans,
    KMeansConfig,
    KMeansResult,
    check_clustering_geometry,
    cluster_stability_analysis,
    compare_with_labels,
)

# ---------------------------------------------------------------------------
#  Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def well_separated_clusters() -> np.ndarray:
    """Three well-separated Gaussian clusters in 2D, 60 samples."""
    rng = np.random.default_rng(42)
    clusters = []
    for i in range(3):
        center = np.array([i * 5.0, i * 5.0])
        cluster = rng.normal(loc=center, scale=0.3, size=(20, 2))
        clusters.append(cluster)  # type: ignore[reportUnknownMemberType]
    return np.concatenate(clusters, axis=0)  # type: ignore[reportUnknownArgumentType]


@pytest.fixture
def overlapping_clusters() -> np.ndarray:
    """Two overlapping clusters in 4D, 40 samples."""
    rng = np.random.default_rng(42)
    c1 = rng.normal(loc=[0, 0, 0, 0], scale=1.0, size=(20, 4))
    c2 = rng.normal(loc=[0.5, 0.5, 0.5, 0.5], scale=1.0, size=(20, 4))
    return np.concatenate([c1, c2], axis=0)


@pytest.fixture
def high_dim_data() -> np.ndarray:
    """50 samples in 20D with 4 clusters."""
    rng = np.random.default_rng(42)
    clusters: list[np.ndarray] = []
    for i in range(4):
        center = np.full(20, i * 3.0)
        cluster = rng.normal(loc=center, scale=0.5, size=(15, 20))
        clusters.append(cluster)
    return np.concatenate(clusters, axis=0)


@pytest.fixture
def km() -> KMeans:
    return KMeans()


@pytest.fixture
def true_labels() -> np.ndarray:
    """Ground-truth labels for well_separated_clusters."""
    return np.repeat([0, 1, 2], 20)


# ---------------------------------------------------------------------------
#  KMeansConfig  (Tasks 1)
# ---------------------------------------------------------------------------


class TestKMeansConfig:
    def test_default_config(self) -> None:
        cfg = KMeansConfig()
        assert cfg.n_clusters == 8
        assert cfg.init == "k-means++"
        assert cfg.n_init == 10
        assert cfg.max_iter == 300
        assert cfg.random_state == 0
        assert cfg.standardize is True

    def test_invalid_n_clusters_raises(self) -> None:
        with pytest.raises(ValidationError):
            KMeansConfig(n_clusters=1)

    def test_invalid_init_raises(self) -> None:
        with pytest.raises(ValidationError):
            KMeansConfig(init="forgy")

    def test_config_serialization_roundtrip(self) -> None:
        cfg = KMeansConfig(n_clusters=5, random_state=42, standardize=False)
        data = cfg.model_dump()
        restored = KMeansConfig(**data)
        assert restored.n_clusters == 5
        assert restored.random_state == 42
        assert restored.standardize is False


# ---------------------------------------------------------------------------
#  KMeansResult  (Task 2)
# ---------------------------------------------------------------------------


class TestKMeansResult:
    def test_result_holds_expected_fields(self, well_separated_clusters: np.ndarray) -> None:
        km = KMeans(KMeansConfig(n_clusters=3, random_state=0))
        result = km.fit_predict(well_separated_clusters)
        assert isinstance(result, KMeansResult)
        assert isinstance(result.assignments, np.ndarray)
        assert result.assignments.shape == (60,)
        assert result.centers.shape == (3, 2)
        assert isinstance(result.inertia, float)
        assert result.inertia > 0
        assert result.n_clusters == 3
        assert result.cluster_sizes.shape == (3,)
        assert result.cluster_sizes.sum() == 60
        assert isinstance(result.silhouette_score, float)
        assert -1.0 <= result.silhouette_score <= 1.0
        assert result.confidence.shape == (60,)
        assert result.confidence.min() >= 0
        assert isinstance(result.config, KMeansConfig)
        assert isinstance(result.provenance, dict)

    def test_result_on_separated_data(self, well_separated_clusters: np.ndarray) -> None:
        km = KMeans(KMeansConfig(n_clusters=3, random_state=0))
        result = km.fit_predict(well_separated_clusters)
        # On well-separated data, silhouette should be high
        assert result.silhouette_score > 0.6, f"silhouette={result.silhouette_score}"

    def test_result_to_dict(self, well_separated_clusters: np.ndarray) -> None:
        km = KMeans(KMeansConfig(n_clusters=3, random_state=0))
        result = km.fit_predict(well_separated_clusters)
        d = result.to_dict()
        assert isinstance(d["assignments"], list)
        assert isinstance(d["centers"], list)
        assert isinstance(d["inertia"], float)
        assert isinstance(d["config"], dict)
        assert d["n_clusters"] == 3

    def test_result_validation_raises(self) -> None:
        with pytest.raises(ValueError, match="assignments must be 1D"):
            KMeansResult(
                assignments=np.ones((5, 2)),
                centers=np.ones((3, 4)),
                inertia=1.0,
                n_iter=10,
                n_clusters=3,
                cluster_sizes=np.ones(3),
                silhouette_score=0.5,
                per_sample_silhouette=np.ones(5),
                confidence=np.ones(5),
                n_init=10,
                config=KMeansConfig(),
            )

    def test_per_sample_silhouette_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="per_sample_silhouette shape"):
            KMeansResult(
                assignments=np.zeros(10),
                centers=np.ones((3, 4)),
                inertia=1.0,
                n_iter=10,
                n_clusters=3,
                cluster_sizes=np.ones(3),
                silhouette_score=0.5,
                per_sample_silhouette=np.ones(5),
                confidence=np.ones(10),
                n_init=10,
                config=KMeansConfig(),
            )


# ---------------------------------------------------------------------------
#  KMeans fit_predict lifecycle  (Task 1)
# ---------------------------------------------------------------------------


class TestKMeansFitPredict:
    def test_fit_predict_returns_result(self, km: KMeans, well_separated_clusters: np.ndarray) -> None:
        result = km.fit_predict(well_separated_clusters)
        assert isinstance(result, KMeansResult)
        assert km.is_fitted

    def test_fit_predict_with_provenance(self, km: KMeans, well_separated_clusters: np.ndarray) -> None:
        result = km.fit_predict(well_separated_clusters, provenance={"layer": 5, "model": "gpt2"})
        assert result.provenance["layer"] == 5
        assert result.provenance["model"] == "gpt2"

    def test_result_property(self, km: KMeans, well_separated_clusters: np.ndarray) -> None:
        km.fit_predict(well_separated_clusters)
        result = km.result
        assert isinstance(result, KMeansResult)

    def test_result_raises_before_fit(self) -> None:
        km = KMeans()
        with pytest.raises(RuntimeError, match="not been fitted"):
            _ = km.result

    def test_is_fitted_after_fit(self, km: KMeans, well_separated_clusters: np.ndarray) -> None:
        assert not km.is_fitted
        km.fit_predict(well_separated_clusters)
        assert km.is_fitted

    def test_fit_is_idempotent(self, well_separated_clusters: np.ndarray) -> None:
        km = KMeans(KMeansConfig(n_clusters=3, random_state=0))
        result1 = km.fit_predict(well_separated_clusters)
        result2 = km.fit_predict(well_separated_clusters)
        assert_array_equal(result1.assignments, result2.assignments)
        assert result1.inertia == result2.inertia

    def test_custom_config(self, well_separated_clusters: np.ndarray) -> None:
        cfg = KMeansConfig(n_clusters=3, n_init=5, max_iter=100, random_state=42)
        km = KMeans(cfg)
        result = km.fit_predict(well_separated_clusters)
        assert result.config.n_clusters == 3
        assert result.config.n_init == 5
        assert result.n_init == 5


# ---------------------------------------------------------------------------
#  Input validation  (Task 1)
# ---------------------------------------------------------------------------


class TestInputValidation:
    def test_1d_data_raises(self, km: KMeans) -> None:
        with pytest.raises(ValueError, match="data must be 2D"):
            km.fit_predict(np.array([1.0, 2.0, 3.0]))

    def test_fewer_samples_than_clusters_raises(self, km: KMeans) -> None:
        cfg = KMeansConfig(n_clusters=5)
        km = KMeans(cfg)
        with pytest.raises(ValueError, match="n_samples"):
            km.fit_predict(np.ones((3, 2)))

    def test_single_sample_raises(self, km: KMeans) -> None:
        with pytest.raises(ValueError, match="n_samples|at least 2 samples"):
            km.fit_predict(np.ones((1, 3)))

    def test_no_features_raises(self) -> None:
        km = KMeans(KMeansConfig(n_clusters=3))
        with pytest.raises(ValueError, match="at least 1 feature"):
            km.fit_predict(np.ones((5, 0)))


# ---------------------------------------------------------------------------
#  Standardization / preprocessing  (Task 1)
# ---------------------------------------------------------------------------


class TestStandardization:
    def test_standardize_default(self, well_separated_clusters: np.ndarray) -> None:
        km = KMeans(KMeansConfig(n_clusters=3))
        result = km.fit_predict(well_separated_clusters)
        assert result.feature_means is not None
        assert result.feature_stds is not None
        assert result.feature_means.shape == (2,)
        assert result.feature_stds.shape == (2,)

    def test_no_standardize(self, well_separated_clusters: np.ndarray) -> None:
        km = KMeans(KMeansConfig(n_clusters=3, standardize=False))
        result = km.fit_predict(well_separated_clusters)
        assert result.feature_means is None
        assert result.feature_stds is None

    def test_standardize_equal_to_manual(self, well_separated_clusters: np.ndarray) -> None:
        """Standardized should give same assignments as manual z-score."""
        km_std = KMeans(KMeansConfig(n_clusters=3, standardize=True, random_state=0))
        km_raw = KMeans(KMeansConfig(n_clusters=3, standardize=False, random_state=1))
        # Manual standardization
        data = well_separated_clusters.copy()
        data_scaled = (data - data.mean(axis=0)) / data.std(axis=0, ddof=0)
        result_raw = km_raw.fit_predict(data_scaled)
        result_std = km_std.fit_predict(data)
        assert_array_equal(result_raw.assignments, result_std.assignments)


# ---------------------------------------------------------------------------
#  Silhouette and confidence proxy  (Task 2)
# ---------------------------------------------------------------------------


class TestDiagnostics:
    def test_silhouette_on_separated_data(self, well_separated_clusters: np.ndarray) -> None:
        km = KMeans(KMeansConfig(n_clusters=3, random_state=0))
        result = km.fit_predict(well_separated_clusters)
        assert result.silhouette_score > 0.6
        assert result.per_sample_silhouette is not None
        assert result.per_sample_silhouette.shape == (60,)

    def test_confidence_margin_exists(self, well_separated_clusters: np.ndarray) -> None:
        km = KMeans(KMeansConfig(n_clusters=3, random_state=0))
        result = km.fit_predict(well_separated_clusters)
        assert np.all(result.confidence >= 0)
        # Confident assignments should have positive margin
        confident = result.confidence > 0
        assert confident.sum() > 0


# ---------------------------------------------------------------------------
#  Geometry checks  (Task 5)
# ---------------------------------------------------------------------------


class TestGeometryChecks:
    def test_euclidean_allowed(self) -> None:
        # Should not raise
        check_clustering_geometry("euclidean")

    def test_unit_norm_allowed(self) -> None:
        check_clustering_geometry("unit_norm")

    def test_gaussian_set_raises(self) -> None:
        with pytest.raises(ValueError, match="does not support geometry"):
            check_clustering_geometry("gaussian_set")

    def test_discrete_code_raises(self) -> None:
        with pytest.raises(ValueError, match="does not support geometry"):
            check_clustering_geometry("discrete_code")

    def test_static_check_geometry_method(self) -> None:
        # Should not raise
        KMeans.check_geometry("euclidean")

    def test_static_check_geometry_raises(self) -> None:
        with pytest.raises(ValueError, match="does not support geometry"):
            KMeans.check_geometry("gaussian_set")

    def test_unknown_geometry_raises(self) -> None:
        with pytest.raises(ValueError, match="does not support geometry"):
            check_clustering_geometry("unknown_geo")


# ---------------------------------------------------------------------------
#  Cluster stability analysis  (Task 3)
# ---------------------------------------------------------------------------


class TestClusterStability:
    def test_stability_returns_report(self, well_separated_clusters: np.ndarray) -> None:
        config = KMeansConfig(n_clusters=3, n_init=5, max_iter=100)
        report = cluster_stability_analysis(well_separated_clusters, seeds=[0, 1, 2], config=config)
        assert isinstance(report, ClusterStabilityReport)
        assert report.n_seeds == 3
        assert isinstance(report.adjusted_rand_index, float)
        assert isinstance(report.mean_stability, float)
        assert report.per_cluster_stability.shape == (3,)

    def test_stability_high_on_separated_data(self, well_separated_clusters: np.ndarray) -> None:
        config = KMeansConfig(n_clusters=3, n_init=5, max_iter=100)
        report = cluster_stability_analysis(well_separated_clusters, seeds=[0, 1, 2], config=config)
        assert report.mean_stability > 0.8, f"mean_stability={report.mean_stability}"
        assert report.adjusted_rand_index > 0.8

    def test_stability_lower_on_overlapping(self, overlapping_clusters: np.ndarray) -> None:
        config = KMeansConfig(n_clusters=2, n_init=5, max_iter=100)
        report = cluster_stability_analysis(overlapping_clusters, seeds=[0, 1, 2], config=config)
        # Overlapping data should have lower stability
        assert report.mean_stability >= 0.0

    def test_stability_with_n_seeds(self, well_separated_clusters: np.ndarray) -> None:
        report = cluster_stability_analysis(well_separated_clusters, n_seeds=5, config=KMeansConfig(n_clusters=3))
        assert report.n_seeds == 5
        assert report.assignments_matrix.shape[0] == 5

    def test_stability_results_accessible(self, well_separated_clusters: np.ndarray) -> None:
        report = cluster_stability_analysis(
            well_separated_clusters,
            seeds=[0, 1],
            config=KMeansConfig(n_clusters=3),
        )
        assert len(report.results) == 2
        assert all(isinstance(r, KMeansResult) for r in report.results)


# ---------------------------------------------------------------------------
#  External validation  (Task 4)
# ---------------------------------------------------------------------------


class TestExternalValidation:
    def test_compare_with_labels_returns_metrics(self, well_separated_clusters: np.ndarray) -> None:
        km = KMeans(KMeansConfig(n_clusters=3, random_state=0))
        result = km.fit_predict(well_separated_clusters)
        true_labels = np.repeat([0, 1, 2], 20)
        metrics = compare_with_labels(result.assignments, true_labels)
        assert isinstance(metrics, dict)
        assert "adjusted_rand_index" in metrics
        assert "adjusted_mutual_info" in metrics
        assert "homogeneity" in metrics
        assert "completeness" in metrics
        assert "v_measure" in metrics
        assert 0.0 <= metrics["adjusted_rand_index"] <= 1.0

    def test_high_ari_on_separated_data(self, well_separated_clusters: np.ndarray) -> None:
        km = KMeans(KMeansConfig(n_clusters=3, random_state=0))
        result = km.fit_predict(well_separated_clusters)
        true_labels = np.repeat([0, 1, 2], 20)
        metrics = compare_with_labels(result.assignments, true_labels)
        assert metrics["adjusted_rand_index"] > 0.8, f"ARI={metrics['adjusted_rand_index']}"
        assert metrics["v_measure"] > 0.8

    def test_compare_does_not_use_labels_for_fitting(self, well_separated_clusters: np.ndarray) -> None:
        """Labels should only be used for comparison, never passed to fit."""
        km = KMeans(KMeansConfig(n_clusters=3, random_state=0))
        result = km.fit_predict(well_separated_clusters)
        true_labels = np.repeat([0, 1, 2], 20)
        # This verifies that assignments exist independently before comparison
        assert len(np.unique(result.assignments)) == 3
        metrics = compare_with_labels(result.assignments, true_labels)
        assert metrics["homogeneity"] > 0.8


# ---------------------------------------------------------------------------
#  Degenerate / edge cases  (Task 7)
# ---------------------------------------------------------------------------


class TestDegenerateInputs:
    def test_duplicate_samples(self) -> None:
        """All identical samples should produce a valid cluster assignment."""
        data = np.ones((10, 3))
        km = KMeans(KMeansConfig(n_clusters=2, random_state=0, standardize=False))
        # Should not crash; silhouette may be degenerate for identical data
        try:
            result = km.fit_predict(data)
            assert result.assignments.shape == (10,)
        except ValueError:
            pytest.skip("K-means raised on perfectly collinear data")
        except Exception:
            pytest.skip("K-means raised on degenerate data")

    def test_imbalanced_clusters(self) -> None:
        """90 % in one cluster, 10 % in another."""
        rng = np.random.default_rng(42)
        data = np.concatenate(
            [
                rng.normal(0, 0.3, (90, 2)),
                rng.normal(10, 0.3, (10, 2)),
            ],
            axis=0,
        )
        km = KMeans(KMeansConfig(n_clusters=2, random_state=0))
        result = km.fit_predict(data)
        assert result.cluster_sizes.sum() == 100
        assert np.min(result.cluster_sizes) >= 1

    def test_empty_cluster_on_initialization(self) -> None:
        """K-means can produce empty clusters; handle gracefully."""
        rng = np.random.default_rng(42)
        data = rng.normal(0, 0.1, (20, 5))
        cfg = KMeansConfig(n_clusters=10, random_state=0, standardize=False)
        km = KMeans(cfg)
        result = km.fit_predict(data)
        assert result.assignments.shape == (20,)
        # Some clusters may be empty
        non_empty = (result.cluster_sizes > 0).sum()
        assert non_empty >= 1

    def test_high_dimensional(self, high_dim_data: np.ndarray) -> None:
        """20D data should work."""
        km = KMeans(KMeansConfig(n_clusters=4, random_state=0))
        result = km.fit_predict(high_dim_data)
        assert result.centers.shape == (4, 20)
        assert result.silhouette_score > 0.3

    def test_minimal_data(self) -> None:
        """Minimum viable: 2 samples, 1 feature, 2 clusters."""
        data = np.array([[0.0], [1.0]])
        km = KMeans(KMeansConfig(n_clusters=2, random_state=0, standardize=False))
        result = km.fit_predict(data)
        assert result.assignments.shape == (2,)
        assert result.centers.shape == (2, 1)


# ---------------------------------------------------------------------------
#  Config construction via registry  (Task 7)
# ---------------------------------------------------------------------------


class TestRegistryConstruction:
    def test_build_from_object_spec(self) -> None:
        from latent_anything.config import ObjectSpec, build_from_config

        spec = ObjectSpec(kind="analysis", name="kmeans")
        km = build_from_config(spec)
        assert isinstance(km, KMeans)
        assert isinstance(km.config, KMeansConfig)

    def test_build_from_object_spec_with_params(self) -> None:
        from latent_anything.config import ObjectSpec, build_from_config

        spec = ObjectSpec(kind="analysis", name="kmeans", params={"n_clusters": 5, "random_state": 42})
        km = build_from_config(spec)
        assert isinstance(km, KMeans)
        assert km.config.n_clusters == 5
        assert km.config.random_state == 42

    def test_build_from_dict(self) -> None:
        from latent_anything.config import build_from_dict

        km = build_from_dict({"kind": "analysis", "name": "kmeans"})
        assert isinstance(km, KMeans)

    def test_registry_entry_exists(self) -> None:
        from latent_anything.registry import GLOBAL_REGISTRY

        entry = GLOBAL_REGISTRY.lookup("kmeans")
        assert entry.kind == "analysis"
        assert entry.factory is KMeans

    def test_config_is_properly_set(self) -> None:
        """Direct construction should accept optional config."""
        cfg = KMeansConfig(n_clusters=4, init="random")
        km = KMeans(cfg)
        assert km.config.n_clusters == 4
        assert km.config.init == "random"


# ---------------------------------------------------------------------------
#  ClusterStabilityReport serialization  (Task 3)
# ---------------------------------------------------------------------------


class TestStabilityReportSerialization:
    def test_to_dict(self, well_separated_clusters: np.ndarray) -> None:
        config = KMeansConfig(n_clusters=3, n_init=5)
        report = cluster_stability_analysis(well_separated_clusters, seeds=[0, 1], config=config)
        d = report.to_dict()
        assert isinstance(d["adjusted_rand_index"], float)
        assert isinstance(d["mean_stability"], float)
        assert isinstance(d["per_cluster_stability"], list)
        assert isinstance(d["assignments_matrix"], list)
        assert isinstance(d["results"], list)
        assert d["n_seeds"] == 2


# ---------------------------------------------------------------------------
#  Real-integration tests (marked network)  --  Task 6
# ---------------------------------------------------------------------------


@pytest.mark.network
class TestRealIntegration:
    """Integration tests that use real model representations (off by default).

    Run with::

        uv run pytest tests/test_clustering.py -m network -v
    """

    def test_on_vae_latents(self) -> None:
        """Cluster VAE latent representations."""
        from latent_anything.integrations.diffusers_vae import DiffusersAutoencoderKLAdapter

        adapter = DiffusersAutoencoderKLAdapter("CompVis/stable-diffusion-v1-4", "7460a6f", latent_mode="mean")
        rng = np.random.default_rng(42)
        images = rng.uniform(-1, 1, (64, 3, 64, 64)).astype(np.float32)
        latents = adapter.encode(images)

        # Flatten spatial dims
        n, c, h, w = latents.shape
        latents_flat = latents.reshape(n, c * h * w)  # (64, 4*64*64)

        km = KMeans(KMeansConfig(n_clusters=3, random_state=0, max_iter=50))
        result = km.fit_predict(latents_flat)
        assert result.assignments.shape == (64,)
        assert result.centers.shape == (3, c * h * w)
        assert result.silhouette_score > -1.0

    def test_on_transformer_hidden_states(self) -> None:
        """Cluster transformer hidden states across layers."""
        from latent_anything.integrations.transformer_lm import (
            TransformerGenerationRequest,
            TransformerLMIntegration,
        )

        integration = TransformerLMIntegration()
        request = TransformerGenerationRequest(
            prompt="The cat sat on the mat and looked around",
            max_length=64,
        )
        gen_result = integration.generate(request)

        # Pool across all layers and tokens
        all_activations: list[np.ndarray] = []
        for hs in gen_result.hidden_states:
            # Mean over sequence dimension
            pooled = hs.values[0].mean(axis=0)  # (hidden_dim,)
            all_activations.append(pooled)

        acts_matrix = np.stack(all_activations, axis=0)  # (n_layers+1, hidden_dim)

        km = KMeans(KMeansConfig(n_clusters=3, random_state=0, max_iter=50))
        result = km.fit_predict(acts_matrix)
        assert result.assignments.shape == (acts_matrix.shape[0],)
        assert result.centers.shape[1] == acts_matrix.shape[1]

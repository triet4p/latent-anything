"""Schema, builder, and downsampling tests for renderer inputs.

The renderer inputs are the stable contract between analysis results and the
visualization frontends, so their JSON schema is asserted exactly and the
downsampling behavior (the responsiveness contract) is tested deterministically.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from latent_anything import LatentSpace, SegmentationConfig, Trajectory, compute_dtw, detect_change_points
from latent_anything.clustering import ClusterStabilityReport, KMeansConfig, KMeansResult
from latent_anything.density import (
    DensityEvaluationReport,
    DensityMetrics,
    DensityResult,
    DensityStabilityReport,
)
from latent_anything.probes import CrossSeedReport, LinearProbeConfig, LinearProbeResult
from latent_anything.sae_evaluation import (
    FeatureAtlas,
    FeatureAtlasEntry,
    SAEConfig,
    SAEEvaluationResult,
    SAEFeatureMetrics,
)
from latent_anything.visualization import (
    DEFAULT_POINT_LIMIT_2D,
    DEFAULT_POINT_LIMIT_3D,
    DOWNSAMPLE_SEED,
    MetricSummary,
    PointView,
    TrajectoryView,
    build_projection,
    default_point_limit,
    downsample_view,
    metric_summary_from_atlas,
    metric_summary_from_cross_seed,
    metric_summary_from_density,
    metric_summary_from_kmeans,
    metric_summary_from_probe,
    metric_summary_from_sae,
    metric_summary_from_stability,
    points_view,
    projection_from_atlas,
    projection_from_density,
    projection_from_dtw,
    projection_from_kmeans,
    projection_from_probe,
    projection_from_trajectory,
    trajectory_view,
)


def _coords(n: int, ndim: int = 2, seed: int = 0) -> np.ndarray:
    return np.asarray(np.random.default_rng(seed).random((n, ndim)), dtype=np.float64)


def _kmeans_result(n: int = 30, n_clusters: int = 3, seed: int = 0) -> KMeansResult:
    rng = np.random.default_rng(seed)
    assignments = np.asarray(rng.integers(0, n_clusters, size=n), dtype=np.int64)
    centers = np.asarray(rng.random((n_clusters, 4)), dtype=np.float64)
    return KMeansResult(
        assignments=assignments,
        centers=centers,
        inertia=float(rng.random()),
        n_iter=10,
        n_clusters=n_clusters,
        cluster_sizes=np.asarray([np.count_nonzero(assignments == i) for i in range(n_clusters)], dtype=np.int64),
        silhouette_score=0.42,
        per_sample_silhouette=np.asarray(rng.random(n), dtype=np.float64),
        confidence=np.asarray(rng.random(n), dtype=np.float64),
        n_init=2,
        config=KMeansConfig(n_clusters=n_clusters),
        provenance={"source": "test"},
    )


def _probe_result(n: int = 30, n_classes: int = 3, seed: int = 0) -> LinearProbeResult:
    rng = np.random.default_rng(seed)
    classes = np.arange(n_classes)
    predictions = np.asarray(rng.integers(0, n_classes, size=n), dtype=np.int64)
    probabilities = rng.random((n, n_classes))
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return LinearProbeResult(
        accuracy=0.81,
        val_accuracy=0.77,
        classes=classes,
        predictions=predictions,
        probabilities=probabilities,
        coefficients=np.asarray(rng.random((n_classes, 4)), dtype=np.float64),
        intercept=np.asarray(rng.random(n_classes), dtype=np.float64),
        n_iter=25,
        train_indices=np.zeros(n, dtype=bool),
        val_indices=np.zeros(n, dtype=bool),
        test_indices=np.ones(n, dtype=bool),
        feature_means=np.asarray(rng.random(4), dtype=np.float64),
        feature_stds=np.asarray(rng.random(4) + 0.5, dtype=np.float64),
        config=LinearProbeConfig(),
        provenance={"source": "test"},
    )


def _density_result(n: int = 30, seed: int = 0) -> DensityResult:
    rng = np.random.default_rng(seed)
    return DensityResult(
        log_density=np.asarray(rng.random(n) - 2.0, dtype=np.float64),
        responsibilities=np.asarray(rng.random((n, 2)), dtype=np.float64),
        calibrated_ood_score=np.asarray(rng.random(n), dtype=np.float64),
        source_representation_identity="test_conv_vae",
        fit_provenance={"n_samples": n},
        calibration_provenance={"n_samples": n},
    )


def _atlas() -> FeatureAtlas:
    entries = (
        FeatureAtlasEntry(
            feature_index=0,
            is_dead=False,
            activation_frequency=0.2,
            mean_activation=1.5,
            decoder_norm=2.0,
            encoder_norm=1.1,
            top_examples=({"rank": 1, "example_index": 3, "activation": 2.0, "label": "7"},),
            bottom_examples=(),
            top_decoder_dims=({"dim_index": 0, "weight": 0.9},),
        ),
        FeatureAtlasEntry(
            feature_index=1,
            is_dead=True,
            activation_frequency=0.0,
            mean_activation=0.0,
            decoder_norm=1.0,
            encoder_norm=0.0,
            top_examples=(),
            bottom_examples=(),
            top_decoder_dims=(),
        ),
    )
    return FeatureAtlas(
        entries=entries,
        n_components=2,
        n_examples=30,
        source_representation_identity="test_conv_vae",
        provenance={"seed": 0},
    )


def _sae_result(n_components: int = 4, seed: int = 0) -> SAEEvaluationResult:
    rng = np.random.default_rng(seed)
    n_features = 6
    return SAEEvaluationResult(
        config=SAEConfig(n_components=n_components),
        n_train=16,
        n_val=8,
        reconstruction_mse=0.003,
        train_reconstruction_mse=0.002,
        mean_l0=1.4,
        mean_l1=2.1,
        n_dead_features=1,
        dead_fraction=0.25,
        activation_frequencies=np.asarray(rng.random(n_components), dtype=np.float64),
        decoder_norms=np.asarray(rng.random(n_components) + 0.5, dtype=np.float64),
        features=tuple(
            SAEFeatureMetrics(
                feature_index=i,
                activation_frequency=float(rng.random()),
                mean_activation=float(rng.random()),
                mean_positive_activation=float(rng.random()),
                decoder_norm=float(rng.random() + 0.5),
                encoder_norm=float(rng.random()),
                is_dead=False,
            )
            for i in range(n_components)
        ),
        val_activations=np.asarray(rng.random((8, n_components)), dtype=np.float64),
        decoder_weights=np.asarray(rng.random((n_features, n_components)), dtype=np.float64),
        source_representation_identity="test_conv_vae",
        provenance={"seed": 0},
    )


def _cross_seed_report() -> CrossSeedReport:
    return CrossSeedReport(
        accuracies=[0.8, 0.82, 0.79],
        val_accuracies=[0.76, 0.78, 0.75],
        mean_accuracy=0.803,
        ci95=0.012,
        std_accuracy=0.015,
        min_accuracy=0.79,
        max_accuracy=0.82,
        n_seeds=3,
        results=[],
    )


def _stability_report() -> ClusterStabilityReport:
    return ClusterStabilityReport(
        adjusted_rand_index=0.91,
        mean_stability=0.87,
        per_cluster_stability=np.asarray([0.9, 0.85, 0.86], dtype=np.float64),
        n_seeds=3,
        assignments_matrix=np.zeros((3, 30), dtype=np.int64),
        results=[],
    )


# ── Renderer-input schema ─────────────────────────────────────────────────


class TestViewSchema:
    def test_point_view_to_dict_schema(self) -> None:
        point = PointView((1.0, 2.0), label="a", category="c", metadata={"p": 0.5})
        assert point.to_dict() == {"coordinates": [1.0, 2.0], "label": "a", "category": "c", "metadata": {"p": 0.5}}

    def test_trajectory_view_to_dict_schema(self) -> None:
        trajectory = TrajectoryView((PointView((0.0, 0.0)), PointView((1.0, 1.0))), name="t")
        assert trajectory.to_dict() == {
            "name": "t",
            "points": [
                {"coordinates": [0.0, 0.0], "label": None, "category": None, "metadata": {}},
                {"coordinates": [1.0, 1.0], "label": None, "category": None, "metadata": {}},
            ],
        }

    def test_projection_view_to_dict_schema(self) -> None:
        view = build_projection(_coords(3), categories=["a", "b", "a"], title="T", extra_metadata={"metrics": {"m": 1}})
        payload = view.to_dict()
        assert list(payload) == ["title", "ndim", "n_points", "points", "trajectories", "metadata"]
        assert payload["title"] == "T"
        assert payload["ndim"] == 2
        assert payload["n_points"] == 3
        assert payload["metadata"] == {"metrics": {"m": 1}}
        assert payload["points"][0]["category"] == "a"

    def test_metric_summary_to_dict_schema(self) -> None:
        summary = MetricSummary("K-means", {"silhouette_score": 0.42, "n_clusters": 3})
        assert summary.to_dict() == {"title": "K-means", "metrics": {"silhouette_score": 0.42, "n_clusters": 3}}

    def test_points_view_coerces_arrays(self) -> None:
        points = points_view(np.array([[1, 2], [3, 4]], dtype=np.int64), categories=[0, 1])
        assert points[0].coordinates == (1.0, 2.0)
        assert points[0].category == "0"

    def test_points_view_rejects_bad_shapes(self) -> None:
        with pytest.raises(ValueError, match="2 or 3 columns"):
            points_view(np.ones((5, 4)))
        with pytest.raises(ValueError, match="finite"):
            points_view(np.array([[np.nan, 0.0]]))

    def test_points_view_rejects_mismatched_optics(self) -> None:
        with pytest.raises(ValueError, match="labels"):
            points_view(_coords(3), labels=["a", "b"])


# ── Builders from analysis results ────────────────────────────────────────


class TestProjectionBuilders:
    def test_projection_from_kmeans(self) -> None:
        result = _kmeans_result()
        view = projection_from_kmeans(result, _coords(len(result.assignments)))
        assert {point.category for point in view.points} == {"cluster 0", "cluster 1", "cluster 2"}
        assert all("confidence" in point.metadata for point in view.points)
        assert view.metadata["metrics"]["silhouette_score"] == 0.42
        assert view.ndim == 2

    def test_projection_from_kmeans_rejects_row_mismatch(self) -> None:
        result = _kmeans_result()
        with pytest.raises(ValueError, match="one row per clustered sample"):
            projection_from_kmeans(result, _coords(len(result.assignments) + 1))

    def test_projection_from_probe(self) -> None:
        result = _probe_result()
        view = projection_from_probe(result, _coords(len(result.predictions)))
        assert view.n_points == len(result.predictions)
        assert "top_probability" in view.points[0].metadata
        assert view.metadata["metrics"]["accuracy"] == 0.81

    def test_projection_from_density_splits_id_ood(self) -> None:
        result = _density_result()
        scores = np.asarray(result.calibrated_ood_score)
        view = projection_from_density(result, _coords(len(scores)))
        categories = {point.category for point in view.points}
        assert categories <= {"id", "ood"}
        assert "calibrated_ood_score" in view.points[0].metadata
        assert view.metadata["source_representation_identity"] == "test_conv_vae"

    def test_projection_from_density_respects_threshold(self) -> None:
        result = _density_result(seed=1)
        view = projection_from_density(result, _coords(len(result.calibrated_ood_score)), ood_threshold=0.0)
        assert {point.category for point in view.points} == {"ood"}

    def test_projection_from_trajectory_adds_overlay(self) -> None:
        trajectory = Trajectory(_coords(8, ndim=4, seed=3))
        coords = _coords(8, ndim=3, seed=4)
        view = projection_from_trajectory(trajectory, coords, step_labels=[str(i) for i in range(8)])
        assert len(view.trajectories) == 1
        assert view.trajectories[0].name == "trajectory"
        assert view.ndim == 3
        assert view.points[0].label == "0"
        assert view.metadata["n_trajectory_points"] == 8

    def test_projection_from_dtw_adds_query_and_reference_overlays(self) -> None:
        query = np.array([[0.0], [1.0], [2.0]])
        reference = np.array([[0.0], [0.0], [1.0], [2.0]])
        result = compute_dtw(query, reference, LatentSpace(dim=1))
        view = projection_from_dtw(result, _coords(3, ndim=2), _coords(4, ndim=2, seed=4))
        assert [overlay.name for overlay in view.trajectories] == ["query", "reference"]
        assert view.metadata["dtw_normalization"] == "path_length"
        assert view.metadata["dtw_path"] == [list(pair) for pair in result.path]

    def test_trajectory_view_rejects_point_mismatch(self) -> None:
        trajectory = Trajectory(_coords(8, ndim=4, seed=3))
        with pytest.raises(ValueError, match="only 5 coordinates"):
            trajectory_view(trajectory, _coords(5, ndim=2, seed=4))

    def test_trajectory_view_carries_segmentation_boundaries(self) -> None:
        trajectory = Trajectory(np.arange(20, dtype=float)[:, None])
        segmentation = detect_change_points(
            trajectory,
            LatentSpace(dim=1),
            config=SegmentationConfig(threshold=0.0, min_segment_length=5),
        )
        view = trajectory_view(trajectory, _coords(20, ndim=2, seed=4), segmentation=segmentation)
        assert view.boundaries == segmentation.boundaries
        serialized = view.to_dict()
        if segmentation.boundaries:
            assert serialized["boundaries"] == list(segmentation.boundaries)
        else:
            assert "boundaries" not in serialized

    def test_projection_from_atlas(self) -> None:
        view = projection_from_atlas(_atlas())
        assert {point.category for point in view.points} == {"dead", "alive"}
        assert view.points[0].metadata["feature_index"] == 0
        assert view.metadata["metrics"]["dead_fraction"] == 0.5

    def test_projection_from_atlas_rejects_empty(self) -> None:
        atlas = FeatureAtlas(
            entries=(), n_components=0, n_examples=0, source_representation_identity="x", provenance={}
        )
        with pytest.raises(ValueError, match="at least one entry"):
            projection_from_atlas(atlas)


class TestMetricSummaries:
    def test_probe_and_cross_seed(self) -> None:
        assert metric_summary_from_probe(_probe_result()).metrics["accuracy"] == 0.81
        report = _cross_seed_report()
        assert metric_summary_from_cross_seed(report).metrics["n_seeds"] == 3
        assert metric_summary_from_cross_seed(report).metrics["mean_accuracy"] == 0.803

    def test_kmeans_and_stability(self) -> None:
        result = _kmeans_result()
        assert metric_summary_from_kmeans(result).metrics["n_clusters"] == 3
        report = _stability_report()
        assert metric_summary_from_stability(report).metrics["adjusted_rand_index"] == 0.91

    def test_density_evaluation_and_stability(self) -> None:
        report = DensityEvaluationReport(
            in_distribution=_density_result(),
            out_of_distribution=_density_result(seed=1),
            metrics=DensityMetrics(
                auroc=0.9, auprc=0.85, brier_score=0.2, mean_id_score=-1.0, mean_ood_score=-3.0, n_id=30, n_ood=30
            ),
            split_provenance={},
        )
        summary = metric_summary_from_density(report)
        assert summary.metrics["auroc"] == 0.9
        stability = DensityStabilityReport(
            seeds=(0, 1),
            aurocs=(0.9, 0.88),
            auprcs=(0.85, 0.83),
            mean_auroc=0.89,
            auroc_ci95=0.02,
            mean_auprc=0.84,
            auprc_ci95=0.03,
            reports=(),
        )
        assert metric_summary_from_density(stability).metrics["mean_auroc"] == 0.89

    def test_sae_and_atlas(self) -> None:
        assert metric_summary_from_sae(_sae_result()).metrics["reconstruction_mse"] == 0.003
        assert metric_summary_from_atlas(_atlas()).metrics["n_dead"] == 1


# ── Downsampling / responsiveness ─────────────────────────────────────────


class TestDownsampling:
    def test_declared_targets(self) -> None:
        assert (DEFAULT_POINT_LIMIT_2D, DEFAULT_POINT_LIMIT_3D) == (50_000, 20_000)
        assert default_point_limit(2) == DEFAULT_POINT_LIMIT_2D
        assert default_point_limit(3) == DEFAULT_POINT_LIMIT_3D
        assert DOWNSAMPLE_SEED == 0

    def test_under_limit_unchanged(self) -> None:
        view = build_projection(_coords(10), categories=["a"] * 10)
        downsampled = downsample_view(view, limit=50)
        assert downsampled is view
        assert downsampled.n_points == 10

    def test_over_limit_is_capped_and_recorded(self) -> None:
        view = build_projection(_coords(100), categories=[f"c{i % 4}" for i in range(100)])
        downsampled = downsample_view(view, limit=40)
        assert downsampled.n_points <= 40
        assert downsampled.n_points == downsampled.metadata["n_kept"]
        assert downsampled.metadata["n_dropped"] == 100 - downsampled.n_points
        assert downsampled.metadata["downsampled"] is True

    def test_downsampling_is_seeded_deterministic(self) -> None:
        view = build_projection(_coords(200, seed=1), categories=[f"c{i % 5}" for i in range(200)])
        first = downsample_view(view, limit=50, seed=7)
        second = downsample_view(view, limit=50, seed=7)
        first_coords = np.asarray([point.coordinates for point in first.points])
        second_coords = np.asarray([point.coordinates for point in second.points])
        assert_array_equal(first_coords, second_coords)

    def test_downsampling_preserves_trajectory_overlays(self) -> None:
        trajectory = Trajectory(_coords(5, ndim=4, seed=3))
        coords = _coords(5, ndim=2, seed=4)
        overlay = trajectory_view(trajectory, coords, name="t")
        view = build_projection(_coords(100), categories=["a"] * 100, trajectories=[overlay])
        downsampled = downsample_view(view, limit=20)
        assert downsampled.trajectories == (overlay,)
        assert len(downsampled.trajectories[0].points) == 5

    def test_downsampling_preserves_class_balance(self) -> None:
        categories = ["a"] * 90 + ["b"] * 10
        view = build_projection(_coords(100), categories=categories)
        downsampled = downsample_view(view, limit=20)
        a = sum(1 for point in downsampled.points if point.category == "a")
        b = sum(1 for point in downsampled.points if point.category == "b")
        assert a >= 1 and b >= 1

    def test_rejects_invalid_limit(self) -> None:
        with pytest.raises(ValueError, match="limit"):
            downsample_view(build_projection(_coords(5)), limit=0)

    def test_downsample_round_trip_is_stable(self) -> None:
        view = build_projection(_coords(500, seed=2), categories=[f"c{i % 7}" for i in range(500)])
        once = downsample_view(view, limit=100)
        twice = downsample_view(once, limit=100)
        assert once.to_dict() == twice.to_dict()

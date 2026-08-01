"""Offline tests for representation-bound Gaussian-mixture density."""

import numpy as np
import pytest

from latent_anything.config import ObjectSpec, build_from_config
from latent_anything.density import (
    GaussianMixtureDensity,
    GMMConfig,
    cross_seed_evaluation,
    mahalanobis_baseline,
)


def _datasets() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    train = rng.normal(0, 0.45, (120, 3))
    calibration = rng.normal(0, 0.45, (40, 3))
    in_distribution = rng.normal(0, 0.45, (40, 3))
    out_of_distribution = rng.normal(4, 0.45, (40, 3))
    return train, calibration, in_distribution, out_of_distribution


def test_gmm_fit_calibrate_and_score_is_typed_and_deterministic() -> None:
    train, calibration, in_distribution, _ = _datasets()
    config = GMMConfig(n_components=2, random_state=4)
    first = GaussianMixtureDensity(config).fit(train, source_representation_identity="vae@rev/layer=z")
    first.calibrate(calibration)
    second = GaussianMixtureDensity(config).fit(train, source_representation_identity="vae@rev/layer=z")
    second.calibrate(calibration)
    result = first.score(in_distribution, source_representation_identity="vae@rev/layer=z")
    assert result.responsibilities.shape == (40, 2)
    np.testing.assert_allclose(result.log_density, second.score(in_distribution).log_density)
    assert result.fit_provenance["n_features"] == 3
    snapshot = first.state_snapshot()
    assert snapshot["source_representation_identity"] == "vae@rev/layer=z"
    assert snapshot["n_calibration_samples"] == 40


def test_evaluation_reports_ood_metrics_and_split_provenance() -> None:
    train, calibration, in_distribution, out_of_distribution = _datasets()
    estimator = GaussianMixtureDensity(GMMConfig(n_components=2)).fit(
        train, source_representation_identity="transformer@rev/layer=6"
    )
    estimator.calibrate(calibration, provenance={"purpose": "id calibration"})
    report = estimator.evaluate(
        in_distribution,
        out_of_distribution,
        source_representation_identity="transformer@rev/layer=6",
        split_provenance={"fit": "train", "calibration": "id_calibration", "evaluation": "id_vs_constructed_ood"},
    )
    assert report.metrics.auroc > 0.95
    assert report.metrics.auprc > 0.95
    assert report.split_provenance["evaluation"] == "id_vs_constructed_ood"


def test_cross_seed_report_contains_uncertainty() -> None:
    train, calibration, in_distribution, out_of_distribution = _datasets()
    report = cross_seed_evaluation(
        train,
        calibration,
        in_distribution,
        out_of_distribution,
        source_representation_identity="vae@rev/z",
        seeds=[0, 1, 2],
    )
    assert report.seeds == (0, 1, 2)
    assert report.mean_auroc > 0.95
    assert report.auroc_ci95 >= 0.0


def test_rejects_unsupported_geometry_dimension_and_cross_space() -> None:
    train, calibration, _, _ = _datasets()
    with pytest.raises(ValueError, match="gaussian_set"):
        GaussianMixtureDensity().fit(train, source_representation_identity="x", geometry="gaussian_set")
    with pytest.raises(ValueError, match="more samples than dimensions"):
        mahalanobis_baseline(np.ones((3, 3)), np.ones((2, 3)))
    estimator = GaussianMixtureDensity().fit(train, source_representation_identity="space-a")
    estimator.calibrate(calibration)
    with pytest.raises(ValueError, match="cross-space"):
        estimator.score(train[:2], source_representation_identity="space-b")


def test_direct_and_config_registry_construction() -> None:
    direct = GaussianMixtureDensity(n_components=3)
    configured = build_from_config(
        ObjectSpec(kind="analysis", name="gaussian_mixture_density", params={"n_components": 3})
    )
    assert direct.config.n_components == 3
    assert isinstance(configured, GaussianMixtureDensity)

"""Comprehensive tests for the linear probe module.

Covers:
- Unit tests for config, result, and split helpers
- Fit / predict lifecycle
- Leakage guards (standardisation, stratification)
- Degenerate class handling
- Config construction via registry
- Control baselines
- Cross-seed stability
- Marked real-integration tests
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal
from pydantic import ValidationError

from latent_anything.probes import (
    ControlBaselines,
    CrossSeedReport,
    LinearProbe,
    LinearProbeConfig,
    LinearProbeResult,
    compute_controls,
    cross_seed_evaluation,
    evaluate_layers,
)

# ---------------------------------------------------------------------------
#  Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def binary_data() -> tuple[np.ndarray, np.ndarray]:
    """Well-separated binary classification data (80 samples, 5 features)."""
    rng = np.random.default_rng(42)
    labels = np.repeat([0, 1], 40)
    features = rng.normal(0, 1, (80, 5))
    features[:40] += 2.0  # class 0
    features[40:] -= 2.0  # class 1
    return features, labels


@pytest.fixture
def multiclass_data() -> tuple[np.ndarray, np.ndarray]:
    """Three-class data (60 samples, 4 features)."""
    rng = np.random.default_rng(42)
    labels = np.repeat([0, 1, 2], 20)
    features = rng.normal(0, 1, (60, 4))
    for i in range(3):
        features[i * 20 : (i + 1) * 20] += i * 1.5
    return features, labels


@pytest.fixture
def probe() -> LinearProbe:
    return LinearProbe()


# ---------------------------------------------------------------------------
#  LinearProbeConfig
# ---------------------------------------------------------------------------


class TestLinearProbeConfig:
    def test_default_config(self) -> None:
        cfg = LinearProbeConfig()
        assert cfg.C == 1.0
        assert cfg.solver == "lbfgs"
        assert cfg.test_size == 0.3
        assert cfg.random_state == 0
        assert cfg.standardize is True
        assert cfg.class_weight == "balanced"

    def test_invalid_c_raises(self) -> None:
        with pytest.raises(ValidationError):
            LinearProbeConfig(C=-1.0)

    def test_invalid_test_size_raises(self) -> None:
        with pytest.raises(ValidationError):
            LinearProbeConfig(test_size=0.0)

    def test_invalid_val_size_raises(self) -> None:
        with pytest.raises(ValidationError):
            LinearProbeConfig(val_size=-0.1)

    def test_config_serialization_roundtrip(self) -> None:
        cfg = LinearProbeConfig(C=0.5, random_state=42, test_size=0.2)
        data = cfg.model_dump()
        restored = LinearProbeConfig(**data)
        assert restored.C == 0.5
        assert restored.random_state == 42
        assert restored.test_size == 0.2


# ---------------------------------------------------------------------------
#  LinearProbeResult
# ---------------------------------------------------------------------------


class TestLinearProbeResult:
    def test_result_holds_expected_fields(self, binary_data: tuple[np.ndarray, np.ndarray]) -> None:
        features, labels = binary_data
        probe = LinearProbe()
        result = probe.fit(features, labels)
        assert isinstance(result, LinearProbeResult)
        assert isinstance(result.accuracy, float)
        assert 0.0 <= result.accuracy <= 1.0
        assert isinstance(result.val_accuracy, float)
        assert_array_equal(result.classes, np.array([0, 1]))
        assert result.predictions.shape == result.test_indices.sum()
        assert result.probabilities.shape == (result.test_indices.sum(), 2)
        assert result.coefficients.shape == (5,)  # binary → 1D
        assert result.intercept.shape == (1,)
        assert result.train_indices.dtype == bool
        assert result.test_indices.dtype == bool
        assert isinstance(result.config, LinearProbeConfig)

    def test_result_to_dict(self, binary_data: tuple[np.ndarray, np.ndarray]) -> None:
        features, labels = binary_data
        probe = LinearProbe()
        result = probe.fit(features, labels)
        d = result.to_dict()
        assert isinstance(d["accuracy"], float)
        assert isinstance(d["coefficients"], list)
        assert isinstance(d["config"], dict)
        assert d["config"]["C"] == 1.0


# ---------------------------------------------------------------------------
#  Stratified split (tested via probe.fit)
# ---------------------------------------------------------------------------


class TestStratifiedSplit:
    def test_private_split_import_remains_compatible(self, binary_data: tuple[np.ndarray, np.ndarray]) -> None:
        from latent_anything._probe_split import stratified_split as shared_split
        from latent_anything.probes import _stratified_split as facade_split  # type: ignore[reportPrivateUsage]

        _, labels = binary_data
        shared = shared_split(labels, test_size=0.3, val_size=0.1, random_state=7)
        facade = facade_split(labels, test_size=0.3, val_size=0.1, random_state=7)
        for shared_mask, facade_mask in zip(shared, facade, strict=True):
            assert_array_equal(shared_mask, facade_mask)

    def test_split_sums_to_full_dataset(self, binary_data: tuple[np.ndarray, np.ndarray]) -> None:
        features, labels = binary_data
        probe = LinearProbe()
        probe.fit(features, labels)
        total = probe.result.train_indices.sum() + probe.result.val_indices.sum() + probe.result.test_indices.sum()
        assert total == len(labels)

    def test_each_class_represented_in_train(self, multiclass_data: tuple[np.ndarray, np.ndarray]) -> None:
        features, labels = multiclass_data
        probe = LinearProbe()
        probe.fit(features, labels)
        train_y = labels[probe.result.train_indices]
        assert len(np.unique(train_y)) == 3

    def test_test_size_zero_val_size(self) -> None:
        """Val size of 0 means no validation split."""
        rng = np.random.default_rng(0)
        features = rng.normal(0, 1, (50, 3))
        labels = np.repeat([0, 1], 25)
        cfg = LinearProbeConfig(val_size=0.0, test_size=0.3)
        probe = LinearProbe(cfg)
        result = probe.fit(features, labels)
        assert result.val_indices.sum() == 0
        assert result.train_indices.sum() + result.test_indices.sum() == 50


# ---------------------------------------------------------------------------
#  Fit lifecycle
# ---------------------------------------------------------------------------


class TestLinearProbeFit:
    def test_fit_returns_result(self, probe: LinearProbe, binary_data: tuple[np.ndarray, np.ndarray]) -> None:
        features, labels = binary_data
        result = probe.fit(features, labels)
        assert isinstance(result, LinearProbeResult)
        assert probe.is_fitted

    def test_fit_multiclass(self, probe: LinearProbe, multiclass_data: tuple[np.ndarray, np.ndarray]) -> None:
        features, labels = multiclass_data
        result = probe.fit(features, labels)
        assert result.accuracy > 0.3  # should be above chance (33% for 3 classes)
        assert result.probabilities.shape == (result.test_indices.sum(), 3)

    def test_fit_with_provenance(self, probe: LinearProbe, binary_data: tuple[np.ndarray, np.ndarray]) -> None:
        features, labels = binary_data
        result = probe.fit(features, labels, provenance={"layer": 7, "model": "gpt2"})
        assert result.provenance["layer"] == 7
        assert result.provenance["model"] == "gpt2"

    def test_fit_is_idempotent(self, binary_data: tuple[np.ndarray, np.ndarray]) -> None:
        features, labels = binary_data
        probe = LinearProbe()
        result1 = probe.fit(features, labels)
        result2 = probe.fit(features, labels)
        # Same seed → same split → same accuracy
        assert result1.accuracy == result2.accuracy

    def test_val_accuracy_computed(self) -> None:
        rng = np.random.default_rng(0)
        features = rng.normal(0, 1, (200, 5))
        features[:100] += 2.0
        labels = np.repeat([0, 1], 100)
        cfg = LinearProbeConfig(val_size=0.2, test_size=0.3)
        probe = LinearProbe(cfg)
        result = probe.fit(features, labels)
        assert result.val_accuracy > 0.0
        assert result.val_indices.sum() > 0

    # ── Input validation ─────────────────────────────────────────

    def test_fit_raises_on_1d_features(self, probe: LinearProbe) -> None:
        with pytest.raises(ValueError, match="features must be 2D"):
            probe.fit(np.array([1.0, 2.0, 3.0]), np.array([0, 1, 0]))

    def test_fit_raises_on_2d_labels(self, probe: LinearProbe) -> None:
        with pytest.raises(ValueError, match="labels must be 1D"):
            probe.fit(np.ones((5, 2)), np.ones((5, 1)))

    def test_fit_raises_on_mismatched_samples(self, probe: LinearProbe) -> None:
        with pytest.raises(ValueError, match="same number of samples"):
            probe.fit(np.ones((10, 2)), np.ones(5))

    def test_fit_raises_on_single_class(self, probe: LinearProbe) -> None:
        with pytest.raises(ValueError, match="at least 2 classes"):
            probe.fit(np.ones((10, 2)), np.zeros(10))

    def test_fit_raises_on_one_sample_of_any_class(self, probe: LinearProbe) -> None:
        """One class with only 1 sample should fail."""
        labels = np.array([0, 1, 1, 1])
        with pytest.raises(ValueError, match="at least 2 samples"):
            probe.fit(np.ones((4, 2)), labels)


# ---------------------------------------------------------------------------
#  Predict (after fit)
# ---------------------------------------------------------------------------


class TestLinearProbePredict:
    def test_predict_returns_correct_shape(self, binary_data: tuple[np.ndarray, np.ndarray]) -> None:
        features, labels = binary_data
        probe = LinearProbe()
        probe.fit(features, labels)
        preds = probe.predict(features[:10])
        assert preds.shape == (10,)

    def test_predict_proba_returns_valid_probs(self, binary_data: tuple[np.ndarray, np.ndarray]) -> None:
        features, labels = binary_data
        probe = LinearProbe()
        probe.fit(features, labels)
        probs = probe.predict_proba(features[:10])
        assert probs.shape == (10, 2)
        assert np.all(probs >= 0) and np.all(probs <= 1)
        assert_array_almost_equal(probs.sum(axis=1), np.ones(10))

    def test_predict_raises_before_fit(self) -> None:
        probe = LinearProbe()
        with pytest.raises(RuntimeError, match="not been fitted"):
            probe.predict(np.ones((5, 3)))

    def test_predict_proba_raises_before_fit(self) -> None:
        probe = LinearProbe()
        with pytest.raises(RuntimeError, match="not been fitted"):
            probe.predict_proba(np.ones((5, 3)))

    def test_result_raises_before_fit(self) -> None:
        probe = LinearProbe()
        with pytest.raises(RuntimeError, match="not been fitted"):
            _ = probe.result


# ---------------------------------------------------------------------------
#  Leakage guards
# ---------------------------------------------------------------------------


class TestLeakageGuards:
    def test_standardization_fit_on_train_only(self) -> None:
        """Test that scaler statistics come only from training data."""
        rng = np.random.default_rng(0)
        features = rng.normal(0, 1, (100, 3))
        features[:50] += 5.0  # different mean for first half
        features[50:] -= 5.0
        labels = np.repeat([0, 1], 50)

        probe = LinearProbe()
        result = probe.fit(features, labels)

        # The stored feature means should be close to train data statistics
        train_x = features[result.train_indices]
        assert result.feature_means is not None
        assert_array_almost_equal(result.feature_means, train_x.mean(axis=0), decimal=1)

    def test_no_test_labels_in_train(self) -> None:
        """No test sample has the same index as a training sample."""
        rng = np.random.default_rng(0)
        features = rng.normal(0, 1, (100, 3))
        labels = np.repeat([0, 1], 50)

        probe = LinearProbe()
        result = probe.fit(features, labels)
        overlap = result.train_indices & result.test_indices
        assert overlap.sum() == 0
        assert result.train_indices.sum() + result.test_indices.sum() + result.val_indices.sum() == 100

    def test_no_shuffled_label_leakage(self) -> None:
        """Probe on shuffled labels should be near chance for well-separated data."""
        rng = np.random.default_rng(0)
        features = rng.normal(0, 1, (100, 5))
        features[:50] += 3.0
        labels = np.repeat([0, 1], 50)
        shuffled = rng.permutation(labels)

        probe = LinearProbe()
        real_result = probe.fit(features, labels)
        shuffled_result = probe.fit(features, shuffled)
        # Real probe should be notably higher than shuffled probe
        assert real_result.accuracy > shuffled_result.accuracy + 0.2


# ---------------------------------------------------------------------------
#  Degenerate classes
# ---------------------------------------------------------------------------


class TestDegenerateClasses:
    def test_too_few_samples_raises(self) -> None:
        probe = LinearProbe()
        with pytest.raises(ValueError, match="at least 2 samples"):
            probe.fit(np.ones((1, 3)), np.array([0]))

    def test_minimal_binary_works(self) -> None:
        """2 samples per class, 2 features — the smallest valid case."""
        features = np.array([[0.0, 0.0], [0.1, 0.1], [2.0, 2.0], [2.1, 2.1]])
        labels = np.array([0, 0, 1, 1])
        probe = LinearProbe(LinearProbeConfig(test_size=0.25, val_size=0.0, random_state=0))
        result = probe.fit(features, labels)
        assert result.accuracy >= 0.0  # just checking it doesn't crash

    def test_heavily_imbalanced(self) -> None:
        """90 % class 0, 10 % class 1."""
        rng = np.random.default_rng(0)
        features = rng.normal(0, 1, (100, 3))
        features[:10] += 3.0  # make minority separable
        labels = np.array([1] * 10 + [0] * 90)
        probe = LinearProbe(LinearProbeConfig(class_weight="balanced", random_state=0))
        result = probe.fit(features, labels)
        assert result.accuracy > 0.5


# ---------------------------------------------------------------------------
#  Config construction (registry)
# ---------------------------------------------------------------------------


class TestConfigConstruction:
    def test_build_from_config_class_directly(self) -> None:
        """LinearProbe should be constructable with a config."""
        cfg = LinearProbeConfig(C=0.1, random_state=7)
        probe = LinearProbe(cfg)
        assert probe.config.C == 0.1
        assert probe.config.random_state == 7

    def test_build_from_object_spec(self) -> None:
        """Build via ObjectSpec (registry-backed config)."""
        from latent_anything.config import ObjectSpec, build_from_config

        spec = ObjectSpec(
            kind="analysis",
            name="linear_probe",
        )
        probe = build_from_config(spec)
        assert isinstance(probe, LinearProbe)
        assert isinstance(probe.config, LinearProbeConfig)

    def test_build_from_dict(self) -> None:
        """Build via build_from_dict convenience."""
        from latent_anything.config import build_from_dict

        probe = build_from_dict({"kind": "analysis", "name": "linear_probe"})
        assert isinstance(probe, LinearProbe)

    def test_registry_entry_exists(self) -> None:
        from latent_anything.registry import GLOBAL_REGISTRY

        entry = GLOBAL_REGISTRY.lookup("linear_probe")
        assert entry.kind == "analysis"
        assert entry.factory is LinearProbe


# ---------------------------------------------------------------------------
#  Control baselines
# ---------------------------------------------------------------------------


class TestControlBaselines:
    def test_controls_shape(self, binary_data: tuple[np.ndarray, np.ndarray]) -> None:
        features, labels = binary_data
        probe = LinearProbe()
        result = probe.fit(features, labels)
        controls = compute_controls(
            features,
            labels,
            train_indices=result.train_indices,
            test_indices=result.test_indices,
            random_state=0,
        )
        assert isinstance(controls, ControlBaselines)
        assert 0.0 <= controls.majority_class <= 1.0
        assert 0.0 <= controls.shuffled_label <= 1.0
        assert 0.0 <= controls.raw_input <= 1.0

    def test_majority_class_is_chance_with_balanced_data(self, binary_data: tuple[np.ndarray, np.ndarray]) -> None:
        features, labels = binary_data
        probe = LinearProbe()
        result = probe.fit(features, labels)
        controls = compute_controls(
            features,
            labels,
            train_indices=result.train_indices,
            test_indices=result.test_indices,
            random_state=0,
        )
        # For balanced data, majority class is ~50 %
        assert controls.majority_class > 0.3

    def test_controls_integration(self) -> None:
        """Control baselines should all be lower than a real probe on real data."""
        rng = np.random.default_rng(0)
        features = rng.normal(0, 1, (100, 5))
        features[:50] += 3.0
        labels = np.repeat([0, 1], 50)

        probe = LinearProbe()
        result = probe.fit(features, labels)
        controls = compute_controls(
            features,
            labels,
            train_indices=result.train_indices,
            test_indices=result.test_indices,
            random_state=0,
        )

        assert result.accuracy > controls.shuffled_label
        assert result.accuracy > controls.majority_class
        assert result.accuracy >= controls.raw_input  # latents should be at least as good


# ---------------------------------------------------------------------------
#  Cross-seed stability
# ---------------------------------------------------------------------------


class TestCrossSeedEvaluation:
    def test_cross_seed_returns_report(self, binary_data: tuple[np.ndarray, np.ndarray]) -> None:
        features, labels = binary_data
        report = cross_seed_evaluation(features, labels, seeds=[0, 1, 2])
        assert isinstance(report, CrossSeedReport)
        assert report.n_seeds == 3
        assert len(report.accuracies) == 3
        assert isinstance(report.mean_accuracy, float)
        assert isinstance(report.ci95, float)
        assert report.min_accuracy <= report.mean_accuracy <= report.max_accuracy

    def test_cross_seed_with_n_seeds(self, binary_data: tuple[np.ndarray, np.ndarray]) -> None:
        features, labels = binary_data
        report = cross_seed_evaluation(features, labels, n_seeds=5)
        assert report.n_seeds == 5

    def test_cross_seed_results_accessible(self, binary_data: tuple[np.ndarray, np.ndarray]) -> None:
        features, labels = binary_data
        report = cross_seed_evaluation(features, labels, seeds=[0, 1])
        assert len(report.results) == 2
        assert all(isinstance(r, LinearProbeResult) for r in report.results)


# ---------------------------------------------------------------------------
#  Layer evaluation
# ---------------------------------------------------------------------------


class TestEvaluateLayers:
    def test_evaluate_layers(self, binary_data: tuple[np.ndarray, np.ndarray]) -> None:
        features, labels = binary_data
        layer_features: dict[str, np.ndarray] = {"layer_0": features, "layer_1": np.asarray(features + 0.1)}
        reports = evaluate_layers(layer_features, labels, seeds=[0, 1])  # type: ignore[reportArgumentType]
        assert len(reports) == 2
        assert all(isinstance(r, CrossSeedReport) for r in reports.values())
        assert "layer_0" in reports
        assert "layer_1" in reports


# ---------------------------------------------------------------------------
#  Real-integration tests (marked)
# ---------------------------------------------------------------------------


@pytest.mark.network
class TestRealIntegration:
    """Integration tests that download real models.

    These tests are marked ``network`` and are off by default.  Run with::

        uv run pytest tests/test_probes.py -m network -v
    """

    def test_on_vae_latents(self) -> None:
        """Evaluate probe on real VAE representations (Sprint 35)."""
        from latent_anything.integrations.diffusers_vae import DiffusersAutoencoderKLAdapter

        adapter = DiffusersAutoencoderKLAdapter("CompVis/stable-diffusion-v1-4", "7460a6f", latent_mode="mean")
        rng = np.random.default_rng(42)
        images = rng.uniform(-1, 1, (32, 3, 64, 64)).astype(np.float32)
        latents = adapter.encode(images)  # (32, 4, 64, 64)

        # Pool spatial dimensions for probe input
        n_samples = latents.shape[0]
        n_channels = latents.shape[1]
        latents_flat = latents.reshape(n_samples, n_channels, -1).mean(axis=2)  # (32, 4)

        # Create synthetic labels based on spatial variance
        spatial_var = latents.reshape(n_samples, n_channels, -1).var(axis=2).mean(axis=1)
        labels = (spatial_var > spatial_var.median()).astype(np.int64)

        probe = LinearProbe(LinearProbeConfig(random_state=0))
        result = probe.fit(latents_flat, labels)
        assert result.accuracy > 0.3  # above chance

    def test_on_transformer_hidden_states(self) -> None:
        """Evaluate probe on real transformer hidden states (Sprint 39)."""
        from latent_anything.integrations.transformer_lm import TransformerGenerationRequest, TransformerLMIntegration

        integration = TransformerLMIntegration()
        prompt = "The cat sat on the"
        request = TransformerGenerationRequest(prompt=prompt, max_length=8)
        gen_result = integration.generate(request)

        # Collect hidden states from last layer
        hidden_states = gen_result.hidden_states
        last_layer = max(hs.layer for hs in hidden_states)
        hs_last = [hs for hs in hidden_states if hs.layer == last_layer][0]
        seq_len = hs_last.values.shape[1]

        # Use position in sequence as label (first half vs second half)
        pos_labels = np.array([0 if i < seq_len // 2 else 1 for i in range(seq_len)])

        probe = LinearProbe(LinearProbeConfig(random_state=0, test_size=0.3, val_size=0.0))
        result = probe.fit(hs_last.values[0], pos_labels)
        assert result.accuracy > 0.3

    def test_on_transformer_multi_layer(self) -> None:
        """Cross-seed evaluation across multiple transformer layers."""
        from latent_anything.integrations.transformer_lm import TransformerGenerationRequest, TransformerLMIntegration

        integration = TransformerLMIntegration()
        request = TransformerGenerationRequest(prompt="The meaning of life is", max_length=8)
        gen_result = integration.generate(request)

        layer_features: dict[str | int, np.ndarray] = {}
        for hs in gen_result.hidden_states:
            if hs.layer in (0, 5, 11):  # early, mid, late layers
                # Use first token hidden states
                layer_features[f"layer_{hs.layer}"] = hs.values[0, :, :]

        # Create synthetic labels based on token position
        seq_len = gen_result.hidden_states[0].values.shape[1]
        labels = np.array([0 if i < seq_len // 2 else 1 for i in range(seq_len)])

        reports = evaluate_layers(layer_features, labels, seeds=[0, 1])
        assert len(reports) == 3
        # Hidden state dimensions vary per layer (all should be GPT2_HIDDEN_DIM)
        for _layer_name, report in reports.items():
            assert report.mean_accuracy > 0.3

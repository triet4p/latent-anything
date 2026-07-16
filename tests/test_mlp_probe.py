"""Comprehensive tests for the MLP probe module.

Covers:
- Unit tests for config, result
- Fit / predict lifecycle
- Architecture reporting
- Early stopping
- Determinism
- Degenerate class handling
- Config construction via registry
- Memorization test
- Linear vs nonlinear comparison
- Marked real-integration tests
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from pydantic import ValidationError

from latent_anything.mlp_probe import (
    MLPProbe,
    MLPProbeConfig,
    MLPProbeResult,
    NonlinearControls,
    ProbeComparison,
    compare_probes,
    nonlinear_memorization_test,
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
    features[:40] += 2.0
    features[40:] -= 2.0
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
def probe() -> MLPProbe:
    return MLPProbe()


# ---------------------------------------------------------------------------
#  MLPProbeConfig
# ---------------------------------------------------------------------------


class TestMLPProbeConfig:
    def test_default_config(self) -> None:
        cfg = MLPProbeConfig()
        assert cfg.hidden_sizes == [64]
        assert cfg.activation == "relu"
        assert cfg.max_epochs == 200
        assert cfg.early_stopping_patience == 10
        assert cfg.learning_rate == 1e-3
        assert cfg.weight_decay == 1e-4
        assert cfg.batch_size == 32

    def test_invalid_hidden_sizes_accepts_empty(self) -> None:
        cfg = MLPProbeConfig(hidden_sizes=[])
        assert cfg.hidden_sizes == []

    def test_invalid_activation_raises(self) -> None:
        with pytest.raises(ValidationError):
            MLPProbeConfig(activation="sigmoid")  # only relu and tanh allowed

    def test_config_serialization_roundtrip(self) -> None:
        cfg = MLPProbeConfig(hidden_sizes=[128, 64], learning_rate=1e-4, random_state=7)
        data = cfg.model_dump()
        restored = MLPProbeConfig(**data)
        assert restored.hidden_sizes == [128, 64]
        assert restored.learning_rate == 1e-4
        assert restored.random_state == 7


# ---------------------------------------------------------------------------
#  MLPProbeResult
# ---------------------------------------------------------------------------


class TestMLPProbeResult:
    def test_result_holds_expected_fields(self, binary_data: tuple[np.ndarray, np.ndarray]) -> None:
        features, labels = binary_data
        probe = MLPProbe()
        result = probe.fit(features, labels)
        assert isinstance(result, MLPProbeResult)
        assert isinstance(result.accuracy, float)
        assert 0.0 <= result.accuracy <= 1.0
        assert isinstance(result.val_accuracy, float)
        assert_array_equal(result.classes, np.array([0, 1]))
        assert result.predictions.shape == (result.test_indices.sum(),)
        assert result.probabilities.shape == (result.test_indices.sum(), 2)
        assert isinstance(result.n_epochs, int) and result.n_epochs > 0
        assert isinstance(result.stopped_early, bool)
        assert isinstance(result.architecture, dict)
        assert result.n_params > 0
        assert result.optimizer == "AdamW"
        assert isinstance(result.config, MLPProbeConfig)

    def test_result_to_dict(self, binary_data: tuple[np.ndarray, np.ndarray]) -> None:
        features, labels = binary_data
        probe = MLPProbe()
        result = probe.fit(features, labels)
        d = result.to_dict()
        assert isinstance(d["accuracy"], float)
        assert isinstance(d["n_params"], int)
        assert isinstance(d["config"], dict)


# ---------------------------------------------------------------------------
#  Fit lifecycle
# ---------------------------------------------------------------------------


class TestMLPProbeFit:
    def test_fit_returns_result(self, probe: MLPProbe, binary_data: tuple[np.ndarray, np.ndarray]) -> None:
        features, labels = binary_data
        result = probe.fit(features, labels)
        assert isinstance(result, MLPProbeResult)
        assert probe.is_fitted

    def test_fit_multiclass(self, probe: MLPProbe, multiclass_data: tuple[np.ndarray, np.ndarray]) -> None:
        features, labels = multiclass_data
        result = probe.fit(features, labels)
        assert result.accuracy > 0.3
        assert result.probabilities.shape == (result.test_indices.sum(), 3)

    def test_fit_with_provenance(self, probe: MLPProbe, binary_data: tuple[np.ndarray, np.ndarray]) -> None:
        features, labels = binary_data
        result = probe.fit(features, labels, provenance={"layer": 7, "model": "gpt2"})
        assert result.provenance["layer"] == 7
        assert result.provenance["model"] == "gpt2"

    def test_val_accuracy_computed(self) -> None:
        rng = np.random.default_rng(0)
        features = rng.normal(0, 1, (200, 5))
        features[:100] += 2.0
        labels = np.repeat([0, 1], 100)
        cfg = MLPProbeConfig(val_size=0.2, test_size=0.3, max_epochs=50)
        probe = MLPProbe(cfg)
        result = probe.fit(features, labels)
        assert result.val_accuracy > 0.0
        assert result.val_indices.sum() > 0

    # ── Input validation ─────────────────────────────────────────

    def test_fit_raises_on_1d_features(self, probe: MLPProbe) -> None:
        with pytest.raises(ValueError, match="features must be 2D"):
            probe.fit(np.array([1.0, 2.0, 3.0]), np.array([0, 1, 0]))

    def test_fit_raises_on_2d_labels(self, probe: MLPProbe) -> None:
        with pytest.raises(ValueError, match="labels must be 1D"):
            probe.fit(np.ones((5, 2)), np.ones((5, 1)))

    def test_fit_raises_on_mismatched_samples(self, probe: MLPProbe) -> None:
        with pytest.raises(ValueError, match="same number of samples"):
            probe.fit(np.ones((10, 2)), np.ones(5))

    def test_fit_raises_on_single_class(self, probe: MLPProbe) -> None:
        with pytest.raises(ValueError, match="at least 2 classes"):
            probe.fit(np.ones((10, 2)), np.zeros(10))


# ---------------------------------------------------------------------------
#  Architecture reporting
# ---------------------------------------------------------------------------


class TestArchitecture:
    def test_single_hidden_layer(self, binary_data: tuple[np.ndarray, np.ndarray]) -> None:
        features, labels = binary_data
        cfg = MLPProbeConfig(hidden_sizes=[32])
        probe = MLPProbe(cfg)
        result = probe.fit(features, labels)
        assert result.architecture["type"] == "MLP"
        assert result.architecture["hidden_sizes"] == [32]
        assert result.n_params > 0

    def test_two_hidden_layers(self, binary_data: tuple[np.ndarray, np.ndarray]) -> None:
        features, labels = binary_data
        cfg = MLPProbeConfig(hidden_sizes=[64, 32])
        probe = MLPProbe(cfg)
        result = probe.fit(features, labels)
        assert result.architecture["hidden_sizes"] == [64, 32]
        assert result.architecture["n_hidden_layers"] == 2


# ---------------------------------------------------------------------------
#  Early stopping
# ---------------------------------------------------------------------------


class TestEarlyStopping:
    def test_stops_before_max_epochs_on_easy_data(self, binary_data: tuple[np.ndarray, np.ndarray]) -> None:
        """On well-separated data, early stopping should trigger before max_epochs."""
        features, labels = binary_data
        cfg = MLPProbeConfig(max_epochs=500, early_stopping_patience=5, val_size=0.2)
        probe = MLPProbe(cfg)
        result = probe.fit(features, labels)
        assert result.stopped_early, f"Expected early stopping, ran {result.n_epochs}/{cfg.max_epochs} epochs"
        assert result.n_epochs < cfg.max_epochs

    def test_no_early_stopping_without_validation(self, binary_data: tuple[np.ndarray, np.ndarray]) -> None:
        """Without validation, early stopping cannot trigger."""
        features, labels = binary_data
        cfg = MLPProbeConfig(max_epochs=5, early_stopping_patience=2, val_size=0.0, random_state=0)
        probe = MLPProbe(cfg)
        result = probe.fit(features, labels)
        assert not result.stopped_early
        assert result.n_epochs == 5


# ---------------------------------------------------------------------------
#  Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_seed_same_results(self, binary_data: tuple[np.ndarray, np.ndarray]) -> None:
        features, labels = binary_data
        cfg = MLPProbeConfig(random_state=42, max_epochs=10, val_size=0.0)

        result1 = MLPProbe(cfg).fit(features, labels)
        result2 = MLPProbe(cfg).fit(features, labels)

        assert result1.accuracy == result2.accuracy
        assert_array_equal(result1.predictions, result2.predictions)

    def test_different_seed_different_split(self, binary_data: tuple[np.ndarray, np.ndarray]) -> None:
        """Different seeds produce different splits and possibly different accuracies."""
        features, labels = binary_data
        r1 = MLPProbeConfig(random_state=0, max_epochs=10, val_size=0.0)
        r2 = MLPProbeConfig(random_state=1, max_epochs=10, val_size=0.0)

        result1 = MLPProbe(r1).fit(features, labels)
        result2 = MLPProbe(r2).fit(features, labels)

        # Splits differ (different train/test masks)
        assert not np.array_equal(result1.train_indices, result2.train_indices)


# ---------------------------------------------------------------------------
#  Degenerate / edge cases
# ---------------------------------------------------------------------------


class TestDegenerate:
    def test_too_few_samples_raises(self) -> None:
        probe = MLPProbe()
        with pytest.raises(ValueError, match="at least 2 samples"):
            probe.fit(np.ones((1, 3)), np.array([0]))

    def test_minimal_binary_works(self) -> None:
        """2 samples per class with simple patterns."""
        features = np.array([[0.0, 0.0], [0.1, 0.1], [2.0, 2.0], [2.1, 2.1]])
        labels = np.array([0, 0, 1, 1])
        cfg = MLPProbeConfig(test_size=0.25, val_size=0.0, random_state=0, max_epochs=20)
        probe = MLPProbe(cfg)
        result = probe.fit(features, labels)
        assert result.accuracy >= 0.0  # doesn't crash

    def test_heavily_imbalanced(self) -> None:
        """90 % class 0, 10 % class 1, minority separable."""
        rng = np.random.default_rng(0)
        features = rng.normal(0, 1, (100, 3))
        features[:10] += 3.0
        labels = np.array([1] * 10 + [0] * 90)
        cfg = MLPProbeConfig(random_state=0, max_epochs=50)
        probe = MLPProbe(cfg)
        result = probe.fit(features, labels)
        assert result.accuracy > 0.5


# ---------------------------------------------------------------------------
#  Predict (not implemented)
# ---------------------------------------------------------------------------


class TestPredict:
    def test_predict_raises_not_implemented(self, binary_data: tuple[np.ndarray, np.ndarray]) -> None:
        features, labels = binary_data
        probe = MLPProbe()
        probe.fit(features, labels)
        with pytest.raises(NotImplementedError, match="not available"):
            probe.predict(features[:5])

    def test_result_raises_before_fit(self) -> None:
        probe = MLPProbe()
        with pytest.raises(RuntimeError, match="not been fitted"):
            _ = probe.result


# ---------------------------------------------------------------------------
#  Config construction (registry)
# ---------------------------------------------------------------------------


class TestConfigConstruction:
    def test_build_from_object_spec(self) -> None:
        from latent_anything.config import ObjectSpec, build_from_config

        spec = ObjectSpec(kind="analysis", name="mlp_probe")
        probe = build_from_config(spec)
        assert isinstance(probe, MLPProbe)
        assert isinstance(probe.config, MLPProbeConfig)

    def test_build_from_dict(self) -> None:
        from latent_anything.config import build_from_dict

        probe = build_from_dict({"kind": "analysis", "name": "mlp_probe"})
        assert isinstance(probe, MLPProbe)

    def test_registry_entry_exists(self) -> None:
        from latent_anything.registry import GLOBAL_REGISTRY

        entry = GLOBAL_REGISTRY.lookup("mlp_probe")
        assert entry.kind == "analysis"
        assert entry.factory is MLPProbe


# ---------------------------------------------------------------------------
#  Nonlinear controls / memorization test
# ---------------------------------------------------------------------------


class TestMemorizationTest:
    def test_memorization_returns_controls(self, binary_data: tuple[np.ndarray, np.ndarray]) -> None:
        features, labels = binary_data
        ctl = nonlinear_memorization_test(features, labels)
        assert isinstance(ctl, NonlinearControls)
        assert isinstance(ctl.shuffled_label_accuracy, float)
        assert isinstance(ctl.memorization_ratio, float)
        assert isinstance(ctl.passed_memorization_test, bool)
        assert ctl.chance_accuracy == 0.5

    def test_memorization_test_passes_on_clean_data(self, binary_data: tuple[np.ndarray, np.ndarray]) -> None:
        """Well-separated data with shuffled labels should be near chance (no memorization)."""
        features, labels = binary_data
        ctl = nonlinear_memorization_test(features, labels)
        # Shuffled labels should be near chance
        assert ctl.passed_memorization_test, f"Ratio {ctl.memorization_ratio} exceeds threshold"
        assert ctl.shuffled_label_accuracy < 0.7  # well below 2x chance

    def test_memorization_raises_on_single_class(self) -> None:
        with pytest.raises(ValueError, match="at least 2 classes"):
            nonlinear_memorization_test(np.ones((10, 2)), np.zeros(10))


# ---------------------------------------------------------------------------
#  Linear vs nonlinear comparison
# ---------------------------------------------------------------------------


class TestProbeComparison:
    def test_comparison_returns_results(self, binary_data: tuple[np.ndarray, np.ndarray]) -> None:
        features, labels = binary_data
        result = compare_probes(features, labels, seed=0)
        assert isinstance(result, ProbeComparison)
        assert isinstance(result.linear_accuracy, float)
        assert isinstance(result.nonlinear_accuracy, float)
        assert isinstance(result.gap, float)
        assert isinstance(result.classification, str)
        assert result.classification in ("linear-only", "nonlinear-only", "both", "unsupported", "memorization-prone")

    def test_comparison_both_above_chance(self, binary_data: tuple[np.ndarray, np.ndarray]) -> None:
        """Both probes should do well on well-separated data."""
        features, labels = binary_data
        result = compare_probes(features, labels, seed=0)
        assert result.linear_accuracy > 0.8
        assert result.nonlinear_accuracy > 0.8
        assert result.classification in ("both", "linear-only")


# ---------------------------------------------------------------------------
#  Real-integration tests (marked)
# ---------------------------------------------------------------------------


@pytest.mark.network
class TestRealIntegration:
    """Integration tests that download real models (off by default)."""

    def test_on_vae_latents(self) -> None:
        """Evaluate MLP probe on real VAE representations."""
        from latent_anything.integrations.diffusers_vae import DiffusersAutoencoderKLAdapter

        adapter = DiffusersAutoencoderKLAdapter("CompVis/stable-diffusion-v1-4", "7460a6f", latent_mode="mean")
        rng = np.random.default_rng(42)
        images = rng.uniform(-1, 1, (32, 3, 64, 64)).astype(np.float32)
        latents = adapter.encode(images)
        n_samples, n_channels = latents.shape[0], latents.shape[1]
        latents_flat = latents.reshape(n_samples, n_channels, -1).mean(axis=2)

        spatial_var = latents.reshape(n_samples, n_channels, -1).var(axis=2).mean(axis=1)
        labels = (spatial_var > spatial_var.median()).astype(np.int64)

        cfg = MLPProbeConfig(random_state=0, max_epochs=30, val_size=0.1)
        probe = MLPProbe(cfg)
        result = probe.fit(latents_flat, labels)
        assert result.accuracy > 0.3

    def test_on_transformer_hidden_states(self) -> None:
        """Evaluate MLP probe on real transformer hidden states."""
        from latent_anything.integrations.transformer_lm import TransformerGenerationRequest, TransformerLMIntegration

        integration = TransformerLMIntegration()
        request = TransformerGenerationRequest(prompt="The cat sat on the", max_length=8)
        gen_result = integration.generate(request)

        hidden_states = gen_result.hidden_states
        last_layer = max(hs.layer for hs in hidden_states)
        hs_last = [hs for hs in hidden_states if hs.layer == last_layer][0]
        seq_len = hs_last.values.shape[1]

        pos_labels = np.array([0 if i < seq_len // 2 else 1 for i in range(seq_len)])

        cfg = MLPProbeConfig(random_state=0, test_size=0.3, val_size=0.0, max_epochs=20)
        probe = MLPProbe(cfg)
        result = probe.fit(hs_last.values[0], pos_labels)
        assert result.accuracy > 0.3

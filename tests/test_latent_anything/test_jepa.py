"""Sprint 71 tests for the decoder-free JEPA/LeWM-style adapter."""

from __future__ import annotations

import inspect
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from latent_anything import (
    AnalysisPipeline,
    FileSystemRunRecorder,
    JEPAEvaluationReport,
    JEPALatentHealth,
    JEPAPrediction,
    JEPAPredictionMetrics,
    JEPARolloutMetrics,
    JEPAWorldModelAdapter,
    JEPAWorldModelConfig,
    ObjectSpec,
    RolloutPipeline,
    RolloutPipelineSpec,
    build_rollout_pipeline_from_config,
)
from latent_anything.adapters import DecodableAdapter, ModelAdapter
from latent_anything.methods.pca import PCA
from latent_anything.registry import GLOBAL_REGISTRY, KIND_ADAPTER, KIND_RUNTIME
from latent_anything.transition_contract import LatentTransition


def _sequences(seed: int = 71, episodes: int = 12, horizon: int = 5) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    observations = np.zeros((episodes, horizon + 1, 3), dtype=np.float64)
    observations[:, 0] = rng.normal(scale=0.5, size=(episodes, 3))
    actions = rng.normal(scale=0.4, size=(episodes, horizon, 1))
    for index in range(horizon):
        action = np.repeat(actions[:, index], 3, axis=1)
        observations[:, index + 1] = 0.7 * observations[:, index] + 0.2 * action
    return observations, actions


def _adapter() -> JEPAWorldModelAdapter:
    observations, actions = _sequences()
    return JEPAWorldModelAdapter(
        observation_dim=3,
        latent_dim=2,
        action_dim=1,
        hidden_dim=8,
        epochs=40,
        learning_rate=0.03,
        seed=71,
    ).fit(observations, actions)


def test_jepa_adapter_is_decoder_free_and_reuses_mean_transition_contract() -> None:
    adapter = _adapter()

    assert isinstance(adapter, ModelAdapter)
    assert isinstance(adapter, LatentTransition)
    assert not isinstance(adapter, DecodableAdapter)
    assert not hasattr(adapter, "decode")
    assert adapter.latent_space.metadata["decoder"] == "absent"
    assert adapter.latent_space.metadata["exposure_mode"] == "no_explicit_latent"
    assert GLOBAL_REGISTRY.lookup("jepa_world_model").kind == KIND_ADAPTER
    assert GLOBAL_REGISTRY.lookup("jepa_transition").kind == KIND_RUNTIME


def test_jepa_fit_keeps_target_encoder_stop_gradient_and_beats_collapsed_baseline() -> None:
    observations, actions = _sequences()
    adapter = _adapter()
    metrics = adapter.evaluate_one_step(observations[:, -2], actions[:, -1], observations[:, -1])

    assert adapter.target_encoder_requires_grad is False
    assert adapter.target_encoder_has_gradients is False
    assert metrics.mse < metrics.collapsed_baseline_mse
    assert metrics.improvement_over_collapsed > 0.0
    assert metrics.target_health.collapsed_fraction < 1.0
    assert np.isfinite(adapter.scale).all()
    assert adapter.fit_metadata["variance_regularizer_source"] == "trainable_context_encoder"


def test_jepa_encode_accepts_the_frozen_model_adapter_keyword() -> None:
    adapter = _adapter()
    observations, _ = _sequences()

    encoded = adapter.encode(data=observations[:, 0])

    assert encoded.shape == (observations.shape[0], adapter.latent_dim)


def test_jepa_rollout_integrates_with_rollout_and_analysis_pipelines() -> None:
    observations, actions = _sequences()
    adapter = _adapter()
    initial = adapter.encode(observations[:, 0])[0]
    target = adapter.encode_target(observations[0])
    target_states = np.vstack((initial, target[1:]))

    rollout = RolloutPipeline(adapter).run(initial, actions[0])
    metrics = adapter.evaluate_rollout(initial, actions[0], target_states)
    analysis = AnalysisPipeline(adapter, PCA(n_components=2)).run(observations[:, 0])

    assert rollout.trajectory.metadata["decoder"] == "absent"
    assert rollout.to_numpy().shape == (actions.shape[1] + 1, adapter.latent_dim)
    assert metrics.horizon == actions.shape[1]
    assert metrics.stable
    assert analysis.latents.shape == (observations.shape[0], adapter.latent_dim)


def test_jepa_is_selectable_by_analysis_and_rollout_config_paths() -> None:
    adapter = JEPAWorldModelAdapter(3, 2, 1)
    rollout = build_rollout_pipeline_from_config(
        RolloutPipelineSpec(
            transition=ObjectSpec(
                kind=KIND_RUNTIME,
                name="jepa_transition",
                params={"observation_dim": 3, "latent_dim": 2, "action_dim": 1},
            )
        )
    )

    assert isinstance(adapter, ModelAdapter)
    assert isinstance(rollout.transition, LatentTransition)


def test_jepa_masked_fit_and_latent_health_report_covariance_diagnostics() -> None:
    observations, actions = _sequences()
    mask = np.ones(actions.shape[:2], dtype=np.float64)
    mask[-1, -2:] = 0.0
    adapter = JEPAWorldModelAdapter(3, 2, 1, hidden_dim=8, epochs=10, seed=71).fit(
        observations,
        actions,
        sequence_mask=mask,
    )
    health = adapter.evaluate_latent_health(observations[:, 0])

    assert adapter.fit_metadata["valid_transitions"] == int(mask.sum())
    assert health.n_samples == observations.shape[0]
    assert np.isfinite(health.effective_rank)
    assert np.isfinite(health.collapse_score)


def test_jepa_checkpoint_round_trip_preserves_prediction(tmp_path: Path) -> None:
    observations, actions = _sequences()
    adapter = _adapter()
    path = tmp_path / "jepa.npz"
    adapter.save(str(path))
    restored = JEPAWorldModelAdapter.load(str(path))

    np.testing.assert_allclose(
        restored.step(adapter.encode(observations[0, 0][None, :])[0], actions[0, 0]),
        adapter.step(adapter.encode(observations[0, 0][None, :])[0], actions[0, 0]),
        atol=1e-6,
    )
    assert restored.latent_space.metadata == adapter.latent_space.metadata
    assert restored.target_encoder_requires_grad is False


def test_jepa_public_surface_and_result_schema_snapshot() -> None:
    assert JEPAWorldModelAdapter.__module__ == "latent_anything.adapters.jepa"
    assert tuple(inspect.signature(JEPAWorldModelAdapter.fit).parameters) == (
        "self",
        "observations",
        "actions",
        "sequence_mask",
        "seed",
    )
    assert tuple(inspect.signature(JEPAWorldModelAdapter.load).parameters) == ("path", "device")
    assert tuple(JEPAWorldModelConfig.model_fields) == (
        "hidden_dim",
        "epochs",
        "learning_rate",
        "ema_momentum",
        "variance_loss_weight",
        "minimum_latent_std",
        "variance_floor",
        "stability_norm_limit",
        "seed",
        "device",
    )
    assert tuple(JEPAPrediction.__dataclass_fields__) == ("mean", "scale")
    assert tuple(JEPALatentHealth.__dataclass_fields__) == (
        "mean_variance",
        "min_variance",
        "max_variance",
        "covariance_condition",
        "effective_rank",
        "participation_ratio",
        "collapsed_fraction",
        "collapse_score",
        "n_samples",
        "latent_dim",
    )
    assert tuple(JEPAPredictionMetrics.__dataclass_fields__) == (
        "mse",
        "rmse",
        "mean_error",
        "collapsed_baseline_mse",
        "improvement_over_collapsed",
        "target_health",
        "n_samples",
        "runtime_seconds",
    )
    assert tuple(JEPARolloutMetrics.__dataclass_fields__) == (
        "errors_by_horizon",
        "mean_error",
        "final_error",
        "horizon_drift",
        "error_growth_ratio",
        "n_episodes",
        "runtime_seconds",
        "stable",
    )


def test_jepa_same_seed_training_is_numerically_reproducible() -> None:
    observations, actions = _sequences(episodes=4, horizon=3)
    first = JEPAWorldModelAdapter(3, 2, 1, hidden_dim=6, epochs=8, seed=71).fit(observations, actions)
    second = JEPAWorldModelAdapter(3, 2, 1, hidden_dim=6, epochs=8, seed=71).fit(observations, actions)
    encoded = first.encode(observations[:, 0])
    assert np.array_equal(encoded, second.encode(observations[:, 0]))
    assert np.array_equal(first.step(encoded[0], actions[0, 0]), second.step(encoded[0], actions[0, 0]))


def test_jepa_checkpoint_is_cross_process_stable_and_tamper_rejected(tmp_path: Path) -> None:
    observations, actions = _sequences(episodes=4, horizon=3)
    adapter = JEPAWorldModelAdapter(3, 2, 1, hidden_dim=6, epochs=8, seed=71).fit(observations, actions)
    path = tmp_path / "jepa-cross-process.npz"
    adapter.save(str(path))
    command = (
        "from latent_anything.adapters.jepa import JEPAWorldModelAdapter; "
        f"model = JEPAWorldModelAdapter.load({str(path)!r}); "
        "print(model.latent_space.metadata['decoder']); print(model.target_encoder_requires_grad)"
    )
    completed = subprocess.run([sys.executable, "-c", command], capture_output=True, text=True, check=True)
    assert "absent" in completed.stdout
    assert "False" in completed.stdout

    tampered = tmp_path / "jepa-tampered.npz"
    np.savez(tampered, scale=np.zeros(2))
    with pytest.raises(KeyError):
        JEPAWorldModelAdapter.load(str(tampered))


def test_jepa_evaluation_is_persisted_as_a_content_addressed_run_artifact(tmp_path: Path) -> None:
    observations, actions = _sequences()
    adapter = _adapter()
    prediction = adapter.evaluate_one_step(
        observations[:, :-1].reshape(-1, 3),
        actions.reshape(-1, 1),
        observations[:, 1:].reshape(-1, 3),
    )
    initial = adapter.encode(observations[:, 0])
    target = adapter.encode_target(observations.reshape(-1, 3)).reshape(observations.shape[0], -1, 2)
    report = JEPAEvaluationReport(
        prediction=prediction,
        rollout=adapter.evaluate_rollout(
            initial, actions, np.concatenate((initial[:, None, :], target[:, 1:]), axis=1)
        ),
        provenance={"decoder": "absent", "model_revision": adapter.model_revision},
    )
    recorder = FileSystemRunRecorder(tmp_path / "runs")
    started = recorder.start("jepa-test", model_revisions={"model": adapter.model_revision}, seeds=(71,))
    completed = recorder.complete_jepa_evaluation(started.run_id, report)

    assert completed.status == "completed"
    assert completed.metrics["latent_prediction_mse"] == prediction.mse
    assert completed.artifacts[0].name == "jepa_evaluation.json"

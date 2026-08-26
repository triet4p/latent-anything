"""Tests for the first concrete deterministic latent transition."""

from __future__ import annotations

import inspect
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from numpy.testing import assert_allclose

from latent_anything import (
    DeterministicLatentTransition,
    GaussianPrediction,
    LatentSpace,
    LatentTransition,
    OneStepMetrics,
    RSSMLatentTransition,
    RSSMOneStepMetrics,
    RSSMPrediction,
    RSSMRollout,
    RSSMRolloutMetrics,
    RSSMTransitionConfig,
    StochasticGaussianLatentTransition,
    StochasticRollout,
    Trajectory,
)


def _transition(state_dim: int = 2, action_dim: int = 1) -> DeterministicLatentTransition:
    return DeterministicLatentTransition(
        LatentSpace(state_dim, source_model="synthetic-linear"),
        action_dim,
        source_space_identity="synthetic-linear-v1",
    )


def test_identity_dynamics_fit_and_rollout() -> None:
    states = np.array([[0.0, 1.0], [1.0, -2.0], [2.5, 3.0], [-1.0, 0.5]])
    actions = np.zeros((len(states), 1))
    model = _transition().fit(states, actions, states)

    assert_allclose(model.step(np.array([4.0, -3.0]), np.array([0.0])), [4.0, -3.0], atol=1e-7)
    result = model.rollout(np.array([4.0, -3.0]), np.zeros((3, 1)), metadata={"case": "identity"})
    assert isinstance(result, Trajectory)
    assert_allclose(result.to_numpy(), np.tile([4.0, -3.0], (4, 1)), atol=1e-7)
    assert result.metadata["source_space_identity"] == "synthetic-linear-v1"
    assert result.metadata["rollout_horizon"] == 3


def test_action_conditioned_linear_system() -> None:
    rng = np.random.default_rng(63)
    states = rng.normal(size=(80, 2))
    actions = rng.normal(size=(80, 1))
    matrix = np.array([[0.9, 0.1], [-0.2, 0.8]])
    control = np.array([[0.4], [-0.6]])
    next_states = states @ matrix.T + actions @ control.T
    model = _transition().fit(states, actions, next_states, training_horizon=1)

    test_state = np.array([0.3, -1.1])
    test_action = np.array([0.7])
    assert_allclose(
        model.step(test_state, test_action),
        matrix @ test_state + control[:, 0] * test_action[0],
        atol=1e-7,
    )
    metrics = model.evaluate_one_step(states, actions, next_states)
    assert metrics.rmse < 1e-6
    assert model.fit_metadata["training_horizon"] == 1


def test_rollout_error_is_reported_by_horizon() -> None:
    states = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]])
    actions = np.ones((len(states), 1))
    model = DeterministicLatentTransition(LatentSpace(1), 1, ridge=0).fit(states, actions, states + 1)
    rollout_actions = np.ones((4, 1))
    target = np.arange(5, dtype=np.float64).reshape(-1, 1)
    metrics = model.evaluate_rollout(np.array([0.0]), rollout_actions, target)

    assert metrics.horizon == 4
    assert len(metrics.errors_by_horizon) == 4
    assert metrics.runtime_seconds >= 0.0
    assert metrics.stable


@pytest.mark.parametrize(
    ("space", "action_dim", "match"),
    [
        (LatentSpace(2, geometry="unit_norm"), 1, "flat Euclidean"),
        (LatentSpace(2), 0, "action_dim"),
    ],
)
def test_constructor_validation(space: LatentSpace, action_dim: int, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        DeterministicLatentTransition(space, action_dim)


def test_shape_validation_and_unfitted_guard() -> None:
    model = _transition()
    with pytest.raises(RuntimeError, match="fitted"):
        model.step(np.zeros(2), np.zeros(1))
    with pytest.raises(ValueError, match="shape"):
        model.fit(np.zeros((2, 3)), np.zeros((2, 1)), np.zeros((2, 2)))
    with pytest.raises(ValueError, match="same number"):
        model.fit(np.zeros((2, 2)), np.zeros((3, 1)), np.zeros((2, 2)))
    model.fit(np.zeros((2, 2)), np.zeros((2, 1)), np.zeros((2, 2)))
    with pytest.raises(ValueError, match="shape"):
        model.step(np.zeros(3), np.zeros(1))


def test_fit_and_rollout_are_repeatable() -> None:
    rng = np.random.default_rng(63)
    states = rng.normal(size=(30, 2))
    actions = rng.normal(size=(30, 1))
    next_states = states + 0.2 * states + actions @ np.array([[0.3, -0.4]])
    action_sequence = rng.normal(size=(6, 1))
    first = _transition().fit(states, actions, next_states).rollout(states[0], action_sequence)
    second = _transition().fit(states, actions, next_states).rollout(states[0], action_sequence)
    assert_allclose(first.to_numpy(), second.to_numpy())


def _stochastic_transition(
    state_dim: int = 2, action_dim: int = 1, *, variance_floor: float = 1e-8
) -> StochasticGaussianLatentTransition:
    return StochasticGaussianLatentTransition(
        LatentSpace(state_dim, source_model="synthetic-stochastic"),
        action_dim,
        source_space_identity="synthetic-stochastic-v1",
        variance_floor=variance_floor,
        ridge=1e-10,
    )


def _noisy_samples(seed: int = 640) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    states = rng.normal(size=(500, 2))
    actions = rng.normal(size=(500, 1))
    mean = states @ np.array([[0.88, 0.06], [-0.12, 0.91]]) + actions @ np.array([[0.25, -0.35]])
    next_states = mean + rng.normal(scale=[0.08, 0.18], size=mean.shape)
    return states, actions, next_states


def test_stochastic_prediction_exposes_positive_scale_and_covariance() -> None:
    states, actions, next_states = _noisy_samples()
    model = _stochastic_transition().fit(states, actions, next_states)

    prediction = model.predict(np.array([0.2, -0.4]), np.array([0.7]))
    assert isinstance(prediction, GaussianPrediction)
    assert prediction.mean.shape == (2,)
    assert np.all(prediction.scale > 0)
    assert_allclose(prediction.covariance, np.diag(prediction.variance))
    assert model.fit_metadata["distribution_family"] == "diagonal_gaussian"
    assert model.fit_metadata["covariance_parameterization"] == "diagonal_scale"


def test_stochastic_sampling_is_seeded_and_keeps_distribution_explicit() -> None:
    states, actions, next_states = _noisy_samples()
    model = _stochastic_transition().fit(states, actions, next_states)
    prediction = model.predict(np.zeros(2), np.zeros(1))

    first = prediction.sample(seed=64, n_samples=12)
    second = prediction.sample(seed=64, n_samples=12)
    third = prediction.sample(seed=65, n_samples=12)
    assert_allclose(first, second)
    assert not np.array_equal(first, third)
    assert np.std(first, axis=0).mean() > 0
    assert np.isfinite(prediction.log_prob(np.zeros(2)))
    assert np.isfinite(prediction.log_prob(first)).all()


@given(st.integers(min_value=1, max_value=8))
def test_stochastic_sampling_shape_and_seed_property(n_samples: int) -> None:
    states, actions, next_states = _noisy_samples()
    model = _stochastic_transition().fit(states, actions, next_states)
    prediction = model.predict(np.zeros(2), np.zeros(1))

    samples = prediction.sample(seed=64, n_samples=n_samples)
    repeated = prediction.sample(seed=64, n_samples=n_samples)
    assert samples.shape == (n_samples, 2)
    assert_allclose(samples, repeated)


def test_stochastic_degenerate_noise_is_reproducibly_deterministic() -> None:
    states = np.array([[0.0], [1.0], [2.0], [3.0]])
    actions = np.ones((4, 1))
    next_states = states + 1.0
    model = StochasticGaussianLatentTransition(LatentSpace(1), 1, variance_floor=0.0, ridge=0).fit(
        states, actions, next_states
    )
    prediction = model.predict(np.array([4.0]), np.array([1.0]))
    assert_allclose(prediction.scale, [0.0], atol=1e-12)
    assert_allclose(prediction.sample(seed=10, n_samples=20), np.full((20, 1), prediction.mean))
    assert np.isfinite(prediction.log_prob(prediction.mean))


def test_stochastic_rollout_supports_batch_shaped_targets_and_uncertainty_bands() -> None:
    states, actions, next_states = _noisy_samples()
    model = _stochastic_transition().fit(states, actions, next_states)
    rng = np.random.default_rng(6401)
    initial_states = rng.normal(size=(4, 2))
    rollout_actions = rng.normal(size=(4, 5, 1))
    target_states = np.empty((4, 6, 2))
    target_states[:, 0] = initial_states
    for step in range(5):
        mean = np.vstack(
            [model.predict(target_states[index, step], rollout_actions[index, step]).mean for index in range(4)]
        )
        target_states[:, step + 1] = mean + rng.normal(scale=model.scale, size=(4, 2))

    rollout = model.rollout(initial_states[0], rollout_actions[0], n_samples=32, seed=64)
    assert isinstance(rollout, StochasticRollout)
    assert rollout.samples.shape == (32, 6, 2)
    assert rollout.lower.shape == rollout.upper.shape == (6, 2)
    assert np.all(rollout.lower <= rollout.upper)

    metrics = model.evaluate_rollout(initial_states, rollout_actions, target_states, n_samples=96, seed=64)
    assert metrics.horizon == 5
    assert len(metrics.nll_by_horizon) == 5
    assert len(metrics.coverage_by_horizon) == 5
    assert len(metrics.sample_diversity_by_horizon) == 5
    assert metrics.mean_sample_diversity > 0
    assert metrics.stable


def test_stochastic_one_step_metrics_report_nll_coverage_and_diversity() -> None:
    states, actions, next_states = _noisy_samples()
    model = _stochastic_transition().fit(states, actions, next_states)
    metrics = model.evaluate_one_step(states, actions, next_states, n_diversity_samples=32, seed=64)

    assert metrics.nll == pytest.approx(metrics.negative_log_likelihood)
    assert metrics.nll < 0.0
    assert 0.85 < metrics.coverage < 1.0
    assert metrics.interval_width > 0.0
    assert metrics.sample_diversity > 0.0
    assert metrics.mean_error < 0.4


def _rssm_sequences(seed: int = 65, episodes: int = 12, horizon: int = 7) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    states = np.empty((episodes, horizon + 1, 2), dtype=np.float64)
    actions = rng.normal(scale=0.4, size=(episodes, horizon, 1))
    states[:, 0] = rng.normal(scale=0.5, size=(episodes, 2))
    memory = np.zeros((episodes, 1), dtype=np.float64)
    for index in range(horizon):
        memory = 0.82 * memory + 0.25 * states[:, index, :1] + 0.18 * actions[:, index]
        states[:, index + 1, 0] = 0.76 * states[:, index, 0] + 0.35 * memory[:, 0] + 0.2 * actions[:, index, 0]
        states[:, index + 1, 1] = 0.9 * states[:, index, 1] - 0.1 * actions[:, index, 0]
    mask = np.ones((episodes, horizon), dtype=np.float64)
    mask[-1, -2:] = 0.0
    return states, actions, mask


def _rssm() -> RSSMLatentTransition:
    return RSSMLatentTransition(
        LatentSpace(2, source_model="synthetic-rssm"),
        1,
        source_space_identity="synthetic-rssm-v1",
        config=RSSMTransitionConfig(hidden_dim=6, epochs=35, learning_rate=0.03, seed=65),
    )


def test_rssm_fit_supports_masked_sequences_and_reports_temporal_metrics() -> None:
    states, actions, mask = _rssm_sequences()
    model = _rssm().fit(states, actions, sequence_mask=mask)
    one_step = model.evaluate_one_step(states, actions, states, sequence_mask=mask)

    assert model.is_fitted
    assert isinstance(one_step, RSSMOneStepMetrics)
    assert one_step.n_samples == int(mask.sum())
    assert np.isfinite(one_step.kl_divergence)
    assert 0.0 <= one_step.coverage <= 1.0
    assert model.fit_metadata["valid_transitions"] == int(mask.sum())


def test_rssm_reset_makes_stateful_execution_reproducible() -> None:
    states, actions, _ = _rssm_sequences()
    model = _rssm().fit(states, actions)
    first = model.step(states[0, 0], actions[0, 0])
    model.reset()
    second = model.step(states[0, 0], actions[0, 0])
    assert_allclose(first, second)
    prediction = model.predict(states[0, 1], actions[0, 1])
    assert isinstance(prediction, RSSMPrediction)
    assert prediction.deterministic_state.shape == model.hidden_shape


def test_rssm_runtime_state_and_all_valid_mask_are_parity_stable() -> None:
    states, actions, _ = _rssm_sequences(episodes=4, horizon=4)
    full_mask = np.ones(actions.shape[:2], dtype=np.float64)
    unmasked = _rssm().fit(states, actions, seed=65)
    masked = _rssm().fit(states, actions, sequence_mask=full_mask, seed=65)
    assert_allclose(
        unmasked.step(states[0, 0], actions[0, 0]),
        masked.step(states[0, 0], actions[0, 0]),
        atol=1e-12,
    )
    masked.reset()
    prediction = masked.predict(states[0, 0], actions[0, 0])
    rollout = masked.rollout(states[0, 0], actions[0], n_samples=8, seed=65)
    assert_allclose(prediction.mean, unmasked.mean_rollout(states[0, 0], actions[:1, 0]).to_numpy()[1], atol=1e-12)
    assert_allclose(masked.hidden_state, np.mean(rollout.deterministic_states[:, -1], axis=0), atol=1e-12)


def test_rssm_failed_step_does_not_mutate_recurrent_state() -> None:
    states, actions, _ = _rssm_sequences(episodes=2, horizon=3)
    model = _rssm().fit(states, actions)
    model.step(states[0, 0], actions[0, 0])
    before = model.hidden_state
    with pytest.raises(ValueError, match="state must have shape"):
        model.step(np.zeros(3), actions[0, 0])
    assert_allclose(model.hidden_state, before)


def test_rssm_rollout_is_seeded_and_keeps_deterministic_state() -> None:
    states, actions, _ = _rssm_sequences()
    model = _rssm().fit(states, actions)
    first = model.rollout(states[0, 0], actions[0], n_samples=16, seed=65)
    second = model.rollout(states[0, 0], actions[0], n_samples=16, seed=65)
    assert isinstance(first, RSSMRollout)
    assert_allclose(first.samples, second.samples)
    assert first.deterministic_states.shape == (16, actions.shape[1] + 1, model.config.hidden_dim)
    assert first.metadata["stateful"] is True


def test_rssm_rollout_evaluation_accepts_variable_lengths() -> None:
    states, actions, mask = _rssm_sequences()
    model = _rssm().fit(states, actions, sequence_mask=mask)
    metrics = model.evaluate_rollout(states[:, 0], actions, states, sequence_mask=mask, n_samples=16, seed=65)
    assert isinstance(metrics, RSSMRolloutMetrics)
    assert metrics.horizon == actions.shape[1]
    assert len(metrics.kl_by_horizon) == metrics.horizon
    assert 0.0 <= metrics.mean_coverage <= 1.0


def test_rssm_checkpoint_round_trip_resets_in_flight_state(tmp_path: Path) -> None:
    states, actions, _ = _rssm_sequences()
    model = _rssm().fit(states, actions)
    model.step(states[0, 0], actions[0, 0])
    path = tmp_path / "rssm.npz"
    model.save(path)
    restored = RSSMLatentTransition.load(path)
    assert restored.hidden_state.shape == (restored.config.hidden_dim,)
    assert_allclose(
        restored.step(states[0, 0], actions[0, 0]),
        model.mean_rollout(states[0, 0], actions[:1, 0]).to_numpy()[1],
        atol=1e-5,
    )
    assert restored.to_config().model_dump() == model.to_config().model_dump()


def test_rssm_checkpoint_round_trip_is_cross_process_stable(tmp_path: Path) -> None:
    states, actions, _ = _rssm_sequences()
    model = _rssm().fit(states, actions)
    path = tmp_path / "rssm-cross-process.npz"
    model.save(path)
    command = (
        "from pathlib import Path; "
        "from latent_anything.rssm import RSSMLatentTransition; "
        f"model = RSSMLatentTransition.load(Path({str(path)!r})); "
        "print(model.hidden_state.shape); print(model.to_config().model_dump_json())"
    )
    completed = subprocess.run([sys.executable, "-c", command], capture_output=True, text=True, check=True)
    assert "(6,)" in completed.stdout
    assert '"hidden_dim":6' in completed.stdout


def test_rssm_public_surface_state_and_result_schema_snapshot() -> None:
    assert RSSMLatentTransition.__module__ == "latent_anything.rssm"
    assert tuple(inspect.signature(RSSMLatentTransition.fit).parameters) == (
        "self",
        "states",
        "actions",
        "sequence_mask",
        "seed",
    )
    assert tuple(inspect.signature(RSSMLatentTransition.load).parameters) == ("path", "device")
    assert tuple(RSSMTransitionConfig.model_fields) == (
        "hidden_dim",
        "epochs",
        "learning_rate",
        "variance_floor",
        "posterior_scale_factor",
        "stability_norm_limit",
        "seed",
        "device",
    )
    assert tuple(RSSMPrediction.__dataclass_fields__) == ("mean", "scale", "deterministic_state")
    assert tuple(RSSMRollout.__dataclass_fields__) == ("samples", "deterministic_states", "interval_level", "metadata")
    assert tuple(RSSMOneStepMetrics.__dataclass_fields__) == (
        "mse",
        "rmse",
        "negative_log_likelihood",
        "kl_divergence",
        "coverage",
        "mean_error",
        "n_samples",
        "runtime_seconds",
    )
    assert tuple(RSSMRolloutMetrics.__dataclass_fields__) == (
        "errors_by_horizon",
        "kl_by_horizon",
        "coverage_by_horizon",
        "mean_error",
        "final_error",
        "mean_kl",
        "mean_coverage",
        "runtime_seconds",
        "stable",
    )


def test_rssm_rejects_nonbinary_masks_and_tampered_checkpoint(tmp_path: Path) -> None:
    states, actions, _ = _rssm_sequences(episodes=2, horizon=3)
    model = _rssm().fit(states, actions)
    with pytest.raises(ValueError, match="sequence_mask"):
        model.fit(states, actions, sequence_mask=np.full((2, 3), 0.5))

    tampered = tmp_path / "rssm-tampered.npz"
    np.savez(tampered, recurrent_weights=np.zeros((1, 1)))
    with pytest.raises(KeyError):
        RSSMLatentTransition.load(tampered)


def test_transition_public_surface_and_result_schema_snapshot() -> None:
    assert DeterministicLatentTransition.__module__ == "latent_anything.transition"
    assert StochasticGaussianLatentTransition.__module__ == "latent_anything.transition"
    assert tuple(inspect.signature(DeterministicLatentTransition.fit).parameters) == (
        "self",
        "states",
        "actions",
        "next_states",
        "training_horizon",
        "source_space_identity",
    )
    assert tuple(inspect.signature(StochasticGaussianLatentTransition.rollout).parameters) == (
        "self",
        "initial_state",
        "actions",
        "n_samples",
        "seed",
        "rng",
        "interval_level",
        "metadata",
    )
    assert tuple(GaussianPrediction.__dataclass_fields__) == ("mean", "scale")
    assert tuple(StochasticRollout.__dataclass_fields__) == ("samples", "interval_level", "metadata")
    assert tuple(OneStepMetrics.__dataclass_fields__) == (
        "mse",
        "rmse",
        "max_error",
        "n_samples",
        "runtime_seconds",
    )


def test_transition_facade_import_order_and_pickle_identity(tmp_path: Path) -> None:
    value_path = tmp_path / "prediction.pkl"
    value_path.write_bytes(pickle.dumps(GaussianPrediction(np.array([1.0, 2.0]), np.array([0.1, 0.2]))))
    load_code = (
        "import pickle; "
        f"value = pickle.loads(open({str(value_path)!r}, 'rb').read()); "
        "print(type(value).__module__, type(value).__name__, value.distribution_family)"
    )
    loaded = subprocess.run([sys.executable, "-c", load_code], capture_output=True, text=True, check=True)
    assert loaded.stdout.strip() == "latent_anything.transition GaussianPrediction diagonal_gaussian"

    import_orders = (
        "import latent_anything.transition; import latent_anything._transition_types",
        "import latent_anything._transition_types; import latent_anything.transition",
    )
    for imports in import_orders:
        code = (
            f"{imports}; from latent_anything.transition import GaussianPrediction; "
            "print(GaussianPrediction.__module__)"
        )
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
        assert result.stdout.strip() == "latent_anything.transition"


def test_three_transition_instances_satisfy_only_the_mean_contract() -> None:
    states, actions, _ = _rssm_sequences()
    rssm = _rssm().fit(states, actions)
    deterministic = _transition().fit(states[:, 0], actions[:, 0], states[:, 1])
    stochastic = _stochastic_transition().fit(states[:, 0], actions[:, 0], states[:, 1])
    assert isinstance(deterministic, LatentTransition)
    assert isinstance(stochastic, LatentTransition)
    assert isinstance(rssm, LatentTransition)

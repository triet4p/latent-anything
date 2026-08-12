from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from latent_anything import (
    FileSystemRunRecorder,
    LatentSpace,
    LinearRewardScorer,
    MonteCarloValueEstimator,
    ObjectSpec,
    RewardValueEvaluationSpec,
    RewardValueEvaluator,
    RolloutPipelineSpec,
    Trajectory,
    build_rollout_pipeline_from_config,
    compute_discounted_returns,
)
from latent_anything.registry import KIND_RUNTIME, Registry
from latent_anything.reward_value import HoldoutEvaluation
from latent_anything.transition import DeterministicLatentTransition


def _dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    states = np.asarray(
        [
            [[0.0], [0.0]],
            [[1.0], [1.0]],
            [[2.0], [2.0]],
            [[3.0], [3.0]],
        ]
    )
    actions = np.zeros((4, 1, 1), dtype=np.float64)
    rewards = states[:, :-1, 0] + 1.0
    return states, actions, rewards


def _fitted_evaluator() -> RewardValueEvaluator:
    states, actions, rewards = _dataset()
    reward_scorer = LinearRewardScorer(1, 1, source_space_identity="analytic-mdp")
    reward_scorer.fit(
        states[:, :-1].reshape(-1, 1),
        actions.reshape(-1, 1),
        rewards.reshape(-1),
        policy_id="fixed-zero",
        data_distribution="analytic-state-grid",
    )
    estimator = MonteCarloValueEstimator(
        1,
        discount=0.5,
        horizon=1,
        policy_id="fixed-zero",
        data_distribution="analytic-state-grid",
    )
    estimator.fit_trajectories(states, rewards)
    return RewardValueEvaluator(reward_scorer, estimator)


def test_discounted_returns_honor_terminal_and_padding() -> None:
    rewards = np.asarray([[1.0, 2.0, 3.0, 99.0], [4.0, 5.0, 6.0, 7.0]])
    masks = np.asarray([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=bool)
    terminals = np.asarray([[0, 1, 0, 0], [0, 0, 0, 0]], dtype=bool)
    returns = compute_discounted_returns(rewards, discount=0.5, masks=masks, terminals=terminals)
    np.testing.assert_allclose(returns, [[2.0, 2.0, 3.0, 0.0], [6.5, 5.0, 0.0, 0.0]])


def test_holdout_evaluation_reports_calibration_and_bellman_consistency() -> None:
    states, actions, rewards = _dataset()
    evaluator = _fitted_evaluator()
    result = evaluator.evaluate_holdout(states, actions, rewards)
    assert isinstance(result, HoldoutEvaluation)
    assert result.diagnostics.reward_rmse < 1e-8
    assert result.diagnostics.value_calibration.rmse < 1e-8
    assert result.diagnostics.bellman_residual_rmse < 1e-8
    assert result.provenance["discount"] == 0.5


def test_evaluation_result_contains_per_step_scores_and_real_imagined_bias() -> None:
    evaluator = _fitted_evaluator()
    actions = np.zeros((1, 1), dtype=np.float64)
    real = Trajectory(np.zeros((2, 1)), metadata={"state_source": "observed", "source_space_identity": "analytic-mdp"})
    imagined = Trajectory(
        np.ones((2, 1)), metadata={"state_source": "predicted", "source_space_identity": "analytic-mdp"}
    )
    real_result = evaluator.evaluate(real, actions, source="real")
    assert real_result.rewards.shape == (1,)
    assert real_result.returns.shape == (1,)
    assert real_result.values.shape == (1,)
    assert real_result.valid_steps == 1
    comparison = evaluator.compare_real_imagined(real, imagined, actions)
    assert comparison.reward_mae > 0.0
    assert comparison.valid_steps == 1


def test_evaluator_rejects_trajectory_from_another_source_space() -> None:
    evaluator = _fitted_evaluator()
    trajectory = Trajectory(np.zeros((2, 1)), metadata={"source_space_identity": "wrong-space"})

    with pytest.raises(ValueError, match="source_space_identity"):
        evaluator.evaluate(trajectory, np.zeros((1, 1)))


def test_rollout_config_builds_reward_value_evaluator_and_scores_imagination() -> None:
    evaluator = _fitted_evaluator()
    transition = DeterministicLatentTransition(LatentSpace(1, source_model="analytic-mdp"), 1).fit(
        np.zeros((2, 1)), np.zeros((2, 1)), np.zeros((2, 1))
    )
    registry = Registry("reward-value")
    registry.register(KIND_RUNTIME, "transition", lambda: transition)
    registry.register(KIND_RUNTIME, "reward", lambda: evaluator.reward_scorer)
    registry.register(KIND_RUNTIME, "value", lambda: evaluator.value_estimator)
    spec = RolloutPipelineSpec(
        transition=ObjectSpec(kind=KIND_RUNTIME, name="transition"),
        reward_value=RewardValueEvaluationSpec(
            reward_scorer=ObjectSpec(kind=KIND_RUNTIME, name="reward"),
            value_estimator=ObjectSpec(kind=KIND_RUNTIME, name="value"),
        ),
    )
    pipeline = build_rollout_pipeline_from_config(spec, registry=registry)
    result = pipeline.run(np.zeros(1), np.zeros((1, 1)))
    assert result.evaluation is not None
    assert result.evaluation.source == "imagined"


def test_run_recorder_persists_reward_value_metrics_and_artifact(tmp_path: Path) -> None:
    evaluator = _fitted_evaluator()
    states, actions, rewards = _dataset()
    result = evaluator.evaluate_holdout(states, actions, rewards)
    recorder = FileSystemRunRecorder(tmp_path)
    started = recorder.start("reward-value", config={"discount": 0.5})
    completed = recorder.complete_evaluation(started.run_id, result)
    assert completed.status == "completed"
    assert completed.metrics["value_rmse"] < 1e-8
    assert len(completed.artifacts) == 1
    assert recorder.read_artifact(completed.artifacts[0])

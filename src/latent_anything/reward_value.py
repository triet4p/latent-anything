"""Reward scoring and value diagnostics for real and imagined trajectories.

The first Sprint 67 instance is deliberately small and NumPy-backed.  A
linear reward head and a Monte-Carlo value estimator make the contract
testable on analytic MDPs without pretending to be a full RL learner.  The
module owns return construction, terminal/padding semantics, calibration,
Bellman residuals, and real-versus-imagined score comparisons.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol, cast

import numpy as np

from latent_anything.trajectory import Trajectory


def _finite_array(value: np.ndarray, *, name: str) -> np.ndarray:
    """Validate and return a numeric finite array."""

    if not np.issubdtype(value.dtype, np.number):
        raise TypeError(f"{name} must have a numeric dtype")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values")
    return np.asarray(value, dtype=np.float64)


def _matrix(value: np.ndarray, *, name: str, width: int | None = None) -> np.ndarray:
    """Validate a finite two-dimensional batch."""

    result = _finite_array(value, name=name)
    if result.ndim != 2 or (width is not None and result.shape[1] != width):
        suffix = f", {width}" if width is not None else ""
        raise ValueError(f"{name} must have shape (n{suffix}), got {result.shape}")
    return result


def _vector(value: np.ndarray, *, name: str, length: int | None = None) -> np.ndarray:
    """Validate a finite one-dimensional batch."""

    result = _finite_array(value, name=name)
    if result.ndim != 1 or (length is not None and result.shape[0] != length):
        raise ValueError(f"{name} must have shape ({length or 'n'},), got {result.shape}")
    return result


def _bool_vector(value: np.ndarray | None, *, name: str, length: int, default: bool) -> np.ndarray:
    if value is None:
        return np.full(length, default, dtype=bool)
    if value.ndim != 1 or value.shape[0] != length:
        raise ValueError(f"{name} must have shape ({length},), got {getattr(value, 'shape', None)}")
    return np.asarray(value, dtype=bool).copy()


def _freeze_float_array(value: np.ndarray) -> np.ndarray:
    copied = np.asarray(value, dtype=np.float64).copy()
    copied.setflags(write=False)
    return copied


def _freeze_bool_array(value: np.ndarray) -> np.ndarray:
    copied = np.asarray(value, dtype=bool).copy()
    copied.setflags(write=False)
    return copied


def _validate_discount(discount: float) -> float:
    if isinstance(discount, bool) or not np.isfinite(discount) or not 0.0 <= discount < 1.0:
        raise ValueError(f"discount must be finite and in [0, 1), got {discount}")
    return float(discount)


def compute_discounted_returns(
    rewards: np.ndarray,
    *,
    discount: float,
    masks: np.ndarray | None = None,
    terminals: np.ndarray | None = None,
) -> np.ndarray:
    """Compute masked, terminal-aware discounted returns.

    ``rewards`` may be ``(horizon,)`` or ``(episodes, horizon)``.  ``masks``
    marks valid transitions; invalid/padded positions return zero.  A true
    ``terminals[t]`` keeps the reward at ``t`` but prevents bootstrapping from
    ``t + 1``.  Padding also prevents bootstrapping, even when no terminal
    flag is supplied.
    """

    gamma = _validate_discount(discount)
    values = _finite_array(rewards, name="rewards")
    if values.ndim not in {1, 2}:
        raise ValueError(f"rewards must be 1D or 2D, got {values.shape}")
    was_vector = values.ndim == 1
    episodes = values[None, :] if was_vector else values
    episode_count, horizon = episodes.shape
    if masks is None:
        valid = np.ones_like(episodes, dtype=bool)
    else:
        mask_values = np.asarray(masks)
        expected = values.shape
        if mask_values.shape != expected:
            raise ValueError(f"masks must have shape {expected}, got {mask_values.shape}")
        valid = mask_values[None, :] if was_vector else mask_values
        valid = np.asarray(valid, dtype=bool)
    if terminals is None:
        terminal_values = np.zeros_like(episodes, dtype=bool)
    else:
        terminal_array = np.asarray(terminals)
        if terminal_array.shape != values.shape:
            raise ValueError(f"terminals must have shape {values.shape}, got {terminal_array.shape}")
        terminal_values = terminal_array[None, :] if was_vector else terminal_array
        terminal_values = np.asarray(terminal_values, dtype=bool)

    result = np.zeros_like(episodes, dtype=np.float64)
    for episode in range(episode_count):
        running = 0.0
        for step in range(horizon - 1, -1, -1):
            if not valid[episode, step]:
                running = 0.0
                continue
            if terminal_values[episode, step] or step == horizon - 1 or not valid[episode, step + 1]:
                running = float(episodes[episode, step])
            else:
                running = float(episodes[episode, step]) + gamma * running
            result[episode, step] = running
    return result[0] if was_vector else result


class _RewardScorer(Protocol):
    """Unfrozen internal shape for the first reward scorer instance."""

    state_dim: int
    action_dim: int
    source_space_identity: str

    def predict(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """Predict one scalar reward per state/action pair."""

        ...


class _ValueEstimator(Protocol):
    """Unfrozen internal shape for the first value estimator instance."""

    state_dim: int
    discount: float
    horizon: int | None
    policy_id: str
    data_distribution: str

    def predict(self, states: np.ndarray) -> np.ndarray:
        """Predict one scalar value per state."""

        ...


class LinearRewardScorer:
    """Fit a linear scalar reward head over latent state and action."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        *,
        source_space_identity: str = "unknown",
        ridge: float = 1e-8,
    ) -> None:
        if state_dim < 1 or action_dim < 0:
            raise ValueError("state_dim must be >= 1 and action_dim must be >= 0")
        if ridge < 0 or not np.isfinite(ridge):
            raise ValueError("ridge must be finite and >= 0")
        if not source_space_identity.strip():
            raise ValueError("source_space_identity must not be empty")
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.source_space_identity = source_space_identity
        self.ridge = float(ridge)
        self._weights: np.ndarray | None = None
        self._residual_scale = 0.0
        self._fit_metadata: Mapping[str, object] = MappingProxyType({})

    @property
    def is_fitted(self) -> bool:
        return self._weights is not None

    @property
    def fit_metadata(self) -> Mapping[str, object]:
        return self._fit_metadata

    def fit(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        *,
        policy_id: str = "unknown",
        data_distribution: str = "unknown",
    ) -> LinearRewardScorer:
        """Fit the reward head on observed transition samples."""

        state_values = _matrix(states, name="states", width=self.state_dim)
        action_values = _matrix(actions, name="actions", width=self.action_dim)
        reward_values = _vector(rewards, name="rewards", length=len(state_values))
        if len(action_values) != len(state_values):
            raise ValueError("states, actions, and rewards must have the same number of samples")
        design = self._features(state_values, action_values)
        if self.ridge == 0:
            weights = np.linalg.lstsq(design, reward_values, rcond=None)[0]
        else:
            gram = design.T @ design
            weights = np.linalg.solve(
                gram + self.ridge * np.eye(gram.shape[0], dtype=np.float64),
                design.T @ reward_values,
            )
        predictions = design @ weights
        self._weights = np.asarray(weights, dtype=np.float64)
        self._residual_scale = float(np.sqrt(np.mean(np.square(predictions - reward_values))))
        self._fit_metadata = MappingProxyType(
            {
                "source_space_identity": self.source_space_identity,
                "state_dim": self.state_dim,
                "action_dim": self.action_dim,
                "n_samples": len(state_values),
                "policy_id": policy_id,
                "data_distribution": data_distribution,
                "fit_kind": "linear_state_action_reward",
                "ridge": self.ridge,
            }
        )
        return self

    def predict(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """Predict scalar reward for a batch of state/action pairs."""

        state_values = _matrix(states, name="states", width=self.state_dim)
        action_values = _matrix(actions, name="actions", width=self.action_dim)
        if len(state_values) != len(action_values):
            raise ValueError("states and actions must have the same number of samples")
        if self._weights is None:
            raise RuntimeError("reward scorer must be fitted before prediction")
        return np.asarray(self._features(state_values, action_values) @ self._weights, dtype=np.float64)

    def predict_with_uncertainty(self, states: np.ndarray, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        predictions = self.predict(states, actions)
        return predictions, np.full_like(predictions, self._residual_scale)

    def _features(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        return np.concatenate([states, actions, np.ones((len(states), 1), dtype=np.float64)], axis=1)


class MonteCarloValueEstimator:
    """Fit a linear state-value estimator to declared discounted returns.

    ``discount``, ``horizon``, ``policy_id``, and ``data_distribution`` are
    part of the estimator's provenance.  The estimator is therefore not a
    generic scalar regressor whose outputs can be compared across incompatible
    return definitions.
    """

    def __init__(
        self,
        state_dim: int,
        *,
        discount: float = 0.99,
        horizon: int | None = None,
        policy_id: str = "unknown",
        data_distribution: str = "unknown",
        ridge: float = 1e-8,
    ) -> None:
        if state_dim < 1:
            raise ValueError("state_dim must be >= 1")
        if horizon is not None and horizon < 1:
            raise ValueError("horizon must be >= 1 when supplied")
        if ridge < 0 or not np.isfinite(ridge):
            raise ValueError("ridge must be finite and >= 0")
        self.state_dim = state_dim
        self.discount = _validate_discount(discount)
        self.horizon = horizon
        self.policy_id = policy_id
        self.data_distribution = data_distribution
        self.ridge = float(ridge)
        self._weights: np.ndarray | None = None
        self._residual_scale = 0.0
        self._fit_metadata: Mapping[str, object] = MappingProxyType({})

    @property
    def is_fitted(self) -> bool:
        return self._weights is not None

    @property
    def fit_metadata(self) -> Mapping[str, object]:
        return self._fit_metadata

    def fit(self, states: np.ndarray, returns: np.ndarray) -> MonteCarloValueEstimator:
        """Fit the value head on state/return pairs."""

        state_values = _matrix(states, name="states", width=self.state_dim)
        return_values = _vector(returns, name="returns", length=len(state_values))
        design = np.concatenate([state_values, np.ones((len(state_values), 1), dtype=np.float64)], axis=1)
        if self.ridge == 0:
            weights = np.linalg.lstsq(design, return_values, rcond=None)[0]
        else:
            gram = design.T @ design
            weights = np.linalg.solve(
                gram + self.ridge * np.eye(gram.shape[0], dtype=np.float64),
                design.T @ return_values,
            )
        predictions = design @ weights
        self._weights = np.asarray(weights, dtype=np.float64)
        self._residual_scale = float(np.sqrt(np.mean(np.square(predictions - return_values))))
        self._fit_metadata = MappingProxyType(
            {
                "state_dim": self.state_dim,
                "discount": self.discount,
                "horizon": self.horizon,
                "policy_id": self.policy_id,
                "data_distribution": self.data_distribution,
                "n_samples": len(state_values),
                "fit_kind": "linear_monte_carlo_value",
                "ridge": self.ridge,
            }
        )
        return self

    def fit_trajectories(
        self,
        states: np.ndarray,
        rewards: np.ndarray,
        *,
        masks: np.ndarray | None = None,
        terminals: np.ndarray | None = None,
    ) -> MonteCarloValueEstimator:
        """Fit from ``(episodes, horizon + 1, state_dim)`` trajectories."""

        state_values = _finite_array(states, name="states")
        reward_values = _finite_array(rewards, name="rewards")
        if state_values.ndim != 3 or state_values.shape[2] != self.state_dim:
            raise ValueError("states must have shape (episodes, horizon + 1, state_dim)")
        if reward_values.ndim != 2 or reward_values.shape != (state_values.shape[0], state_values.shape[1] - 1):
            raise ValueError("rewards must have shape (episodes, horizon)")
        returns = compute_discounted_returns(
            reward_values,
            discount=self.discount,
            masks=masks,
            terminals=terminals,
        )
        valid = np.ones_like(reward_values, dtype=bool) if masks is None else np.asarray(masks, dtype=bool)
        return self.fit(state_values[:, :-1][valid], returns[valid])

    def predict(self, states: np.ndarray) -> np.ndarray:
        """Predict one state value per row."""

        state_values = _matrix(states, name="states", width=self.state_dim)
        if self._weights is None:
            raise RuntimeError("value estimator must be fitted before prediction")
        features = np.concatenate([state_values, np.ones((len(state_values), 1), dtype=np.float64)], axis=1)
        return np.asarray(features @ self._weights, dtype=np.float64)

    def predict_with_uncertainty(self, states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        predictions = self.predict(states)
        return predictions, np.full_like(predictions, self._residual_scale)


@dataclass(frozen=True, slots=True)
class ValueCalibration:
    """Calibration summary against held-out Monte-Carlo return targets."""

    rmse: float
    mae: float
    bias: float
    mean_prediction: float
    mean_target: float
    coverage: float | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "rmse": self.rmse,
            "mae": self.mae,
            "bias": self.bias,
            "mean_prediction": self.mean_prediction,
            "mean_target": self.mean_target,
            "coverage": self.coverage,
        }


@dataclass(frozen=True, slots=True)
class RewardValueDiagnostics:
    """Held-out reward, value-calibration, and Bellman-consistency metrics."""

    reward_rmse: float
    reward_mae: float
    reward_bias: float
    value_calibration: ValueCalibration
    bellman_residual_rmse: float
    bellman_residual_mae: float
    bellman_residual_bias: float
    n_steps: int
    source: str

    def to_dict(self) -> dict[str, object]:
        return {
            "reward_rmse": self.reward_rmse,
            "reward_mae": self.reward_mae,
            "reward_bias": self.reward_bias,
            "value_calibration": self.value_calibration.to_dict(),
            "bellman_residual_rmse": self.bellman_residual_rmse,
            "bellman_residual_mae": self.bellman_residual_mae,
            "bellman_residual_bias": self.bellman_residual_bias,
            "n_steps": self.n_steps,
            "source": self.source,
        }


@dataclass(frozen=True, slots=True)
class RewardValueEvaluationResult:
    """Typed per-step scoring result for one real or imagined trajectory."""

    rewards: np.ndarray
    returns: np.ndarray
    values: np.ndarray
    masks: np.ndarray
    terminals: np.ndarray
    reward_uncertainty: np.ndarray
    value_uncertainty: np.ndarray
    bellman_residuals: np.ndarray
    discount: float
    horizon: int
    source: str
    provenance: Mapping[str, object]

    def __post_init__(self) -> None:
        arrays = {
            "rewards": self.rewards,
            "returns": self.returns,
            "values": self.values,
            "reward_uncertainty": self.reward_uncertainty,
            "value_uncertainty": self.value_uncertainty,
            "bellman_residuals": self.bellman_residuals,
        }
        shape = self.rewards.shape
        if any(value.shape != shape for value in arrays.values()):
            raise ValueError("all score arrays must have the same shape")
        if self.masks.shape != shape or self.terminals.shape != shape:
            raise ValueError("masks and terminals must match score array shape")
        for name, value in arrays.items():
            if not np.isfinite(value).all():
                raise ValueError(f"{name} must contain only finite values")
        if len(shape) == 0 or self.horizon != shape[-1]:
            raise ValueError("horizon must match the final score-array dimension")
        object.__setattr__(self, "rewards", _freeze_float_array(self.rewards))
        object.__setattr__(self, "returns", _freeze_float_array(self.returns))
        object.__setattr__(self, "values", _freeze_float_array(self.values))
        object.__setattr__(self, "reward_uncertainty", _freeze_float_array(self.reward_uncertainty))
        object.__setattr__(self, "value_uncertainty", _freeze_float_array(self.value_uncertainty))
        object.__setattr__(self, "bellman_residuals", _freeze_float_array(self.bellman_residuals))
        object.__setattr__(self, "masks", _freeze_bool_array(self.masks))
        object.__setattr__(self, "terminals", _freeze_bool_array(self.terminals))
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))

    @property
    def valid_steps(self) -> int:
        return int(np.count_nonzero(self.masks))

    def to_metrics(self) -> dict[str, float]:
        valid = self.masks
        if not np.any(valid):
            return {"valid_steps": 0.0}
        return {
            "reward_mean": float(np.mean(self.rewards[valid])),
            "return_mean": float(np.mean(self.returns[valid])),
            "value_mean": float(np.mean(self.values[valid])),
            "bellman_residual_rmse": float(np.sqrt(np.mean(np.square(self.bellman_residuals[valid])))),
            "valid_steps": float(np.count_nonzero(valid)),
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "rewards": self.rewards.tolist(),
            "returns": self.returns.tolist(),
            "values": self.values.tolist(),
            "masks": self.masks.tolist(),
            "terminals": self.terminals.tolist(),
            "reward_uncertainty": self.reward_uncertainty.tolist(),
            "value_uncertainty": self.value_uncertainty.tolist(),
            "bellman_residuals": self.bellman_residuals.tolist(),
            "discount": self.discount,
            "horizon": self.horizon,
            "source": self.source,
            "provenance": dict(self.provenance),
        }


@dataclass(frozen=True, slots=True)
class TrajectoryScoreComparison:
    """Quantify score drift between a real and imagined trajectory."""

    reward_mae: float
    reward_bias: float
    return_mae: float
    return_bias: float
    value_mae: float
    value_bias: float
    bellman_residual_mae_delta: float
    valid_steps: int
    provenance: Mapping[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "reward_mae": self.reward_mae,
            "reward_bias": self.reward_bias,
            "return_mae": self.return_mae,
            "return_bias": self.return_bias,
            "value_mae": self.value_mae,
            "value_bias": self.value_bias,
            "bellman_residual_mae_delta": self.bellman_residual_mae_delta,
            "valid_steps": self.valid_steps,
            "provenance": dict(self.provenance),
        }


@dataclass(frozen=True, slots=True)
class HoldoutEvaluation:
    """Predictions and diagnostics for a held-out trajectory batch."""

    predicted_rewards: np.ndarray
    target_rewards: np.ndarray
    predicted_values: np.ndarray
    target_returns: np.ndarray
    bellman_residuals: np.ndarray
    masks: np.ndarray
    diagnostics: RewardValueDiagnostics
    provenance: Mapping[str, object]

    def to_metrics(self) -> dict[str, float]:
        """Return flat metrics suitable for :class:`RunRecord`."""

        diagnostics = self.diagnostics
        return {
            "reward_rmse": diagnostics.reward_rmse,
            "reward_mae": diagnostics.reward_mae,
            "value_rmse": diagnostics.value_calibration.rmse,
            "value_mae": diagnostics.value_calibration.mae,
            "value_bias": diagnostics.value_calibration.bias,
            "bellman_residual_rmse": diagnostics.bellman_residual_rmse,
            "bellman_residual_mae": diagnostics.bellman_residual_mae,
            "valid_steps": float(diagnostics.n_steps),
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "predicted_rewards": self.predicted_rewards.tolist(),
            "target_rewards": self.target_rewards.tolist(),
            "predicted_values": self.predicted_values.tolist(),
            "target_returns": self.target_returns.tolist(),
            "bellman_residuals": self.bellman_residuals.tolist(),
            "masks": self.masks.tolist(),
            "diagnostics": self.diagnostics.to_dict(),
            "provenance": dict(self.provenance),
        }


class RewardValueEvaluator:
    """Evaluate a fitted reward scorer and value estimator together."""

    def __init__(self, reward_scorer: _RewardScorer, value_estimator: _ValueEstimator) -> None:
        if reward_scorer.state_dim != value_estimator.state_dim:
            raise ValueError("reward scorer and value estimator state dimensions must match")
        self.reward_scorer = reward_scorer
        self.value_estimator = value_estimator

    @property
    def discount(self) -> float:
        return self.value_estimator.discount

    def evaluate(
        self,
        trajectory: Trajectory,
        actions: np.ndarray,
        *,
        masks: np.ndarray | None = None,
        terminals: np.ndarray | None = None,
        source: str = "unknown",
    ) -> RewardValueEvaluationResult:
        """Score one trajectory using predicted rewards and values."""

        states = _matrix(trajectory.to_numpy(), name="trajectory", width=self.value_estimator.state_dim)
        if len(states) < 2:
            raise ValueError("trajectory must contain an initial state and at least one transition")
        action_values = _matrix(actions, name="actions", width=self.reward_scorer.action_dim)
        if len(action_values) != len(states) - 1:
            raise ValueError("actions must contain one row per transition in trajectory")
        valid = _bool_vector(masks, name="masks", length=len(action_values), default=True)
        terminal_values = _bool_vector(terminals, name="terminals", length=len(action_values), default=False)
        rewards, reward_uncertainty = self._predict_rewards(states[:-1], action_values)
        values, value_uncertainty = self._predict_values(states[:-1])
        next_values = self.value_estimator.predict(states[1:])
        returns = compute_discounted_returns(
            rewards,
            discount=self.discount,
            masks=valid,
            terminals=terminal_values,
        )
        residuals = self._bellman_residuals(
            rewards,
            values,
            next_values,
            valid,
            terminal_values,
        )
        provenance: dict[str, object] = {
            "source": source,
            "trajectory_metadata": dict(trajectory.metadata),
            "source_space_identity": self.reward_scorer.source_space_identity,
            "discount": self.discount,
            "horizon": len(action_values),
            "policy_id": self.value_estimator.policy_id,
            "data_distribution": self.value_estimator.data_distribution,
        }
        return RewardValueEvaluationResult(
            rewards=rewards,
            returns=returns,
            values=values,
            masks=valid,
            terminals=terminal_values,
            reward_uncertainty=reward_uncertainty,
            value_uncertainty=value_uncertainty,
            bellman_residuals=residuals,
            discount=self.discount,
            horizon=len(action_values),
            source=source,
            provenance=provenance,
        )

    def evaluate_holdout(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        *,
        masks: np.ndarray | None = None,
        terminals: np.ndarray | None = None,
        source: str = "real",
    ) -> HoldoutEvaluation:
        """Measure reward prediction, value calibration, and Bellman error."""

        state_values = _finite_array(states, name="states")
        action_values = _finite_array(actions, name="actions")
        reward_values = _finite_array(rewards, name="rewards")
        if state_values.ndim != 3 or action_values.ndim != 3 or reward_values.ndim != 2:
            raise ValueError(
                "states/actions/rewards must have shapes "
                "(episodes, horizon + 1, dim), (episodes, horizon, dim), "
                "(episodes, horizon)"
            )
        if state_values.shape[0] != action_values.shape[0] or state_values.shape[0] != reward_values.shape[0]:
            raise ValueError("states, actions, and rewards must have the same episode count")
        if state_values.shape[1] != action_values.shape[1] + 1 or reward_values.shape != action_values.shape[:2]:
            raise ValueError("states must have one more time point than actions and rewards")
        episode_count, horizon, _ = action_values.shape
        valid = (
            np.ones((episode_count, horizon), dtype=bool)
            if masks is None
            else self._batch_bool(masks, name="masks", shape=(episode_count, horizon))
        )
        terminal_values = (
            np.zeros((episode_count, horizon), dtype=bool)
            if terminals is None
            else self._batch_bool(terminals, name="terminals", shape=(episode_count, horizon))
        )
        flat_states = state_values[:, :-1].reshape(-1, self.value_estimator.state_dim)
        flat_actions = action_values.reshape(-1, self.reward_scorer.action_dim)
        predicted_rewards = self.reward_scorer.predict(flat_states, flat_actions).reshape(episode_count, horizon)
        predicted_values = self.value_estimator.predict(flat_states).reshape(episode_count, horizon)
        next_values = self.value_estimator.predict(
            state_values[:, 1:].reshape(-1, self.value_estimator.state_dim)
        ).reshape(episode_count, horizon)
        target_returns = compute_discounted_returns(
            reward_values,
            discount=self.discount,
            masks=valid,
            terminals=terminal_values,
        )
        residuals = self._bellman_residuals_batch(
            predicted_rewards,
            predicted_values,
            next_values,
            valid,
            terminal_values,
        )
        reward_metrics = self._metrics(predicted_rewards, reward_values, valid)
        value_metrics = self._metrics(predicted_values, target_returns, valid)
        residual_metrics = self._metrics(residuals, np.zeros_like(residuals), valid)
        calibration = ValueCalibration(
            rmse=value_metrics[0],
            mae=value_metrics[1],
            bias=value_metrics[2],
            mean_prediction=value_metrics[3],
            mean_target=value_metrics[4],
        )
        diagnostics = RewardValueDiagnostics(
            reward_rmse=reward_metrics[0],
            reward_mae=reward_metrics[1],
            reward_bias=reward_metrics[2],
            value_calibration=calibration,
            bellman_residual_rmse=residual_metrics[0],
            bellman_residual_mae=residual_metrics[1],
            bellman_residual_bias=residual_metrics[2],
            n_steps=int(np.count_nonzero(valid)),
            source=source,
        )
        provenance = {
            "source": source,
            "discount": self.discount,
            "horizon": horizon,
            "policy_id": self.value_estimator.policy_id,
            "data_distribution": self.value_estimator.data_distribution,
            "source_space_identity": self.reward_scorer.source_space_identity,
        }
        return HoldoutEvaluation(
            predicted_rewards=predicted_rewards,
            target_rewards=reward_values,
            predicted_values=predicted_values,
            target_returns=target_returns,
            bellman_residuals=residuals,
            masks=valid,
            diagnostics=diagnostics,
            provenance=provenance,
        )

    def compare_real_imagined(
        self,
        real_trajectory: Trajectory,
        imagined_trajectory: Trajectory,
        actions: np.ndarray,
        *,
        masks: np.ndarray | None = None,
        terminals: np.ndarray | None = None,
    ) -> TrajectoryScoreComparison:
        """Compare reward/value scores from real and imagined latent paths."""

        real = self.evaluate(real_trajectory, actions, masks=masks, terminals=terminals, source="real")
        imagined = self.evaluate(imagined_trajectory, actions, masks=masks, terminals=terminals, source="imagined")
        valid = real.masks & imagined.masks
        if not np.any(valid):
            raise ValueError("at least one valid transition is required for comparison")
        return TrajectoryScoreComparison(
            reward_mae=float(np.mean(np.abs(imagined.rewards[valid] - real.rewards[valid]))),
            reward_bias=float(np.mean(imagined.rewards[valid] - real.rewards[valid])),
            return_mae=float(np.mean(np.abs(imagined.returns[valid] - real.returns[valid]))),
            return_bias=float(np.mean(imagined.returns[valid] - real.returns[valid])),
            value_mae=float(np.mean(np.abs(imagined.values[valid] - real.values[valid]))),
            value_bias=float(np.mean(imagined.values[valid] - real.values[valid])),
            bellman_residual_mae_delta=float(
                np.mean(np.abs(imagined.bellman_residuals[valid])) - np.mean(np.abs(real.bellman_residuals[valid]))
            ),
            valid_steps=int(np.count_nonzero(valid)),
            provenance={"real_source": "real", "imagined_source": "imagined", "discount": self.discount},
        )

    def _predict_rewards(self, states: np.ndarray, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        predictor = getattr(self.reward_scorer, "predict_with_uncertainty", None)
        if callable(predictor):
            typed_predictor = cast(Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]], predictor)
            predictions, uncertainty = typed_predictor(states, actions)
            return np.asarray(predictions, dtype=np.float64), np.asarray(uncertainty, dtype=np.float64)
        predictions = self.reward_scorer.predict(states, actions)
        return predictions, np.zeros_like(predictions)

    def _predict_values(self, states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        predictor = getattr(self.value_estimator, "predict_with_uncertainty", None)
        if callable(predictor):
            typed_predictor = cast(Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]], predictor)
            predictions, uncertainty = typed_predictor(states)
            return np.asarray(predictions, dtype=np.float64), np.asarray(uncertainty, dtype=np.float64)
        predictions = self.value_estimator.predict(states)
        return predictions, np.zeros_like(predictions)

    def _bellman_residuals(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        next_values: np.ndarray,
        masks: np.ndarray,
        terminals: np.ndarray,
    ) -> np.ndarray:
        continuation = masks & ~terminals
        if len(continuation) > 1:
            continuation[:-1] &= masks[1:]
        continuation[-1] = False
        residuals = values - (rewards + self.discount * continuation * next_values)
        residuals[~masks] = 0.0
        return residuals

    def _bellman_residuals_batch(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        next_values: np.ndarray,
        masks: np.ndarray,
        terminals: np.ndarray,
    ) -> np.ndarray:
        continuation = masks & ~terminals
        if continuation.shape[1] > 1:
            continuation[:, :-1] &= masks[:, 1:]
        continuation[:, -1] = False
        residuals = values - (rewards + self.discount * continuation * next_values)
        residuals[~masks] = 0.0
        return residuals

    @staticmethod
    def _batch_bool(value: np.ndarray, *, name: str, shape: tuple[int, int]) -> np.ndarray:
        if value.shape != shape:
            raise ValueError(f"{name} must have shape {shape}, got {getattr(value, 'shape', None)}")
        return np.asarray(value, dtype=bool).copy()

    @staticmethod
    def _metrics(
        predicted: np.ndarray, target: np.ndarray, masks: np.ndarray
    ) -> tuple[float, float, float, float, float]:
        valid = masks.astype(bool)
        if not np.any(valid):
            raise ValueError("at least one valid transition is required")
        difference = predicted[valid] - target[valid]
        return (
            float(np.sqrt(np.mean(np.square(difference)))),
            float(np.mean(np.abs(difference))),
            float(np.mean(difference)),
            float(np.mean(predicted[valid])),
            float(np.mean(target[valid])),
        )


def compare_real_imagined_scores(
    evaluator: RewardValueEvaluator,
    real_trajectory: Trajectory,
    imagined_trajectory: Trajectory,
    actions: np.ndarray,
    *,
    masks: np.ndarray | None = None,
    terminals: np.ndarray | None = None,
) -> TrajectoryScoreComparison:
    """Convenience wrapper for real-versus-imagined score comparison."""

    return evaluator.compare_real_imagined(
        real_trajectory,
        imagined_trajectory,
        actions,
        masks=masks,
        terminals=terminals,
    )


__all__ = [
    "HoldoutEvaluation",
    "LinearRewardScorer",
    "MonteCarloValueEstimator",
    "RewardValueDiagnostics",
    "RewardValueEvaluationResult",
    "RewardValueEvaluator",
    "TrajectoryScoreComparison",
    "ValueCalibration",
    "compare_real_imagined_scores",
    "compute_discounted_returns",
]

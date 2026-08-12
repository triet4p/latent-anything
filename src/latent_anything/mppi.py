"""Model Predictive Path Integral planning for bounded continuous actions."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import TYPE_CHECKING

import numpy as np
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator

from latent_anything.runtime.profiling import RuntimeProfile, RuntimeProfiler

if TYPE_CHECKING:
    from latent_anything.latent_value import LatentValue
    from latent_anything.reward_value import RewardValueEvaluator
    from latent_anything.rollout_pipeline import RolloutPipeline


def _vector(value: Sequence[float] | np.ndarray | None, *, name: str) -> tuple[float, ...] | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional sequence")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return tuple(float(item) for item in array)


def compute_mppi_weights(returns: np.ndarray, temperature: float) -> np.ndarray:
    """Return numerically stable softmax weights over candidate returns."""

    values = np.asarray(returns, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("returns must be a non-empty one-dimensional array")
    if not np.isfinite(values).all():
        raise ValueError("returns must contain only finite values")
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and > 0")
    scaled = (values - np.max(values)) / temperature
    exponentials = np.exp(scaled)
    normalizer = float(np.sum(exponentials))
    if not np.isfinite(normalizer) or normalizer <= 0.0:
        raise ValueError("temperature weighting produced an invalid normalizer")
    return exponentials / normalizer


class MPPIConfig(BaseModel):
    """Validated configuration for bounded MPPI action-sequence planning."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    horizon: int
    action_dim: int
    lower_bounds: tuple[float, ...]
    upper_bounds: tuple[float, ...]
    population_size: int = 256
    iterations: int = 6
    temperature: float = Field(default=1.0, validation_alias=AliasChoices("temperature", "lambda", "lambda_"))
    noise_std: tuple[float, ...] = (0.5,)
    initial_mean: tuple[float, ...] | None = None
    seed: int | None = None

    @model_validator(mode="before")
    @classmethod
    def _coerce_vectors(cls, values: object) -> object:
        if not isinstance(values, dict):
            return values
        result = dict(values)
        for name in ("lower_bounds", "upper_bounds", "noise_std", "initial_mean"):
            if name in result or name in ("lower_bounds", "upper_bounds"):
                raw = result.get(name)
                if name == "noise_std" and raw is not None and np.asarray(raw).ndim == 0:
                    raw = [raw]
                vector = _vector(raw, name=name)
                if name == "noise_std" and vector is not None and len(vector) == 1:
                    action_dim = result.get("action_dim")
                    if isinstance(action_dim, int) and action_dim > 1:
                        vector = vector * action_dim
                result[name] = vector
        return result

    @model_validator(mode="after")
    def _validate(self) -> MPPIConfig:
        if self.horizon < 1 or self.action_dim < 1:
            raise ValueError("horizon and action_dim must be >= 1")
        if len(self.lower_bounds) != self.action_dim or len(self.upper_bounds) != self.action_dim:
            raise ValueError("lower_bounds and upper_bounds must match action_dim")
        lower = np.asarray(self.lower_bounds, dtype=np.float64)
        upper = np.asarray(self.upper_bounds, dtype=np.float64)
        if np.any(lower >= upper):
            raise ValueError("every lower_bound must be strictly less than its upper_bound")
        if self.population_size < 2:
            raise ValueError("population_size must be >= 2")
        if self.iterations < 1:
            raise ValueError("iterations must be >= 1")
        if not np.isfinite(self.temperature) or self.temperature <= 0.0:
            raise ValueError("temperature must be finite and > 0")
        if len(self.noise_std) == 1 and self.action_dim > 1:
            object.__setattr__(self, "noise_std", self.noise_std * self.action_dim)
        if len(self.noise_std) != self.action_dim:
            raise ValueError("noise_std must match action_dim")
        if np.any(np.asarray(self.noise_std) < 0.0):
            raise ValueError("noise_std must contain finite non-negative values")
        if self.initial_mean is not None:
            if len(self.initial_mean) != self.action_dim:
                raise ValueError("initial_mean must match action_dim")
            mean = np.asarray(self.initial_mean, dtype=np.float64)
            if np.any(mean < lower) or np.any(mean > upper):
                raise ValueError("initial_mean must lie within action bounds")
        if self.seed is not None and isinstance(self.seed, bool):
            raise ValueError("seed must be an integer or None")
        return self


@dataclass(frozen=True, slots=True)
class MPPIIteration:
    """Summary of one soft-weighted MPPI update."""

    iteration: int
    population_size: int
    mean_return: float
    std_return: float
    best_return: float
    weighted_return: float
    effective_sample_size: float
    weight_entropy: float
    mean_action: np.ndarray

    def __post_init__(self) -> None:
        value = np.asarray(self.mean_action, dtype=np.float64).copy()
        value.setflags(write=False)
        object.__setattr__(self, "mean_action", value)

    def to_dict(self) -> dict[str, object]:
        return {
            "iteration": self.iteration,
            "population_size": self.population_size,
            "mean_return": self.mean_return,
            "std_return": self.std_return,
            "best_return": self.best_return,
            "weighted_return": self.weighted_return,
            "effective_sample_size": self.effective_sample_size,
            "weight_entropy": self.weight_entropy,
            "mean_action": self.mean_action.tolist(),
        }


@dataclass(frozen=True, slots=True)
class MPPIPlanResult:
    """Selected MPPI nominal sequence and soft-weight convergence evidence."""

    actions: np.ndarray
    predicted_return: float
    candidate_statistics: tuple[MPPIIteration, ...]
    convergence_history: tuple[float, ...]
    runtime_profile: RuntimeProfile
    seed: int | None
    sample_count: int

    def __post_init__(self) -> None:
        actions = np.asarray(self.actions, dtype=np.float64).copy()
        actions.setflags(write=False)
        object.__setattr__(self, "actions", actions)

    @property
    def selected_actions(self) -> np.ndarray:
        """Compatibility alias naming the action sequence selected by MPPI."""

        return self.actions

    @property
    def effective_sample_size(self) -> float:
        """Return the final iteration's effective sample size."""

        return self.candidate_statistics[-1].effective_sample_size

    def to_dict(self) -> dict[str, object]:
        return {
            "actions": self.actions.tolist(),
            "predicted_return": self.predicted_return,
            "candidate_statistics": [item.to_dict() for item in self.candidate_statistics],
            "convergence_history": list(self.convergence_history),
            "runtime_profile": {
                "total_seconds": self.runtime_profile.total_seconds,
                "stage_totals": self.runtime_profile.stage_totals(),
            },
            "seed": self.seed,
            "sample_count": self.sample_count,
        }


@dataclass(frozen=True, slots=True)
class MPPIRecedingHorizonResult:
    """Executed actions and state trace from a receding-horizon MPPI loop."""

    actions: np.ndarray
    states: np.ndarray
    plans: tuple[MPPIPlanResult, ...]
    runtime_profile: RuntimeProfile

    def __post_init__(self) -> None:
        for name in ("actions", "states"):
            value = np.asarray(getattr(self, name), dtype=np.float64).copy()
            value.setflags(write=False)
            object.__setattr__(self, name, value)

    def to_dict(self) -> dict[str, object]:
        return {
            "actions": self.actions.tolist(),
            "states": self.states.tolist(),
            "plans": [plan.to_dict() for plan in self.plans],
            "runtime_profile": {
                "total_seconds": self.runtime_profile.total_seconds,
                "stage_totals": self.runtime_profile.stage_totals(),
            },
        }


class MPPIPlanner:
    """Optimize bounded continuous actions with soft importance weighting."""

    def __init__(self, config: MPPIConfig) -> None:
        self.config = config

    def plan(
        self,
        objective: Callable[[np.ndarray], np.ndarray],
        *,
        initial_mean: np.ndarray | None = None,
        noise_std: np.ndarray | None = None,
        profiler: RuntimeProfiler | None = None,
        seed: int | None = None,
    ) -> MPPIPlanResult:
        """Optimize ``objective`` over ``(population, horizon, action_dim)`` candidates."""

        active_profiler = profiler or RuntimeProfiler()
        config = self.config
        lower = np.asarray(config.lower_bounds, dtype=np.float64)
        upper = np.asarray(config.upper_bounds, dtype=np.float64)
        nominal = self._initial_nominal(initial_mean, config.initial_mean, lower, upper)
        std = self._noise_vector(noise_std, config.noise_std)
        rng_seed = config.seed if seed is None else seed
        rng = np.random.default_rng(rng_seed)
        summaries: list[MPPIIteration] = []
        convergence: list[float] = []
        best_return = -np.inf
        planning_start = perf_counter()
        sample_count = 0

        for iteration in range(config.iterations):
            noise = rng.normal(0.0, std, size=(config.population_size, config.horizon, config.action_dim))
            candidates = np.clip(nominal[None, :, :] + noise, lower, upper)
            effective_noise = candidates - nominal[None, :, :]
            score_start = perf_counter()
            returns = np.asarray(objective(candidates), dtype=np.float64)
            active_profiler.record("evaluation", perf_counter() - score_start, iteration=iteration)
            sample_count += config.population_size
            self._validate_returns(returns, config.population_size)
            weights = compute_mppi_weights(returns, config.temperature)
            nominal = np.clip(nominal + np.einsum("n,nha->ha", weights, effective_noise), lower, upper)
            weighted_return = float(np.dot(weights, returns))
            best_return = max(best_return, float(np.max(returns)))
            convergence.append(best_return)
            summaries.append(
                MPPIIteration(
                    iteration=iteration,
                    population_size=config.population_size,
                    mean_return=float(np.mean(returns)),
                    std_return=float(np.std(returns)),
                    best_return=float(np.max(returns)),
                    weighted_return=weighted_return,
                    effective_sample_size=float(1.0 / np.sum(np.square(weights))),
                    weight_entropy=float(-np.sum(weights * np.log(np.maximum(weights, np.finfo(float).tiny)))),
                    mean_action=nominal.copy(),
                )
            )

        score_start = perf_counter()
        selected_score = np.asarray(objective(nominal[None, :, :]), dtype=np.float64)
        active_profiler.record("evaluation", perf_counter() - score_start, iteration=config.iterations)
        self._validate_returns(selected_score, 1)
        active_profiler.record("planning", perf_counter() - planning_start, iterations=config.iterations)
        return MPPIPlanResult(
            actions=nominal,
            predicted_return=float(selected_score[0]),
            candidate_statistics=tuple(summaries),
            convergence_history=tuple(convergence),
            runtime_profile=active_profiler.snapshot(),
            seed=rng_seed,
            sample_count=sample_count,
        )

    def plan_rollouts(
        self,
        initial_state: np.ndarray | LatentValue,
        rollout_pipeline: RolloutPipeline,
        *,
        evaluator: RewardValueEvaluator | None = None,
        metadata: Mapping[str, object] | None = None,
        initial_mean: np.ndarray | None = None,
        noise_std: np.ndarray | None = None,
        profiler: RuntimeProfiler | None = None,
        seed: int | None = None,
    ) -> MPPIPlanResult:
        """Plan by scoring candidates through imagined rollout evaluation."""

        active_evaluator = evaluator or rollout_pipeline.evaluator
        if active_evaluator is None:
            raise ValueError("plan_rollouts requires a reward/value evaluator")
        active_profiler = profiler or RuntimeProfiler()

        def objective(candidates: np.ndarray) -> np.ndarray:
            scores = np.empty(len(candidates), dtype=np.float64)
            discount = active_evaluator.discount
            for index, actions in enumerate(candidates):
                result = rollout_pipeline.run(
                    initial_state,
                    actions,
                    profiler=active_profiler,
                    metadata=metadata,
                )
                evaluation = (
                    active_evaluator.evaluate(result.trajectory, actions, source="imagined")
                    if evaluator is not None
                    else result.evaluation or active_evaluator.evaluate(result.trajectory, actions, source="imagined")
                )
                weights = np.power(discount, np.arange(len(evaluation.rewards), dtype=np.float64))
                reward_return = float(np.sum(weights * evaluation.rewards))
                terminal_state = result.trajectory.to_numpy()[-1][None, :]
                terminal_value = float(active_evaluator.value_estimator.predict(terminal_state)[0])
                scores[index] = reward_return + float(discount ** len(actions)) * terminal_value
            return scores

        return self.plan(
            objective,
            initial_mean=initial_mean,
            noise_std=noise_std,
            profiler=active_profiler,
            seed=seed,
        )

    def plan_receding_horizon(
        self,
        initial_state: np.ndarray | LatentValue,
        rollout_pipeline: RolloutPipeline,
        *,
        steps: int,
        action_steps: int = 1,
        environment_step: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
        initial_mean: np.ndarray | None = None,
        metadata: Mapping[str, object] | None = None,
        profiler: RuntimeProfiler | None = None,
    ) -> MPPIRecedingHorizonResult:
        """Repeatedly plan, execute a prefix, and shift the nominal sequence.

        ``environment_step`` may replay a real environment.  When omitted, the
        transition's predictive ``step`` is used as a deterministic local
        execution loop, which keeps the helper useful for offline evidence.
        """

        if steps < 1:
            raise ValueError("steps must be >= 1")
        if action_steps < 1 or action_steps > self.config.horizon:
            raise ValueError("action_steps must be in [1, horizon]")
        active_profiler = profiler or RuntimeProfiler()
        stepper = environment_step or rollout_pipeline.transition.step
        state = self._coerce_initial_state(initial_state, rollout_pipeline)
        nominal = self._initial_nominal(
            initial_mean,
            self.config.initial_mean,
            np.asarray(self.config.lower_bounds),
            np.asarray(self.config.upper_bounds),
        )
        executed_actions: list[np.ndarray] = []
        states = [state.copy()]
        plans: list[MPPIPlanResult] = []
        for step_index in range(steps):
            plan = self.plan_rollouts(
                state,
                rollout_pipeline,
                initial_mean=nominal,
                metadata=metadata,
                profiler=active_profiler,
                seed=None if self.config.seed is None else self.config.seed + step_index,
            )
            plans.append(plan)
            prefix = plan.actions[:action_steps]
            for action in prefix:
                state = np.asarray(stepper(state, action), dtype=np.float64)
                if state.shape != (rollout_pipeline.transition.state_dim,) or not np.isfinite(state).all():
                    raise ValueError("environment_step must return a finite state with the transition state shape")
                executed_actions.append(action.copy())
                states.append(state.copy())
            nominal = np.concatenate([plan.actions[action_steps:], np.repeat(plan.actions[-1:], action_steps, axis=0)])[
                : self.config.horizon
            ]
        return MPPIRecedingHorizonResult(
            actions=np.asarray(executed_actions, dtype=np.float64),
            states=np.asarray(states, dtype=np.float64),
            plans=tuple(plans),
            runtime_profile=active_profiler.snapshot(),
        )

    def _initial_nominal(
        self,
        supplied: np.ndarray | None,
        configured: tuple[float, ...] | None,
        lower: np.ndarray,
        upper: np.ndarray,
    ) -> np.ndarray:
        fallback = np.broadcast_to((lower + upper) / 2.0, (self.config.horizon, self.config.action_dim))
        value = np.asarray(supplied if supplied is not None else configured if configured is not None else fallback)
        if value.shape == (self.config.action_dim,):
            value = np.broadcast_to(value, fallback.shape)
        if value.shape != fallback.shape:
            raise ValueError(
                "initial_mean must have shape "
                f"({self.config.action_dim},) or ({self.config.horizon}, {self.config.action_dim}), got {value.shape}"
            )
        if not np.isfinite(value).all() or np.any(value < lower) or np.any(value > upper):
            raise ValueError("initial_mean must be finite and lie within action bounds")
        return np.asarray(value, dtype=np.float64).copy()

    def _noise_vector(self, supplied: np.ndarray | None, configured: tuple[float, ...]) -> np.ndarray:
        value = np.asarray(supplied if supplied is not None else configured, dtype=np.float64)
        if value.shape != (self.config.action_dim,):
            raise ValueError(f"noise_std must have shape ({self.config.action_dim},), got {value.shape}")
        if not np.isfinite(value).all() or np.any(value < 0.0):
            raise ValueError("noise_std must contain finite non-negative values")
        return value.copy()

    @staticmethod
    def _validate_returns(returns: np.ndarray, expected_size: int) -> None:
        if returns.shape != (expected_size,):
            raise ValueError(
                "objective must return one finite score per candidate with shape "
                f"({expected_size},), got {returns.shape}"
            )
        if not np.isfinite(returns).all():
            raise ValueError("objective must return only finite scores")

    @staticmethod
    def _coerce_initial_state(initial_state: np.ndarray | LatentValue, rollout_pipeline: RolloutPipeline) -> np.ndarray:
        if isinstance(initial_state, np.ndarray):
            state = np.asarray(initial_state, dtype=np.float64)
        else:
            state = np.asarray(initial_state.to_numpy(), dtype=np.float64)
        if state.shape != (rollout_pipeline.transition.state_dim,) or not np.isfinite(state).all():
            raise ValueError("initial_state must be finite and match the transition state shape")
        return state.copy()


__all__ = [
    "MPPIConfig",
    "MPPIIteration",
    "MPPIPlanResult",
    "MPPIPlanner",
    "MPPIRecedingHorizonResult",
    "compute_mppi_weights",
]

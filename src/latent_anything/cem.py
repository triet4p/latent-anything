"""Cross-Entropy Method planning primitives for bounded continuous actions."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel, ConfigDict, model_validator

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


class CEMConfig(BaseModel):
    """Validated configuration for bounded diagonal-Gaussian CEM."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    horizon: int
    action_dim: int
    lower_bounds: tuple[float, ...]
    upper_bounds: tuple[float, ...]
    population_size: int = 256
    elite_count: int | None = None
    elite_fraction: float = 0.1
    iterations: int = 6
    smoothing: float = 0.1
    min_std: float = 1e-3
    initial_mean: tuple[float, ...] | None = None
    initial_std: tuple[float, ...] | None = None
    seed: int | None = None

    @model_validator(mode="before")
    @classmethod
    def _coerce_vectors(cls, values: object) -> object:
        if not isinstance(values, dict):
            return values
        result = dict(values)
        for name in ("lower_bounds", "upper_bounds", "initial_mean", "initial_std"):
            result[name] = _vector(result.get(name), name=name)
        return result

    @model_validator(mode="after")
    def _validate(self) -> CEMConfig:
        if self.horizon < 1 or self.action_dim < 1:
            raise ValueError("horizon and action_dim must be >= 1")
        if len(self.lower_bounds) != self.action_dim or len(self.upper_bounds) != self.action_dim:
            raise ValueError("lower_bounds and upper_bounds must match action_dim")
        lower = np.asarray(self.lower_bounds)
        upper = np.asarray(self.upper_bounds)
        if np.any(lower >= upper):
            raise ValueError("every lower_bound must be strictly less than its upper_bound")
        if self.population_size < 2:
            raise ValueError("population_size must be >= 2")
        if self.elite_count is not None and not 1 <= self.elite_count <= self.population_size:
            raise ValueError("elite_count must be between 1 and population_size")
        if not 0.0 < self.elite_fraction <= 1.0:
            raise ValueError("elite_fraction must be in (0, 1]")
        if self.elite_count is None and int(np.ceil(self.elite_fraction * self.population_size)) < 1:
            raise ValueError("elite_fraction selects no candidates")
        if self.iterations < 1:
            raise ValueError("iterations must be >= 1")
        if not 0.0 <= self.smoothing < 1.0:
            raise ValueError("smoothing must be in [0, 1)")
        if not np.isfinite(self.min_std) or self.min_std <= 0.0:
            raise ValueError("min_std must be finite and > 0")
        for name, value in (("initial_mean", self.initial_mean), ("initial_std", self.initial_std)):
            if value is not None and len(value) != self.action_dim:
                raise ValueError(f"{name} must match action_dim")
        if self.initial_mean is not None:
            mean = np.asarray(self.initial_mean)
            if np.any(mean < lower) or np.any(mean > upper):
                raise ValueError("initial_mean must lie within action bounds")
        if self.initial_std is not None:
            std = np.asarray(self.initial_std)
            if np.any(~np.isfinite(std)) or np.any(std <= 0.0):
                raise ValueError("initial_std must contain finite positive values")
        if self.seed is not None and isinstance(self.seed, bool):
            raise ValueError("seed must be an integer or None")
        return self

    @property
    def resolved_elite_count(self) -> int:
        """Return the number of candidates retained per iteration."""

        return self.elite_count or max(1, int(np.ceil(self.elite_fraction * self.population_size)))


@dataclass(frozen=True, slots=True)
class CEMIteration:
    """Summary of one CEM sampling and elite-refit iteration."""

    iteration: int
    population_size: int
    elite_count: int
    mean_return: float
    std_return: float
    best_return: float
    elite_threshold: float
    mean_action: np.ndarray
    std_action: np.ndarray

    def __post_init__(self) -> None:
        for name in ("mean_action", "std_action"):
            value = np.asarray(getattr(self, name), dtype=np.float64).copy()
            value.setflags(write=False)
            object.__setattr__(self, name, value)

    def to_dict(self) -> dict[str, object]:
        return {
            "iteration": self.iteration,
            "population_size": self.population_size,
            "elite_count": self.elite_count,
            "mean_return": self.mean_return,
            "std_return": self.std_return,
            "best_return": self.best_return,
            "elite_threshold": self.elite_threshold,
            "mean_action": self.mean_action.tolist(),
            "std_action": self.std_action.tolist(),
        }


@dataclass(frozen=True, slots=True)
class CEMPlanResult:
    """Selected bounded action sequence and CEM convergence evidence."""

    actions: np.ndarray
    predicted_return: float
    candidate_statistics: tuple[CEMIteration, ...]
    convergence_history: tuple[float, ...]
    runtime_profile: RuntimeProfile
    seed: int | None

    def __post_init__(self) -> None:
        actions = np.asarray(self.actions, dtype=np.float64).copy()
        actions.setflags(write=False)
        object.__setattr__(self, "actions", actions)

    @property
    def selected_actions(self) -> np.ndarray:
        """Compatibility alias naming the action sequence selected by CEM."""

        return self.actions

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
        }


class CEMPlanner:
    """Optimize a batched continuous-action objective with bounded CEM."""

    def __init__(self, config: CEMConfig) -> None:
        self.config = config

    def plan(
        self,
        objective: Callable[[np.ndarray], np.ndarray],
        *,
        initial_mean: np.ndarray | None = None,
        initial_std: np.ndarray | None = None,
        profiler: RuntimeProfiler | None = None,
    ) -> CEMPlanResult:
        """Optimize ``objective`` over ``(population, horizon, action_dim)`` candidates."""

        active_profiler = profiler or RuntimeProfiler()
        config = self.config
        lower = np.asarray(config.lower_bounds, dtype=np.float64)
        upper = np.asarray(config.upper_bounds, dtype=np.float64)
        mean = self._initial_vector(initial_mean, config.initial_mean, (lower + upper) / 2.0, "initial_mean")
        span = upper - lower
        std = self._initial_vector(initial_std, config.initial_std, span / 2.0, "initial_std")
        std = np.maximum(std, config.min_std)
        rng = np.random.default_rng(config.seed)
        best_return = -np.inf
        best_actions: np.ndarray | None = None
        summaries: list[CEMIteration] = []
        convergence: list[float] = []
        planning_start = perf_counter()

        for iteration in range(config.iterations):
            samples = rng.normal(mean, std, size=(config.population_size, config.horizon, config.action_dim))
            samples = np.clip(samples, lower, upper)
            score_start = perf_counter()
            returns = np.asarray(objective(samples), dtype=np.float64)
            active_profiler.record("evaluation", perf_counter() - score_start, iteration=iteration)
            if returns.shape != (config.population_size,):
                raise ValueError(
                    "objective must return one finite score per candidate with shape "
                    f"({config.population_size},), got {returns.shape}"
                )
            if not np.isfinite(returns).all():
                raise ValueError("objective must return only finite scores")
            order = np.argsort(returns)[::-1]
            elite = samples[order[: config.resolved_elite_count]]
            elite_returns = returns[order[: config.resolved_elite_count]]
            iteration_best = int(order[0])
            if float(returns[iteration_best]) > best_return:
                best_return = float(returns[iteration_best])
                best_actions = samples[iteration_best].copy()
            elite_mean = elite.mean(axis=0)
            elite_std = np.maximum(elite.std(axis=0), config.min_std)
            mean = config.smoothing * mean + (1.0 - config.smoothing) * elite_mean
            std = config.smoothing * std + (1.0 - config.smoothing) * elite_std
            mean = np.clip(mean, lower, upper)
            std = np.minimum(np.maximum(std, config.min_std), span / 2.0)
            summaries.append(
                CEMIteration(
                    iteration=iteration,
                    population_size=config.population_size,
                    elite_count=config.resolved_elite_count,
                    mean_return=float(np.mean(returns)),
                    std_return=float(np.std(returns)),
                    best_return=float(returns[iteration_best]),
                    elite_threshold=float(np.min(elite_returns)),
                    mean_action=mean.copy(),
                    std_action=std.copy(),
                )
            )
            convergence.append(best_return)

        if best_actions is None:
            raise RuntimeError("CEM completed without selecting an action sequence")
        active_profiler.record("planning", perf_counter() - planning_start, iterations=config.iterations)
        return CEMPlanResult(
            actions=best_actions,
            predicted_return=best_return,
            candidate_statistics=tuple(summaries),
            convergence_history=tuple(convergence),
            runtime_profile=active_profiler.snapshot(),
            seed=config.seed,
        )

    def plan_rollouts(
        self,
        initial_state: np.ndarray | LatentValue,
        rollout_pipeline: RolloutPipeline,
        *,
        evaluator: RewardValueEvaluator | None = None,
        metadata: Mapping[str, object] | None = None,
        profiler: RuntimeProfiler | None = None,
    ) -> CEMPlanResult:
        """Plan by scoring each candidate through imagined rollout evaluation.

        The objective is the evaluator's discounted predicted reward plus its
        terminal state-value prediction.  The rollout pipeline remains the
        owner of transition execution and cache/profiling behavior.
        """

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

        return self.plan(objective, profiler=active_profiler)

    def _initial_vector(
        self,
        supplied: np.ndarray | None,
        configured: tuple[float, ...] | None,
        fallback: np.ndarray,
        name: str,
    ) -> np.ndarray:
        value = np.asarray(supplied if supplied is not None else configured if configured is not None else fallback)
        if value.shape != (self.config.action_dim,):
            raise ValueError(f"{name} must have shape ({self.config.action_dim},), got {value.shape}")
        if not np.isfinite(value).all():
            raise ValueError(f"{name} must contain only finite values")
        return np.asarray(value, dtype=np.float64).copy()


__all__ = ["CEMConfig", "CEMIteration", "CEMPlanResult", "CEMPlanner"]

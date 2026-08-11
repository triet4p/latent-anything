"""Pipeline #3: execute a latent transition over an action sequence."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from time import perf_counter
from typing import cast

import numpy as np

from latent_anything.latent_space import LatentSpace
from latent_anything.latent_value import LatentValue
from latent_anything.pipeline_contract import PipelineContract, PipelineKind
from latent_anything.pipeline_models import RolloutResult
from latent_anything.runtime.cache import CacheKey, InMemoryCache, make_cache_key
from latent_anything.runtime.profiling import RuntimeProfiler
from latent_anything.trajectory import Trajectory
from latent_anything.transition_contract import LatentTransition


class RolloutPipeline(PipelineContract):
    """Compose an initial latent state, actions, and a latent transition.

    The pipeline executes the transition's proven predictive-mean rollout
    surface. Distribution-valued and recurrent-specific behavior remains the
    responsibility of the concrete transition, so this class does not invent
    a generic world-model protocol.
    """

    pipeline_kind: PipelineKind = "rollout"

    def __init__(self, transition: LatentTransition, cache: InMemoryCache | None = None) -> None:
        self.transition = transition
        self.cache = cache
        latent_space = getattr(transition, "latent_space", None)
        if not isinstance(latent_space, LatentSpace):
            latent_space = LatentSpace(dim=transition.state_dim, source_model=transition.source_space_identity)
        self._latent_space = latent_space

    @property
    def latent_space(self) -> LatentSpace:
        """Return the transition's source latent space."""

        return self._latent_space

    def run(
        self,
        initial_state: np.ndarray | LatentValue,
        actions: np.ndarray,
        *,
        profiler: RuntimeProfiler | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> RolloutResult:
        """Run one rollout and return its trajectory plus provenance."""

        initial, action_values = self._validate_inputs(initial_state, actions)
        key = self._cache_key(initial, action_values) if self.cache is not None else None
        if key is not None:
            cached = self._cache_get(key, profiler=profiler)
            if isinstance(cached, dict) and isinstance(cached.get("data"), np.ndarray):
                cached_trajectory = Trajectory(
                    cached["data"],  # type: ignore[arg-type]
                    metadata=cached.get("metadata") if isinstance(cached.get("metadata"), Mapping) else None,
                )
                return RolloutResult(
                    initial.copy(), action_values.copy(), cached_trajectory, self._latent_space, cache_hit=True
                )

        trajectory = self._execute(initial, action_values, profiler=profiler, metadata=metadata)
        if key is not None:
            self._cache_set(key, trajectory, profiler=profiler)
        return RolloutResult(initial.copy(), action_values.copy(), trajectory, self._latent_space, cache_hit=False)

    async def run_async(
        self,
        initial_state: np.ndarray | LatentValue,
        actions: np.ndarray,
        *,
        profiler: RuntimeProfiler | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> RolloutResult:
        """Run the synchronous implementation in a cancellable worker thread.

        ``asyncio.CancelledError`` is deliberately re-raised.  Transition
        validation and execution exceptions are also left untouched so callers
        can handle the concrete error contract of their transition.
        """

        try:
            return await asyncio.to_thread(self.run, initial_state, actions, profiler=profiler, metadata=metadata)
        except asyncio.CancelledError:
            raise

    def _execute(
        self,
        initial: np.ndarray,
        actions: np.ndarray,
        *,
        profiler: RuntimeProfiler | None,
        metadata: Mapping[str, object] | None,
    ) -> Trajectory:
        values = {} if metadata is None else dict(metadata)
        values.update(
            {
                "pipeline": self.__class__.__name__,
                "pipeline_kind": self.pipeline_kind,
                "source_space_identity": self.transition.source_space_identity,
            }
        )

        def operation() -> Trajectory:
            mean_rollout = cast(Callable[..., Trajectory], self.transition.mean_rollout)
            return mean_rollout(initial, actions, metadata=values)

        if profiler is None:
            return operation()
        start = perf_counter()
        result = operation()
        profiler.record("transition", perf_counter() - start, component=type(self.transition).__name__)
        return result

    def _validate_inputs(
        self, initial_state: np.ndarray | LatentValue, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if isinstance(initial_state, LatentValue):
            if initial_state.is_batch:
                raise ValueError("initial_state must contain exactly one latent point, not a batch")
            supplied_space = initial_state.space
            if (
                supplied_space.geometry != self._latent_space.geometry
                or supplied_space.shape != self._latent_space.shape
                or supplied_space.source_model != self._latent_space.source_model
            ):
                raise ValueError("initial_state LatentSpace does not match the transition latent space")
            initial = initial_state.to_numpy()
        else:
            initial = np.asarray(initial_state)
        if initial.ndim != 1 or initial.shape != (self.transition.state_dim,):
            raise ValueError(f"initial_state must have shape ({self.transition.state_dim},), got {initial.shape}")
        if not np.issubdtype(initial.dtype, np.number) or not np.isfinite(initial).all():
            raise ValueError("initial_state must contain finite numeric values")
        action_values = np.asarray(actions)
        if action_values.ndim != 2 or action_values.shape[1] != self.transition.action_dim:
            raise ValueError(
                f"actions must have shape (horizon, {self.transition.action_dim}), got {action_values.shape}"
            )
        if not np.issubdtype(action_values.dtype, np.number) or not np.isfinite(action_values).all():
            raise ValueError("actions must contain finite numeric values")
        return np.asarray(initial, dtype=np.float64), np.asarray(action_values, dtype=np.float64)

    def _cache_key(self, initial: np.ndarray, actions: np.ndarray) -> CacheKey:
        payload = np.concatenate([initial.ravel(), actions.ravel()])
        return make_cache_key(
            namespace="rollout_pipeline",
            operation="transition.mean_rollout",
            component=self.transition,
            data=payload,
        )

    def _cache_get(self, key: CacheKey, *, profiler: RuntimeProfiler | None) -> object | None:
        assert self.cache is not None
        if profiler is None:
            return self.cache.get_object(key)
        start = perf_counter()
        value = self.cache.get_object(key)
        profiler.record("cache", perf_counter() - start, operation="get", cache_hit=value is not None)
        return value

    def _cache_set(self, key: CacheKey, value: Trajectory, *, profiler: RuntimeProfiler | None) -> None:
        assert self.cache is not None
        if profiler is None:
            self.cache.set_object(key, {"data": value.to_numpy(), "metadata": dict(value.metadata)})
            return
        start = perf_counter()
        self.cache.set_object(key, {"data": value.to_numpy(), "metadata": dict(value.metadata)})
        profiler.record("cache", perf_counter() - start, operation="set", cache_hit=False)

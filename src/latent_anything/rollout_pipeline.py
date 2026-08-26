"""Pipeline #3: execute a latent transition over an action sequence."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Callable, Generator, Iterable, Mapping
from contextlib import suppress
from time import perf_counter
from typing import Any

import numpy as np

from latent_anything.latent_space import LatentSpace
from latent_anything.latent_value import LatentValue
from latent_anything.pipeline_contract import PipelineContract, PipelineKind
from latent_anything.pipeline_models import RolloutResult
from latent_anything.reward_value import RewardValueEvaluationResult, RewardValueEvaluator
from latent_anything.runtime.cache import CacheKey, InMemoryCache, make_cache_key
from latent_anything.runtime.profiling import RuntimeProfiler
from latent_anything.trajectory import Trajectory
from latent_anything.transition_contract import LatentTransition

_STREAM_END = object()


async def _settled_thread_call[T](function: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """Run one blocking operation and settle it before propagating cancellation."""

    worker = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError:
        with suppress(BaseException):
            await worker
        raise


def _close_iterator(iterator: object, owner: object | None = None) -> None:
    """Close an iterator and distinct owning iterable, if they expose hooks."""

    seen: set[int] = set()
    for candidate in (iterator, owner):
        if id(candidate) in seen:
            continue
        seen.add(id(candidate))
        close = getattr(candidate, "close", None)
        if callable(close):
            with suppress(Exception):
                close()


class RolloutPipeline(PipelineContract):
    """Compose an initial latent state, actions, and a latent transition.

    The pipeline executes the transition's proven predictive-mean rollout
    surface. Distribution-valued and recurrent-specific behavior remains the
    responsibility of the concrete transition, so this class does not invent
    a generic world-model protocol.
    """

    pipeline_kind: PipelineKind = "rollout"

    def __init__(
        self,
        transition: LatentTransition,
        cache: InMemoryCache | None = None,
        evaluator: RewardValueEvaluator | None = None,
    ) -> None:
        self.transition = transition
        self.cache = cache
        self.evaluator = evaluator
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
        key = self._cache_key(initial, action_values, metadata=metadata) if self.cache is not None else None
        if key is not None:
            cached = self._cache_get(key, profiler=profiler)
            if isinstance(cached, dict) and isinstance(cached.get("data"), np.ndarray):
                cached_trajectory = Trajectory(
                    cached["data"],  # type: ignore[arg-type]
                    metadata=cached.get("metadata") if isinstance(cached.get("metadata"), Mapping) else None,
                )
                return RolloutResult(
                    initial.copy(),
                    action_values.copy(),
                    cached_trajectory,
                    self._latent_space,
                    cache_hit=True,
                    evaluation=self._evaluate(cached_trajectory, action_values),
                )

        trajectory = self._execute(initial, action_values, profiler=profiler, metadata=metadata)
        if key is not None:
            self._cache_set(key, trajectory, profiler=profiler)
        return RolloutResult(
            initial.copy(),
            action_values.copy(),
            trajectory,
            self._latent_space,
            cache_hit=False,
            evaluation=self._evaluate(trajectory, action_values),
        )

    def evaluate(self, result: RolloutResult, *, source: str = "imagined") -> RewardValueEvaluationResult:
        """Score an existing rollout result with the configured evaluator."""

        if self.evaluator is None:
            raise RuntimeError("rollout pipeline has no reward/value evaluator configured")
        return self.evaluator.evaluate(result.trajectory, result.actions, source=source)

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

    def stream(
        self,
        initial_state: np.ndarray | LatentValue,
        action_chunks: Iterable[object],
        *,
        max_chunk_rows: int = 1024,
        profiler: RuntimeProfiler | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> Generator[Trajectory, None, None]:
        """Yield predictive states for sequential action chunks.

        Chunks are disjoint and ordered.  The initial state is carried into
        the first chunk but is not repeated in its output; flattening the
        yielded trajectories therefore equals ``run(...).trajectory[1:]``.
        The next input chunk is not requested until the current chunk has
        been fully processed, which gives this concrete rollout story
        one-chunk backpressure and no prefetch queue.  A transition error
        fails the current chunk before it is yielded; previously yielded
        chunks remain the caller's partial result.
        """

        if type(max_chunk_rows) is not int:
            raise TypeError("max_chunk_rows must be a positive integer")
        if max_chunk_rows < 1:
            raise ValueError("max_chunk_rows must be positive")
        state = self._stream_initial(initial_state)
        iterator = iter(action_chunks)
        chunk_index = 0
        profile_seconds = 0.0
        profile_chunks = 0
        profile_rows = 0
        try:
            for raw_chunk in iterator:
                actions = self._validate_stream_actions(raw_chunk, max_chunk_rows=max_chunk_rows)
                if actions.shape[0] == 0:
                    chunk_index += 1
                    continue
                state, values, elapsed = self._execute_stream_chunk(
                    state,
                    actions,
                )
                profile_seconds += elapsed
                profile_chunks += 1
                profile_rows += actions.shape[0]
                yield Trajectory(values, metadata=self._stream_metadata(metadata, chunk_index, actions.shape[0]))
                chunk_index += 1
        finally:
            close = getattr(iterator, "close", None)
            if callable(close):
                with suppress(Exception):
                    close()
            if profiler is not None and profile_chunks:
                profiler.record(
                    "transition",
                    profile_seconds,
                    component=type(self.transition).__name__,
                    chunk_count=profile_chunks,
                    row_count=profile_rows,
                )

    async def stream_async(
        self,
        initial_state: np.ndarray | LatentValue,
        action_chunks: Iterable[object],
        *,
        max_chunk_rows: int = 1024,
        profiler: RuntimeProfiler | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> AsyncGenerator[Trajectory, None]:
        """Asynchronously yield :meth:`stream` chunks without blocking.

        Synchronous producers and transition work run in worker threads.  No
        chunk is prefetched, so consumer pacing applies directly.  Cancellation
        is observed at the next await boundary; an already-running worker
        finishes its current bounded chunk but its result is discarded.
        """

        if type(max_chunk_rows) is not int:
            raise TypeError("max_chunk_rows must be a positive integer")
        if max_chunk_rows < 1:
            raise ValueError("max_chunk_rows must be positive")
        chunk_index = 0
        profile_seconds = 0.0
        profile_chunks = 0
        profile_rows = 0
        iterator: object | None = None
        try:
            state = await _settled_thread_call(self._stream_initial, initial_state)
            iterator = await _settled_thread_call(iter, action_chunks)
            while True:
                raw_chunk = await _settled_thread_call(next, iterator, _STREAM_END)
                if raw_chunk is _STREAM_END:
                    return
                actions = await _settled_thread_call(
                    self._validate_stream_actions, raw_chunk, max_chunk_rows=max_chunk_rows
                )
                if actions.shape[0] == 0:
                    chunk_index += 1
                    continue
                next_state, values, elapsed = await _settled_thread_call(
                    self._execute_stream_chunk,
                    state,
                    actions,
                )
                state = next_state
                profile_seconds += elapsed
                profile_chunks += 1
                profile_rows += actions.shape[0]
                yield Trajectory(values, metadata=self._stream_metadata(metadata, chunk_index, actions.shape[0]))
                chunk_index += 1
        finally:
            close_target = action_chunks if iterator is None else iterator
            with suppress(Exception):
                await _settled_thread_call(_close_iterator, close_target, action_chunks)
            if profiler is not None and profile_chunks:
                profiler.record(
                    "transition",
                    profile_seconds,
                    component=type(self.transition).__name__,
                    chunk_count=profile_chunks,
                    row_count=profile_rows,
                )

    def _stream_initial(self, initial_state: np.ndarray | LatentValue) -> np.ndarray:
        reset = getattr(self.transition, "reset", None)
        if callable(reset):
            reset()
        elif getattr(self.transition, "stream_state_contract", None) != "explicit":
            raise TypeError("streaming requires transition reset() or stream_state_contract='explicit'")
        empty_actions = np.empty((0, self.transition.action_dim), dtype=np.float64)
        initial, _ = self._validate_inputs(initial_state, empty_actions)
        return initial

    def _validate_stream_actions(self, actions: object, *, max_chunk_rows: int) -> np.ndarray:
        if type(actions) is not np.ndarray:
            raise TypeError("stream action chunks must be exact numpy.ndarray values")
        action_values = actions
        if action_values.ndim != 2 or action_values.shape[1] != self.transition.action_dim:
            raise ValueError(
                f"stream action chunks must have shape (n, {self.transition.action_dim}), got {action_values.shape}"
            )
        if action_values.shape[0] > max_chunk_rows:
            raise ValueError("stream action chunk exceeds max_chunk_rows")
        if not np.issubdtype(action_values.dtype, np.number) or not np.isfinite(action_values).all():
            raise ValueError("stream action chunks must contain finite numeric values")
        return np.asarray(action_values, dtype=np.float64)

    def _execute_stream_chunk(
        self,
        state: np.ndarray,
        actions: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        start = perf_counter()
        current = state
        values: list[np.ndarray] = []
        for action in actions:
            current = np.asarray(self.transition.step(current, action), dtype=np.float64)
            if current.shape != (self.transition.state_dim,) or not np.isfinite(current).all():
                raise ValueError("transition produced an invalid stream state")
            values.append(current.copy())
        result = np.stack(values, axis=0)
        return current, result, perf_counter() - start

    def _stream_metadata(
        self, metadata: Mapping[str, object] | None, chunk_index: int, chunk_rows: int
    ) -> dict[str, object]:
        values = {} if metadata is None else dict(metadata)
        values.update(
            {
                "pipeline": self.__class__.__name__,
                "pipeline_kind": self.pipeline_kind,
                "stream": True,
                "chunk_index": chunk_index,
                "chunk_rows": chunk_rows,
                "source_space_identity": self.transition.source_space_identity,
            }
        )
        return values

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
            """Run the transition's predictive-mean rollout."""
            return self.transition.mean_rollout(initial, actions, metadata=values)

        if profiler is None:
            return operation()
        start = perf_counter()
        result = operation()
        profiler.record("transition", perf_counter() - start, component=type(self.transition).__name__)
        return result

    def _evaluate(self, trajectory: Trajectory, actions: np.ndarray) -> RewardValueEvaluationResult | None:
        if self.evaluator is None:
            return None
        return self.evaluator.evaluate(trajectory, actions, source="imagined")

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

    def _cache_key(
        self,
        initial: np.ndarray,
        actions: np.ndarray,
        *,
        metadata: Mapping[str, object] | None,
    ) -> CacheKey:
        payload = np.concatenate([initial.ravel(), actions.ravel()])
        return make_cache_key(
            namespace="rollout_pipeline",
            operation="transition.mean_rollout",
            component=self.transition,
            data=payload,
            extra=dict(metadata or {}),
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

"""Pipeline #2: focused orchestration for Layer B manipulation stories."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from time import perf_counter
from typing import TypeVar, cast

import numpy as np

from latent_anything.adapters.protocols import FlatBatchDecodableAdapter
from latent_anything.latent_space import LatentSpace
from latent_anything.methods.b_protocols import BMethod
from latent_anything.pipeline_contract import PipelineContract, PipelineKind
from latent_anything.runtime.profiling import RuntimeProfiler, RuntimeStage
from latent_anything.trajectory import Trajectory

_T = TypeVar("_T")


class ManipulationPipeline(PipelineContract):
    """Compose a B-method with either data-space or trajectory execution."""

    pipeline_kind: PipelineKind = "manipulation"

    def __init__(self, method: BMethod, adapter: FlatBatchDecodableAdapter | None = None) -> None:
        self._method = method
        self._adapter = adapter
        method_space = getattr(method, "space", None)
        self._latent_space: LatentSpace | None = (
            method_space if method_space is not None else (adapter.latent_space if adapter is not None else None)
        )

    @property
    def method(self) -> BMethod:
        return self._method

    @property
    def adapter(self) -> FlatBatchDecodableAdapter | None:
        return self._adapter

    @property
    def latent_space(self) -> LatentSpace | None:
        return self._latent_space

    def fit(self, *args: object, **kwargs: object) -> None:
        """Delegate fitting to stateful B-methods."""

        fit_fn = getattr(self._method, "fit", None)
        if fit_fn is None:
            raise TypeError(
                f"{type(self._method).__name__} has no fit method (stateless methods like Lerp skip the fit phase)"
            )
        fit_fn(*args, **kwargs)

    def run_data(self, data: np.ndarray, *, profiler: RuntimeProfiler | None = None) -> np.ndarray:
        """Run the adapter-mediated data-space story."""

        if self._adapter is None:
            raise RuntimeError(
                "No adapter provided — cannot run data-space pipeline. "
                "Provide a FlatBatchDecodableAdapter at construction."
            )
        adapter = self._adapter
        if hasattr(self._method, "apply_latent"):
            latents = self._profile_sync(
                "encode", lambda: adapter.encode(data), profiler=profiler, component=type(adapter).__name__
            )
            patched = self._profile_sync(
                "method",
                lambda: self._apply_method_latent(latents),
                profiler=profiler,
                component=type(self._method).__name__,
            )
            return self._profile_sync(
                "decode", lambda: adapter.decode(patched), profiler=profiler, component=type(adapter).__name__
            )
        return self._profile_sync(
            "method", lambda: self._call_data_method(data), profiler=profiler, component=type(self._method).__name__
        )

    async def run_data_async(self, data: np.ndarray, *, profiler: RuntimeProfiler | None = None) -> np.ndarray:
        """Run the data-space story asynchronously; cancellation propagates."""

        if self._adapter is None:
            raise RuntimeError(
                "No adapter provided — cannot run data-space pipeline. "
                "Provide a FlatBatchDecodableAdapter at construction."
            )
        adapter = self._adapter
        if hasattr(self._method, "apply_latent"):
            latents = await self._profile_async(
                "encode", lambda: adapter.encode(data), profiler=profiler, component=type(adapter).__name__
            )
            patched = await self._profile_async(
                "method",
                lambda: self._apply_method_latent(latents),
                profiler=profiler,
                component=type(self._method).__name__,
            )
            return await self._profile_async(
                "decode", lambda: adapter.decode(patched), profiler=profiler, component=type(adapter).__name__
            )
        return await self._profile_async(
            "method", lambda: self._call_data_method(data), profiler=profiler, component=type(self._method).__name__
        )

    def run_trajectory(
        self, trajectory: Trajectory, *, profiler: RuntimeProfiler | None = None, **kwargs: object
    ) -> np.ndarray | Trajectory:
        """Apply a B-method to a latent trajectory."""

        apply_trajectory = cast(Callable[..., np.ndarray | Trajectory], self._method.apply_trajectory)
        return self._profile_sync(
            "method",
            lambda: apply_trajectory(trajectory, **kwargs),
            profiler=profiler,
            component=type(self._method).__name__,
        )

    async def run_trajectory_async(
        self, trajectory: Trajectory, *, profiler: RuntimeProfiler | None = None, **kwargs: object
    ) -> np.ndarray | Trajectory:
        """Apply a B-method to a latent trajectory asynchronously."""

        apply_trajectory = cast(Callable[..., np.ndarray | Trajectory], self._method.apply_trajectory)
        return await self._profile_async(
            "method",
            lambda: apply_trajectory(trajectory, **kwargs),
            profiler=profiler,
            component=type(self._method).__name__,
        )

    def fit_run_data(
        self,
        fit_args: tuple[object, ...] = (),
        fit_kwargs: dict[str, object] | None = None,
        data: np.ndarray | None = None,
    ) -> np.ndarray:
        self.fit(*fit_args, **(fit_kwargs or {}))
        return self.run_data(data) if data is not None else np.array([])

    def fit_run_trajectory(
        self,
        fit_args: tuple[object, ...] = (),
        fit_kwargs: dict[str, object] | None = None,
        trajectory: Trajectory | None = None,
        **apply_kwargs: object,
    ) -> np.ndarray | Trajectory | None:
        self.fit(*fit_args, **(fit_kwargs or {}))
        if trajectory is None:
            return None
        profiler = cast(RuntimeProfiler | None, apply_kwargs.pop("profiler", None))
        return self.run_trajectory(trajectory, profiler=profiler, **apply_kwargs)

    def _apply_method_latent(self, latents: np.ndarray) -> np.ndarray:
        apply_latent = getattr(self._method, "apply_latent", None)
        if not callable(apply_latent):
            raise RuntimeError(f"{type(self._method).__name__} does not expose apply_latent()")
        return cast(Callable[[np.ndarray], np.ndarray], apply_latent)(latents)

    def _call_data_method(self, data: np.ndarray) -> np.ndarray:
        return cast(Callable[[np.ndarray], np.ndarray], self._method)(data)

    @staticmethod
    def _profile_sync(
        stage: RuntimeStage, operation: Callable[[], _T], *, profiler: RuntimeProfiler | None, component: str
    ) -> _T:
        return operation() if profiler is None else profiler.measure(stage, operation, component=component)

    @staticmethod
    async def _profile_async(
        stage: RuntimeStage, operation: Callable[[], _T], *, profiler: RuntimeProfiler | None, component: str
    ) -> _T:
        if profiler is None:
            return await asyncio.to_thread(operation)
        start = perf_counter()
        result = await asyncio.to_thread(operation)
        profiler.record(stage, perf_counter() - start, component=component)
        return result

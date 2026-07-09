"""BatchExecutor #1 — eager, synchronous first-axis batching.

This module starts Layer C with a deliberately small runtime primitive:
split numpy arrays into deterministic first-axis chunks, call an existing
adapter or method on each chunk, and concatenate the outputs in original
order.

No cache, async execution, worker pool, prefetching, or DAG abstraction is
introduced in this sprint.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterator
from time import perf_counter
from typing import cast

import numpy as np

from latent_anything.adapters.protocols import DecodableAdapter, ModelAdapter
from latent_anything.methods.protocols import Method
from latent_anything.runtime.profiling import RuntimeProfiler


class BatchExecutor:
    """Eager first-axis batch executor for numpy array calls.

    Parameters
    ----------
    batch_size : int
        Maximum number of first-axis samples passed to each operation call.
        Must be a positive integer.

    Notes
    -----
    This is Runtime instance #1. It is intentionally concrete and narrow:
    one input array, one numpy output array, executed synchronously in the
    current thread.
    """

    def __init__(self, batch_size: int) -> None:
        if type(batch_size) is not int:
            msg = f"batch_size must be a positive integer, got {batch_size!r}"
            raise TypeError(msg)
        if batch_size < 1:
            msg = f"batch_size must be >= 1, got {batch_size}"
            raise ValueError(msg)
        self.batch_size = batch_size

    def iter_chunks(self, data: np.ndarray) -> Iterator[np.ndarray]:
        """Yield deterministic first-axis chunks from ``data``.

        Parameters
        ----------
        data : np.ndarray
            Array with at least one dimension. The first axis is treated
            as the batch axis.

        Yields
        ------
        np.ndarray
            Views into ``data`` with at most ``batch_size`` samples.
        """
        self._validate_batchable(data)
        for start in range(0, data.shape[0], self.batch_size):
            stop = min(start + self.batch_size, data.shape[0])
            yield data[start:stop]

    def map_array(self, operation: Callable[[np.ndarray], object], data: np.ndarray) -> np.ndarray:
        """Apply ``operation`` to each chunk and concatenate outputs.

        Parameters
        ----------
        operation : Callable[[np.ndarray], np.ndarray]
            Synchronous function that accepts one numpy array chunk and
            returns one numpy array with a matching first-axis length.
        data : np.ndarray
            Input array batched along axis 0.

        Returns
        -------
        np.ndarray
            Concatenated outputs in the same order as ``data``.
        """
        self._validate_batchable(data)
        if data.shape[0] == 0:
            return self._call_operation(operation, data)

        outputs = [self._call_operation(operation, chunk) for chunk in self.iter_chunks(data)]
        return self._concatenate_outputs(outputs, expected_rows=data.shape[0])

    async def map_array_async(self, operation: Callable[[np.ndarray], object], data: np.ndarray) -> np.ndarray:
        """Asynchronously apply ``operation`` to each chunk and concatenate outputs."""
        return await asyncio.to_thread(self.map_array, operation, data)

    def encode(
        self,
        adapter: ModelAdapter,
        data: np.ndarray,
        *,
        profiler: RuntimeProfiler | None = None,
    ) -> np.ndarray:
        """Batch an adapter ``encode`` call."""
        if profiler is None:
            return self.map_array(adapter.encode, data)
        return profiler.measure(
            "encode",
            lambda: self.map_array(adapter.encode, data),
            component=type(adapter).__name__,
        )

    async def encode_async(
        self,
        adapter: ModelAdapter,
        data: np.ndarray,
        *,
        profiler: RuntimeProfiler | None = None,
    ) -> np.ndarray:
        """Asynchronously batch an adapter ``encode`` call."""
        if profiler is None:
            return await self.map_array_async(adapter.encode, data)
        start = perf_counter()
        result = await self.map_array_async(adapter.encode, data)
        profiler.record("encode", perf_counter() - start, component=type(adapter).__name__)
        return result

    def decode(
        self,
        adapter: DecodableAdapter,
        latent: np.ndarray,
        *,
        profiler: RuntimeProfiler | None = None,
    ) -> np.ndarray:
        """Batch an adapter ``decode`` call."""
        if profiler is None:
            return self.map_array(adapter.decode, latent)
        return profiler.measure(
            "decode",
            lambda: self.map_array(adapter.decode, latent),
            component=type(adapter).__name__,
        )

    async def decode_async(
        self,
        adapter: DecodableAdapter,
        latent: np.ndarray,
        *,
        profiler: RuntimeProfiler | None = None,
    ) -> np.ndarray:
        """Asynchronously batch an adapter ``decode`` call."""
        if profiler is None:
            return await self.map_array_async(adapter.decode, latent)
        start = perf_counter()
        result = await self.map_array_async(adapter.decode, latent)
        profiler.record("decode", perf_counter() - start, component=type(adapter).__name__)
        return result

    def transform(
        self,
        method: Method,
        data: np.ndarray,
        *,
        profiler: RuntimeProfiler | None = None,
    ) -> np.ndarray:
        """Batch a fitted Layer A method ``transform`` call."""
        if profiler is None:
            return self.map_array(method.transform, data)
        return profiler.measure(
            "method",
            lambda: self.map_array(method.transform, data),
            component=type(method).__name__,
        )

    async def transform_async(
        self,
        method: Method,
        data: np.ndarray,
        *,
        profiler: RuntimeProfiler | None = None,
    ) -> np.ndarray:
        """Asynchronously batch a fitted Layer A method ``transform`` call."""
        if profiler is None:
            return await self.map_array_async(method.transform, data)
        start = perf_counter()
        result = await self.map_array_async(method.transform, data)
        profiler.record("method", perf_counter() - start, component=type(method).__name__)
        return result

    @staticmethod
    def _validate_batchable(data: np.ndarray) -> None:
        if data.ndim < 1:
            msg = "data must have at least one dimension for first-axis batching"
            raise ValueError(msg)

    @staticmethod
    def _call_operation(operation: Callable[[np.ndarray], object], chunk: np.ndarray) -> np.ndarray:
        output = operation(chunk)
        if not isinstance(output, np.ndarray):
            msg = f"batched operation must return numpy.ndarray, got {type(output).__name__}"
            raise TypeError(msg)
        output_array = cast(np.ndarray, output)
        if output_array.ndim < 1:
            msg = "batched operation output must have at least one dimension"
            raise ValueError(msg)
        if output_array.shape[0] != chunk.shape[0]:
            msg = (
                "batched operation must preserve the first-axis length per chunk: "
                f"input chunk has {chunk.shape[0]} rows, output has {output_array.shape[0]}"
            )
            raise ValueError(msg)
        return output_array

    @staticmethod
    def _concatenate_outputs(outputs: list[np.ndarray], *, expected_rows: int) -> np.ndarray:
        result = outputs[0] if len(outputs) == 1 else np.concatenate(outputs, axis=0)
        if result.shape[0] != expected_rows:
            msg = f"batched output row count mismatch: expected {expected_rows}, got {result.shape[0]}"
            raise ValueError(msg)
        return result

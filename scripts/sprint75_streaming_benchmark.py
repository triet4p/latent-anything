"""Offline CPU benchmark for bounded RolloutPipeline streaming."""

from __future__ import annotations

import hashlib
import json
import time
import tracemalloc

import numpy as np

from latent_anything import DeterministicLatentTransition, LatentSpace, RolloutPipeline
from latent_anything.runtime import RuntimeProfiler


def _transition() -> DeterministicLatentTransition:
    states = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    actions = np.ones((4, 1), dtype=np.float64)
    return DeterministicLatentTransition(LatentSpace(2, source_model="sprint75-benchmark"), 1).fit(
        states, actions, states + np.array([1.0, 0.0])
    )


def run_benchmark(*, horizon: int = 4096, chunk_rows: int = 64) -> dict[str, object]:
    if horizon < 1 or chunk_rows < 1:
        raise ValueError("horizon and chunk_rows must be positive")
    actions = np.ones((horizon, 1), dtype=np.float64)
    pipeline = RolloutPipeline(_transition())

    eager_start = time.perf_counter()
    eager = pipeline.run(np.zeros(2), actions)
    eager_seconds = time.perf_counter() - eager_start
    eager_tail = eager.to_numpy()[1:]
    eager_digest = hashlib.sha256(eager_tail.tobytes()).hexdigest()

    def action_chunks():
        for start in range(0, horizon, chunk_rows):
            yield np.ones((min(chunk_rows, horizon - start), 1), dtype=np.float64)

    profiler = RuntimeProfiler()
    max_chunk_bytes = 0
    streamed_rows = 0
    stream_digest = hashlib.sha256()
    tracemalloc.start()
    stream_start = time.perf_counter()
    for chunk in pipeline.stream(np.zeros(2), action_chunks(), max_chunk_rows=chunk_rows, profiler=profiler):
        values = chunk.to_numpy()
        max_chunk_bytes = max(max_chunk_bytes, values.nbytes)
        streamed_rows += values.shape[0]
        stream_digest.update(values.tobytes())
        del values
    stream_seconds = time.perf_counter() - stream_start
    _, stream_peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    result: dict[str, object] = {
        "status": "pass" if streamed_rows == horizon and stream_digest.hexdigest() == eager_digest else "fail",
        "horizon": horizon,
        "chunk_rows": chunk_rows,
        "queue_capacity": 1,
        "streamed_rows": streamed_rows,
        "eager_seconds": round(eager_seconds, 6),
        "stream_seconds": round(stream_seconds, 6),
        "eager_output_bytes": int(eager_tail.nbytes),
        "stream_max_chunk_bytes": int(max_chunk_bytes),
        "stream_peak_tracemalloc_bytes": int(stream_peak_bytes),
        "profile_events": len(profiler.snapshot().events),
        "eager_digest": eager_digest,
        "stream_digest": stream_digest.hexdigest(),
    }
    print(json.dumps(result, sort_keys=True))
    return result


if __name__ == "__main__":
    run_benchmark()

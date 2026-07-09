"""End-to-end demo: InMemoryCache speedup through AnalysisPipeline.

Usage:
    uv run python scripts/end_to_end_cache_demo.py

This script uses small local components with deliberate sleep delays so
the cache benefit is visible and repeatable. The exercised path is real:
``AnalysisPipeline(adapter, method, cache=InMemoryCache()).run(data)``.
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter, sleep

import numpy as np

from latent_anything import InMemoryCache, LatentSpace
from latent_anything.pipeline import AnalysisPipeline


class SlowAdapter:
    """Small adapter with deliberately slow encode."""

    def __init__(self, *, input_dim: int, latent_dim: int, delay_seconds: float) -> None:
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.delay_seconds = delay_seconds

    @property
    def latent_space(self) -> LatentSpace:
        return LatentSpace(dim=self.latent_dim, source_model="slow_demo_adapter")

    def encode(self, data: np.ndarray) -> np.ndarray:
        sleep(self.delay_seconds)
        return np.tanh(data[:, : self.latent_dim])


class SlowMethod:
    """Small Layer A method with deliberately slow fit and transform."""

    def __init__(self, *, delay_seconds: float) -> None:
        self.delay_seconds = delay_seconds
        self._mean: np.ndarray | None = None

    def fit(self, data: np.ndarray) -> None:
        sleep(self.delay_seconds)
        self._mean = data.mean(axis=0)

    def transform(self, data: np.ndarray) -> np.ndarray:
        if self._mean is None:
            msg = "SlowMethod must be fitted before transform"
            raise RuntimeError(msg)
        sleep(self.delay_seconds)
        return data - self._mean


def timed_run(pipeline: AnalysisPipeline, data: np.ndarray) -> tuple[float, tuple[int, int]]:
    """Run the pipeline and return elapsed seconds plus output shapes."""
    start = perf_counter()
    result = pipeline.run(data)
    elapsed = perf_counter() - start
    return elapsed, (result.latents.shape[0], result.transformed.shape[1])


def main() -> None:
    """Run the repeated-call cache demo."""
    rng = np.random.default_rng(42)
    data = rng.normal(size=(1_000, 8)).astype(np.float64)
    cache = InMemoryCache()
    pipeline = AnalysisPipeline(
        adapter=SlowAdapter(input_dim=8, latent_dim=4, delay_seconds=0.02),
        method=SlowMethod(delay_seconds=0.02),
        cache=cache,
    )

    first_seconds, output_shape_hint = timed_run(pipeline, data)
    second_seconds, _ = timed_run(pipeline, data)
    speedup = first_seconds / second_seconds if second_seconds > 0 else float("inf")
    stats = cache.stats

    lines = [
        "Sprint 23 InMemoryCache demo",
        f"data_shape={data.shape}",
        f"output_shape_hint={output_shape_hint}",
        f"first_run={first_seconds * 1000:.3f} ms",
        f"second_run_cached={second_seconds * 1000:.3f} ms",
        f"speedup={speedup:.2f}x",
        f"cache_stats=hits:{stats.hits},misses:{stats.misses},sets:{stats.sets},size:{stats.size}",
    ]

    for line in lines:
        print(line)

    output_path = Path("artifacts/cache_demo_summary.txt")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nSummary written to {output_path}")


if __name__ == "__main__":
    main()

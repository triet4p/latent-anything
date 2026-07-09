"""End-to-end demo: async runtime wrappers + profiling hooks.

Usage:
    uv run python scripts/end_to_end_async_runtime_demo.py

Runs two independent pipeline jobs concurrently:

1. ``AnalysisPipeline.run_async()`` with cache + profiling.
2. ``ManipulationPipeline.run_data_async()`` with staged encode/method/decode profiling.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from time import perf_counter, sleep

import numpy as np

from latent_anything import InMemoryCache, LatentSpace, RuntimeProfiler
from latent_anything.pipeline import AnalysisPipeline, ManipulationPipeline
from latent_anything.trajectory import Trajectory


class SlowFlatBatchAdapter:
    """Small adapter with deliberate delays so async overlap is visible."""

    def __init__(
        self,
        *,
        input_dim: int,
        latent_dim: int,
        encode_delay_seconds: float,
        decode_delay_seconds: float,
    ) -> None:
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encode_delay_seconds = encode_delay_seconds
        self.decode_delay_seconds = decode_delay_seconds

    @property
    def latent_space(self) -> LatentSpace:
        return LatentSpace(dim=self.latent_dim, source_model="async_runtime_demo")

    @property
    def supports_flat_batch(self) -> bool:
        return True

    def encode(self, data: np.ndarray) -> np.ndarray:
        sleep(self.encode_delay_seconds)
        return np.tanh(data[:, : self.latent_dim])

    def decode(self, latent: np.ndarray) -> np.ndarray:
        sleep(self.decode_delay_seconds)
        return latent + 0.25


class SlowMethod:
    """Small Layer A method with visible fit/transform latency."""

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


class SlowPatchMethod:
    """Small adapter-mediated method with visible latent patch latency."""

    def __init__(self, *, delta: np.ndarray, delay_seconds: float) -> None:
        self._delta = delta
        self.delay_seconds = delay_seconds

    @property
    def space(self) -> LatentSpace | None:
        return None

    @property
    def is_fitted(self) -> bool:
        return True

    def apply_latent(self, latent: np.ndarray) -> np.ndarray:
        sleep(self.delay_seconds)
        return latent + self._delta

    def apply_trajectory(self, trajectory: Trajectory, **kwargs: object) -> Trajectory:
        _ = kwargs
        return Trajectory(data=self.apply_latent(trajectory.to_numpy()))

    def __call__(self, data: np.ndarray) -> np.ndarray:
        _ = data
        msg = "SlowPatchMethod should use apply_latent through ManipulationPipeline"
        raise RuntimeError(msg)


def format_stage_breakdown(title: str, profiler: RuntimeProfiler) -> list[str]:
    """Return human-readable stage totals for one profiler snapshot."""
    profile = profiler.snapshot()
    totals = profile.stage_totals()
    lines = [title]
    for stage in ("cache", "encode", "method", "decode"):
        if stage in totals:
            lines.append(f"  {stage}: {totals[stage] * 1000:.3f} ms")
    lines.append(f"  total_recorded: {profile.total_seconds * 1000:.3f} ms")
    return lines


async def main() -> None:
    """Run the async runtime demo and write a text summary artifact."""
    rng = np.random.default_rng(42)

    analysis_data = rng.normal(size=(1_000, 8)).astype(np.float64)
    manipulation_data = rng.normal(size=(800, 8)).astype(np.float64)

    analysis_profiler = RuntimeProfiler()
    analysis_pipeline = AnalysisPipeline(
        adapter=SlowFlatBatchAdapter(
            input_dim=8,
            latent_dim=4,
            encode_delay_seconds=0.03,
            decode_delay_seconds=0.01,
        ),
        method=SlowMethod(delay_seconds=0.03),
        cache=InMemoryCache(),
    )

    manipulation_profiler = RuntimeProfiler()
    manipulation_pipeline = ManipulationPipeline(
        method=SlowPatchMethod(delta=np.array([0.2, -0.1, 0.05, 0.3], dtype=np.float64), delay_seconds=0.03),
        adapter=SlowFlatBatchAdapter(
            input_dim=8,
            latent_dim=4,
            encode_delay_seconds=0.03,
            decode_delay_seconds=0.03,
        ),
    )

    wall_start = perf_counter()
    analysis_result, manipulation_result = await asyncio.gather(
        analysis_pipeline.run_async(analysis_data, profiler=analysis_profiler),
        manipulation_pipeline.run_data_async(manipulation_data, profiler=manipulation_profiler),
    )
    wall_seconds = perf_counter() - wall_start

    lines = [
        "Sprint 24 async runtime demo",
        f"analysis_latents_shape={analysis_result.latents.shape}",
        f"analysis_transformed_shape={analysis_result.transformed.shape}",
        f"manipulation_output_shape={manipulation_result.shape}",
        f"concurrent_wall_time={wall_seconds * 1000:.3f} ms",
    ]
    lines.extend(format_stage_breakdown("AnalysisPipeline breakdown", analysis_profiler))
    lines.extend(format_stage_breakdown("ManipulationPipeline breakdown", manipulation_profiler))

    for line in lines:
        print(line)

    output_path = Path("artifacts/async_runtime_demo_summary.txt")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nSummary written to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())

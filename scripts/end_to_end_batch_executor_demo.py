"""End-to-end demo: BatchExecutor direct vs batched paths on synthetic data.

Usage:
    uv run python scripts/end_to_end_batch_executor_demo.py

This demo is intentionally benchmark-ish rather than a rigorous benchmark:
it gives rough local timings for direct and batched execution while proving
that the batched outputs preserve shape and order.
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np

from latent_anything.adapters import RandomProjection
from latent_anything.methods import PCA
from latent_anything.runtime import BatchExecutor


def mean_runtime_seconds(operation: object, repeats: int = 5) -> float:
    """Return mean runtime for a zero-argument callable."""
    if not callable(operation):
        msg = "operation must be callable"
        raise TypeError(msg)
    timings: list[float] = []
    for _ in range(repeats):
        start = perf_counter()
        operation()
        timings.append(perf_counter() - start)
    return float(np.mean(timings))


def format_row(name: str, direct_seconds: float, batched_seconds: float, max_abs_diff: float) -> str:
    """Format one result row for console and artifact output."""
    ratio = batched_seconds / direct_seconds if direct_seconds > 0 else float("inf")
    return (
        f"{name}: direct={direct_seconds * 1000:.3f} ms, "
        f"batched={batched_seconds * 1000:.3f} ms, "
        f"ratio={ratio:.2f}x, max_abs_diff={max_abs_diff:.3e}"
    )


def main() -> None:
    """Run the synthetic direct-vs-batched demo."""
    rng = np.random.default_rng(42)
    data = rng.normal(size=(20_000, 64)).astype(np.float64)
    executor = BatchExecutor(batch_size=512)

    adapter = RandomProjection(input_dim=64, latent_dim=16, random_state=42)
    direct_encode = adapter.encode(data)
    batched_encode = executor.encode(adapter, data)
    encode_diff = float(np.max(np.abs(direct_encode - batched_encode)))

    pca = PCA(n_components=8)
    pca.fit(direct_encode[:5_000])
    direct_transform = pca.transform(direct_encode)
    batched_transform = executor.transform(pca, direct_encode)
    transform_diff = float(np.max(np.abs(direct_transform - batched_transform)))

    encode_direct_seconds = mean_runtime_seconds(lambda: adapter.encode(data))
    encode_batched_seconds = mean_runtime_seconds(lambda: executor.encode(adapter, data))
    transform_direct_seconds = mean_runtime_seconds(lambda: pca.transform(direct_encode))
    transform_batched_seconds = mean_runtime_seconds(lambda: executor.transform(pca, direct_encode))

    lines = [
        "Sprint 22 BatchExecutor demo",
        f"data_shape={data.shape}",
        f"batch_size={executor.batch_size}",
        format_row("RandomProjection.encode", encode_direct_seconds, encode_batched_seconds, encode_diff),
        format_row("PCA.transform", transform_direct_seconds, transform_batched_seconds, transform_diff),
    ]

    for line in lines:
        print(line)

    output_path = Path("artifacts/batch_executor_demo_summary.txt")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nSummary written to {output_path}")


if __name__ == "__main__":
    main()

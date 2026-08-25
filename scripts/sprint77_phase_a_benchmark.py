#!/usr/bin/env python3
"""Reproducible, offline CPU performance evidence for Sprint 77 Phase A.

The harness deliberately measures framework-owned calls around fixed NumPy and
small Torch fixtures.  It records robust latency statistics and a correctness
digest, but never treats model or provider time as framework performance.  No
wall-clock assertion from this script is part of the default test suite.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import tempfile
import time
import tracemalloc
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

import numpy as np
import torch
from torch import nn

from latent_anything import (
    CEMConfig,
    CEMPlanner,
    DTWConfig,
    LatentSpace,
    LatentValue,
    MPPIConfig,
    MPPIPlanner,
    RolloutPipeline,
    compute_dtw,
)
from latent_anything.artifact_store import ArtifactStore
from latent_anything.capture import ActivationCaptureSession
from latent_anything.experiment_recorder import LocalExperimentRecorder
from latent_anything.geodesic import DensityGeodesic, GeodesicConfig
from latent_anything.integrations.lerobot import LeRobotCapturedLatent
from latent_anything.plugin_discovery import list_entry_points
from latent_anything.portable import decode_portable, encode_portable
from latent_anything.runtime import CacheKey
from latent_anything.runtime.disk_cache import SQLiteDiskCache, make_disk_cache_key
from latent_anything.transition import DeterministicLatentTransition

SCHEMA_VERSION = "sprint77-phase-a-v1"
SEED = 771


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    category: str
    attribution: str
    operation: Callable[[], object]


def _digest(value: object) -> str:
    """Hash deterministic result content without serializing arbitrary code."""

    digest = hashlib.sha256()

    def visit(item: object) -> None:
        if isinstance(item, np.ndarray):
            contiguous = np.ascontiguousarray(item)
            digest.update(b"array")
            digest.update(str(contiguous.dtype).encode())
            digest.update(repr(contiguous.shape).encode())
            digest.update(contiguous.tobytes())
        elif isinstance(item, (bytes, bytearray, memoryview)):
            digest.update(b"bytes")
            digest.update(bytes(item))
        elif isinstance(item, Mapping):
            digest.update(b"mapping")
            for key in sorted(item, key=str):
                digest.update(str(key).encode())
                visit(item[key])
        elif isinstance(item, (tuple, list)):
            digest.update(type(item).__name__.encode())
            for child in item:
                visit(child)
        elif isinstance(item, (float, int, str, bool, type(None))):
            digest.update(repr(item).encode())
        else:
            to_numpy = getattr(item, "to_numpy", None)
            if callable(to_numpy):
                visit(to_numpy())
            else:
                digest.update(type(item).__qualname__.encode())
                digest.update(repr(item).encode())

    visit(value)
    return digest.hexdigest()


def _environment() -> dict[str, object]:
    """Return allowlisted, non-secret execution metadata."""

    versions: dict[str, str] = {}
    for package in ("numpy", "torch", "pyarrow", "latent-anything"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "unknown"
    thread_vars = {
        key: os.environ[key]
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        if key in os.environ
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "python": platform.python_version(),
        "platform": platform.platform(aliased=True),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "cpu_count": os.cpu_count(),
        "versions": versions,
        "torch_threads": int(torch.get_num_threads()),
        "thread_environment": thread_vars,
        "seed": SEED,
    }


def _rss_bytes() -> int | None:
    """Return a best-effort peak RSS value without adding a dependency."""

    try:
        import resource

        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return value * (1024 if platform.system() != "Darwin" else 1)
    except (ImportError, AttributeError, OSError):
        return None


def _measure(case: BenchmarkCase, *, warmups: int, repetitions: int) -> dict[str, object]:
    for _ in range(warmups):
        case.operation()
    samples_us: list[float] = []
    peak = 0
    result: object = None
    tracemalloc.start()
    try:
        for _ in range(repetitions):
            started = time.perf_counter_ns()
            result = case.operation()
            samples_us.append((time.perf_counter_ns() - started) / 1_000.0)
            _, current_peak = tracemalloc.get_traced_memory()
            peak = max(peak, int(current_peak))
    finally:
        tracemalloc.stop()
    ordered = sorted(samples_us)
    p95_index = min(len(ordered) - 1, max(0, int(np.ceil(len(ordered) * 0.95)) - 1))
    return {
        "name": case.name,
        "category": case.category,
        "attribution": case.attribution,
        "warmups": warmups,
        "repetitions": repetitions,
        "latency_us": {
            "median": statistics.median(samples_us),
            "p95": ordered[p95_index],
            "mean": statistics.fmean(samples_us),
            "stdev": statistics.stdev(samples_us) if len(samples_us) > 1 else 0.0,
            "samples": samples_us,
        },
        "peak_tracemalloc_bytes": peak,
        "peak_rss_bytes": _rss_bytes(),
        "correctness_digest": _digest(result),
    }


def _transition() -> DeterministicLatentTransition:
    states = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    actions = np.ones((4, 1), dtype=np.float64)
    return DeterministicLatentTransition(LatentSpace(2, source_model="sprint77-phase-a"), 1).fit(
        states, actions, states + np.array([1.0, 0.0])
    )


def build_cases() -> tuple[BenchmarkCase, ...]:
    """Build fixed offline workloads covering existing framework seams."""

    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    space = LatentSpace(dim=16, source_model="sprint77-phase-a")
    left = rng.normal(size=16)
    right = rng.normal(size=16)
    query = rng.normal(size=(24, 16))
    reference = rng.normal(size=(32, 16))
    ring_a = np.array([1.8, 0.9])
    ring_b = np.array([1.8, -0.9])
    geodesic = DensityGeodesic(
        GeodesicConfig(n_points=16, max_iter=50, step_size=0.2, tol=1e-6, density_exponent=1.0)
    ).attach_density(lambda point: -float((np.linalg.norm(point) - 2.0) ** 2))
    transition = _transition()
    pipeline = RolloutPipeline(transition)
    actions = np.ones((32, 1), dtype=np.float64)
    value = LatentValue(rng.normal(size=(32, 16)), space)
    payload = encode_portable(value)
    cache_key = make_disk_cache_key(
        CacheKey("phase-a", "portable", "fixture", "config", "state", "data", "0.1"),
        plugin_identity="phase-a-plugin@1",
        checkpoint_identity="phase-a-checkpoint@1",
        behavior_state_identity="phase-a-state@1",
    )
    model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 4))
    model.eval()
    model_input = torch.from_numpy(rng.normal(size=(16, 16)).astype(np.float32))
    cem = CEMPlanner(
        CEMConfig(
            horizon=8,
            action_dim=2,
            lower_bounds=(-1.0, -1.0),
            upper_bounds=(1.0, 1.0),
            population_size=64,
            iterations=3,
            seed=SEED,
        )
    )
    mppi = MPPIPlanner(
        MPPIConfig(
            horizon=8,
            action_dim=2,
            lower_bounds=(-1.0, -1.0),
            upper_bounds=(1.0, 1.0),
            population_size=64,
            iterations=3,
            seed=SEED,
        )
    )
    target = np.zeros((8, 2), dtype=np.float64)

    def capture() -> object:
        with torch.no_grad(), ActivationCaptureSession(model, ["0"], source_model_version="phase-a") as session:
            model(model_input)
            return session.captures[0].values

    def stream() -> object:
        chunks = (np.ones((8, 1), dtype=np.float64) for _ in range(4))
        outputs = [chunk.to_numpy() for chunk in pipeline.stream(np.zeros(2), chunks, max_chunk_rows=8)]
        return np.concatenate(outputs, axis=0)

    def artifact_cache() -> object:
        with tempfile.TemporaryDirectory(prefix="latent-anything-phase-a-") as directory:
            root = Path(directory)
            store = ArtifactStore(root / "artifacts")
            cache = SQLiteDiskCache(root / "cache.sqlite")
            store.write("value.la", payload, artifact_type="latent-value", metadata={"lane": "phase-a"})
            cache.set_portable(cache_key, payload)
            return decode_portable(cache.get_portable(cache_key))

    def record() -> object:
        with tempfile.TemporaryDirectory(prefix="latent-anything-phase-a-recorder-") as directory:
            recorder = LocalExperimentRecorder(Path(directory))
            run = recorder.start_run("phase-a", config={"seed": SEED}, tags={"lane": "offline"})
            run.log_metrics({"latency": 1.0}, step=0)
            return run.finish().run_id

    return (
        BenchmarkCase(
            "geometry_distance", "geometry", "framework: LatentSpace.distance", lambda: space.distance(left, right)
        ),
        BenchmarkCase(
            "trajectory_dtw",
            "trajectory_alignment",
            "framework: compute_dtw; NumPy point costs and Python traceback",
            lambda: compute_dtw(query, reference, space, config=DTWConfig(max_cells=2_000_000)).distance,
        ),
        BenchmarkCase(
            "density_geodesic",
            "geometry",
            "framework: bounded density path optimizer",
            lambda: geodesic.optimize(ring_a, ring_b).path,
        ),
        BenchmarkCase(
            "activation_capture",
            "activation",
            "framework: hook and NumPy capture; model forward excluded from attribution",
            capture,
        ),
        BenchmarkCase(
            "rollout",
            "rollout",
            "framework: RolloutPipeline and transition execution",
            lambda: pipeline.run(np.zeros(2), actions).trajectory.to_numpy(),
        ),
        BenchmarkCase(
            "cem_planning",
            "planning",
            "framework: CEM sampling/update; vectorized objective fixture",
            lambda: cem.plan(lambda candidates: -np.sum(np.square(candidates - target), axis=(1, 2))).actions,
        ),
        BenchmarkCase(
            "mppi_planning",
            "planning",
            "framework: MPPI sampling/update; vectorized objective fixture",
            lambda: mppi.plan(lambda candidates: -np.sum(np.square(candidates - target), axis=(1, 2))).actions,
        ),
        BenchmarkCase(
            "portable_encode_decode",
            "portable",
            "framework: Arrow portable codec",
            lambda: decode_portable(encode_portable(value)),
        ),
        BenchmarkCase(
            "artifact_and_disk_cache",
            "serialization_cache",
            "framework: ArtifactStore plus SQLiteDiskCache; filesystem/SQLite included",
            artifact_cache,
        ),
        BenchmarkCase(
            "bounded_streaming", "streaming", "framework: RolloutPipeline.stream; one chunk in flight", stream
        ),
        BenchmarkCase("local_recorder", "recorder", "framework: LocalExperimentRecorder filesystem lifecycle", record),
        BenchmarkCase(
            "plugin_listing",
            "plugins",
            "framework: metadata-only importlib entry-point listing",
            lambda: list_entry_points(provider=lambda: {}),
        ),
        BenchmarkCase(
            "lerobot_numpy_boundary",
            "lerobot",
            "framework: offline LeRobot captured-latent NumPy boundary; no policy/model claim",
            lambda: LeRobotCapturedLatent(left).to_numpy(),
        ),
    )


def run_suite(*, warmups: int = 2, repetitions: int = 12, output: Path | None = None) -> dict[str, object]:
    """Run all fixed cases and optionally write a versioned JSON artifact."""

    if warmups < 0 or repetitions < 2:
        raise ValueError("warmups must be >= 0 and repetitions must be >= 2")
    torch.set_num_threads(1)
    report: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "benchmark": "sprint77_phase_a",
        "environment": _environment(),
        "workload_contract": {
            "offline": True,
            "seed": SEED,
            "warmups": warmups,
            "repetitions": repetitions,
            "default_ci_policy": "semantic digest checks only; latency is advisory and environment-scoped",
        },
        "cases": [_measure(case, warmups=warmups, repetitions=repetitions) for case in build_cases()],
    }
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=12)
    parser.add_argument("--output", type=Path, default=Path("artifacts/sprint77_phase_a_benchmark.json"))
    args = parser.parse_args()
    report = run_suite(warmups=args.warmups, repetitions=args.repetitions, output=args.output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

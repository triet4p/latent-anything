#!/usr/bin/env python3
"""Offline CPU size/latency comparison for Sprint 74 portable paths."""

from __future__ import annotations

import json
import statistics
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from latent_anything.artifact_store import ArtifactStore
from latent_anything.latent_space import LatentSpace
from latent_anything.latent_value import LatentValue
from latent_anything.portable import decode_portable, encode_portable
from latent_anything.runtime import CacheKey
from latent_anything.runtime.disk_cache import SQLiteDiskCache, make_disk_cache_key


def _measure(operation: Callable[[], object], *, repeats: int = 10) -> float:
    for _ in range(2):
        operation()
    samples: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter_ns()
        operation()
        samples.append((time.perf_counter_ns() - started) / 1_000.0)
    return statistics.mean(samples)


def main() -> int:
    value = LatentValue(np.arange(4096, dtype=np.float32).reshape(64, 64), LatentSpace(dim=64))
    payload = encode_portable(value)
    key = make_disk_cache_key(
        CacheKey("benchmark", "portable", "fixture", "config", "state", "data", "0.1"),
        plugin_identity="fixture@1",
        checkpoint_identity="checkpoint@1",
        behavior_state_identity="state@1",
    )
    with tempfile.TemporaryDirectory(prefix="latent-anything-sprint74-bench-") as temporary:
        root = Path(temporary)
        store = ArtifactStore(root / "artifacts")
        cache = SQLiteDiskCache(root / "cache.sqlite")
        store.write("value.la", payload, artifact_type="latent-value", metadata={"fixture": "cpu"})
        cache.set_portable(key, payload)
        metrics: dict[str, float | int] = {
            "payload_bytes": len(payload),
            "stored_artifact_bytes": (root / "artifacts" / "value.la").stat().st_size,
            "arrow_encode_us": _measure(lambda: encode_portable(value)),
            "arrow_decode_us": _measure(lambda: decode_portable(payload)),
            "artifact_write_us": _measure(
                lambda: store.write("value.la", payload, artifact_type="latent-value", metadata={"fixture": "cpu"})
            ),
            "artifact_read_us": _measure(lambda: store.read("value.la")),
            "cache_set_us": _measure(lambda: cache.set_portable(key, payload)),
            "cache_get_us": _measure(lambda: cache.get_portable(key)),
            "in_memory_copy_us": _measure(lambda: value.to_numpy()),
        }
        print(json.dumps(metrics, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

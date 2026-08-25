#!/usr/bin/env python3
"""Offline CPU reproduction for Sprint 74 portable artifacts and cache parity."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from latent_anything.artifact_store import ArtifactStore
from latent_anything.cem import CEMIteration, CEMPlanResult
from latent_anything.latent_space import LatentSpace
from latent_anything.latent_value import LatentValue
from latent_anything.portable import decode_portable, encode_portable
from latent_anything.portable_results import decode_result_envelope, encode_result_envelope
from latent_anything.runtime import CacheKey
from latent_anything.runtime.disk_cache import SQLiteDiskCache, make_disk_cache_key
from latent_anything.runtime.profiling import ProfileEvent, RuntimeProfile
from latent_anything.trajectory import Trajectory


def _child(root: Path, cache_key: str) -> dict[str, object]:
    store = ArtifactStore(root / "artifacts")
    restored_value = decode_portable(store.read("value.la").payload)
    restored_trajectory = decode_portable(store.read("trajectory.la").payload)
    restored_result = decode_result_envelope(store.read("result.la").payload)
    cache = SQLiteDiskCache(root / "cache.sqlite")
    cached = cache.get_portable(cache_key)
    if not isinstance(restored_value, LatentValue) or not isinstance(restored_trajectory, Trajectory):
        raise RuntimeError("domain values did not restore in child process")
    if not isinstance(restored_result.value, CEMPlanResult):
        raise RuntimeError("typed result did not restore in child process")
    if cached is None:
        raise RuntimeError("disk cache miss in child process")
    cached_result = decode_result_envelope(cached)
    if not isinstance(cached_result.value, CEMPlanResult):
        raise RuntimeError("cached typed result did not restore")
    if not np.array_equal(restored_value.to_numpy(), np.array([1.0, 2.0], dtype=np.float32)):
        raise RuntimeError("latent value behavior changed after restore")
    if restored_trajectory.shape != (2, 2) or restored_result.value.actions.shape != (1, 2):
        raise RuntimeError("restored shape behavior changed")
    return {"child": "pass", "cache_hit": True, "result_identity": cached_result.identity}


def _parent() -> dict[str, object]:
    started = time.perf_counter()
    space = LatentSpace(dim=2, source_model="sprint74", metadata={"revision": "r1"})
    value = LatentValue(np.array([1.0, 2.0], dtype=np.float32), space, {"source": "fixture"})
    trajectory = Trajectory(np.arange(4, dtype=np.float32).reshape(2, 2), metadata={"source": "fixture"})
    iteration = CEMIteration(0, 4, 1, 1.0, 0.1, 1.2, 1.1, np.array([0.1, 0.2]), np.array([0.3, 0.4]))
    result = CEMPlanResult(
        np.array([[0.1, 0.2]], dtype=np.float64),
        1.2,
        (iteration,),
        (1.0, 1.2),
        RuntimeProfile((ProfileEvent("planning", 0.01, {}),)),
        7,
    )
    result_payload = encode_result_envelope(
        result,
        provenance={"plugin": "builtin-cem", "version": "0.1.0b1"},
        behavior_state={"checkpoint": "fixture-v1", "config_hash": "config-fixture"},
    )
    with tempfile.TemporaryDirectory(prefix="latent-anything-sprint74-") as temporary:
        root = Path(temporary)
        store = ArtifactStore(root / "artifacts")
        store.write("value.la", encode_portable(value), artifact_type="latent-value")
        store.write("trajectory.la", encode_portable(trajectory), artifact_type="trajectory")
        result_artifact = store.write("result.la", result_payload, artifact_type="cem-result")
        key = make_disk_cache_key(
            CacheKey("planning", "cem", "builtin-cem", "config-fixture", "state-fixture", "data-fixture", "0.1.0b1"),
            plugin_identity="builtin-cem@0.1.0b1",
            checkpoint_identity="fixture-v1",
            behavior_state_identity="state-fixture",
        )
        SQLiteDiskCache(root / "cache.sqlite").set_portable(key, result_payload)
        child = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--child", str(root), "--key", key],
            check=True,
            capture_output=True,
            text=True,
        )
        child_result = json.loads(child.stdout)
        elapsed = time.perf_counter() - started
        return {
            "status": "pass",
            "child": child_result,
            "artifact_bytes": len(result_payload),
            "artifact_identity": result_artifact.identity,
            "elapsed_seconds": round(elapsed, 6),
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", type=Path)
    parser.add_argument("--key")
    arguments = parser.parse_args()
    if arguments.child is not None:
        if arguments.key is None:
            parser.error("--key is required with --child")
        result = _child(arguments.child, arguments.key)
    else:
        result = _parent()
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

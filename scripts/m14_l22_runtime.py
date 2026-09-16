"""Exercise the local M14 L22 portable/runtime filesystem contract."""
from __future__ import annotations

import asyncio
import hashlib
import importlib.metadata
import json
import platform
import tempfile
import time
from pathlib import Path

import numpy as np
import psutil

from latent_anything.artifact_store import ArtifactStore, ArtifactStoreError
from latent_anything.config import ObjectSpec
from latent_anything.experiment_recorder import LocalExperimentRecorder, RecorderContractError
from latent_anything.latent_space import LatentSpace
from latent_anything.portable import decode_portable, encode_portable
from latent_anything.portable_results import decode_result_envelope, encode_result_envelope
from latent_anything.rollout_pipeline import RolloutPipeline
from latent_anything.runtime import InMemoryCache
from latent_anything.runtime.cache import CacheKey
from latent_anything.runtime.disk_cache import SQLiteDiskCache, make_disk_cache_key
from latent_anything.transition import DeterministicLatentTransition

RUN_COMMAND = "uv run python scripts/m14_l22_runtime.py"
SEED = 2201


def _transition() -> DeterministicLatentTransition:
    space = LatentSpace(2, source_model="m14-l22-runtime-fixture-v1")
    states = np.array([[0.0, 0.0], [1.0, 0.5], [2.0, 1.0], [3.0, 1.5]], dtype=np.float64)
    actions = np.ones((3, 1), dtype=np.float64)
    next_states = states[1:]
    return DeterministicLatentTransition(space, 1, source_space_identity="m14-l22-runtime-fixture-v1").fit(states[:-1], actions, next_states)


def main() -> None:
    started = time.perf_counter()
    process = psutil.Process()
    rng = np.random.default_rng(SEED)
    with tempfile.TemporaryDirectory(prefix="latent-anything-m14-l22-") as temp:
        root = Path(temp)
        store = ArtifactStore(root / "artifacts", max_artifact_bytes=1024 * 1024)
        payload = encode_portable({"array": rng.normal(size=(4, 2)), "schema": "portable-node-v1"})
        written = store.write("nested/evidence.arrow", payload, artifact_type="portable", metadata={"seed": SEED})
        restored = store.read("nested/evidence.arrow")
        traversal_rejected = False
        try:
            store.read("../escape")
        except ArtifactStoreError:
            traversal_rejected = True
        tamper_path = root / "artifacts" / "tamper.arrow"
        store.write("tamper.arrow", b"tamper-check", artifact_type="raw")
        tamper_bytes = bytearray(tamper_path.read_bytes())
        tamper_bytes[-1] ^= 1
        tamper_path.write_bytes(tamper_bytes)
        checksum_rejected = False
        try:
            store.read("tamper.arrow")
        except ArtifactStoreError:
            checksum_rejected = True
        portable_value = decode_portable(restored.payload)
        result_value = ObjectSpec(kind="method_a", name="pca", params={"n_components": 2})
        envelope = decode_result_envelope(encode_result_envelope(result_value, provenance={"lane": "L22"}, behavior_state={"seed": SEED}))
        result_roundtrip = envelope.value == result_value and envelope.provenance == {"lane": "L22"}

        cache_path = root / "cache" / "runtime.sqlite"
        key = make_disk_cache_key(
            CacheKey("m14", "portable", "fixture", "cfg-v1", "state-v1", "data-v1", "torch-2.10.0"),
            plugin_identity="builtin-runtime@1",
            checkpoint_identity="none",
            behavior_state_identity="seed-2201",
        )
        disk = SQLiteDiskCache(cache_path, max_bytes=1024 * 1024, max_entries=8)
        disk.set(key, payload)
        disk_hit = disk.get(key) == payload
        disk_reopened = SQLiteDiskCache(cache_path).get(key) == payload
        disk_miss = disk.get("0" * 64) is None

        pipeline_cache = InMemoryCache()
        pipeline = RolloutPipeline(_transition(), cache=pipeline_cache)
        initial = np.array([0.0, 0.0])
        actions = np.ones((4, 1), dtype=np.float64)
        first = pipeline.run(initial, actions, metadata={"episode": 1})
        second = pipeline.run(initial, actions, metadata={"episode": 1})
        changed = pipeline.run(initial, actions, metadata={"episode": 2})
        chunks = list(pipeline.stream(initial, [actions[:2], np.empty((0, 1)), actions[2:]], max_chunk_rows=2, metadata={"episode": 3}))
        streamed = np.concatenate([chunk.to_numpy() for chunk in chunks])
        eager = pipeline.run(initial, actions, metadata={"episode": 3}).trajectory.to_numpy()[1:]
        streaming_ok = np.array_equal(streamed, eager) and [chunk.metadata["chunk_index"] for chunk in chunks] == [0, 2]
        async_result = asyncio.run(pipeline.run_async(initial, actions, metadata={"episode": 4}))

        recorder = LocalExperimentRecorder(root / "runs")
        parent = recorder.start_run("m14-l22", config={"seed": SEED}, tags={"lane": "L22"}, code_version="git:fixture", framework_version="latent-anything", seeds=(SEED,))
        parent.log_params({"schema": "v1"})
        parent.log_metrics({"roundtrip": 1.0}, step=0)
        artifact = parent.log_artifact(b"runtime-evidence", name="runtime/evidence.bin")
        child = parent.child("stream-child", config={"seed": SEED})
        child.log_metrics({"chunks": float(len(chunks))}, step=0)
        child.finish()
        resumed = recorder.start_run("m14-l22", resume_run_id=parent.info.run_id, config={"seed": SEED}, tags={"lane": "L22"}, code_version="git:fixture", framework_version="latent-anything", seeds=(SEED,))
        resumed.log_metrics({"roundtrip": 2.0}, step=1)
        resumed.finish()
        record = recorder.get_record(parent.info.run_id)
        recorder_artifact_ok = any(recorder.read_artifact(item) == b"runtime-evidence" for item in record.artifacts)
        recorder_ok = record.status == "completed" and record.child_run_ids == (child.info.run_id,) and recorder_artifact_ok
        unsafe_name_rejected = False
        try:
            parent.log_artifact(b"x", name="../escape.bin")
        except RecorderContractError:
            unsafe_name_rejected = True

        rss_peak = process.memory_info().rss
        checks = {
            "artifact_roundtrip": restored.payload == payload and restored.identity == written.identity,
            "portable_schema_v1_roundtrip": portable_value["schema"] == "portable-node-v1",
            "result_envelope_roundtrip": result_roundtrip,
            "artifact_path_traversal_rejected": traversal_rejected,
            "artifact_checksum_tamper_rejected": checksum_rejected,
            "sqlite_cache_roundtrip": disk_hit and disk_reopened and disk_miss,
            "rollout_cache_hit": first.cache_hit is False and second.cache_hit is True and changed.cache_hit is False,
            "stream_bounded_ordered": streaming_ok,
            "async_rollout_complete": async_result.trajectory.shape == (5, 2),
            "recorder_parent_child_resume": recorder_ok,
            "recorder_artifact_path_rejected": unsafe_name_rejected,
            "finite_outputs": bool(np.isfinite(np.asarray(streamed)).all()),
        }
        payload_out = {
            "schema_version": "m14-l22-run-v1",
            "source_sha": __import__("subprocess").check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "command": RUN_COMMAND,
            "target": {"filesystem": "isolated temporary Arrow/SQLite-compatible paths", "device": "cpu", "schema_versions": ["portable-node-v1", "artifact-envelope-v1", "disk-cache-v1", "result-envelope-v1"]},
            "recorder_parent_child_resume_detail": {"status": str(record.status), "child_run_ids": list(record.child_run_ids), "expected_child": child.info.run_id, "artifact_count": len(record.artifacts), "artifact_ok": recorder_artifact_ok},
            "checks": checks,
            "cache": {"sqlite_path": "<temporary>/cache/runtime.sqlite", "key": key, "disk_stats": disk.stats.__dict__ if hasattr(disk.stats, "__dict__") else {"hits": disk.stats.hits, "misses": disk.stats.misses, "sets": disk.stats.sets, "entries": disk.stats.entries, "bytes": disk.stats.bytes}, "in_memory_stats": {"hits": pipeline_cache.stats.hits, "misses": pipeline_cache.stats.misses, "sets": pipeline_cache.stats.sets, "size": pipeline_cache.stats.size}},
            "recorder": {"backend": "filesystem", "parent_run_id": parent.info.run_id, "child_run_id": child.info.run_id, "artifact_digest": artifact.digest},
            "resource_budget": {"max_artifact_bytes": 1024 * 1024, "max_cache_bytes": 1024 * 1024, "max_cache_entries": 8, "rss_peak_bytes": rss_peak, "elapsed_seconds": time.perf_counter() - started},
            "optional_tracking": {"mlflow": "external service/credentials not claimed; local filesystem contract covered", "wandb": "external service/credentials not claimed; local filesystem contract covered"},
            "acceptance": checks,
            "cleanup": "Temporary Arrow/SQLite/run roots and all scratch files were removed by TemporaryDirectory; only this JSON artifact is retained.",
        }
    output = Path("artifacts/m14/l22-runtime-run.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload_out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload_out, sort_keys=True))


if __name__ == "__main__":
    main()

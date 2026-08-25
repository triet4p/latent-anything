# Sprint 74 Task 08 — Offline CPU size/latency measurement

Status: Complete (2026-08-25)

## Scope

Added a deterministic offline benchmark with warmups and ten repeated samples
for a 64x64 float32 `LatentValue`. It compares in-memory NumPy copy,
Arrow-node encode/decode, checksummed artifact write/read, and SQLite cache
set/get. Metrics are declared as mean microseconds and byte counts; no
performance gate is inferred from this small CPU fixture.

## Files

- `scripts/sprint74_artifact_benchmark.py`
- `tests/test_sprint74_benchmark.py`
- `docs/sprint-plans/sprint-74.md`

## Focused validation and measured result

```text
uv run ruff check scripts/sprint74_artifact_benchmark.py tests/test_sprint74_benchmark.py
All checks passed!
uv run pyright scripts/sprint74_artifact_benchmark.py
0 errors, 0 warnings, 0 informations
uv run pytest -q tests/test_sprint74_benchmark.py
1 passed in 5.14s
uv run python scripts/sprint74_artifact_benchmark.py
{"arrow_decode_us": 807.62, "arrow_encode_us": 337.97999999999996, "artifact_read_us": 254.18, "artifact_write_us": 6691.7, "cache_get_us": 14776.53, "cache_set_us": 18548.16, "in_memory_copy_us": 1.3599999999999999, "payload_bytes": 18466, "stored_artifact_bytes": 18770}
```

The cache timings include SQLite connection/WAL setup per operation, as
implemented, and are not comparable to an in-memory dictionary hit.

Post-remediation rerun after the validated portable cache seam and path
hardening:

```text
{"arrow_decode_us": 180.78, "arrow_encode_us": 258.95, "artifact_read_us": 229.04, "artifact_write_us": 5329.06, "cache_get_us": 16718.24, "cache_set_us": 17532.86, "in_memory_copy_us": 1.04, "payload_bytes": 18466, "stored_artifact_bytes": 18770}
```

## Graph refresh

Pending at artifact creation; `graphify update .` is required immediately
after this atomic completion and its exact result is recorded below.

Refresh completed immediately after task completion:

```text
graphify update .
[graphify watch] Rebuilt: 9326 nodes, 18210 edges, 843 communities
Code graph updated.
```

Graphify reported the known 42 zero-node JSON/source warning and rebuilt the
aggregate view because the graph exceeds 5,000 nodes. No refresh failure was
observed.

# Sprint 74 Task 06 — Cross-process behavior parity

Status: Complete (2026-08-25)

## Scope

Added an offline CPU reproduction script that writes a `LatentValue`,
`Trajectory`, and CEM planning result as portable artifacts, stores the result
payload under a state-aware SQLite key, and opens all of them in a fresh Python
subprocess. The child verifies domain/result types, shapes, numeric behavior,
typed-result identity, and a cache hit. The reproduction also exposed and
fixed explicit SQLite connection closure needed for Windows temporary-file
cleanup; the lesson is recorded append-only in project memory.

## Files

- `scripts/sprint74_portable_roundtrip.py`
- `tests/test_sprint74_roundtrip.py`
- `src/latent_anything/runtime/disk_cache.py` (explicit connection sessions)
- `.agents/memory/lessons-learned.md`
- `docs/sprint-plans/sprint-74.md`

## Focused validation

```text
uv run ruff check scripts/sprint74_portable_roundtrip.py src/latent_anything/runtime/disk_cache.py tests/test_sprint74_roundtrip.py
All checks passed!
uv run pyright scripts/sprint74_portable_roundtrip.py src/latent_anything/runtime/disk_cache.py
0 errors, 0 warnings, 0 informations
uv run pytest -q tests/test_sprint74_roundtrip.py
1 passed in 8.44s
uv run python scripts/sprint74_portable_roundtrip.py
{"artifact_bytes": 4530, "artifact_identity": "90574c480495db1d1f1f3a3f74df383675300e4d87c0b45d8bf4a1578e36ea93", "child": {"cache_hit": true, "child": "pass", "result_identity": "186a10512d3e1ee9e3780ffe39f113dbb6a7c49a2741349992445e5f73b3a55b"}, "elapsed_seconds": 4.288528, "status": "pass"}
```

## Graph refresh

Pending at artifact creation; `graphify update .` is required immediately
after this atomic completion and its exact result is recorded below.

Refresh completed immediately after task completion:

```text
graphify update .
[graphify watch] Rebuilt: 9301 nodes, 18139 edges, 823 communities
Code graph updated.
```

Graphify reported the known 42 zero-node JSON/source warning and rebuilt the
aggregate view because the graph exceeds 5,000 nodes. No refresh failure was
observed.

Post-remediation rerun through `SQLiteDiskCache.set_portable/get_portable`:

```text
{"artifact_bytes": 4978, "artifact_identity": "b3909afc71f2af30141ccd8fd7a2fb027def91cccf560b892c3e36d1a250cdb2", "child": {"cache_hit": true, "child": "pass", "result_identity": "186a10512d3e1ee9e3780ffe39f113dbb6a7c49a2741349992445e5f73b3a55b"}, "elapsed_seconds": 6.881989, "status": "pass"}
```

# Sprint 74 Task 05 — Failure and concurrency validation

Status: Complete (2026-08-25)

## Scope

Expanded the focused validation lane for artifact and cache failure modes:
truncated and schema-mismatched envelopes, checksum corruption, bounded
allocation guards, corrupt SQLite rows treated as misses, deterministic
eviction, malformed keys, cross-process reopening, and concurrent SQLite
writers under WAL/busy-timeout.

## Files

- `tests/test_artifact_store.py`
- `tests/test_disk_cache.py`
- `docs/sprint-plans/sprint-74.md`

## Focused validation

```text
uv run ruff format src/latent_anything/runtime/disk_cache.py tests/test_disk_cache.py tests/test_artifact_store.py
3 files left unchanged
uv run ruff check src/latent_anything/artifact_store.py src/latent_anything/portable.py src/latent_anything/portable_results.py src/latent_anything/runtime/disk_cache.py tests/test_artifact_store.py tests/test_disk_cache.py tests/test_portable.py tests/test_portable_results.py
All checks passed!
uv run pyright src/latent_anything/artifact_store.py src/latent_anything/portable.py src/latent_anything/portable_results.py src/latent_anything/runtime/disk_cache.py
0 errors, 0 warnings, 0 informations
uv run pytest -q tests/test_artifact_store.py tests/test_disk_cache.py tests/test_portable.py tests/test_portable_results.py
16 passed in 17.70s
```

## Graph refresh

Pending at artifact creation; `graphify update .` is required immediately
after this atomic completion and its exact result is recorded below.

Refresh completed immediately after task completion:

```text
graphify update .
[graphify watch] Rebuilt: 9286 nodes, 18094 edges, 799 communities
Code graph updated.
```

Graphify reported the known 42 zero-node JSON/source warning and rebuilt the
aggregate view because the graph exceeds 5,000 nodes. No refresh failure was
observed.

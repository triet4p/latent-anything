# Sprint 74 Task 04 — SQLite disk cache

Status: Complete (2026-08-25)

## Scope

Added the approved custom stdlib SQLite backend (`SQLiteDiskCache`) with WAL,
busy-timeout, per-operation connections, transactional writes, checksummed
opaque bytes, deterministic LRU eviction (`accessed_ns`, `created_ns`, key),
and bounded entries/bytes. `make_disk_cache_key` hashes the existing
`CacheKey` together with required non-empty plugin, checkpoint, and
behavior-state identities; framework callers use the validated
`set_portable/get_portable` seam before caching fitted/stateful outputs. The
low-level byte primitive remains explicitly non-coherent. No pickle or cache
Protocol was introduced.

## Files

- `src/latent_anything/runtime/disk_cache.py`
- `tests/test_disk_cache.py`
- `docs/sprint-plans/sprint-74.md`

## Focused validation

```text
uv run ruff check src/latent_anything/runtime/disk_cache.py tests/test_disk_cache.py
All checks passed!
uv run ruff format --check src/latent_anything/runtime/disk_cache.py tests/test_disk_cache.py
2 files already formatted
uv run pyright src/latent_anything/runtime/disk_cache.py
0 errors, 0 warnings, 0 informations
uv run pytest -q tests/test_disk_cache.py
3 passed in 8.30s
```

Tests cover key separation for state/provenance, cross-process reopening,
deterministic eviction, occupancy/counters, corruption-as-miss, and malformed
key rejection.

## Graph refresh

Pending at artifact creation; `graphify update .` is required immediately
after this atomic completion and its exact result is recorded below.

Refresh completed immediately after task completion:

```text
graphify update .
[graphify watch] Rebuilt: 9278 nodes, 18080 edges, 832 communities
Code graph updated.
```

Graphify reported the known 42 zero-node JSON/source warning and rebuilt the
aggregate view because the graph exceeds 5,000 nodes. No refresh failure was
observed.

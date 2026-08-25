# Sprint 74 remediation 05 — bounded and coherent disk cache

Status: Complete (2026-08-25)

## Scope

SQLite cache reads now begin a snapshot transaction, validate the stored size,
digest type, and configured byte bound before loading the BLOB, then verify
actual length and SHA-256. Oversized or malformed rows are deleted as misses.
`make_disk_cache_key` rejects empty/whitespace identities for plugin,
checkpoint, and behavior state. The low-level byte API remains available for
internal cache primitives, while `set_portable/get_portable` is the required
framework seam and validates Arrow portable envelopes on write and restore.
Cross-process and benchmark scripts now use that coherent seam.

## Files

- `src/latent_anything/runtime/disk_cache.py`
- `tests/test_disk_cache.py`
- `scripts/sprint74_portable_roundtrip.py`
- `scripts/sprint74_artifact_benchmark.py`
- `docs/PORTABLE_ARTIFACTS.md`
- `docs/sprint-plans/sprint-74.md`

## Focused validation

```text
uv run ruff check --fix tests/test_disk_cache.py
Found 1 error (1 fixed, 0 remaining).
uv run ruff format src/latent_anything/runtime/disk_cache.py scripts/sprint74_portable_roundtrip.py scripts/sprint74_artifact_benchmark.py tests/test_disk_cache.py
4 files left unchanged
uv run ruff check src/latent_anything/runtime/disk_cache.py scripts/sprint74_portable_roundtrip.py scripts/sprint74_artifact_benchmark.py tests/test_disk_cache.py
All checks passed!
uv run pyright src/latent_anything/runtime/disk_cache.py scripts/sprint74_portable_roundtrip.py scripts/sprint74_artifact_benchmark.py
0 errors, 0 warnings, 0 informations
uv run pytest -q tests/test_disk_cache.py tests/test_sprint74_roundtrip.py tests/test_sprint74_benchmark.py
9 passed in 40.93s
```

Tests cover empty identity rejection, invalid portable payloads, oversized
stored rows, corruption, eviction, concurrent writers, and symlink paths.

## Graph refresh

Pending at artifact creation; the required immediate refresh follows this
focused completion.

Refresh completed immediately after the concern:

```text
graphify update .
[graphify watch] Rebuilt: 9392 nodes, 18346 edges, 843 communities
Code graph updated.
```

The known 42 zero-node JSON/source warning and community-label drift warning
were reported; no refresh failure occurred.

# Sprint 74 remediation 04 — path and reparse safety

Status: Complete (2026-08-25)

## Scope

Artifact and cache paths now reject ordinary symlinks, Windows junctions,
and Windows reparse-point components, including dangling links, and recheck
after directory creation. Conditional Windows junction coverage was added
alongside POSIX/Windows-compatible symlink tests. The portable API limitation
is documented honestly: `pathlib` check/open/replace cannot eliminate a
hostile concurrent path swap, so callers must use private directories and OS
permissions for adversarial filesystems.

## Files

- `src/latent_anything/artifact_store.py`
- `src/latent_anything/runtime/disk_cache.py`
- `tests/test_artifact_store.py`
- `tests/test_disk_cache.py`
- `docs/PORTABLE_ARTIFACTS.md`
- `docs/sprint-plans/sprint-74.md`

## Focused validation

```text
uv run ruff format src/latent_anything/artifact_store.py src/latent_anything/runtime/disk_cache.py tests/test_artifact_store.py tests/test_disk_cache.py
4 files left unchanged
uv run ruff check --fix tests/test_disk_cache.py
Found 1 error (1 fixed, 0 remaining).
uv run ruff check src/latent_anything/artifact_store.py src/latent_anything/runtime/disk_cache.py tests/test_artifact_store.py tests/test_disk_cache.py
All checks passed!
uv run pyright src/latent_anything/artifact_store.py src/latent_anything/runtime/disk_cache.py
0 errors, 0 warnings, 0 informations
uv run pytest -q tests/test_artifact_store.py tests/test_disk_cache.py tests/test_run_record_portable.py
11 passed in 23.63s
```

## Graph refresh

Pending at artifact creation; the required immediate refresh follows this
focused completion.

Refresh completed immediately after the concern:

```text
graphify update .
[graphify watch] Rebuilt: 9380 nodes, 18311 edges, 837 communities
Code graph updated.
```

The known 42 zero-node JSON/source warning and community-label drift warning
were reported; no refresh failure occurred.

# Sprint 74 Task 03 — Versioned artifact storage

Status: Complete (2026-08-25)

## Scope

Added `ArtifactStore`, a storage envelope for already-encoded portable
payloads. Each file contains a magic/version marker, canonical JSON header,
artifact type, SHA-256 payload digest, payload size, canonical identity, and
JSON metadata. Writes use a same-directory temporary file, flush/fsync, and
atomic replace. Reads validate path, symlink, file/header/payload bounds,
schema, metadata, size, checksum, and identity before returning bytes.

Paths are relative to an explicit non-symlink root and reject traversal and
symlink components. The API remains opaque-bytes at this layer, so callers can
store Task 01 values or Task 02 typed envelopes without exposing Arrow in
public domain APIs.

## Files

- `src/latent_anything/artifact_store.py`
- `tests/test_artifact_store.py`
- `docs/sprint-plans/sprint-74.md`

## Focused validation

```text
uv run ruff check src/latent_anything/artifact_store.py tests/test_artifact_store.py
All checks passed!
uv run ruff format --check src/latent_anything/artifact_store.py tests/test_artifact_store.py
2 files already formatted
uv run pyright src/latent_anything/artifact_store.py
0 errors, 0 warnings, 0 informations
uv run pytest -q tests/test_artifact_store.py
3 passed in 3.81s
```

Tests cover round-trip identity, traversal/symlink rejection, checksum
tampering, size bounds, and truncated envelopes.

## Graph refresh

Pending at artifact creation; `graphify update .` is required immediately
after this atomic completion and its exact result is recorded below.

Refresh completed immediately after task completion:

```text
graphify update .
[graphify watch] Rebuilt: 9240 nodes, 18020 edges, 825 communities
Code graph updated.
```

Graphify reported the known 42 zero-node JSON/source warning and rebuilt the
aggregate view because the graph exceeds 5,000 nodes. No refresh failure was
observed.

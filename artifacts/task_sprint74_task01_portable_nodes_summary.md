# Sprint 74 Task 01 — Portable Arrow nodes

Status: Complete (2026-08-25)

## Scope

Implemented the first bounded value-layer wire format for Sprint 74. The
format is Arrow IPC `portable-node-v1`: one row per NumPy buffer with explicit
array id, dtype, shape, and binary payload columns, plus a canonical JSON
manifest in schema metadata. Public encode/decode functions return bytes and
NumPy/domain objects; PyArrow objects do not appear in the public boundary.

Supported values are JSON scalars, bytes, mappings with string keys,
list/tuple/frozenset, non-object NumPy arrays (including explicit byte order),
`LatentSpace`, `LatentValue`, `Trajectory`, and `CovarianceState`. Cycles,
object arrays, unsupported types, non-finite metadata floats, malformed shape
or payload lengths, excessive depth/node counts, and excessive array bytes are
rejected without pickle fallback.

## Files

- `src/latent_anything/portable.py`
- `tests/test_portable.py`
- `docs/sprint-plans/sprint-74.md`
- `pyproject.toml`, `uv.lock` (approved `pyarrow>=19,<26` dependency)

## Focused validation

Commands run from the repository root:

```text
uv run ruff check src/latent_anything/portable.py tests/test_portable.py
All checks passed!
uv run ruff format --check src/latent_anything/portable.py tests/test_portable.py
2 files already formatted
uv run pyright src/latent_anything/portable.py
0 errors, 0 warnings, 0 informations
uv run pytest -q tests/test_portable.py
5 passed in 4.09s
```

The test suite covers dtype/shape/endianness, nested values, domain-object
round trips and immutable metadata, cycle and size guards, object-dtype
rejection, and Arrow-container/version presence.

## Graph refresh

Pending at artifact creation; `graphify update .` is required immediately
after this atomic completion and its exact result is recorded below.

Refresh completed immediately after task completion:

```text
graphify update .
[graphify watch] Rebuilt: 9180 nodes, 17793 edges, 812 communities
Code graph updated.
```

Graphify also reported 42 source files producing zero nodes (the known JSON
and non-code warning) and rebuilt the aggregate graph because it is above the
5,000-node limit. No graphify failure occurred.

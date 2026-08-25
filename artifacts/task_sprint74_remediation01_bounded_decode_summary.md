# Sprint 74 remediation 01 — bounded Arrow decoding

Status: Complete (2026-08-25)

## Scope

The owner audit found that `read_all()` materialized an untrusted Arrow table
before applying `PortableLimits`. The decoder now bounds the input bytes,
manifest bytes, record-batch count, array-row count, and shape rank before
restoring values. It reads bounded record batches instead of calling
`read_all()`, validates array IDs, normalizes malformed dtypes to
`PortableNodeError`, rejects object dtypes on decode, and uses Python-integer
size arithmetic for allocation checks.

## Files

- `src/latent_anything/portable.py`
- `tests/test_portable.py`
- `docs/sprint-plans/sprint-74.md`

## Focused validation

```text
uv run ruff check --fix tests/test_portable.py
Found 1 error (1 fixed, 0 remaining).
uv run ruff check tests/test_portable.py src/latent_anything/portable.py
All checks passed!
uv run ruff format tests/test_portable.py src/latent_anything/portable.py
2 files left unchanged
uv run pyright src/latent_anything/portable.py
0 errors, 0 warnings, 0 informations
uv run pytest -q tests/test_portable.py tests/test_portable_results.py
10 passed in 5.37s
```

The adversarial tests cover early input and manifest limits, rank and row
limits, malformed dtypes, object dtypes, cycles, and allocation bounds.

## Graph refresh

Pending at artifact creation; the required immediate refresh follows this
focused completion.

Refresh completed immediately after the concern:

```text
graphify update .
[graphify watch] Rebuilt: 9352 nodes, 18267 edges, 833 communities
Code graph updated.
```

Graphify reported the known 42 zero-node JSON/source warning and a community
label drift warning; no refresh failure occurred.

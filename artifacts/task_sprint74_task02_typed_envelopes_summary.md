# Sprint 74 Task 02 — Typed result/config envelopes

Status: Complete (2026-08-25)

## Scope

Added `result-envelope-v1` on top of the Task 01 Arrow node codec, with an
explicit tested `result-envelope-v0` to v1 migration hook. Typed
results/configs are reconstructed only through an explicit built-in allowlist;
the decoder never imports a module named in an artifact. Dataclass fields and
Pydantic model fields are recursively represented through the safe node layer.
The envelope carries canonical provenance and behavior-affecting state, plus a
SHA-256 identity over the type and both metadata mappings. Identity mismatch,
unknown type markers, malformed metadata, validation failures, and unsupported
root values fail closed. Imports of optional integrations remain lazy until
the envelope API is called.

## Files

- `src/latent_anything/portable_results.py`
- `tests/test_portable_results.py`
- `docs/sprint-plans/sprint-74.md`

## Focused validation

```text
uv run ruff check src/latent_anything/portable_results.py tests/test_portable_results.py
All checks passed!
uv run ruff format --check src/latent_anything/portable_results.py tests/test_portable_results.py
2 files already formatted
uv run pyright src/latent_anything/portable_results.py
0 errors, 0 warnings, 0 informations
uv run pytest -q tests/test_portable_results.py
4 passed (including explicit v0 migration)
```

The tests cover nested CEM dataclasses and immutable arrays, Pydantic
`ObjectSpec`, state/provenance restoration, root allowlist rejection, identity
tampering, and unallowlisted type markers.

## Graph refresh

Pending at artifact creation; `graphify update .` is required immediately
after this atomic completion and its exact result is recorded below.

Refresh completed immediately after task completion:

```text
graphify update .
[graphify watch] Rebuilt: 9209 nodes, 17966 edges, 797 communities
Code graph updated.
```

Graphify reported the known 42 zero-node JSON/source warning and rebuilt the
aggregate view because the graph exceeds 5,000 nodes. No refresh failure was
observed.

After the explicit v0 migration hardening, the required follow-up refresh was:

```text
graphify update .
[graphify watch] Rebuilt: 9340 nodes, 18250 edges, 838 communities
Code graph updated.
```

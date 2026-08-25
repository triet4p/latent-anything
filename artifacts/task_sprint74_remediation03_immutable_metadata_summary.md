# Sprint 74 remediation 03 — recursive metadata immutability

Status: Complete (2026-08-25)

## Scope

Decoded typed-envelope provenance and behavior-state metadata, plus stored
artifact metadata, is now recursively immutable. Nested mappings become
mapping proxies and nested lists become tuples after identity/checksum
validation; canonical JSON identity and serialization remain unchanged.

## Files

- `src/latent_anything/portable_results.py`
- `src/latent_anything/artifact_store.py`
- `tests/test_portable_results.py`
- `tests/test_artifact_store.py`
- `docs/sprint-plans/sprint-74.md`

## Focused validation

```text
uv run ruff format src/latent_anything/portable_results.py src/latent_anything/artifact_store.py tests/test_portable_results.py tests/test_artifact_store.py
4 files left unchanged
uv run ruff check src/latent_anything/portable_results.py src/latent_anything/artifact_store.py tests/test_portable_results.py tests/test_artifact_store.py
All checks passed!
uv run pyright src/latent_anything/portable_results.py src/latent_anything/artifact_store.py
0 errors, 0 warnings, 0 informations
uv run pytest -q tests/test_portable_results.py tests/test_artifact_store.py tests/test_run_record_portable.py
12 passed in 5.94s
```

## Graph refresh

Pending at artifact creation; the required immediate refresh follows this
focused completion.

Refresh completed immediately after the concern:

```text
graphify update .
[graphify watch] Rebuilt: 9369 nodes, 18293 edges, 835 communities
Code graph updated.
```

The known 42 zero-node JSON/source warning and community-label drift warning
were reported; no refresh failure occurred.

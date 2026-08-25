# Sprint 74 remediation 08 — final closure

Status: Complete (2026-08-25)

## Scope

All seven audit concern groups and the final traceability task are complete:
bounded Arrow decoding, exact typed sequences, recursive metadata freezing,
path/reparse safety, bounded/coherent cache reads, historical documentation,
ledger traceability, and closure gates. At this Sprint 74 closure snapshot,
Sprint 75 was not started; it subsequently completed its own bounded
streaming sprint.

## Final validation

```text
uv run ruff check src tests scripts
All checks passed!
uv run ruff format --check src tests scripts
251 files already formatted
uv run pyright
0 errors, 0 warnings, 0 informations
uv run pytest -q
1413 passed, 32 skipped, 39 warnings in 226.37s (0:03:46)
uv run python scripts/validate_evidence_ledger.py
Inventory: 107 capabilities; core 23/63 (36.5%); overall 23/65 (35.4%); exit 0
uv run python scripts/sprint74_portable_roundtrip.py
status=pass; cache_hit=true; elapsed_seconds=6.881989
uv run python scripts/sprint74_artifact_benchmark.py
payload_bytes=18466; stored_artifact_bytes=18770
arrow_encode_us=258.95; arrow_decode_us=180.78
artifact_write_us=5329.06; artifact_read_us=229.04
cache_set_us=17532.86; cache_get_us=16718.24
uv run --extra docs mkdocs build --strict
Documentation built in 27.18 seconds; exit 0 (upstream Material advisory only)
git diff --check
Passed (only expected Git LF-to-CRLF working-copy notices)
```

The repository-wide `uv run ruff check .` remains outside the actual release
scope and retains the known 1,920 pre-existing `.agents`/theory violations;
the changed `src`/`tests`/`scripts` gate is clean.

## Graph refresh

Pending at artifact creation; the required immediate final refresh follows
this closure completion.

Final refresh completed after the closure implementation and documentation:

```text
graphify update .
[graphify watch] Rebuilt: 9411 nodes, 18366 edges, 842 communities
Code updated.
```

The known 42 zero-node JSON/source warning and community-label drift warning
were reported; no refresh failure occurred.

A final path-ancestor hardening follow-up also passed its focused checks and
refreshed graphify to 9411 nodes and 18366 edges; community regrouping changed
the aggregate count to 842, with the same known warnings.

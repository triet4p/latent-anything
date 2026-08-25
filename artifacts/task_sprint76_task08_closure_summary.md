# Sprint 76 Task 08 — Closure gates

Status: Complete (2026-08-25)

## Closure validation

```text
uv run ruff check src tests scripts
All checks passed!
uv run ruff format --check src tests scripts
261 files already formatted
uv run pyright
0 errors, 0 warnings, 0 informations
uv run --extra viz --extra tracking pytest -q
1441 passed, 32 skipped, 39 warnings in 208.84s
uv run python scripts/validate_evidence_ledger.py
Inventory: 107 capabilities; core 23/63 (36.5%); overall 23/65 (35.4%)
uv run --extra docs mkdocs build --strict
Documentation built in 27.81 seconds (exit 0; upstream Material warning only)
uv run --extra tracking pytest -q -m integration tests/test_tracking_parity.py
2 passed, 1 deselected in 53.68s
git diff --check
passed (CRLF conversion warnings are Git working-copy notices, not errors)
```

The supported full-suite command above includes the approved `viz` and
`tracking` extras. A bare `uv run pytest -q` after `uv sync --locked --offline`
without `viz` failed only because Plotly was absent. The full-suite warnings
are pre-existing sklearn convergence, registry
deprecation, and UMAP seed warnings. After enabling the approved tracking
and viz extras, the full suite also exercised the two real local/offline SDK
tests and visualization tests. A bare run after `uv sync --locked --offline`
without the viz extra failed only because Plotly was absent; the supported
closure command with `--extra viz --extra tracking` passed. No remote server,
credentials, or cloud API was used. No Sprint 77 work was started.

## Graph refresh

`graphify update .` completed after the closure update. It re-extracted 48/48
uncached code files and reported the existing warning that 42 JSON/source
files produce zero graph nodes and are absent from the graph; the process
then completed. No Sprint 76 source was omitted from AST extraction.

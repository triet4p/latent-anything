# Sprint 76 Remediation 10 — MLflow `Path` root lexical safety

Status: Complete (2026-08-25)

## Change

MLflow tracking roots now apply platform-independent lexical checks to both
`str` and `Path` inputs before filesystem resolution or optional SDK setup.
UNC/device forms, encoded components, traversal/dot segments, ambiguous
drive-relative paths, URI syntax, and Windows alternate-data-stream syntax are
rejected while valid native absolute `Path` roots remain supported.

## Focused validation

```text
uv run pytest -q tests/test_mlflow_recorder.py
12 passed
```

The regression matrix covers Windows-style UNC/device, encoded, ADS,
traversal, and URI-looking `Path` values on the current Windows runner.

## Graph refresh

`graphify update .` was run immediately after this atomic completion:

```text
Rebuilt: 9877 nodes, 19295 edges, 868 communities
Warning: 42 source/JSON files produced zero graph nodes; graphify retried them.
```

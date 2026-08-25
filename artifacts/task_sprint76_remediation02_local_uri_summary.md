# Sprint 76 Remediation 02 — Canonical MLflow local roots

Status: Complete (2026-08-25)

The focused count below is the atomic-boundary snapshot; later remediation
tasks added tests. The final current totals are recorded in Remediation 09.

## Change

MLflow tracking roots now normalize to canonical local file URIs with empty
authority. Remote/UNC authorities, encoded paths, query/fragment components,
backslash ambiguity, symlink components, and Windows reparse-point components
are rejected before the optional SDK is imported or configured. The residual
hostile-process TOCTOU limitation is documented: a process that changes a path
after validation cannot be prevented by ordinary Python path APIs.

## Focused validation

```text
uv run pytest -q tests/test_mlflow_recorder.py
4 passed
```

Coverage includes remote/alternate URI forms, encoded traversal/separators,
UNC syntax, and a conditional symlink-root test.

## Graph refresh

`graphify update .` completed immediately after this atomic completion:

```text
Rebuilt: 9814 nodes, 19181 edges, 880 communities
Warning: 42 source/JSON files produced zero graph nodes; graphify retried them.
```

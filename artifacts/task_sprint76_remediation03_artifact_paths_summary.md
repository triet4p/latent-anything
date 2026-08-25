# Sprint 76 Remediation 03 — Canonical artifact paths

Status: Complete (2026-08-25)

The focused count below is the atomic-boundary snapshot; later remediation
tasks added tests. The final current totals are recorded in Remediation 09.

## Change

Artifact names are now canonical POSIX-relative paths: separators, empty/dot
segments, traversal, percent-encoded components, Windows drives, UNC forms,
and backslashes are rejected. Both optional adapters validate before reading
content and use a root-contained path helper that rejects existing symlink or
reparse components. A hostile process can still race a validated path using
ordinary filesystem APIs; that residual TOCTOU limitation is not presented as
fully preventable.

## Focused validation

```text
uv run pytest -q tests/test_experiment_recorder.py tests/test_mlflow_recorder.py tests/test_wandb_recorder.py
29 passed
```

## Graph refresh

`graphify update .` completed immediately after this atomic completion:

```text
Rebuilt: 9822 nodes, 19197 edges, 870 communities
Warning: 42 source/JSON files produced zero graph nodes; graphify retried them.
```

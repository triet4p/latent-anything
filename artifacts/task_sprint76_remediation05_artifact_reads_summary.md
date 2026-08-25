# Sprint 76 Remediation 05 — Bounded artifact reads

Status: Complete (2026-08-25)

The focused count below is the atomic-boundary snapshot; later remediation
tasks added tests. The final current totals are recorded in Remediation 09.

## Change

Artifact content is now accepted only from exact bytes/bytearray/memoryview
values or paths. Byte-like sizes are checked before copying; arbitrary
`__bytes__` objects are rejected. Path reads use a bounded descriptor read,
regular-file and reparse checks, POSIX `O_NOFOLLOW` where available, and
pre/post descriptor identity, size, and mtime checks. A hostile process can
still race ordinary cross-platform path APIs after validation; that residual
limitation is explicit.

## Focused validation

```text
uv run pytest -q tests/test_experiment_recorder.py tests/test_mlflow_recorder.py tests/test_wandb_recorder.py
35 passed
```

Adversarial coverage proves oversized memoryviews/files are rejected before a
successful return and arbitrary byte-conversion protocols are never invoked.

## Graph refresh

`graphify update .` completed immediately after this atomic completion:

```text
Rebuilt: 9844 nodes, 19258 edges, 860 communities
Warning: 42 source/JSON files produced zero graph nodes; graphify retried them.
```

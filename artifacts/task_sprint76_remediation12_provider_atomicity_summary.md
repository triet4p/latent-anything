# Sprint 76 Remediation 12 — Provider state commit atomicity

Status: Complete (2026-08-25)

## Change

MLflow and W&B now prepare prospective params, metrics, and tags without
mutating adapter state, invoke the provider, and commit the local state only
after provider success. Existing `record_*` helpers retain their direct
validated behavior, while adapter calls use the prepare/commit path. W&B tag
provenance is generated from the prospective candidate state.

## Focused validation

```text
uv run pytest -q tests/test_mlflow_recorder.py tests/test_wandb_recorder.py
21 passed
```

Provider-failure retry tests prove that failed params and metric calls do not
consume immutable parameter keys or metric steps in either adapter.

## Graph refresh

`graphify update .` completed immediately after this atomic completion:

```text
Rebuilt: 9899 nodes, 19330 edges, 858 communities
Warning: 42 source/JSON files produced zero graph nodes; graphify retried them.
```

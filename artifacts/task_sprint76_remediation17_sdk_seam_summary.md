# Sprint 76 Remediation 17 — Private SDK test seam

Status: Complete (2026-08-25)

## Change

Optional recorder constructors no longer expose a public `sdk=` parameter.
Deterministic tests use the underscore-prefixed `_sdk` injection seam; public
run/recorder operations still return only backend-neutral recorder values and
no SDK objects. Base import isolation and lazy optional loading are unchanged.

## Files

- `src/latent_anything/integrations/mlflow_recorder.py`
- `src/latent_anything/integrations/wandb_recorder.py`
- `tests/test_mlflow_recorder.py`
- `tests/test_wandb_recorder.py`
- `tests/test_tracking_parity.py`
- `docs/sprint-plans/sprint-76.md`

## Focused validation

```text
uv run pytest -q tests/test_integrations.py tests/test_mlflow_recorder.py tests/test_wandb_recorder.py tests/test_tracking_parity.py
34 passed in 15.83s
```

## Graph refresh

`graphify update .` completed immediately after this atomic completion:

```text
Rebuilt: 9937 nodes, 19381 edges, 882 communities
Warning: 42 source/JSON files produced zero graph nodes; graphify retried them.
```

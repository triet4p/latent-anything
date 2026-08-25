# Sprint 76 Remediation 15 — Provider resume-ID continuity

Status: Complete (2026-08-25)

## Change

MLflow and W&B resume now require the provider-returned run ID to exactly equal
the requested `resume_run_id`, in addition to validating canonical provenance.
Unexpected provider-created runs are best-effort finished as failed before a
`RecorderContractError` is raised. Cleanup failures remain contract errors and
no adapter run state is constructed. Regression tests cover ignored IDs,
cleanup success, cleanup failure, and retry with the original ID.

## Files

- `src/latent_anything/integrations/mlflow_recorder.py`
- `src/latent_anything/integrations/wandb_recorder.py`
- `tests/test_mlflow_recorder.py`
- `tests/test_wandb_recorder.py`
- `docs/sprint-plans/sprint-76.md`

## Focused validation

```text
uv run pytest -q tests/test_mlflow_recorder.py tests/test_wandb_recorder.py
25 passed in 4.41s
```

## Graph refresh

`graphify update .` completed immediately after this atomic completion:

```text
Rebuilt: 9913 nodes, 19358 edges, 893 communities
Warning: 42 source/JSON files produced zero graph nodes; graphify retried them.
```

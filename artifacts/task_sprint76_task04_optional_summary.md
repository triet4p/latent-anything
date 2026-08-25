# Sprint 76 Task 04 — Optional SDK isolation

Status: Complete (2026-08-25)

## Scope

Added `tracking-mlflow`, `tracking-wandb`, and combined `tracking` optional
extras with compatible ranges. The base package and integration boundaries
remain lazy: importing `latent_anything` does not import either SDK, and
missing optional modules continue to produce an actionable `uv sync --extra`
message. MLflow is restricted to local file tracking and W&B to offline or
disabled mode in this sprint.

## Files

- `pyproject.toml`
- `uv.lock`
- `tests/test_integrations.py`

## Focused validation

```text
uv lock --offline
Resolved 272 packages in 2.46s; added mlflow 3.14.0 and resolved tracking extras.
uv run pytest -q tests/test_integrations.py tests/test_experiment_recorder.py tests/test_mlflow_recorder.py tests/test_wandb_recorder.py
14 passed in 17.06s
```

The isolation test runs a fresh Python process and asserts neither optional
module is present after importing the base package. The tests inject fake SDKs
for provider behavior and make no network or credential calls.

## Graph refresh

`graphify update .` completed immediately after this atomic completion. The
refresh re-extracted 47/47 uncached code files and reported the existing
warning that 42 JSON/source files produce zero graph nodes and are absent
from the graph; the process then completed. No Sprint 76 source was omitted
from AST extraction.

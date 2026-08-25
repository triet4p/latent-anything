# Sprint 76 Task 02 — Optional MLflow recorder

Status: Complete (2026-08-25)

## Scope

Added `MLflowRecorder` and `MLflowExperimentRun` behind a lazy optional import.
The adapter permits only local file tracking URIs, maps config/params, ordered
finite metrics, tags, content-addressed bounded artifacts, parent/child runs,
resume identity tags, and terminal success/failure states. No MLflow SDK type is
part of the public contract and no remote tracking URI is accepted.

The implementation follows the current MLflow fluent API: `start_run`,
`log_params`, `log_metrics(..., step=...)`, `set_tags`, `log_artifact`, and
`end_run`. References: [MLflow Python API](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.html)
and [MLflow tracking API](https://mlflow.org/docs/latest/ml/tracking/tracking-api).

## Files

- `src/latent_anything/integrations/mlflow_recorder.py`
- `src/latent_anything/integrations/_tracking_common.py`
- `tests/test_mlflow_recorder.py`
- `src/latent_anything/experiment_recorder.py`

## Focused validation

```text
uv run pytest -q tests/test_experiment_recorder.py tests/test_mlflow_recorder.py
7 passed
uv run ruff check <recorder contract/common/MLflow/tests scope>
All checks passed!
uv run ruff format --check <recorder contract/common/MLflow/tests scope>
5 files already formatted
uv run pyright <recorder contract/common/MLflow/tests scope>
0 errors, 0 warnings, 0 informations
```

## Graph refresh

`graphify update .` completed immediately after this atomic completion:

```text
Re-extracting code files in . (no LLM needed)...
AST extraction: 43/43 uncached files (100%) [8 workers]
warning: 42 source file(s) produced zero nodes and are absent from the graph
[graphify watch] No code-graph topology changes detected; outputs left untouched.
Code graph updated.
```

The warning is the repository's existing JSON/source zero-node condition; no
Sprint 76 source was omitted from the code graph.

# Sprint 76 Task 06 — Identity, resume, and metric-step integrity

Status: Complete (2026-08-25)

## Scope

Strengthened the fake-provider contract tests to prove that MLflow and W&B
resume paths reject a changed config under the same run ID, while equivalent
runs retain the same canonical identity. Existing tests also prove monotonic
metric steps, finite values, checksum-bearing artifacts, and double-finish
rejection for both adapters and the local implementation.

## Focused validation

```text
uv run pytest -q tests/test_experiment_recorder.py tests/test_mlflow_recorder.py tests/test_wandb_recorder.py tests/test_tracking_parity.py
11 passed, 2 skipped in 5.72s
uv run ruff check <recorder adapter/test scope>
All checks passed!
uv run pyright <recorder adapter/test scope>
0 errors, 0 warnings, 0 informations
```

## Graph refresh

`graphify update .` completed immediately after this atomic completion. The
refresh re-extracted 45/45 uncached code files and reported the existing
warning that 43 JSON/source files produce zero graph nodes and are absent
from the graph; the process then completed. No Sprint 76 source was omitted
from AST extraction.

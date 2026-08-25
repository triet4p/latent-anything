# Sprint 76 Task 03 — W&B offline recorder

Status: Complete (2026-08-25)

Historical delivery record: final post-audit remediation supersedes the
offline provider-continuation wording where W&B does not persist adapter
identity; the current real lane fails closed rather than claiming resume.

## Scope

Implemented `WandbRecorder` and `WandbExperimentRun` behind the validated
experiment-recorder contract. The adapter permits only W&B `offline` or
`disabled` modes, lazy-loads the optional SDK, preserves identity/config/tags,
supports ordered finite metrics, checksum-bearing artifacts, resume identity,
finish/fail lifecycle, and parent relationships through W&B groups plus an
explicit parent tag. W&B does not provide a portable nested-run contract;
that limitation is intentional and documented by the group mapping.

## Files

- `src/latent_anything/integrations/wandb_recorder.py`
- `tests/test_wandb_recorder.py`
- `src/latent_anything/experiment_recorder.py`
- `src/latent_anything/integrations/_tracking_common.py`

## Focused validation

```text
uv run pytest -q tests/test_experiment_recorder.py tests/test_mlflow_recorder.py tests/test_wandb_recorder.py
10 passed in 4.67s
uv run ruff check <recorder/common/MLflow/W&B/tests scope>
All checks passed!
uv run ruff format --check <recorder/common/MLflow/W&B/tests scope>
7 files already formatted
uv run pyright <recorder/common/MLflow/W&B/tests scope>
0 errors, 0 warnings, 0 informations
```

The tests inject a local fake SDK; they do not install, authenticate, or call
the W&B service. The adapter rejects `online` mode before SDK initialization.

## Graph refresh

`graphify update .` completed immediately after this atomic completion. The
refresh re-extracted 47/47 uncached code files and reported the existing
warning that 42 JSON/source files produce zero graph nodes and are absent
from the graph; the graphify process then completed without a topology-change
summary. No Sprint 76 source was omitted from AST extraction.

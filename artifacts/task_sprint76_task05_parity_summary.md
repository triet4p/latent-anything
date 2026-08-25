# Sprint 76 Task 05 — Contract parity and opt-in integration lanes

Status: Complete (2026-08-25)

## Scope

Added a representative parity lane across the filesystem recorder, injected
MLflow adapter, and injected W&B adapter. It checks equal stable identities,
metric step behavior, equal artifact checksums, parent-child linkage, and
terminal lifecycle. During this task, the common identity helper was aligned
with the local recorder's default code/framework version resolution so the
three backends cannot silently compute different identities for equivalent
inputs.

Added opt-in local MLflow file-store and W&B offline tests. They use
`pytest.importorskip`, temporary directories, no credentials, and no cloud
endpoint. In the base suite the two tests skip because the optional SDKs are
not installed; the fake parity test remains required.

## Files

- `src/latent_anything/experiment_recorder.py`
- `tests/test_tracking_parity.py`
- `tests/test_mlflow_recorder.py`
- `tests/test_wandb_recorder.py`
- `pyproject.toml` (integration marker)

## Focused validation

```text
uv run pytest -q tests/test_tracking_parity.py
1 passed, 2 skipped in 4.14s
uv run ruff check <contract/parity scope>
All checks passed!
uv run pyright <contract/parity scope>
0 errors, 0 warnings, 0 informations
```

The skipped tests are the explicit optional-SDK local integration lanes, not
claimed provider evidence. No network or credential path was used.

## Graph refresh

`graphify update .` completed immediately after this atomic completion. The
refresh re-extracted 51/51 uncached code files and reported the existing
warning that 43 JSON/source files produce zero graph nodes and are absent
from the graph; the process then completed. No Sprint 76 source was omitted
from AST extraction.

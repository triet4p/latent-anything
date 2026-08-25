# Sprint 76 Remediation 06 — SDK boundary closure

Status: Complete (2026-08-25)

The focused count below is the atomic-boundary snapshot; later remediation
tasks added tests. The final current totals are recorded in Remediation 09.

## Change

The public MLflow `call()` and W&B `artifact_factory()` escape hatches were
removed. Adapter-owned provider calls and artifact construction are private;
the public contract returns only `RecorderRunInfo` and checksum-bearing
`RecorderArtifact` values. API isolation tests assert the removed names are not
present on the public recorder classes.

## Focused validation

```text
uv run pytest -q tests/test_integrations.py tests/test_mlflow_recorder.py tests/test_wandb_recorder.py tests/test_tracking_parity.py
16 passed
```

## Graph refresh

`graphify update .` completed immediately after this atomic completion:

```text
Rebuilt: 9850 nodes, 19263 edges, 866 communities
Warning: 42 source/JSON files produced zero graph nodes; graphify retried them.
```

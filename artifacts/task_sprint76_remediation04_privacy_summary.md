# Sprint 76 Remediation 04 — Privacy-safe bounded provenance

Status: Complete (2026-08-25)

The focused count below is the atomic-boundary snapshot; later remediation
tasks added tests. The final current totals are recorded in Remediation 09.

## Change

Recorder mappings now enforce bounded entry counts, recursive depth and sequence
length, string/key limits, string-only revision metadata, non-negative bounded
seeds, canonical serialized size, and normalized `RecorderContractError`
failures. Secret-like keys and common credential/token/private-key value forms
are rejected recursively across config, metadata, environment, params, and
tags. Environment metadata remains explicit caller input; the process
environment is never captured as run metadata. External params, metrics, and
tags are bounded cumulatively before state mutation.

## Focused validation

```text
uv run pytest -q tests/test_experiment_recorder.py tests/test_mlflow_recorder.py tests/test_wandb_recorder.py
32 passed
```

## Graph refresh

`graphify update .` completed immediately after this atomic completion:

```text
Rebuilt: 9834 nodes, 19239 edges, 852 communities
Warning: 42 source/JSON files produced zero graph nodes; graphify retried them.
```

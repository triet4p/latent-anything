# Sprint 76 Remediation 01 — Local resume identity validation

Status: Complete (2026-08-25)

The focused count below is the atomic-boundary snapshot; later remediation
tasks added tests. The final current totals are recorded in Remediation 09.

## Change

`LocalExperimentRecorder.start_run()` now validates every explicitly supplied
resume field against the stored run: name, config, tags, parent, code and
framework versions, model and dataset revisions, seeds, environment, and
metadata. Omitted optional fields retain the existing resume convenience, while
the stored identity is still checked for corruption. No provider SDK is
involved in this local-only concern.

## Focused validation

```text
uv run pytest -q tests/test_experiment_recorder.py
15 passed
```

The parameterized mismatch matrix covers every identity field and the existing
metric-history/double-finish tests remain green.

## Graph refresh

`graphify update .` completed immediately after this atomic completion:

```text
Rebuilt: 9806 nodes, 19168 edges, 872 communities
Warning: 42 source/JSON files produced zero graph nodes; graphify retried them.
```

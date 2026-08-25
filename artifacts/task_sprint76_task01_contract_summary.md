# Sprint 76 Task 01 — Recorder contract and local adapter

Status: Complete (2026-08-25)

## Scope

Compared the Sprint 62 `FileSystemRunRecorder` with the common MLflow and W&B
operations: run start/resume, immutable parameters, ordered finite metrics,
string tags, bounded checksummed artifacts, parent/child linkage, and terminal
success/failure lifecycle. Added the SDK-free `ExperimentRecorder` /
`ExperimentRun` Protocols and a `LocalExperimentRecorder` adapter. Existing
`FileSystemRunRecorder` behavior remains available unchanged; local logging
state is persisted as bounded content-addressed recorder-state artifacts.

The public contract exposes only canonical JSON-compatible values, NumPy-free
run metadata, and SHA-256 artifact references. Provider SDK objects are not
part of the contract.

## Files

- `src/latent_anything/experiment_recorder.py`
- `tests/test_experiment_recorder.py`
- `docs/PLAN.md`
- `docs/sprint-plans/sprint-76.md`
- `.agents/memory/decisions.md`

## Focused validation

```text
uv run pytest -q tests/test_experiment_recorder.py
4 passed
uv run ruff check src/latent_anything/experiment_recorder.py tests/test_experiment_recorder.py
All checks passed!
uv run pyright src/latent_anything/experiment_recorder.py tests/test_experiment_recorder.py
0 errors, 0 warnings, 0 informations
```

## Graph refresh

`graphify update .` is required immediately after this atomic completion.

Refresh completed with the known 42 zero-node JSON/source warning:

```text
graphify update .
Rebuilt graph: 9635 nodes, 18777 edges, 849 communities
```

Graphify reported a community-label refresh and backed up the semantic/curated
graph; no graphify failure occurred.

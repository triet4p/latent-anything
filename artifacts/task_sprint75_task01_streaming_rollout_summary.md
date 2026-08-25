# Sprint 75 Task 01 — Concrete chunked rollout story

Status: Complete (2026-08-25)

## Scope

Extended the existing concrete `RolloutPipeline` with synchronous streaming
over ordered, disjoint action chunks. Each chunk is validated and fully
processed before the next source chunk is requested. The current state carries
across chunk boundaries, empty chunks are skipped, and each yielded
`Trajectory` contains only the states produced by its own actions. Flattened
outputs therefore match the eager rollout trajectory excluding its initial
state. The implementation uses the existing `LatentTransition.step` seam and
adds no runtime Protocol or generic stream framework. Optional transition
`reset()` hooks are called before a stream so stateful concrete transitions
start from a defined state.

## Files

- `src/latent_anything/rollout_pipeline.py`
- `tests/test_latent_anything/test_rollout_pipeline.py`
- `docs/PLAN.md`
- `docs/sprint-plans/sprint-75.md`

## Focused validation

```text
uv run pytest -q tests/test_latent_anything/test_rollout_pipeline.py
10 passed in 11.51s
uv run ruff check src/latent_anything/rollout_pipeline.py tests/test_latent_anything/test_rollout_pipeline.py
All checks passed!
uv run ruff format src/latent_anything/rollout_pipeline.py tests/test_latent_anything/test_rollout_pipeline.py
2 files left unchanged
uv run pyright src/latent_anything/rollout_pipeline.py tests/test_latent_anything/test_rollout_pipeline.py
0 errors, 0 warnings, 0 informations
```

Tests cover eager-order equivalence, variable and empty chunks, per-chunk
metadata/profiling, one-chunk backpressure, early source close, and rejection
of oversized chunks before transition execution.

## Graph refresh

`graphify update .` is required immediately after this atomic completion.

Refresh completed immediately after task completion:

```text
graphify update .
AST extraction: 48/48 uncached files (100%)
Rebuilt graph: 9411 nodes, 18366 edges, 842 communities
```

The known 42 zero-node JSON/source warning was reported; no graphify failure
occurred. The graph topology count is unchanged from the prior Sprint 74
snapshot because this first implementation is represented in the existing
source graph aggregate.

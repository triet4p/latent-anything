# Sprint 75 Task 03 — Non-blocking async streaming

Status: Complete (2026-08-25)

## Scope

Added `RolloutPipeline.stream_async()` as an async-generator wrapper over the
concrete chunk story. Synchronous source iteration, validation, optional reset,
and transition execution run in shielded worker calls; the event loop is never
used for blocking producer or CPU work. There is no prefetch and therefore no
unbounded task queue. Cancellation waits for the current bounded worker call,
discards its result, closes the source, and re-raises `CancelledError`.
Output ordering and state carry match synchronous streaming.

## Files

- `src/latent_anything/rollout_pipeline.py`
- `tests/test_latent_anything/test_rollout_pipeline.py`
- `docs/PIPELINES.md`
- `docs/sprint-plans/sprint-75.md`

## Focused validation

```text
uv run pytest -q tests/test_latent_anything/test_rollout_pipeline.py
14 passed in 6.11s
uv run ruff check src/latent_anything/rollout_pipeline.py tests/test_latent_anything/test_rollout_pipeline.py
All checks passed!
uv run pyright src/latent_anything/rollout_pipeline.py tests/test_latent_anything/test_rollout_pipeline.py
0 errors, 0 warnings, 0 informations
```

Tests prove ordered async output, event-loop responsiveness, worker settling
under cancellation, and source cleanup after cancellation.

## Graph refresh

`graphify update .` is required immediately after this atomic completion.

Refresh completed immediately after task completion:

```text
graphify update .
Rebuilt graph: 9446 nodes, 18446 edges, 829 communities
```

The known 42 zero-node JSON/source warning was reported; no graphify failure
occurred.

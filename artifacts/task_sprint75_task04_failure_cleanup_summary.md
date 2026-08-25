# Sprint 75 Task 04 — Failure, cancellation, and cleanup coverage

Status: Complete (2026-08-25)

## Scope

Expanded the streaming regression suite for producer failures, transition
failures before chunk publication, async consumer close, cancellation cleanup,
source finalizers, oversized chunks, empty/final chunks, and stateful reset plus
cross-chunk carry. The one-in-flight/no-prefetch behavior is asserted by
observing that the source has produced only the first chunk before the first
result is consumed.

## Files

- `tests/test_latent_anything/test_rollout_pipeline.py`
- `docs/sprint-plans/sprint-75.md`

## Focused validation

```text
uv run pytest -q tests/test_latent_anything/test_rollout_pipeline.py
15 passed in 6.69s
uv run ruff check src/latent_anything/rollout_pipeline.py tests/test_latent_anything/test_rollout_pipeline.py
All checks passed!
uv run ruff format src/latent_anything/rollout_pipeline.py tests/test_latent_anything/test_rollout_pipeline.py
1 file reformatted, 1 file left unchanged
uv run pyright src/latent_anything/rollout_pipeline.py tests/test_latent_anything/test_rollout_pipeline.py
0 errors, 0 warnings, 0 informations
```

## Graph refresh

`graphify update .` is required immediately after this atomic completion.

Refresh completed immediately after task completion:

```text
graphify update .
Rebuilt graph: 9462 nodes, 18468 edges, 827 communities
```

The known 42 zero-node JSON/source warning was reported; no graphify failure
occurred.

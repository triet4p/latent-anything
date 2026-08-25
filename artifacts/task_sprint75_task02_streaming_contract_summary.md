# Sprint 75 Task 02 — Chunk and state contract

Status: Complete (2026-08-25)

## Scope

Documented and regression-tested the concrete rollout-stream contract. Chunks
are ordered, disjoint, and variable-length; empty chunks are consumed and
skipped; overlap/window semantics are intentionally unsupported. There is no
prefetch queue: the producer is advanced only after the consumer requests the
next result, and `max_chunk_rows` bounds the one in-flight chunk. Transition
errors occur before the failing chunk is yielded, producer errors propagate,
and previously yielded chunks remain an explicit partial result. Optional
`reset()` hooks make stateful transitions start cleanly while carrying their
hidden state across chunks.

## Files

- `src/latent_anything/rollout_pipeline.py`
- `tests/test_latent_anything/test_rollout_pipeline.py`
- `docs/PIPELINES.md`
- `docs/sprint-plans/sprint-75.md`

## Focused validation

```text
uv run pytest -q tests/test_latent_anything/test_rollout_pipeline.py
12 passed in 5.97s
uv run ruff check src/latent_anything/rollout_pipeline.py tests/test_latent_anything/test_rollout_pipeline.py
All checks passed!
uv run ruff format src/latent_anything/rollout_pipeline.py tests/test_latent_anything/test_rollout_pipeline.py
2 files left unchanged
uv run pyright src/latent_anything/rollout_pipeline.py tests/test_latent_anything/test_rollout_pipeline.py
0 errors, 0 warnings, 0 informations
```

Coverage includes empty/final/variable chunks, one-chunk backpressure, source
cleanup, oversized-chunk rejection before transition execution, resettable
state, transition errors, and producer errors.

## Graph refresh

`graphify update .` is required immediately after this atomic completion.

Refresh completed immediately after task completion:

```text
graphify update .
Rebuilt graph: 9428 nodes, 18411 edges, 859 communities
```

The known 42 zero-node JSON/source warning was reported; no graphify failure
occurred.

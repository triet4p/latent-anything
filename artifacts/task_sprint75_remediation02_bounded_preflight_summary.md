# Sprint 75 Remediation 02 — Fail-closed action-chunk preflight

Status: Complete (2026-08-25)

## Scope

Streaming action chunks now require an exact `numpy.ndarray`. Rank, action
width, and row count are inspected before any dtype conversion/copy. Chunks
that are arbitrary lists, array-protocol objects, wrong-rank arrays, object
dtypes, non-finite values, or larger than `max_chunk_rows` fail closed. This
keeps the public boundary NumPy-only and prevents an unbounded conversion from
preceding the row limit.

## Files

- `src/latent_anything/rollout_pipeline.py`
- `tests/test_latent_anything/test_rollout_pipeline.py`
- `docs/sprint-plans/sprint-75.md`

## Focused validation

```text
uv run pytest -q tests/test_latent_anything/test_rollout_pipeline.py tests/test_sprint75_streaming.py
20 passed in 4.14s
uv run ruff check src/latent_anything/rollout_pipeline.py tests/test_latent_anything/test_rollout_pipeline.py tests/test_sprint75_streaming.py scripts/sprint75_streaming_benchmark.py
All checks passed!
uv run ruff format --check src/latent_anything/rollout_pipeline.py tests/test_latent_anything/test_rollout_pipeline.py tests/test_sprint75_streaming.py scripts/sprint75_streaming_benchmark.py
4 files already formatted
uv run pyright src/latent_anything/rollout_pipeline.py tests/test_latent_anything/test_rollout_pipeline.py tests/test_sprint75_streaming.py scripts/sprint75_streaming_benchmark.py
0 errors, 0 warnings, 0 informations
```

The adversarial tests prove that nested sequences and an object implementing
`__array__` are rejected without invoking conversion.

## Graph refresh

`graphify update .` is required immediately after this atomic completion.
Refresh completed with the known 42 zero-node JSON/source warning:

```text
graphify update .
Rebuilt graph: 9531 nodes, 18551 edges, 865 communities
```

Graphify reported a community-label refresh (830 saved labels, 865 current
communities, 124 renamed hubs) and backed up the semantic/curated graph. No
graphify failure occurred.

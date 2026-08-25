# Sprint 75 Task 05 — Streaming benchmark and eager equivalence

Status: Complete (2026-08-25)

## Scope

Added an offline CPU reproduction and regression test for a 4096-step rollout
with 64-row action chunks. The stream keeps one chunk in flight, emits 4096
rows in order, and matches the eager tail byte-for-byte by SHA-256 digest. The
benchmark consumes each output chunk immediately rather than accumulating a
second full trajectory; `tracemalloc` is reported as supplemental Python
allocation evidence while the explicit NumPy chunk-byte bound is the primary
memory claim.

## Files

- `scripts/sprint75_streaming_benchmark.py`
- `tests/test_sprint75_streaming.py`
- `docs/sprint-plans/sprint-75.md`

## Focused validation and measurements

```text
uv run pytest -q tests/test_sprint75_streaming.py tests/test_latent_anything/test_rollout_pipeline.py
16 passed in 4.79s
uv run ruff check src/latent_anything/rollout_pipeline.py scripts/sprint75_streaming_benchmark.py tests/test_sprint75_streaming.py tests/test_latent_anything/test_rollout_pipeline.py
All checks passed!
uv run pyright src/latent_anything/rollout_pipeline.py scripts/sprint75_streaming_benchmark.py tests/test_sprint75_streaming.py
0 errors, 0 warnings, 0 informations
uv run python scripts/sprint75_streaming_benchmark.py
status=pass; horizon=4096; chunk_rows=64; queue_capacity=1; streamed_rows=4096
eager_seconds=0.067632; stream_seconds=0.282119
eager_output_bytes=65536; stream_max_chunk_bytes=1024
stream_peak_tracemalloc_bytes=50384; profile_events=64
eager_digest=stream_digest=2b103dd24ec88375fbdbe76c7bb92a00abc5e6472454a7ba867f9d3087a8bf00
```

The benchmark is synthetic, offline, CPU-only evidence; it does not claim
LeRobot or real-model throughput.

After the final bounded-profile refinement, the rerun remained `status=pass`
with `eager_seconds=0.075836`, `stream_seconds=0.211525`,
`stream_peak_tracemalloc_bytes=28576`, and `profile_events=1`. The earlier
64-event result is retained as historical pre-refinement output; the current
contract aggregates one profile event per stream.

After Sprint 75 audit remediation, the final rerun remained `status=pass`
with `eager_seconds=0.119332`, `stream_seconds=0.420138`,
`stream_peak_tracemalloc_bytes=28576`, `stream_max_chunk_bytes=1024`, and
`profile_events=1`; the eager and stream SHA-256 digests remained identical.

A post-cleanup-helper verification rerun also remained `status=pass` with
`eager_seconds=0.263273`, `stream_seconds=0.588445`,
`stream_peak_tracemalloc_bytes=28576`, `stream_max_chunk_bytes=1024`, and
`profile_events=1`; the same eager/stream digest was reproduced. Timing varies
with local CPU load, so the benchmark is used for boundedness and parity, not a
throughput claim.

## Graph refresh

`graphify update .` is required immediately after this atomic completion.

Refresh completed immediately after task completion:

```text
graphify update .
Rebuilt graph: 9469 nodes, 18474 edges, 815 communities
```

The known 42 zero-node JSON/source warning was reported; no graphify failure
occurred.

# Sprint 75 Task 08 — Final closure gates

Status: Complete after post-closure remediation (2026-08-25)

## Scope

Closed Sprint 75 after the concrete rollout streaming story, failure and
cancellation tests, benchmark, Rule-of-Three decision, docs, typed ledger,
changelog, and final quality gates passed. Sprint 76 was not started. The
subsequent closure audit remediation bounded async iterator setup/cleanup,
made action-chunk validation fail closed before conversion, strengthened
event-loop regression coverage, and reconciled state and evidence scope.

## Final validation

```text
uv run ruff check src tests scripts
All checks passed!
uv run ruff format --check src tests scripts
253 files already formatted
uv run pyright
Completed with no diagnostics; changed-scope strict Pyright: 0 errors, 0 warnings, 0 informations
uv run pytest -q
1427 passed, 32 skipped, 39 warnings in 167.83s (0:02:47)
uv run python scripts/validate_evidence_ledger.py
Inventory: 107; core 23/63 (36.5%); overall 23/65 (35.4%); exit 0
uv run python scripts/sprint75_streaming_benchmark.py
status=pass; horizon=4096; chunk_rows=64; queue_capacity=1; streamed_rows=4096
eager_seconds=0.119332; stream_seconds=0.420138
eager_output_bytes=65536; stream_max_chunk_bytes=1024; profile_events=1
eager_digest == stream_digest
uv run --extra docs mkdocs build --strict
PASS (Material for MkDocs 2.0 upstream advisory only)
git diff --check
PASS (only expected LF-to-CRLF working-copy notices)
```

The benchmark is synthetic offline CPU evidence. Stream latency was higher
than eager on this small fixture (`0.420138s` versus `0.119332s`), while the
bounded output chunk remained 1024 bytes versus 65536 eager-tail bytes. This
is a memory/order contract proof, not a throughput claim. Profiling emits one
bounded aggregate event per stream rather than one event per chunk; memory
scope is explicit NumPy chunk bytes plus supplemental `tracemalloc`, not native
RSS.

A post-cleanup-helper benchmark verification also passed with
`eager_seconds=0.263273`, `stream_seconds=0.588445`,
`stream_peak_tracemalloc_bytes=28576`, `stream_max_chunk_bytes=1024`, and one
aggregate profile event; the eager/stream digest remained identical. Timing
varies with local CPU load and is not used as a throughput claim.

## Post-closure remediation records

- `artifacts/task_sprint75_remediation01_async_boundary_summary.md`
- `artifacts/task_sprint75_remediation02_bounded_preflight_summary.md`
- `artifacts/task_sprint75_remediation03_event_loop_test_summary.md`
- `artifacts/task_sprint75_remediation04_scope_docs_summary.md`

All four remediation tasks passed focused tests, and the final closure gates
below were rerun after their changes.

## Graph refresh

`graphify update .` is required immediately after this closure artifact.

Refresh completed immediately after closure:

```text
graphify update .
Rebuilt graph: 9495 nodes, 18507 edges, 843 communities
```

The known 42 zero-node JSON/source warning was reported; no graphify failure
occurred.

After the final remediation refresh, graphify reported `9549 nodes, 18575
edges, 845 communities`; the same 42 zero-node warning remained. This final
refresh included the completed source/docs reconciliation.

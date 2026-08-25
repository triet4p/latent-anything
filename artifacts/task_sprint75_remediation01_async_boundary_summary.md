# Sprint 75 Remediation 01 — Async iterator and cleanup boundary

Status: Complete (2026-08-25)

## Scope

The closure audit found that `stream_async()` dispatched `next()` but still
constructed the iterator and called its finalizer on the event-loop thread.
Iterator construction and cleanup now use the same settled worker boundary as
validation and transition execution. Cancellation settles the active worker,
then settles cleanup before re-raising; no worker result is committed after
cancellation.

## Files

- `src/latent_anything/rollout_pipeline.py`
- `tests/test_latent_anything/test_rollout_pipeline.py`
- `docs/sprint-plans/sprint-75.md`

## Focused validation

```text
uv run pytest -q tests/test_latent_anything/test_rollout_pipeline.py
17 passed in 4.21s
```

The regression suite now exercises blocking `__iter__`, blocking `close`,
cancellation while transition work is active, early async-generator close,
source finalization, and completion of the worker tasks.

## Graph refresh

`graphify update .` is required immediately after this atomic completion.
Refresh completed with the known 42 zero-node JSON/source warning:

```text
graphify update .
Rebuilt graph: 9521 nodes, 18534 edges, 830 communities
```

Graphify also reported that the community set changed (855 saved labels,
830 current communities, 138 renamed hubs) and backed up the semantic/curated
graph. No graphify failure occurred.

After the final cleanup-owner refinement, a focused rerun still passed and the
refresh reported `9549 nodes, 18575 edges, 843 communities`; the same 42
zero-node warning remained. Distinct iterators and owning iterables are now
both finalized without double-closing.

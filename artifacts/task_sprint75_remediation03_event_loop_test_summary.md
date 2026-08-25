# Sprint 75 Remediation 03 — Event-loop responsiveness regression

Status: Complete (2026-08-25)

## Scope

The former heartbeat assertion could pass after only one initial scheduling
yield and therefore did not detect a later blocking transition. The test now
starts a delayed transition in a task, synchronizes on worker start and
completion, and asserts ticker progress specifically during the delayed
operation. It retains ordered output assertions for the complete async stream.

## Files

- `tests/test_latent_anything/test_rollout_pipeline.py`
- `docs/sprint-plans/sprint-75.md`

## Focused validation

```text
uv run pytest -q tests/test_latent_anything/test_rollout_pipeline.py tests/test_sprint75_streaming.py
20 passed in 4.25s
```

The delayed transition is still the existing CPU-only worker fixture; the
regression now fails if that work executes synchronously on the event loop.

## Graph refresh

`graphify update .` is required immediately after this atomic completion.
Refresh completed with the known 42 zero-node JSON/source warning:

```text
graphify update .
Rebuilt graph: 9537 nodes, 18556 edges, 852 communities
```

Graphify reported a community-label refresh (865 saved labels, 852 current
communities, 152 renamed hubs) and backed up the semantic/curated graph. No
graphify failure occurred.

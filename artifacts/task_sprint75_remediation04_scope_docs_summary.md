# Sprint 75 Remediation 04 — State and evidence scope reconciliation

Status: Complete (2026-08-25)

## Scope

The streaming contract now fails before source consumption when a transition
has neither `reset()` nor the explicit `stream_state_contract = "explicit"`
marker. Built-in predictive-mean transitions declare their explicit state
surface; RSSM-style stateful transitions use reset. Documentation and evidence
now state that masks/padding and seeded sampling are outside this action-chunk
story, that streaming intentionally bypasses cache/run-record persistence, and
that profiling and provenance are bounded to per-stream counters/metadata.
Benchmark memory claims are explicitly limited to NumPy chunk bytes and
supplemental `tracemalloc`, not native RSS.

## Files

- `src/latent_anything/rollout_pipeline.py`
- `src/latent_anything/transition.py`
- `src/latent_anything/adapters/jepa.py`
- `src/latent_anything/tokenized_world_model.py`
- `tests/test_latent_anything/test_rollout_pipeline.py`
- `docs/PIPELINES.md`
- `docs/EVIDENCE_LEDGER.md`
- `README.md`
- `CHANGELOG.md`
- `.agents/memory/decisions.md`
- `.agents/memory/lessons-learned.md`
- `docs/sprint-plans/sprint-75.md`

## Focused validation

```text
uv run pytest -q tests/test_latent_anything/test_rollout_pipeline.py tests/test_sprint75_streaming.py
21 passed in 7.72s
uv run ruff check <changed Sprint 75 source/tests/scripts scope>
All checks passed!
uv run ruff format --check <changed Sprint 75 source/tests/scripts scope>
8 files already formatted
uv run pyright <changed Sprint 75 source/tests/scripts scope>
0 errors, 0 warnings, 0 informations
```

## Graph refresh

`graphify update .` is required immediately after this atomic completion.
Refresh completed with the known 42 zero-node JSON/source warning:

```text
graphify update .
Rebuilt graph: 9548 nodes, 18574 edges, 838 communities
```

Graphify reported a community-label refresh (852 saved labels, 838 current
communities, 137 renamed hubs) and backed up the semantic/curated graph. No
graphify failure occurred.

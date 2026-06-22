# Sprint 20 — Pipeline #1 Summary

**Date:** 2026-06-22
**Commit:** *(pending)*
**Verdict:** PASS

## Summary

Added **Pipeline #1** — the first concrete `Pipeline` instance in the
latent-anything framework. The `AnalysisPipeline` class chains a model
adapter's `encode` step with a Layer A method's `fit` + `transform`,
returning a typed `PipelineResult` frozen dataclass.

## Scope

- **Pipeline #1** is deliberately narrow: adapter + Layer A method only.
  No Layer B support, no generic DAG/executor abstraction.
- Config-backed construction via `PipelineSpec` (pydantic) +
  `build_pipeline_from_config`, using Sprint 18 config machinery.
- Works with both decodable (VAE) and non-decodable (HiddenStateAdapter)
  adapters.

## Files created

| File | Purpose |
|---|---|
| `src/latent_anything/pipeline.py` | `AnalysisPipeline`, `PipelineResult`, `PipelineSpec`, `build_pipeline_from_config` |
| `tests/test_latent_anything/test_pipeline.py` | 21 tests across 5 test classes |
| `scripts/end_to_end_pipeline_demo.py` | Reproduction script with direct + config construction paths |

## Files modified

| File | Change |
|---|---|
| `src/latent_anything/__init__.py` | Added pipeline exports |
| `CHANGELOG.md` | Added Sprint 20 entries |
| `docs/PLAN.md` | Moved Sprint 20 to Completed |
| `docs/sprint-plans/sprint-20.md` | All tasks marked `[x]` |

## Test results

- **pytest:** 523 passed (502 existing + 21 new), 0 failed
- **ruff check:** All checks passed
- **ruff format --check:** All files already formatted
- **pyright strict:** 0 errors, 0 warnings, 0 informations

## Rule of Three check

Pipeline instance #1 stays concrete. No:
- Generic DAG/executor abstraction
- New registry kind (`KIND_PIPELINE`)
- Broad workflow engine
- `Pipeline` Protocol/ABC

Generalisation waits for Pipeline #2 (Sprint 21) when at least two
distinct execution stories exist.

## ADR reconciliation

Not applicable — this is a composition round, not an instance-adding
round. All three 2026-06-16 ADRs remain:
- `LatentSpace` geometry-keyed: **validated** (no change)
- Geometry-dispatch: **validated** (no change)
- `ModelAdapter` 3-mode: **validated** (no change)

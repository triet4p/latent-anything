# Sprint 20 (Round 17) — Review Notes

**Commit:** `6046509` `feat(pipeline): implement Sprint 20 - AnalysisPipeline #1 with config-backed construction`

**Verdict:** PASS

## Key facts

- **Instance count:** Pipeline #1 (AnalysisPipeline) — single concrete instance. No Protocol/ABC introduced. Generalisation waits for Pipeline #2 (Sprint 21).
- **Public surface:** `AnalysisPipeline`, `PipelineResult`, `PipelineSpec`, `build_pipeline_from_config` exported from top-level `__init__` and `__all__`.
- **All 3 ADRs remain unchanged:** `LatentSpace` geometry-keyed (validated), Geometry-dispatch (validated), `ModelAdapter` 3-mode (validated). No ADR reconciliation needed — composition-only round.
- **Pipeline stays concrete:** No new registry kind (`KIND_PIPELINE`), no DAG/executor abstraction, no Layer B support.
- **Config-backed construction:** `PipelineSpec` (pydantic) wraps two `ObjectSpec` instances. `build_pipeline_from_config` resolves through registry.
- **Works with both decodable and non-decodable adapters:** VAE/PCA and HiddenStateAdapter/PCA tested.

## Tooling gate

- `ruff check` / `ruff format --check` / `pyright strict` / `pytest` — all clean (523 tests, 21 new).

## Things to watch in future reviews

- `PipelineSpec` uses pydantic auto-coercion of dicts to `ObjectSpec` — this works at runtime but pyright flags `reportArgumentType`. Tests use `# pyright: ignore` comments.
- `adapter.fit(data)` is called explicitly in tests before `pipeline.run(data)` because `fit` is not part of `ModelAdapter` Protocol. VAE needs pre-fitting; HiddenStateAdapter doesn't. This is correct — the pipeline does not (and should not) assume adapters have `fit`.
- `PipelineResult` is a frozen dataclass — no mutation. Callers must create new instances for different results.

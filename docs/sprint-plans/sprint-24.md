# Sprint 24 Plan

## Sprint Goal
Increment thứ hai mươi mốt (Round 21): add **async execution and profiling hooks** as the next Layer C runtime increment. This makes runtime behavior observable before deciding on heavier optimization.

## Atomic Tasks
Status legend: [ ] pending / [~] in progress / [x] done

- [x] Task 1: Add async wrappers for selected pipeline/executor paths using `asyncio`.
- [x] Task 2: Keep sync convenience wrappers where existing scripts need them.
- [x] Task 3: Add profiling hooks that report latency per stage: encode, method, decode, cache.
- [x] Task 4: Add tests for async result parity with sync paths.
- [x] Task 5: Add tests that profiling records stage names and non-negative durations.
- [x] Task 6: Add a demo that runs two independent pipeline jobs concurrently and prints a latency breakdown.
- [x] Task 7: Run `ruff check`, `ruff format`, `pyright`, and full pytest.
- [x] Task 8: Rule check: async/profiling is runtime instance #3-ish only if BatchExecutor + Cache exposed enough shared runtime shape; freeze a `RuntimeExecutor` Protocol only if the code demands it.
- [x] Task 9: Update artifact summary, `CHANGELOG.md`, and `docs/PLAN.md`.

## Notes / Blockers
* Do not add distributed execution.
* Do not add Rust. Rust core remains evidence-based and later.
* Completed with thin async wrappers on existing concrete runtime paths: `AnalysisPipeline.run_async()`, `ManipulationPipeline.run_data_async()/run_trajectory_async()`, and `BatchExecutor.encode_async()/decode_async()/transform_async()`.
* Profiling stays hook-based and concrete through `RuntimeProfiler` / `RuntimeProfile` / `ProfileEvent`; no `RuntimeExecutor` Protocol was frozen because BatchExecutor + Cache + async wrappers still share too little invariant surface.
* Demo artifact: `artifacts/async_runtime_demo_summary.txt` captured a concurrent run where `AnalysisPipeline` reported `cache/encode/method` timings and `ManipulationPipeline` reported `encode/method/decode` timings.
* Verification: `uv run ruff format src tests scripts`, `uv run ruff check src tests scripts`, `uv run pyright`, and full `uv run pytest` all pass. Final suite: 594 passed, 9 existing UMAP warnings.

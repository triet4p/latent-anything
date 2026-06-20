# Sprint 24 Plan

## Sprint Goal
Increment thứ hai mươi mốt (Round 21): add **async execution and profiling hooks** as the next Layer C runtime increment. This makes runtime behavior observable before deciding on heavier optimization.

## Atomic Tasks
Status legend: [ ] pending / [~] in progress / [x] done

- [ ] Task 1: Add async wrappers for selected pipeline/executor paths using `asyncio`.
- [ ] Task 2: Keep sync convenience wrappers where existing scripts need them.
- [ ] Task 3: Add profiling hooks that report latency per stage: encode, method, decode, cache.
- [ ] Task 4: Add tests for async result parity with sync paths.
- [ ] Task 5: Add tests that profiling records stage names and non-negative durations.
- [ ] Task 6: Add a demo that runs two independent pipeline jobs concurrently and prints a latency breakdown.
- [ ] Task 7: Run `ruff check`, `ruff format`, `pyright`, and full pytest.
- [ ] Task 8: Rule check: async/profiling is runtime instance #3-ish only if BatchExecutor + Cache exposed enough shared runtime shape; freeze a `RuntimeExecutor` Protocol only if the code demands it.
- [ ] Task 9: Update artifact summary, `CHANGELOG.md`, and `docs/PLAN.md`.

## Notes / Blockers
* Do not add distributed execution.
* Do not add Rust. Rust core remains evidence-based and later.

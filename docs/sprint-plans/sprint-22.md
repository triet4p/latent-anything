# Sprint 22 Plan

## Sprint Goal
Increment thứ mười chín (Round 19): start Layer C with **BatchExecutor #1**. The executor handles explicit batching for adapter/method calls without cache or async yet.

## Atomic Tasks
Status legend: [ ] pending / [~] in progress / [x] done

- [x] Task 1: Implement a small `BatchExecutor` with deterministic chunking over numpy arrays.
- [x] Task 2: Support adapter `encode`/`decode` batching first.
- [x] Task 3: Support one method path that has clear batch semantics, such as PCA transform or ActivationPatch call.
- [x] Task 4: Preserve output order and shape exactly.
- [x] Task 5: Add tests for exact divisibility, remainder batch, batch size 1, batch size larger than data, and invalid batch sizes.
- [x] Task 6: Add benchmark-ish demo numbers for direct vs batched path on synthetic data.
- [x] Task 7: Run `ruff check`, `ruff format`, `pyright`, and full pytest.
- [x] Task 8: Rule check: runtime instance #1 stays eager/sync; no cache or async in this sprint.
- [x] Task 9: Update artifact summary, `CHANGELOG.md`, and `docs/PLAN.md`.

## Notes / Blockers
* This sprint is correctness-first. Performance optimization follows evidence.
* Completed with `BatchExecutor` as Runtime #1: eager/sync only, no cache, async, worker pool, or DAG abstraction.
* Verification: CI-style `ruff check src/ tests/`, `ruff format --check src/ tests/`, `pyright`, and full `pytest` pass (575 tests).

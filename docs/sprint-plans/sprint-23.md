# Sprint 23 Plan

## Sprint Goal
Increment thứ hai mươi (Round 20): add **in-memory cache #1** for encoded latents and fitted method outputs. Disk cache waits until the in-memory key shape is proven.

## Atomic Tasks
Status legend: [ ] pending / [~] in progress / [x] done

- [ ] Task 1: Define a stable cache key structure using input data hash, adapter/method name, config hash, and framework version when available.
- [ ] Task 2: Implement an in-memory cache backend with `get`, `set`, `clear`, and stats.
- [ ] Task 3: Integrate cache into one concrete path: adapter encode through Pipeline #1 or BatchExecutor.
- [ ] Task 4: Add tests for cache hit, miss, invalidation by config, invalidation by data, and no mutation of cached arrays.
- [ ] Task 5: Add a small demo showing measurable repeated-call speedup.
- [ ] Task 6: Run `ruff check`, `ruff format`, `pyright`, and full pytest.
- [ ] Task 7: Rule check: cache backend #1 stays memory-only; no diskcache dependency until backend #2.
- [ ] Task 8: Update artifact summary, `CHANGELOG.md`, and `docs/PLAN.md`.

## Notes / Blockers
* Cache invalidation is part of the feature, not a follow-up.
* Do not use pickle for public cache format decisions in this sprint.

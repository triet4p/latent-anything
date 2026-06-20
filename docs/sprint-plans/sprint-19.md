# Sprint 19 Plan

## Sprint Goal
Increment thứ mười sáu (Round 16): convert built-in adapters and methods to **registry-first built-ins** and prove behavior parity against direct imports. This completes the practical plugin-extraction baseline before entry points.

## Atomic Tasks
Status legend: [ ] pending / [~] in progress / [x] done

- [ ] Task 1: Move built-in registrations into stable import locations that avoid import cycles.
- [ ] Task 2: Ensure direct imports still work exactly as before.
- [ ] Task 3: Add parity tests: direct constructor vs registry constructor for representative adapters/methods.
- [ ] Task 4: Add smoke tests that old demo scripts still run or their core helpers still pass.
- [ ] Task 5: Document the internal plugin extraction contract in module docstrings.
- [ ] Task 6: Decide whether entry points are now justified; if not, record why in artifact summary rather than implementing them.
- [ ] Task 7: Run `ruff check`, `ruff format`, `pyright`, and full pytest.
- [ ] Task 8: ADR check: registry + config matches the no-vector-DB architecture decision; append a decision only if a new irreversible choice appears.
- [ ] Task 9: Update artifact summary, `CHANGELOG.md`, and `docs/PLAN.md`.

## Notes / Blockers
* Registry is implementation infrastructure. It should not force users to abandon normal Python imports.

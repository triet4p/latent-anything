# Sprint 25 Plan

## Sprint Goal

Resolve every blocking and advisory finding from the Sprint 17–24 review, restore the full changed-scope tooling gate, and commit only after a clean re-review.

## Atomic Tasks

Status legend: [ ] pending / [~] in progress / [x] done

- [x] Task 1: Make cache keys distinguish component runtime/model state and add cross-instance regression coverage.
- [x] Task 2: Keep stateful Layer A methods fitted when `AnalysisPipeline` serves cached transformations and add shared-cache regression coverage.
- [x] Task 3: Align the frozen `BMethod` Protocol with all three concrete implementations and remove related strict typing failures.
- [x] Task 4: Clear every remaining Pyright error in Python files changed since Sprint 17.
- [x] Task 5: Consolidate the `[Unreleased]` changelog structure and record the review fixes.
- [x] Task 6: Record cache correctness lessons and produce the Sprint 25 task summary artifact.
- [x] Task 7: Run the full latent-anything review gate and inspect ADR, Rule-of-Three, and test integrity.
- [x] Task 8: Mark Sprint 25 complete, update the global plan, and create a Conventional Commit if the review passes.

## Notes / Blockers

* This is a corrective sprint, not a new runtime abstraction increment.
* The historical non-Conventional commit `49d9e1a` will not be rewritten because rewriting shared history is riskier than preserving it; Sprint 25 records and follows the correct convention.
* Final review: Ruff check passed, Ruff format check passed, changed-scope Pyright reported 0 errors, and all 596 tests passed with 9 existing UMAP warnings.

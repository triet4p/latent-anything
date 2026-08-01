# Sprint 45 Plan

## Sprint Goal

Add activation-space Integrated Gradients from a selected transformer residual-layer activation to a scalar next-token logit, with completeness and sensitivity checks.

## Entry Criteria

- Sprint 39 provides a revision-pinned causal language model, token semantics, scalar logit target, and gradient-preserving intervention seam.
- Sprint 42 provides concept/intervention evidence suitable for a bounded observational comparison.

## Atomic Tasks

Status legend: [ ] pending / [~] in progress / [x] done

- [x] Define one concrete attribution path from a selected residual-layer/token activation to a declared scalar next-token logit; defer input-token attribution to separate scope.
- [x] Define a bounded baseline family, integration rule, and step-count range with full layer/token/target/model provenance.
- [x] Preserve PyTorch gradients internally while returning a typed NumPy attribution result with convergence delta/completeness error.
- [x] Test completeness on analytic models and quantify approximation error against declared tolerances.
- [x] Evaluate step sensitivity, baseline sensitivity, target specificity, and parameter-randomization sanity checks.
- [x] Integrate through the hook/intervention seam without leaked hooks, retained graphs, or globally enabled gradients after completion.
- [x] Produce real-transformer positive, negative, and unstable examples and compare observational attribution with Sprint 39 intervention and Sprint 42 concept evidence.
- [x] Add direct/config construction, deterministic and marked integration tests, then update evidence/ADR/changelog/artifact/gates.

## Notes / Blockers

Attribution magnitude is observational evidence, not causal proof. Restricting the first implementation to one activation-space path keeps completeness, masking, and baseline semantics testable.

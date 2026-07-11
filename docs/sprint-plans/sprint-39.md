# Sprint 39 Plan

## Sprint Goal

Add a label-aware linear probe with controlled evaluation that answers what information is linearly accessible from a representation.

## Atomic Tasks

Status legend: [ ] pending / [~] in progress / [x] done

- [ ] Implement classification and/or regression `LinearProbe` behavior with explicit train/validation/test splits.
- [ ] Define typed labels, predictions, coefficients, scores, and split metadata without overloading the dimensionality-reduction protocol.
- [ ] Add regularization, class-balance, seed, and feature-standardization controls.
- [ ] Evaluate against majority, shuffled-label, and raw-input baselines on VAE and one hidden-state integration.
- [ ] Add leakage checks and tests that fitting never sees evaluation labels.
- [ ] Register the probe under the semantic analysis kind and support config construction.
- [ ] Produce coefficient/stability artifacts with confidence intervals.
- [ ] Apply Rule of Three/ADR checks and update evidence/changelog/artifact/gates.

## Notes / Blockers

A high test score alone is not interpretability evidence; controls and split integrity are required.


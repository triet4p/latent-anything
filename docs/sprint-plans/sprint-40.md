# Sprint 40 Plan

## Sprint Goal

Add a label-aware linear classification probe with split integrity and controlled evaluation on real VAE and transformer representations.

## Entry Criteria

- Sprint 35 exposes real VAE representations with stable provenance.
- Sprint 39 exposes real transformer hidden states and valid token masks.

## Atomic Tasks

Status legend: [ ] pending / [~] in progress / [x] done

- [x] Define a classification-only `LinearProbe` configuration and typed result containing labels, predictions, probabilities/scores, coefficients, split metadata, and provenance.
- [x] Implement explicit train/validation/test splitting with leakage guards, deterministic seeds, feature standardization fit on training data only, regularization, and class-balance controls.
- [x] Keep construction direct and registry-by-config under the semantic `analysis` kind; do not force label-aware fitting through the current unlabeled `Method`/`AnalysisPipeline` lifecycle.
- [x] Compare against majority-class, shuffled-label, and raw-input controls on the same splits.
- [x] Evaluate the probe on one real VAE representation dataset and one real transformer hidden-state dataset from Sprint 39.
- [x] Report confidence intervals and coefficient/score stability across seeds and declared representation layers.
- [x] Reconcile or migrate the Sprint 36 centroid-based `probe_accuracy` helper so the project has one unambiguous meaning for linear probing.
- [x] Add unit, leakage, degenerate-class, config-construction, and marked real-integration tests.
- [x] Apply the Rule of Three/ADR check and update theory evidence, changelog, artifact index, and gates.

## Notes / Blockers

The registry's `analysis` kind is a semantic taxonomy, not proof of a shared callable protocol. A high held-out score is evidence of accessible information only when split integrity and controls pass.


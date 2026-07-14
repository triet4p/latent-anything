# Sprint 41 Plan

## Sprint Goal

Add a bounded nonlinear classification probe as an information-accessibility upper bound without confusing probe capacity with representation quality.

## Entry Criteria

- Sprint 40 split, dataset, baseline, and typed-result conventions are stable enough to reuse directly.

## Atomic Tasks

Status legend: [ ] pending / [~] in progress / [x] done

- [x] Implement one small MLP classifier with a bounded architecture, deterministic initialization, early stopping, and NumPy-facing typed results.
- [x] Reuse Sprint 40 train/validation/test and preprocessing invariants while keeping nonlinear-model-specific state out of the linear result.
- [x] Report architecture, parameter count, optimizer, training steps, stopping decision, seed, and representation provenance.
- [x] Compare linear and nonlinear accessibility on identical splits for the real VAE and transformer datasets.
- [x] Use a predeclared selectivity control and shuffled-label memorization test; remove vague MDL-inspired claims unless a complete MDL protocol is separately planned.
- [x] Report uncertainty and classify findings as linear, nonlinear-only, unsupported, or memorization-prone under explicit thresholds.
- [x] Add component-local config/state round-trip tests without introducing framework-wide model serialization.
- [x] Add capacity, overfit, degenerate-label, determinism, and marked real-integration tests, then update evidence/ADR/changelog/artifact/gates.

## Notes / Blockers

This is the second probe implementation, so shared probe abstractions remain provisional. TCAV has different inputs, target semantics, and statistics and must not be counted mechanically as a third interchangeable probe.


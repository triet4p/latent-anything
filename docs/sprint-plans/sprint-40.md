# Sprint 40 Plan

## Sprint Goal

Add a nonlinear probe as an information upper bound while controlling capacity so probe power is not confused with representation quality.

## Atomic Tasks

Status legend: [ ] pending / [~] in progress / [x] done

- [ ] Implement one small MLP probe with bounded architecture, early stopping, and NumPy-facing results.
- [ ] Reuse the split/evaluation contract from Sprint 39 without forcing linear coefficients onto nonlinear models.
- [ ] Add capacity, parameter-count, training-step, and random-seed reporting.
- [ ] Compare linear vs nonlinear accessibility and shuffled-label memorization across the same latent datasets.
- [ ] Add selectivity or minimum-description-length-inspired controls where feasible.
- [ ] Add config/registry construction and model-state serialization tests.
- [ ] Produce a benchmark that marks features as linear, nonlinear-only, or unsupported rather than declaring all probe success interpretable.
- [ ] Update evidence/ADR/changelog/artifact and run the gate.

## Notes / Blockers

This is Probe #2. Shared probe interfaces remain provisional until the concept/TCAV instance stresses them in Sprint 41.


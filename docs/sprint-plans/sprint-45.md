# Sprint 45 Plan

## Sprint Goal

Add a real transformer hidden-state and logit-lens integration that tracks information across layers with correct token/axis semantics.

## Atomic Tasks

Status legend: [ ] pending / [~] in progress / [x] done

- [ ] Select and revision-pin a compact pretrained transformer with a clear output head.
- [ ] Adapt Sprint 32 capture to layer-indexed residual/hidden states and token masks.
- [ ] Implement direct or tuned logit-lens decoding with explicit normalization/head assumptions.
- [ ] Validate layer outputs and final logits against the backend's native forward pass.
- [ ] Measure token-level rank/probability trajectories and stability over prompt perturbations.
- [ ] Compare observational lens results with one activation intervention or patching result.
- [ ] Add optional-extra tests, offline checkpoint handling, and a reproducible artifact.
- [ ] Update evidence/ADR/changelog/artifact and gates.

## Notes / Blockers

This sprint upgrades `HiddenStateAdapter` from a synthetic projection example to a real model integration; the old adapter may remain as a unit-test fixture.


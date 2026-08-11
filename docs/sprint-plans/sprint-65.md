# Sprint 65 Plan

## Sprint Goal

Add an RSSM-style recurrent stochastic transition as the third differing instance and extract the proven transition contract.

## Atomic Tasks

Status legend: [ ] pending / [~] in progress / [x] done

- [x] Implement deterministic recurrent state plus stochastic latent state with explicit reset/sequence semantics.
- [x] Train/evaluate on a compact temporal dataset and report reconstruction/prediction, KL, calibration, and horizon drift.
- [x] Compare stateful RSSM execution with the deterministic and memoryless stochastic instances.
- [x] Extract only invariant transition/rollout surfaces proven across all three instances and migrate prior call sites together.
- [x] Define serialization/checkpoint/config contracts for stateful transitions.
- [x] Add sequence masks, variable-length, reset, device, and reproducibility tests.
- [x] Publish a three-transition comparison artifact and failure analysis.
- [x] Log the freeze/revision ADR and update evidence/changelog/artifact/gates.

## Notes / Blockers

This is the Rule-of-Three decision point. If the three instances do not share a useful surface, keep separate concrete contracts and record why.

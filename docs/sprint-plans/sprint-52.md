# Sprint 52 Plan

## Sprint Goal

Add Dynamic Time Warping trajectory similarity for unequal-length latent sequences with geometry-aware point costs.

## Atomic Tasks

Status legend: [ ] pending / [~] in progress / [x] done

- [x] Implement DTW distance and alignment path using the trajectory's declared geometry/metric.
- [x] Add window/step constraints, normalization policy, and deterministic tie behavior.
- [x] Return typed alignment results with cost matrix summary and provenance.
- [x] Add analytic tests for identical, shifted, stretched, unequal-length, empty, and invalid-space cases.
- [x] Compare DTW with index-wise Euclidean distance on controlled and real policy trajectories.
- [x] Profile memory/time and add a bounded-memory implementation or documented size limit.
- [x] Integrate the result with the interactive trajectory renderer.
- [x] Update evidence/ADR/changelog/artifact and gates.

## Notes / Blockers

Distance values require normalization/context to be interpretable across lengths; the result contract must make that explicit.

# Sprint 63 Plan

## Sprint Goal

Add the first deterministic latent transition instance and execute multi-step rollout end-to-end on controlled dynamics.

## Atomic Tasks

Status legend: [ ] pending / [~] in progress / [x] done

- [x] Implement one concrete deterministic transition `z_t, a_t -> z_{t+1}` without a speculative public protocol.
- [x] Define action/state shape validation, source-space identity, horizon, and rollout metadata.
- [x] Add one-step training/evaluation and recursive multi-step rollout behavior.
- [x] Measure one-step error, horizon-dependent drift, runtime, and stability on known synthetic dynamics.
- [x] Return immutable latent trajectories compatible with DTW/segmentation analysis.
- [x] Add property/unit tests for identity dynamics, linear systems, shape errors, and deterministic seeds.
- [x] Produce a rollout artifact with error-vs-horizon analysis.
- [x] Apply Rule of Three (#1 hardcoded) and update evidence/ADR/changelog/gates.

## Notes / Blockers

This is Transition #1. Keep the implementation narrow and concrete.

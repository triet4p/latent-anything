# Sprint 56 Plan

## Sprint Goal

Ship a clean `latent_anything[lerobot]` boundary with pinned compatibility, import isolation, and a minimal bridge package skeleton.

## Atomic Tasks

Status legend: [ ] pending / [~] in progress / [x] done

- [ ] Audit the current stable LeRobot release, Python/Torch/NumPy constraints, policy APIs, processors, dataset APIs, evaluation CLI, and plugin mechanism at sprint activation time.
- [ ] Add the `lerobot` optional dependency extra with an explicit supported-version window and conflict diagnostics.
- [ ] Create an integration namespace that imports no LeRobot modules until the extra is used.
- [ ] Add base-install, extra-install, unsupported-version, and CPU-only smoke tests.
- [ ] Define bridge-owned data/result types while reusing LeRobot policy, processor, dataset, and environment objects rather than wrapping them wholesale.
- [ ] Document supported seams and explicitly rejected reimplementation scope.
- [ ] Add a compatibility CI lane and an upstream-upgrade checklist.
- [ ] Record the integration ADR, evidence/changelog/artifact, and gates.

## Notes / Blockers

Official LeRobot APIs are moving quickly. This sprint must refresh assumptions from upstream source rather than treating the July 2026 plan as a permanent API spec.


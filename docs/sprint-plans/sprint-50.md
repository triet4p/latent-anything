# Sprint 50 Plan

## Sprint Goal

Add one non-Euclidean geodesic interpolation implementation backed by density or decoder pullback geometry and compare it against lerp/slerp.

## Atomic Tasks

Status legend: [ ] pending / [~] in progress / [x] done

- [ ] Select one tractable concrete geometry: learned density-penalized paths or VAE decoder pullback metric.
- [ ] Implement path optimization with deterministic initialization, convergence reporting, and bounded compute.
- [ ] Return the full path plus length, density/reconstruction diagnostics, and optimization status.
- [ ] Add analytic tests on a known curved manifold and failure tests for non-convergence.
- [ ] Compare lerp, slerp where applicable, and geodesic paths on real VAE decoded quality and density.
- [ ] Add cache/profiling integration because path optimization is expensive.
- [ ] Document when the method is justified and when simpler interpolation is preferable.
- [ ] Update evidence/ADR/changelog/artifact and gates.

## Notes / Blockers

This sprint should implement one well-validated path method, not a generic Riemannian optimization library.


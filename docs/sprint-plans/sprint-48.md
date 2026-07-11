# Sprint 48 Plan

## Sprint Goal

Add anisotropic Gaussian geometry with covariance-aware validation, Mahalanobis distance, and statistically correct interpolation behavior.

## Atomic Tasks

Status legend: [ ] pending / [~] in progress / [x] done

- [ ] Define the anisotropic latent-space metadata and covariance ownership/fitting contract.
- [ ] Implement positive-definite covariance validation, regularization, Mahalanobis distance, whitening, and inverse transforms.
- [ ] Decide and document interpolation semantics instead of silently applying Euclidean lerp under anisotropy.
- [ ] Add analytic/property tests for affine invariance, singular covariance handling, and distance symmetry.
- [ ] Compare Euclidean vs Mahalanobis neighbors/OOD scores on a controlled anisotropic dataset and real latents.
- [ ] Route the implementation through the geometry modules extracted in Sprint 30.
- [ ] Add serialization/config support for fitted covariance provenance.
- [ ] Update evidence/ADR/changelog/artifact and gates.

## Notes / Blockers

If covariance is learned from data, the geometry is stateful and must be bound to the fitting dataset/model version.


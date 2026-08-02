# Sprint 49 Plan

## Sprint Goal

Add subspace projection, concept removal, and latent arithmetic with explicit coordinate-system and source-model compatibility checks.

## Atomic Tasks

Status legend: [ ] pending / [~] in progress / [x] done

- [x] Implement projection onto and removal from one fitted orthonormal subspace.
- [x] Implement latent arithmetic only for values proven to share space identity, geometry, shape, and source-model revision.
- [x] Preserve immutable latent values and attach operation/provenance metadata to outputs.
- [x] Add analytic tests for idempotence, orthogonality, reconstruction, and invalid cross-space operations.
- [x] Evaluate concept removal for target suppression, off-target preservation, and decode degradation.
- [x] Compare projection bases from PCA, probe coefficients, and concept directions without treating them as interchangeable.
- [x] Add registry/config support under canonical transformation vocabulary.
- [x] Update evidence/ADR/changelog/artifact and gates.

## Notes / Blockers

The framework should reject mathematically invalid arithmetic across unrelated coordinate systems rather than returning plausible-looking arrays.


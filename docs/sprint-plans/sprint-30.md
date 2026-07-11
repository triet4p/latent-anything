# Sprint 30 Plan

## Sprint Goal

Add discrete-code geometry as the fourth materially different geometry and extract geometry-specific algorithms from `LatentSpace` only where running code proves the seam.

## Atomic Tasks

Status legend: [ ] pending / [~] in progress / [x] done

- [x] Implement discrete-code validation, distance, interpolation policy, normalization semantics, and metadata on the latent value path.
- [x] Add tests for categorical/codebook constraints, invalid codes, Hamming-like distance, and the absence of misleading continuous interpolation.
- [x] Run existing Euclidean, unit-norm, Gaussian-set, and new discrete geometry tests through one shared conformance matrix.
- [x] Extract geometry-specific helpers or strategies from `latent_space.py` if the fourth branch confirms a stable boundary.
- [x] Keep `LatentSpace` as a small public facade that delegates behavior and preserves beta construction compatibility.
- [x] Move Gaussian-set helper logic into its focused geometry module if extraction is validated.
- [x] Benchmark dispatch overhead and verify no material regression on existing paths.
- [x] Reconcile the geometry ADR, update evidence/changelog/artifact, and run the full gate.

## Notes / Blockers

The sprint must not invent a large abstract geometry hierarchy. Extraction is justified only by the fourth distinct case and must leave a smaller, clearer facade.

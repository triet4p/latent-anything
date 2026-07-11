# Sprint 30 Plan

## Sprint Goal

Add discrete-code geometry as the fourth materially different geometry and extract geometry-specific algorithms from `LatentSpace` only where running code proves the seam.

## Atomic Tasks

Status legend: [ ] pending / [~] in progress / [x] done

- [ ] Implement discrete-code validation, distance, interpolation policy, normalization semantics, and metadata on the latent value path.
- [ ] Add tests for categorical/codebook constraints, invalid codes, Hamming-like distance, and the absence of misleading continuous interpolation.
- [ ] Run existing Euclidean, unit-norm, Gaussian-set, and new discrete geometry tests through one shared conformance matrix.
- [ ] Extract geometry-specific helpers or strategies from `latent_space.py` if the fourth branch confirms a stable boundary.
- [ ] Keep `LatentSpace` as a small public facade that delegates behavior and preserves beta construction compatibility.
- [ ] Move Gaussian-set helper logic into its focused geometry module if extraction is validated.
- [ ] Benchmark dispatch overhead and verify no material regression on existing paths.
- [ ] Reconcile the geometry ADR, update evidence/changelog/artifact, and run the full gate.

## Notes / Blockers

The sprint must not invent a large abstract geometry hierarchy. Extraction is justified only by the fourth distinct case and must leave a smaller, clearer facade.


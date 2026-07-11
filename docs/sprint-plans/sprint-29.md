# Sprint 29 Plan

## Sprint Goal

Add one first-class latent value container that can carry flat batches and structured states while keeping `LatentSpace` as the space/schema handle.

## Atomic Tasks

Status legend: [ ] pending / [~] in progress / [x] done

- [x] Derive container requirements from VAE batches, hidden-state sequences, Gaussian sets, and the current 2D-only `Trajectory` limitation.
- [x] Implement the narrowest immutable container with explicit space association, array ownership, shape validation, metadata, and safe NumPy conversion.
- [x] Migrate one flat adapter path and one structured Gaussian path end-to-end without changing their numerical behavior.
- [x] Define how a one-state value, a batch, and a temporal trajectory relate without forcing structured data into a flat 2D matrix.
- [x] Add property-based tests for immutability, shape preservation, copying, slicing, and geometry validation.
- [x] Add compatibility adapters for existing `Trajectory` and raw-array callers; do not remove beta APIs yet.
- [x] Apply the Rule of Three and record whether a broader latent-data protocol is justified.
- [x] Update evidence links, docs, changelog, and the sprint artifact; run the strict gate.

## Notes / Blockers

This sprint resolves a real contract mismatch: `LatentSpace` supports structured geometry while `Trajectory` currently stores only `(steps, dim)` arrays. The implementation name should follow Sprint 28's naming RFC.

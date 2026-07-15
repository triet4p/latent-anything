# Sprint 42 Plan

## Sprint Goal

Add concept activation vectors and TCAV-style directional sensitivity with target-specific gradients, statistical controls, and intervention cross-checks.

## Entry Criteria

- Sprint 39 provides a gradient-preserving transformer intervention seam.
- Real VAE/generative and transformer representation datasets have documented concept/reference provenance.

## Atomic Tasks

Status legend: [ ] pending / [~] in progress / [x] done

- [x] Define deterministic concept/reference dataset handling with sampling, split, source, representation-space, and model-version provenance.
- [x] Learn concept directions using both mean difference and a regularized linear separator baseline, reporting direction stability and held-out separability.
- [x] Define one scalar model target per integration and compute directional gradients at a declared activation location.
- [x] Implement a typed TCAV result with target, layer, concept direction, per-example sensitivities, aggregate score, uncertainty, and provenance.
- [x] Add repeated random-concept baselines, multiple seeds, significance testing, and correction for the declared family of comparisons.
- [x] Validate on one real VAE/generative factor and one Sprint 39 transformer representation.
- [x] Cross-check observational sensitivity with a bounded matched-norm intervention along the learned direction and report agreements and contradictions.
- [x] Support direct/config registry construction, add failure/control tests, and update evidence/ADR/changelog/artifact/gates without extracting a generic probe/concept protocol prematurely.

## Notes / Blockers

TCAV is target- and layer-specific. Concept-direction visualization or separability alone is insufficient, and statistical significance does not by itself establish causal meaning.


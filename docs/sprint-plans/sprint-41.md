# Sprint 41 Plan

## Sprint Goal

Add concept activation vectors and TCAV-style sensitivity scoring with statistical controls on a real model output.

## Atomic Tasks

Status legend: [ ] pending / [~] in progress / [x] done

- [ ] Implement concept/reference dataset handling with deterministic sampling and provenance.
- [ ] Learn concept directions using mean difference and a linear separator baseline.
- [ ] Compute directional sensitivity of a selected model output through the activation-capture seam.
- [ ] Add repeated random-concept baselines, significance testing, and multiple-seed stability reporting.
- [ ] Validate on one VAE/generative factor and one transformer or policy representation.
- [ ] Extract a probe/concept result contract only if the third differing analysis instance proves shared invariants.
- [ ] Add registry/config support and a causal intervention cross-check along the learned direction.
- [ ] Update evidence/ADR/changelog/artifact and gates.

## Notes / Blockers

TCAV scores must be tied to a declared target output and random-concept controls. Direction visualization alone is insufficient.


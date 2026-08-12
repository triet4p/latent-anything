# Sprint 71 Plan

## Sprint Goal

Add a decoder-free JEPA/LeWM-style world-model adapter and validate latent prediction without fabricating a decoder.

## Atomic Tasks

Status legend: [ ] pending / [~] in progress / [x] done

- [x] Select a publicly available JEPA/LeWM-style checkpoint or reproducible compact implementation and pin provenance.
- [x] Implement context/target encoding, prediction, action conditioning where present, and latent-space metadata.
- [x] Preserve the no-decoder exposure mode in types, pipeline selection, and documentation.
- [x] Measure latent prediction error, variance/covariance health, collapse indicators, and horizon drift.
- [x] Add stop-gradient/target-encoder state tests where relevant and compare against trivial collapsed baselines.
- [x] Integrate predicted trajectories with analysis, rollout, and experiment records.
- [x] Add lightweight structural tests plus a marked real-checkpoint benchmark.
- [x] Update evidence/ADR/changelog/artifact and gates.

## Notes / Blockers

The compact reference lane is pinned as `compact-jepa-lewm-v1` over
`synthetic-controlled-latent-dynamics-v1`; the optional public checkpoint
smoke is pinned to `facebook/ijepa_vith14_1k` at revision
`be440b1cac639542ae553e71a9c7afd925ab5fac`. The CPU artifact is D2 synthetic
evidence only: it shows strong one-step improvement over a collapsed baseline,
but records anisotropic health and open-loop drift rather than promoting a
real LeWM or CUDA claim.

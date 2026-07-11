# Sprint 71 Plan

## Sprint Goal

Add a decoder-free JEPA/LeWM-style world-model adapter and validate latent prediction without fabricating a decoder.

## Atomic Tasks

Status legend: [ ] pending / [~] in progress / [x] done

- [ ] Select a publicly available JEPA/LeWM-style checkpoint or reproducible compact implementation and pin provenance.
- [ ] Implement context/target encoding, prediction, action conditioning where present, and latent-space metadata.
- [ ] Preserve the no-decoder exposure mode in types, pipeline selection, and documentation.
- [ ] Measure latent prediction error, variance/covariance health, collapse indicators, and horizon drift.
- [ ] Add stop-gradient/target-encoder state tests where relevant and compare against trivial collapsed baselines.
- [ ] Integrate predicted trajectories with analysis, rollout, and experiment records.
- [ ] Add lightweight structural tests plus a marked real-checkpoint benchmark.
- [ ] Update evidence/ADR/changelog/artifact and gates.

## Notes / Blockers

The exact anchor model must be refreshed at sprint activation. The adapter taxonomy must describe observed model behavior, not retain outdated assumptions about LeWM internals.


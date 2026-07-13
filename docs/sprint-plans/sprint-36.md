# Sprint 36 Plan

## Sprint Goal

Establish an explanation-validity benchmark for VAE latents so framework output is judged by fidelity and causal meaning rather than visual appeal.

## Atomic Tasks

Status legend: [ ] pending / [~] in progress / [x] done

- [x] Define a typed explanation-evaluation result with reconstruction, factor predictability, stability, locality, and intervention-effect fields.
- [x] Add held-out factor predictability on the selected real dataset.
- [x] Measure probe performance against shuffled-label and input-feature baselines.
- [x] Intervene along discovered directions and quantify target-factor change, off-target change, and decode degradation.
- [x] Repeat across seeds/checkpoints to report confidence intervals and unstable conclusions.
- [x] Compare PCA, SAE, and steering outputs under the same evaluation contract.
- [x] Add acceptance thresholds, negative controls, and regression tests using a compact fixture.
- [x] Publish the benchmark artifact and update the D2/D3 evidence ledger, ADR/changelog, and gates.
- [x] Resolve post-sprint review findings for evidence schema, metadata immutability, strict typing, capture lifecycle, and benchmark terminology.

## Notes / Blockers

This benchmark becomes the minimum bar for later headline explanation methods. A method that produces a readable plot but fails controls remains D1.

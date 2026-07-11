# Sprint 36 Plan

## Sprint Goal

Establish an explanation-validity benchmark for VAE latents so framework output is judged by fidelity and causal meaning rather than visual appeal.

## Atomic Tasks

Status legend: [ ] pending / [~] in progress / [x] done

- [ ] Define a typed explanation-evaluation result with reconstruction, factor predictability, stability, locality, and intervention-effect fields.
- [ ] Add DCI or equivalent factor/disentanglement metrics on the selected real dataset.
- [ ] Measure probe performance against shuffled-label and input-feature baselines.
- [ ] Intervene along discovered directions and quantify target-factor change, off-target change, and decode degradation.
- [ ] Repeat across seeds/checkpoints to report confidence intervals and unstable conclusions.
- [ ] Compare PCA, SAE, and steering outputs under the same evaluation contract.
- [ ] Add acceptance thresholds, negative controls, and regression tests using a compact fixture.
- [ ] Publish the benchmark artifact and update the D2/D3 evidence ledger, ADR/changelog, and gates.

## Notes / Blockers

This benchmark becomes the minimum bar for later headline explanation methods. A method that produces a readable plot but fails controls remains D1.


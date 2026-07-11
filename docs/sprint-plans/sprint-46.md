# Sprint 46 Plan

## Sprint Goal

Evaluate sparse-autoencoder features for reconstruction, sparsity, stability, and semantic usefulness, then publish a queryable feature atlas artifact.

## Atomic Tasks

Status legend: [ ] pending / [~] in progress / [x] done

- [ ] Define SAE evaluation results for reconstruction, L0/L1 activity, dead features, activation frequency, and decoder norms.
- [ ] Add train/validation separation and checkpoint serialization for fitted SAE state.
- [ ] Measure feature stability/alignment across seeds and avoid comparing arbitrary feature indices directly.
- [ ] Rank feature examples and counterexamples from one real VAE or transformer activation dataset.
- [ ] Cross-check selected features with probes, concepts, and causal steering/patching.
- [ ] Generate a portable feature-atlas data artifact independent of the visualization frontend.
- [ ] Add regression thresholds and marked full-model evaluation tests.
- [ ] Update evidence/ADR/changelog/artifact and gates.

## Notes / Blockers

Monosemantic labels must not be assigned solely from top-activating examples. Causal and stability evidence should accompany strong claims.


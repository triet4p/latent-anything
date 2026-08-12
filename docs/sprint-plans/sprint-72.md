# Sprint 72 Plan

## Sprint Goal

Validate tokenized world-model next-token prediction and rollout from image trajectories encoded by the fitted VQ-VAE, with codebook, temporal, and task-level metrics.

## Atomic Tasks

Status legend: [ ] pending / [~] in progress / [x] done

- [x] Compose encoded image observations, actions, and a sequence dynamics model through existing adapter/transition seams.
- [x] Implement autoregressive next-token prediction and seeded rollout for one concrete dataset/task.
- [x] Measure token likelihood/perplexity, code usage, multi-step drift, decoded consistency where a decoder exists, and task proxy metrics.
- [x] Compare teacher-forced and free-running behavior to expose compounding error.
- [x] Add invalid-token, mask, padding, horizon, and codebook-version tests.
- [x] Analyze whether tokenized dynamics fits the frozen transition contract; revise via ADR if it does not.
- [x] Produce a reproducible rollout artifact with failure horizons.
- [x] Update theory coverage, ADR/changelog/artifact and gates.

## Notes / Blockers

This sprint is an end-to-end CPU capability benchmark, not a promise to reproduce GAIA or Genie scale; arbitrary synthetic token IDs are reserved for unit tests. The current artifact remains D1 because its measured non-trivial-token-usage gate fails: the fitted tokenizer maps every frame to one code, so perfect token metrics are degenerate and do not validate meaningful dynamics.

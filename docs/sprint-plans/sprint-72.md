# Sprint 72 Plan

## Sprint Goal

Validate tokenized world-model next-token prediction and rollout with codebook, temporal, and task-level metrics.

## Atomic Tasks

Status legend: [ ] pending / [~] in progress / [x] done

- [ ] Compose discrete observation tokens, actions/latent actions, and a sequence dynamics model through existing adapter/transition seams.
- [ ] Implement autoregressive next-token prediction and seeded rollout for one concrete dataset/task.
- [ ] Measure token likelihood/perplexity, code usage, multi-step drift, decoded consistency where a decoder exists, and task proxy metrics.
- [ ] Compare teacher-forced and free-running behavior to expose compounding error.
- [ ] Add invalid-token, mask, padding, horizon, and codebook-version tests.
- [ ] Analyze whether tokenized dynamics fits the frozen transition contract; revise via ADR if it does not.
- [ ] Produce a reproducible rollout artifact with failure horizons.
- [ ] Update theory coverage, ADR/changelog/artifact and gates.

## Notes / Blockers

This sprint is a capability benchmark, not a promise to reproduce GAIA or Genie scale.


# Sprint 66 Plan

## Sprint Goal

Add rollout execution as Pipeline #3 and decompose `pipeline.py` into focused modules using the now-proven three-story seams.

## Atomic Tasks

Status legend: [ ] pending / [~] in progress / [x] done

- [ ] Implement a concrete rollout pipeline composing an initial latent value, actions, a transition, optional cache, and profiling.
- [ ] Support sync and async execution with identical results and explicit cancellation/error behavior.
- [ ] Compare Analysis, Manipulation, and Rollout pipeline responsibilities and identify actual shared invariants.
- [ ] Move pipeline classes, result models, config specs/builders, and execution helpers into focused modules without compatibility breaks.
- [ ] Extract/freeze a shared pipeline contract only if all three stories use it meaningfully.
- [ ] Add import, signature, config, behavior-parity, cache, profiling, and async regression tests.
- [ ] Update public exports/migration docs and measure module complexity reduction.
- [ ] Log the architecture ADR and update evidence/changelog/artifact/gates.

## Notes / Blockers

This is the evidence point requested by the Sprint 26 SRP audit. The goal is focused ownership, not a generic DAG engine.


# Sprint 38 Plan

## Sprint Goal

Demonstrate meaningful diffusion latent intervention with quantitative target, preservation, and generation-quality evidence.

## Atomic Tasks

Status legend: [ ] pending / [~] in progress / [x] done

- [ ] Choose one bounded edit task with measurable target semantics and a non-target preservation objective.
- [ ] Implement a timestep-aware intervention on the concrete Sprint 37 capture path.
- [ ] Compare against prompt-only, random-direction, and no-edit controls using fixed seeds.
- [ ] Measure target change, content preservation, distribution/density drift, and decode/generation quality proxies.
- [ ] Sweep intervention layer, timestep, and strength to identify causal windows and failure modes.
- [ ] Add deterministic smoke tests plus a marked reproducible benchmark with stored configuration.
- [ ] Produce side-by-side outputs, metric tables, and explicit counterexamples where the edit fails.
- [ ] Promote relevant theory items to D2/D3 and update ADR/changelog/artifact/gates.

## Notes / Blockers

Success requires measurable selectivity over controls. A visually pleasing cherry-picked edit does not complete the sprint.


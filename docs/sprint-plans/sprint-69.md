# Sprint 69 Plan

## Sprint Goal

Add MPPI planning for continuous control and compare its smoothness, return, and compute trade-offs against CEM.

## Atomic Tasks

Status legend: [ ] pending / [~] in progress / [x] done

- [ ] Implement MPPI noise sampling, temperature weighting, action constraints, receding horizon, and seeded behavior.
- [ ] Reuse transition/rollout/reward components without adding planner-specific branches to them.
- [ ] Add analytic tests for weighting, zero-noise, bounds, horizon shift, and numerical stability.
- [ ] Compare MPPI, CEM, and random shooting on the same continuous-control tasks.
- [ ] Measure return, action smoothness, sample count, latency, and robustness to transition error.
- [ ] Add config/registry/experiment records and profiler integration.
- [ ] Apply Rule of Three to planner abstractions only if a third materially different planner exists or is added.
- [ ] Update evidence/ADR/changelog/artifact and gates.

## Notes / Blockers

Do not invent a broad planner protocol from only CEM and MPPI unless their shared consumer surface is already unavoidable.


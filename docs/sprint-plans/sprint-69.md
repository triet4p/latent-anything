# Sprint 69 Plan

## Sprint Goal

Add MPPI planning for continuous control and compare its smoothness, return, and compute trade-offs against CEM.

## Atomic Tasks

Status legend: [ ] pending / [~] in progress / [x] done

- [x] Implement MPPI noise sampling, temperature weighting, action constraints, receding horizon, and seeded behavior.
- [x] Reuse transition/rollout/reward components without adding planner-specific branches to them.
- [x] Add analytic tests for weighting, zero-noise, bounds, horizon shift, and numerical stability.
- [x] Compare MPPI, CEM, and random shooting on the same continuous-control tasks.
- [x] Measure return, action smoothness, sample count, latency, and robustness to transition error.
- [x] Add config/registry/experiment records and profiler integration.
- [x] Apply Rule of Three to planner abstractions only if a third materially different planner exists or is added.
- [x] Update evidence/ADR/changelog/artifact and gates.

## Notes / Blockers

Do not invent a broad planner protocol from only CEM and MPPI unless their shared consumer surface is already unavoidable.

Sprint 69 completed with a concrete MPPI planner and a shared consumer path
through the existing rollout and reward/value components. No planner protocol
was frozen because only two materially different planner implementations exist.
Evidence is synthetic CPU D2; the benchmark reports model bias explicitly and
does not require the remote CUDA lane.

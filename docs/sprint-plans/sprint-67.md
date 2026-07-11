# Sprint 67 Plan

## Sprint Goal

Add reward and value evaluation over real or imagined latent trajectories with calibration and Bellman-consistency diagnostics.

## Atomic Tasks

Status legend: [ ] pending / [~] in progress / [x] done

- [ ] Implement one reward scorer from latent state/action to scalar task signal.
- [ ] Implement one value estimator tied to a declared discount, horizon, and policy/data distribution.
- [ ] Return per-step rewards, returns, values, masks, and uncertainty/provenance in a typed result.
- [ ] Test return calculation, terminal handling, padding, discounting, and simple analytic MDPs.
- [ ] Measure reward prediction, value calibration, and Bellman residual on held-out trajectories.
- [ ] Compare real vs imagined trajectory scoring to quantify model bias.
- [ ] Integrate with rollout pipeline config and experiment records.
- [ ] Update evidence/ADR/changelog/artifact and gates.

## Notes / Blockers

If reward and value ownership proves too broad for one increment, activate reward scoring first and split value estimation into a new numbered sprint before implementation.


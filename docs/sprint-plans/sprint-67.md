# Sprint 67 Plan

## Sprint Goal

Add reward and value evaluation over real or imagined latent trajectories with calibration and Bellman-consistency diagnostics.

## Atomic Tasks

Status legend: [ ] pending / [~] in progress / [x] done

- [x] Implement one reward scorer from latent state/action to scalar task signal.
- [x] Implement one value estimator tied to a declared discount, horizon, and policy/data distribution.
- [x] Return per-step rewards, returns, values, masks, and uncertainty/provenance in a typed result.
- [x] Test return calculation, terminal handling, padding, discounting, and simple analytic MDPs.
- [x] Measure reward prediction, value calibration, and Bellman residual on held-out trajectories.
- [x] Compare real vs imagined trajectory scoring to quantify model bias.
- [x] Integrate with rollout pipeline config and experiment records.
- [x] Update evidence/ADR/changelog/artifact and gates.

## Notes / Blockers

The increment keeps both surfaces narrow: a linear reward head, finite-horizon Monte-Carlo value estimator, and diagnostics evaluator. Distributional heads, continuation prediction, bootstrapped TD/lambda returns, and real-model evidence remain future work.

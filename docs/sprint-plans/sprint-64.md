# Sprint 64 Plan

## Sprint Goal

Add a stochastic Gaussian latent transition instance with calibrated uncertainty over rollout horizons.

## Atomic Tasks

Status legend: [ ] pending / [~] in progress / [x] done

- [x] Implement a transition that predicts mean and valid covariance/scale for `p(z_{t+1}|z_t,a_t)`.
- [x] Support seeded sampling and distribution-valued one-step predictions without hiding uncertainty in metadata.
- [x] Evaluate negative log-likelihood, calibration/coverage, sample diversity, and horizon drift.
- [x] Compare deterministic mean rollout and sampled rollout on controlled stochastic dynamics.
- [x] Add tests for positive variance, reproducibility, degenerate noise, batch shapes, and numerical stability.
- [x] Sketch an internal unstable shared transition shape only where #1 and #2 genuinely agree.
- [x] Produce uncertainty-band rollout artifacts.
- [x] Update evidence/ADR/changelog/artifact and gates.

## Notes / Blockers

This is Transition #2. The only shared shape is the internal, unstable prediction vocabulary of
mean plus optional uncertainty summaries; no public transition interface is frozen until Sprint 65.

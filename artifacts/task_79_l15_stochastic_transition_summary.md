# Sprint 79 M14 L15 stochastic-transition reconciliation

## Scope

`THY-T06-STOCHASTIC-TRANSITION` (M14 L15, queue position 10) is promoted from D0 to D2. This is the compact project-authored stochastic Gaussian latent-transition benchmark only. It does not claim a real-world dataset, pretrained temporal model, LeWM, or CUDA execution.

## Evidence

The existing Sprint 64 seeded held-out benchmark was reused without rerunning the experiment, because its source, focused tests, benchmark/configuration, complete D2 artifact, and deterministic seed provenance already satisfy the row contract. The reconciled row artifact is [`m14/l15-stochastic-transition.json`](m14/l15-stochastic-transition.json), with run receipt [`m14/l15-stochastic-transition.run.json`](m14/l15-stochastic-transition.run.json) and explicit thresholds in [`m14/l15-stochastic-transition.config.json`](m14/l15-stochastic-transition.config.json).

Observed held-out metrics from [`stochastic_transition_rollout.json`](stochastic_transition_rollout.json): one-step NLL `-1.6633257720710657`, coverage `0.9513888888888888`, interval width `0.4538665938179671`, sample diversity `0.11438866578096975`, and mean error `0.14813685794583398`; 24-step rollout mean coverage `0.9739583333333334`, final error `0.36706518632313667`, and `stable: true`. All metrics are finite and the seeded benchmark reports reproducible sampling.

Focused validation executed after reconciliation:

```text
uv run pytest tests/test_latent_anything/test_transition.py -q
26 passed
```

The existing M14 L15 deterministic/RSSM receipt remains linked for context, but no RSSM or named Dreamer claim is promoted by this row.

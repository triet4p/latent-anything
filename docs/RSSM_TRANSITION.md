# RSSM-style Latent Transition

`RSSMLatentTransition` is the third concrete transition instance. It keeps a
deterministic recurrent state `h_t` and predicts a diagonal-Gaussian next
latent from the recurrent state, current latent, and action:

```text
h_{t+1} = tanh([h_t, z_t, a_t] W_h + b_h)
p(z_{t+1}) = Normal([h_{t+1}, z_t, a_t, 1] W_z + b_z, diag(s^2))
```

The public boundary remains NumPy arrays. Torch is used for the bounded fit
internally, and `RSSMTransitionConfig` records hidden width, optimizer
settings, seed, variance floor, and device. `reset()` starts a new sequence;
`step()` and `predict()` then advance the recurrent state. `mean_rollout()` and
`rollout()` reset at the beginning of a sequence and carry the recurrent state
through the action horizon.

## Variable-length sequences

Training and evaluation accept `sequence_mask` with shape
`(episodes, horizon)`. A zero entry is padding: it neither updates `h_t` nor
contributes to metrics. This keeps variable-length episodes in one bounded
batch while making reset boundaries explicit.

## Checkpoints

`save()` writes a portable `.npz` checkpoint containing recurrent/emission
parameters, scale, configuration, and fit provenance. `load()` restores the
learned model and always resets in-flight recurrent state. A caller may pass a
new `device` to `load()`; CUDA is accepted only when available.

The KL metric is deliberately labelled a proxy: this compact instance uses an
observation-centred posterior scale rather than a learned posterior encoder or
Dreamer free-bits objective. It is useful for comparing temporal prediction
runs, not evidence of a full RSSM implementation.

## Proven transition contract

The first three instances now satisfy the small runtime-checkable
`LatentTransition` contract: state/action dimensions, source identity, a
predictive-mean `step()`, and `mean_rollout()`. Fitting signatures,
distribution-valued predictions, particle outputs, and recurrent reset are
not forced into that contract because they differ materially across the
deterministic, memoryless Gaussian, and RSSM implementations.

The reproducible comparison is available as:

- [benchmark script](../scripts/rssm_transition_benchmark.py)
- [configuration](../artifacts/rssm_transition_comparison_config.json)
- [measured results and failure analysis](../artifacts/rssm_transition_comparison.json)
- [comparison plot](../artifacts/rssm_transition_comparison.png)

The benchmark is synthetic D2 evidence. On the committed partially observed
system, RSSM improves teacher-forced one-step MSE over the baselines but still
shows substantial open-loop drift and under-coverage. The failure is retained
as evidence: recurrent state alone does not establish a reliable long-horizon
world model.

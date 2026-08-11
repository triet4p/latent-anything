# Stochastic Gaussian Latent Transition

`StochasticGaussianLatentTransition` is the second concrete transition
instance. It is intentionally limited to flat Euclidean latent vectors and
action-conditioned affine residual dynamics:

\[
z_{t+1} = z_t + [z_t, a_t, 1]W + \epsilon_t,
\qquad
\epsilon_t \sim \mathcal{N}(0, \operatorname{diag}(s^2)).
\]

The fitted `GaussianPrediction` exposes `mean`, `scale`, `variance`, and
`covariance` directly. It also provides seeded `sample`, `log_prob`, and
coordinate-wise `interval` operations. A zero variance floor is allowed for
degenerate-noise controls; the sampler then returns the exact mean.

## Rollouts and evaluation

`mean_rollout()` returns the uncertainty-free `Trajectory` baseline. `rollout()`
returns a `StochasticRollout` particle tensor plus mean, scale, lower, and
upper summaries. The stochastic evaluator reports negative log-likelihood,
interval coverage, particle diversity, and mean-path error at every horizon.

The reproducible controlled benchmark is:

- [benchmark script](../scripts/stochastic_transition_benchmark.py)
- [configuration](../artifacts/stochastic_transition_rollout_config.json)
- [measured results](../artifacts/stochastic_transition_rollout.json)
- [uncertainty-band plot](../artifacts/stochastic_transition_uncertainty_band.png)

The benchmark is D2 evidence on a known stochastic linear system. It does not
claim calibration on a real model, epistemic uncertainty, multimodal support,
or manifold-valid sampling. RSSM state and a shared transition contract remain
Sprint 65 scope.

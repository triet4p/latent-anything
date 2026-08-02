# Density-Penalized Geodesic Interpolation

> Guidance for when to use `DensityGeodesic` vs. simpler lerp/slerp/geodesic
> alternatives. The implementation lives in `src/latent_anything/geodesic.py`
> (stateful entry point) and `src/latent_anything/geometry.py` (pure path
> algorithms); the theory background is [Geodesic](../../../latent-anything-theory/01-space-representation/research/05-geodesic.md)
> and [Pullback Metric](../../../latent-anything-theory/01-space-representation/research/06-pullback-metric.md).

## What it does

`DensityGeodesic` treats the latent space as a Riemannian manifold whose metric
is the inverse of a learned data density (a Gaussian mixture fitted on the
representation). The optimizer finds the path between two latent points that
minimizes the density-penalized energy

```
E = sum_i  exp(-alpha * (log p(z_i) - log_ref)) * ||z_{i+1} - z_i||^2
```

so the geodesic bends toward high-density (on-manifold) regions instead of
cutting straight across low-density gaps. It is the tractable realization of
the decoder pullback-metric idea: instead of computing `J^T J` Jacobians it
uses a fitted density oracle (and its analytic gradient), which needs no model
autograd access.

## When the density geodesic is justified

- **The data really is curved.** The straight line between the two endpoints
  departs the data manifold and passes through low-density regions, so the
  decoded interpolations are implausible (blurry, hybrid, off-class). This is
  the classic failure of lerp on a curved VAE latent space.
- **You have a fitted density oracle.** A representation-bound
  `GaussianMixtureDensity` (or any `log_density` + `log_density_gradient`
  pair) is available, and the endpoints are in-distribution.
- **You need the full path plus diagnostics.** The method returns every path
  point, the density-penalized and Euclidean lengths, per-point log-density,
  optional decoded images with a reconstruction diagnostic, and a convergence
  report.

## When simpler interpolation is preferable

- **Flat or locally flat spaces.** If the density is roughly constant between
  the endpoints, the geodesic coincides with lerp and the optimization buys
  nothing — plain `lerp` is cheaper and exact.
- **Short distances.** For close endpoints the chord stays on-manifold, so
  lerp is fine; path optimization adds compute without changing the answer.
- **Known closed-form geometries.** For `unit_norm` latents the spherical
  geodesic has a closed form (slerp) and is exact and cheap; for a constant
  anisotropic covariance the metric geodesic equals the affine lerp in the
  whitened frame (`covariance_interpolate`). Use those instead of numeric path
  optimization.
- **No fitted density.** Without an oracle the method cannot run; do not fall
  back to guessing a density.
- **Budget-constrained pipelines.** Path optimization costs
  `max_iter * n_points` density evaluations; cache identical endpoint/config
  calls with `InMemoryCache` when the same pairs recur.

## Practical guidance

- `density_exponent = 0` recovers the lerp path (a useful calibration check).
- Larger `density_exponent` penalizes low-density crossing more strongly.
- Endpoints must be in the same coordinate system as the fitted density;
  cross-space use is a caller error.
- The optimizer is deterministic (lerp initialization), bounded by `max_iter`
  and `n_points`, and reports `PathOptimizationStatus` so callers can tell a
  converged path from a compute-capped one.

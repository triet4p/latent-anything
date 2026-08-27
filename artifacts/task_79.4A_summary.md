# Task 79.4A — M14 L02 geometry design

Status: design-in-review only. No real benchmark was executed, no accepted
artifact was generated, no evidence level or ledger count was changed, and no
commit/push was made.

## Scope and data

The future bounded runner uses the existing
`latent_anything.adapters.conv_vae.ConvVAE` (latent dimension 4, eight epochs,
random state 42) and `ConvVAE.encode_value`/`ConvVAE.latent_space` on the real
`sklearn.datasets.load_digits` dataset. Images are normalized by `/16.0`; a
`np.random.default_rng(42).permutation` gives 1,437 train and 360 held-out
examples. The adapter and density estimator fit only on train images/latents;
all path endpoints and sequences are made from held-out real images.

Induced sequences are labeled “model-induced latent sequences from held-out
real sklearn digits samples” and explicitly are not recorded physical
trajectories. The stable project has no Fréchet implementation, so this lane
does not claim or invent Fréchet evidence.

## Independent records and predeclared acceptance

The plan maps the six owned gap IDs to six independent records in one shared
future artifact. SLERP is one computation but maps to two separately named gap
records; a shared computation cannot promote a failed record. Partial promotion
is record-local: an accepted record never implies acceptance of its neighbors.

| Record | Gap IDs | Acceptance thresholds |
|---|---|---|
| `manifold_hypothesis` | `THY-T01-MANIFOLD-HYPOTHESIS` | 128 balanced held-out pairs; path-ranking AUC ≥ 0.55; shuffled-label AUC within ±0.10 of 0.50; latent AUC no more than 0.05 below the same-split raw-pixel reference; finite metrics and train-only density fit |
| `slerp_spherical` | `THY-T03-SLERP-SPHERICAL-LINEAR-INTERPOLATION` | endpoint/norm error ≤ 1e-10; angular additivity error ≤ 1e-9; finite float64 output and no mutation |
| `lerp_euclidean` | `THY-T04-LERP-LINEAR-INTERPOLATION` | endpoint and coefficient residual ≤ 1e-12; finite float64 output; no input mutation |
| `riemannian_density_geodesic` | `THY-T03-RIEMANNIAN-GEOMETRY-CO-BAN` | bounded `DensityGeodesic` converges within 300 iterations; endpoint error ≤ 1e-10; finite path; mean held-out path log-density delta ≥ −1e-6 versus LERP; density fit train-only |
| `slerp_latent_operation` | `THY-T04-SLERP` | endpoint/norm error ≤ 1e-10; angular additivity error ≤ 1e-9; finite float64 output and no mutation |
| `trajectory_similarity_dtw` | `THY-T06-TRAJECTORY-SIMILARITY-METRICS` | 128 independent held-out pair-path trials; each 24-point reference has its own monotone 32-point query; median self-to-indexwise ratio ≤ 0.95; median self-to-unrelated deranged-pair ratio ≤ 0.90; ranking AUC ≥ 0.60; finite temporal outputs and no mutation |

The ranking thresholds are conservative screening gates, not significance
claims: AUC 0.55 means a measurable direction above chance, while the ±0.10
shuffled-label envelope catches label leakage. The raw-pixel reference is a
strong same-split comparator; a five-point AUC margin is declared practical
equivalence, not superiority. DTW must beat fixed indexwise alignment and a
deterministically shuffled trajectory. The interpolation tolerances are
numerical contracts, not model-quality claims. Geodesic acceptance allows
optimizer noise at 1e-6 while still requiring convergence and no degradation
against the declared LERP density baseline. Peak RSS is explicitly “not
measured; M14 estimate only”; no unmeasured resource claim is made.

## Existing API integration

The executable runner composes `LatentSpace.distance/interpolate/normalize`,
`latent_anything.methods.Lerp`, `lerp_path` and density path helpers,
train-fitted `GaussianMixtureDensity`/`DensityGeodesic`, `Trajectory`,
triangular-window `smooth_trajectory`, and `compute_dtw` with
`normalization='path_length'` and its bounded `max_cells=4096` contract. The
manifold score is mean train-only GMM log-density along each 24-point held-out
LERP path; labels are evaluation-only. Trajectory controls use all 128
preselected 24-point held-out pair paths, each with a deterministic 32-point
monotone-resampled query, and compare against a deranged unrelated pair path
with no self mapping. Existing
`SubspaceProjection` and SO3/SE3 pose focused controls remain supporting API
coverage; they are not relabeled as physical trajectory evidence and no new
general abstraction is introduced.

## Files and verification

- `scripts/m14_l02_plan.py` — lane-local loader, digest, and declarative-plan
  validator.
- `scripts/m14_l02_data.py`, `scripts/m14_l02_metrics.py`, and
  `scripts/m14_l02_envelope.py` — cohesive data/path, metric/verdict, and
  artifact/run provenance, canonical self-digest, validation, and writing
  responsibilities.
- `scripts/m14_l02_geometry.py` — thin executable orchestrator with
  real-data loading, train-only fitting, held-out paths, six independent
  evaluations, payload construction, and explicit output writing. It was not
  invoked during this design task.
- `tests/test_m14_l02_geometry.py` — unit-only schema, exact gap mapping,
  digest tamper, negative-path, and deferred provenance tests.
- `artifacts/m14/l02-geometry.plan.json` — sole declarative source of truth;
  it also declares the future artifact/run-record schemas. The future accepted
  name remains `artifacts/m14/l02-geometry.json`.
- `docs/M14_REAL_SYSTEM_VALIDATION.md` — L02 is `79.4A design-in-review`.

The real benchmark remains blocked until owner approval and a committed runner;
the design contract has no committed-run SHA, runner digest, timestamp, or
evidence promotion. The plan is explicitly design-only and uses
`deferred_until_committed_run` rather than an artifact digest. A future artifact
will carry dataset/license/content and exact split-index digests, model/density
configs and train-only scopes, package/backend versions, input mutation
digests, independent records, accepted gaps, plan/source/git provenance, and a
canonical self-digest. Its run record will carry the UTC timestamp, command,
status, operational wording, and matching accepted IDs.

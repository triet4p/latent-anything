# Sprint 43 Plan

## Sprint Goal

Add K-means latent structure discovery with explicit geometry assumptions, assignment uncertainty, stability, and external-validity diagnostics.

## Entry Criteria

- Sprint 35 and Sprint 39 make real VAE and transformer representation datasets available with compatible flattening/masking policies.

## Atomic Tasks

Status legend: [ ] pending / [~] in progress / [x] done

- [x] Implement K-means fit/predict behavior on declared flat latent batches with explicit preprocessing, distance metric, initialization, and seed behavior.
- [x] Define a typed cluster result with assignments, centers, inertia, silhouette, cluster sizes, preprocessing, provenance, and a concrete nearest-versus-second-nearest distance-margin confidence proxy.
- [x] Add bootstrap/seed stability and cluster-label alignment before comparing runs.
- [x] Compare discovered clusters with known factors where labels exist without using those labels during fitting.
- [x] Add geometry compatibility checks and reject misleading use on unsupported structured, masked, or discrete spaces.
- [x] Validate on real VAE and Sprint 39 transformer representations across predeclared layers or pooling policies.
- [x] Add direct/config registry construction and tests for degenerate, imbalanced, empty-cluster, duplicate, and high-dimensional inputs.
- [x] Produce an artifact containing stable, unstable, and failed settings, then update evidence/ADR/changelog/artifact/gates.

## Notes / Blockers

The result must communicate uncertainty and stability. Colored projections without geometry checks and quantitative diagnostics remain exploratory only.

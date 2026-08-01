# Sprint 44 Plan

## Sprint Goal

Add Gaussian-mixture density estimation and calibrated out-of-distribution scoring bound to a specific representation space and model version.

## Entry Criteria

- Sprint 43 establishes dataset, preprocessing, geometry-check, provenance, and clustering-result conventions that density estimation can reuse selectively.

## Atomic Tasks

Status legend: [ ] pending / [~] in progress / [x] done

- [x] Implement one GMM estimator with declared covariance choice, regularization, deterministic initialization, and explicit dimension/sample constraints.
- [x] Define a typed density result containing log density, component responsibilities, calibrated OOD score, fit/calibration provenance, and source representation identity.
- [x] Separate fit, held-out calibration, and evaluation data; define in-distribution and constructed/real OOD sets before scoring.
- [x] Report AUROC, AUPRC, calibration diagnostics, and uncertainty across seeds.
- [x] Compare against Euclidean-distance and single-Gaussian/Mahalanobis baselines under identical preprocessing and splits.
- [x] Handle singular covariance, too-few samples, dimension/sample imbalance, unsupported geometry, and cross-space scoring attempts explicitly.
- [x] Validate on real VAE and Sprint 39 transformer representations; add direct/config construction and component-local state snapshot tests.
- [x] Produce calibration and failure artifacts, then update evidence/ADR/changelog/artifact/gates.

## Notes / Blockers

Covariance and Mahalanobis behavior are estimator-local in this sprint; they do not expand `LatentSpace` into general anisotropic geometry, which remains Sprint 48 scope. Density values from different representation spaces are not directly comparable.

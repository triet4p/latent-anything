# Sprint 43 Plan

## Sprint Goal

Add Gaussian-mixture density estimation and out-of-distribution scoring for anisotropic latent distributions.

## Atomic Tasks

Status legend: [ ] pending / [~] in progress / [x] done

- [ ] Implement one GMM density estimator with covariance choice, regularization, and deterministic initialization.
- [ ] Return log-density, component responsibility, and calibrated OOD score results with fit provenance.
- [ ] Add held-out in-distribution vs constructed/real OOD evaluation with AUROC/AUPRC and calibration plots.
- [ ] Compare against Euclidean distance and single-Gaussian Mahalanobis baselines.
- [ ] Add failure handling for singular covariance, too-few samples, and dimension/sample imbalance.
- [ ] Integrate registry/config/state serialization and geometry compatibility checks.
- [ ] Validate on VAE and one hidden-state dataset with a documented OOD definition.
- [ ] Update evidence/ADR/changelog/artifact and gates.

## Notes / Blockers

Densities from different representation spaces are not directly comparable. Metadata must bind every fitted estimator to its source space/model version.


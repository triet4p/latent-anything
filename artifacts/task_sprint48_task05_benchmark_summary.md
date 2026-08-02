# Task Summary: Sprint 48 Task 05 — Euclidean vs Mahalanobis Benchmark

**Sprint:** Sprint 48
**Task:** Compare Euclidean vs Mahalanobis neighbors/OOD scores on a controlled anisotropic dataset and real latents.

## Summary of Work

Added `scripts/anisotropy_benchmark.py`, a reproducible benchmark comparing Euclidean and Mahalanobis metrics on (a) a controlled elongated 2D Gaussian with known covariance `diag(25, 0.25)` and (b) real latents from a compact ConvVAE trained on sklearn digits. It reports k-NN neighbor overlap (Jaccard) and OOD AUROC for both metrics and writes `artifacts/anisotropy_benchmark.json`. Acceptance checks pass: Mahalanobis OOD AUROC strictly beats Euclidean on the controlled set (1.000 vs 0.673) and neighbor sets differ under both metrics.

## Files Modified

- [scripts/anisotropy_benchmark.py](scripts/anisotropy_benchmark.py) - New benchmark script (added to pyright include).
- [artifacts/anisotropy_benchmark.json](artifacts/anisotropy_benchmark.json) - Reproducible artifact.

## Testing

- **Execution Command:** `uv run python scripts/anisotropy_benchmark.py`
- **Status:** Passed (acceptance criteria met)

## Additional Notes

The controlled track demonstrates the value of covariance-aware scoring; the real-latents track records both metrics as required for D2 evidence of the Mahalanobis-distance and isotropy/anisotropy theory topics.

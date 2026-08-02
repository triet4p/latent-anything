# Task Summary: Sprint 49 Task 05 — Concept-removal evaluation

**Sprint:** Sprint 49
**Task:** Evaluate concept removal for target suppression, off-target preservation, and decode degradation.

## Summary of Work

Added `scripts/concept_removal_benchmark.py`, a reproducible benchmark on real ConvVAE digits latents. It removes the supervised probe-coefficient subspace for the "diagonal digits {1,4,7}" concept and measures: target suppression (binary probe accuracy collapses 0.856 → 0.289 toward chance), off-target preservation (digit-parity accuracy drops only 0.139, within the 0.2 bound), decode degradation (reconstruction MSE ratio 1.00, bounded), and a same-dimensionality random-subspace control that suppresses the target far less (0.522 vs 0.289), proving suppression is concept-specific. Acceptance criteria pass and the results are written to `artifacts/concept_removal_benchmark.json`.

## Files Modified

- [scripts/concept_removal_benchmark.py](scripts/concept_removal_benchmark.py) - New benchmark (added to pyright include).
- [artifacts/concept_removal_benchmark.json](artifacts/concept_removal_benchmark.json) - Reproducible artifact.

## Testing

- **Execution Command:** `uv run python scripts/concept_removal_benchmark.py`
- **Status:** Passed (all acceptance criteria met)

## Additional Notes

Provides the D2 benchmark role for the `subspace projection` theory topic and demonstrates that concept removal is specific to the learned direction rather than any dimension removal.

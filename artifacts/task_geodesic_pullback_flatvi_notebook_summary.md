# Task Summary: Geodesic, Pullback Metric & FlatVI Notebook

**Sprint:** lumen-theory / 01-space-representation
**Task:** Create interactive notebook for theory notes 05, 06, and 07

## Summary of Work

Created a 26-cell Jupyter notebook that covers `05-geodesic.md`, `06-pullback-metric.md`,
and `07-flatvi.md` as a unified progression: geodesics expose the interpolation problem,
the pullback metric diagnoses the decoder distortion, and FlatVI provides the cure.
All 26 cells executed without error; all three `assert` blocks passed.

## Files Modified

* [lumen-theory/01-space-representation/notebooks/05_06_07_geodesic_pullback_flatvi.ipynb](lumen-theory/01-space-representation/notebooks/05_06_07_geodesic_pullback_flatvi.ipynb) — new notebook

## Notebook Structure

| Section | Content |
|---|---|
| 1.1 | LERP failure on S^1 — norm drops below 1, exits manifold |
| 1.2 | **Exercise 1 — implement `slerp`**: SLERP formula, angular speed verification |
| 1.3 | Exp/Log maps on S^1 — round-trip proof, equivalence to SLERP |
| 1.4 | Swiss Roll: LERP off-manifold distance table + 3D plot with path |
| 2.1 | **Exercise 2 — implement `pullback_metric`** via finite-difference Jacobian |
| 2.2 | Riemannian unit ball ellipses in latent space + decoded grid comparison |
| 3.1 | **Exercise 3 — implement `flattening_loss`** + `optimal_alpha` |
| 3.2 | Flattening loss landscape — contour + radial slice for polar decoder |
| 3.3 | Speed profile comparison: curved vs flat decoder proves FlatVI claim |
| — | Summary table linking all three theory notes |

## Testing

* **Verification:** `uv run jupyter nbconvert --to notebook --execute` — exit 0, 948 KB output
* **Assertions in notebook:**
  * Exercise 1.2: `assert allclose(slerp_norms, 1.0)` — SLERP stays on S^1
  * Exercise 2.1: `assert allclose(g0_numerical, g0_analytical, atol=1e-5)`
  * Exercise 3.1: four boundary-value assertions on `flattening_loss`

## Additional Notes

No new dependencies added — uses scikit-learn and numpy already present from the previous notebook.
The `decoder_polar` toy decoder (exponential polar mapping) is used throughout Parts 2 and 3
as a simple but instructive example with a known analytical Jacobian for verification.

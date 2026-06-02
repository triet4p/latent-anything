# Task Summary: Manifold Hypothesis & Curse of Dimensionality Notebook

**Sprint:** lumen-theory / 01-space-representation
**Task:** Create interactive notebook for theory notes 03 and 04

## Summary of Work

Created a 32-cell Jupyter notebook covering both `03-manifold-hypothesis.md` and
`04-curse-of-dimensionality.md` with runnable proofs, demos, and two exercises.
Added `scikit-learn` (and transitively `scipy`) to `lumen-theory/pyproject.toml`
since the notebook uses `make_swiss_roll`, `PCA`, `NearestNeighbors`, and
`pairwise_distances`. Also added `nbconvert` as a dev dependency.
All 32 cells executed without error; all three `assert` blocks passed.

## Files Modified

* [lumen-theory/01-space-representation/notebooks/03_manifold_hypothesis_and_curse_of_dimensionality.ipynb](lumen-theory/01-space-representation/notebooks/03_manifold_hypothesis_and_curse_of_dimensionality.ipynb) — new notebook
* [lumen-theory/pyproject.toml](lumen-theory/pyproject.toml) — added scikit-learn dependency

## Notebook Structure

| Section | Content |
|---|---|
| 1.1 | Swiss Roll — 2D manifold in 3D (generate + 3D scatter) |
| 1.2 | Intrinsic dimension via local PCA **(Exercise 1 — implement SVD-based estimator)** |
| 1.3 | Tangent space on a circle — global vs local view |
| 1.4 | Local vs global structure — chart/atlas concept |
| 1.5 | Euclidean vs geodesic distance — misleading pair demo + 3D plot |
| 2.1 | Volume concentration — shell fraction table + plot |
| 2.2 | Distance concentration **(Exercise 2 — implement relative contrast)** |
| 2.3 | Hubness — N_k histogram across dimensions + skewness table |
| 2.4 | kNN search cost **(Exercise 3 — implement side-length formula)** |
| 2.5 | PCA as cure — scree plot + 2D projection of Swiss Roll |
| — | Summary table connecting both theory notes |

## Testing

* **Verification:** `uv run jupyter nbconvert --to notebook --execute` — exit 0
* **Assertions in notebook:**
  * Exercise 1.2: `assert top_dim == 2` — local PCA recovers intrinsic dim = 2
  * Exercise 2.2: `assert rc_2d > rc_100d` — contrast collapses in high-d
  * Exercise 2.4: three boundary-value assertions on `knn_search_side`

## Additional Notes

`nbconvert` was added as a dev dependency for notebook execution verification only.
The notebook uses `rng = np.random.default_rng(42)` throughout for reproducibility.
All matplotlib plots use `plt.style.use('seaborn-v0_8-whitegrid')` matching the style
of the existing `02_covariance_and_mahalanobis_exercise.ipynb`.

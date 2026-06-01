# Task Summary: Covariance and Mahalanobis Exercise Notebook

**Sprint:** N/A
**Task:** Create an exercise notebook for covariance and Mahalanobis distance

## Summary of Work
Created a new Jupyter notebook exercise based on `lumen-theory/01-space-representation/research/02-mahalanobis-distance.md`. The notebook introduces a correlated synthetic dataset, guides the learner through implementing mean centering, covariance estimation, and Mahalanobis distance, and includes validation cells plus visual comparisons against Euclidean distance.

## Files Modified
* [lumen-theory/01-space-representation/notebooks/02_covariance_and_mahalanobis_exercise.ipynb](lumen-theory/01-space-representation/notebooks/02_covariance_and_mahalanobis_exercise.ipynb) - New exercise notebook with TODO-based implementation tasks and plotting cells
* [artifacts/task_covariance_mahalanobis_exercise_summary.md](artifacts/task_covariance_mahalanobis_exercise_summary.md) - Task summary artifact for this atomic task

## Testing
* **Test File:** N/A - notebook JSON structure validated directly
* **Status:** Passed
* **Execution Command:** `Get-Content lumen-theory/01-space-representation/notebooks/02_covariance_and_mahalanobis_exercise.ipynb | ConvertFrom-Json | Out-Null`

## Additional Notes
* The notebook intentionally leaves the core implementation cells incomplete with `TODO` placeholders so the learner can implement the important math steps manually.
* Notebook text is in English to match the repository language policy for artifacts and examples.

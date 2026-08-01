# Task Summary: Sprint 47 Task 01

**Sprint:** Sprint 47
**Task:** Define renderer inputs from stable result objects

Added `src/latent_anything/visualization/data.py` with the pure-data renderer
inputs (`PointView`, `TrajectoryView`, `ProjectionView`, `MetricSummary`) and
builders that convert each stable analysis result into a view:
`projection_from_probe`, `projection_from_kmeans`, `projection_from_density`,
`projection_from_trajectory`, `projection_from_atlas`, plus
`metric_summary_from_*` for every family. Frontends never compute metrics.

**Testing:** 28 schema/builder tests in `tests/test_visualization_data.py` pass.

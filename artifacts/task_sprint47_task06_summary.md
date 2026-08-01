# Task Summary: Sprint 47 Task 06

**Sprint:** Sprint 47
**Task:** Responsiveness target and documented downsampling behavior

Declared `DEFAULT_POINT_LIMIT_2D = 50_000` and `DEFAULT_POINT_LIMIT_3D =
20_000` in `visualization/data.py` with `downsample_view`: deterministic,
category-stratified, seeded (`DOWNSAMPLE_SEED = 0`) reduction that never thins
trajectory overlays, records dropped/kept counts in view metadata, and surfaces
them as a chart annotation. `prepare_view` applies the declared limit by
default. The walkthrough measures a 60k-point render (capped at 50k, ~1s).

**Testing:** downsampling determinism, class-balance preservation, capping, and
declared-target tests pass in `test_visualization_data.py` / `test_visualization_figures.py`.

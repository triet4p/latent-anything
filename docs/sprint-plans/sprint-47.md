# Sprint 47 Plan

## Sprint Goal

Add interactive notebook exploration that renders typed analysis results without embedding plotting logic into every analysis method.

## Atomic Tasks

Status legend: [ ] pending / [~] in progress / [x] done

- [x] Define renderer inputs from stable result objects produced by probes, clusters, densities, trajectories, and feature atlases.
- [x] Implement a Plotly-based 2D/3D projection explorer with selection, labels, trajectory overlays, and metadata inspection.
- [x] Add one notebook widget path that degrades cleanly to static HTML/PNG export.
- [x] Keep visualization optional and prove base-package import isolation.
- [x] Add snapshot/schema tests for renderer data and a browser/manual visual QA checklist.
- [x] Validate responsiveness on a declared point-count target and document downsampling behavior.
- [x] Publish an interactive real-model walkthrough with every chart tied to quantitative metrics.
- [x] Update evidence/ADR/changelog/artifact and gates.

## Notes / Blockers

Visualization consumes analysis results; it must not become the place where metrics or model logic are secretly computed.

- The `viz` optional extra adds `plotly`, `kaleido>=1.0.0`, `ipywidgets`, and `anywidget` (plotly 6.x `FigureWidget` requires anywidget). kaleido 0.2.x has no Windows wheel and hangs on PNG export; 1.3.0 works.
- Renderer inputs are pure data (`ProjectionView`/`TrajectoryView`/`MetricSummary`) with builders per analysis family; frontends render only, they never compute metrics.
- Declared responsiveness targets: `DEFAULT_POINT_LIMIT_2D = 50_000`, `DEFAULT_POINT_LIMIT_3D = 20_000`; downsampling is seeded, category-stratified, deterministic, and never thins trajectory overlays.
- First widget path uses ipywidgets + Plotly `FigureWidget`; a custom anywidget is deferred until ipywidgets cannot express the need (see ADR log).


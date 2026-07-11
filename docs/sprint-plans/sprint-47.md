# Sprint 47 Plan

## Sprint Goal

Add interactive notebook exploration that renders typed analysis results without embedding plotting logic into every analysis method.

## Atomic Tasks

Status legend: [ ] pending / [~] in progress / [x] done

- [ ] Define renderer inputs from stable result objects produced by probes, clusters, densities, trajectories, and feature atlases.
- [ ] Implement a Plotly-based 2D/3D projection explorer with selection, labels, trajectory overlays, and metadata inspection.
- [ ] Add one notebook widget path that degrades cleanly to static HTML/PNG export.
- [ ] Keep visualization optional and prove base-package import isolation.
- [ ] Add snapshot/schema tests for renderer data and a browser/manual visual QA checklist.
- [ ] Validate responsiveness on a declared point-count target and document downsampling behavior.
- [ ] Publish an interactive real-model walkthrough with every chart tied to quantitative metrics.
- [ ] Update evidence/ADR/changelog/artifact and gates.

## Notes / Blockers

Visualization consumes analysis results; it must not become the place where metrics or model logic are secretly computed.


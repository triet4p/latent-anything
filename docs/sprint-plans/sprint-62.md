# Sprint 62 Plan

## Sprint Goal

Add reproducible experiment records and LeRobot-facing inspection commands so model, data, capture, intervention, and evaluation evidence can be compared like tracked ML runs.

## Atomic Tasks

Status legend: [ ] pending / [~] in progress / [x] done

- [x] Define a small versioned run record for config, code/framework version, model/dataset revisions, seeds, environment, metrics, artifacts, and parent/child runs.
- [x] Implement a local filesystem recorder with atomic writes and content-addressed artifact references.
- [x] Record LeRobot dataset inspection, policy capture, intervention, and evaluation runs without modifying LeRobot internals.
- [x] Add CLI commands for listing supported capture points, inspecting a dataset/policy, and replaying a saved run config.
- [x] Add schema validation, interrupted-run recovery, duplicate-run identity, and migration tests.
- [x] Integrate runtime profiling and theory evidence identifiers into run metadata.
- [x] Produce a comparison report across at least two policies or intervention settings.
- [x] Apply Rule of Three before freezing a recorder protocol; update evidence/ADR/changelog/artifact/gates.

## Notes / Blockers

This is "MLflow-like" run evidence, not a replacement for MLflow. External tracking backends are added in Sprint 76 after the local contract is proven.

# Sprint 57 Plan

## Sprint Goal

Bridge LeRobotDataset v3 episodes and streaming samples into typed, provenance-rich inputs for latent analysis.

## Atomic Tasks

Status legend: [ ] pending / [~] in progress / [x] done

- [ ] Map LeRobot feature metadata, normalization stats, tasks, episode/frame indices, timestamps, cameras, states, and actions into bridge-owned descriptors.
- [ ] Implement one episode reader path from `LeRobotDataset` without copying entire datasets.
- [ ] Implement one streaming-sample path from `StreamingLeRobotDataset` with bounded buffering.
- [ ] Preserve processor-ready PyTorch structures internally while converting only captured latent results at the framework boundary.
- [ ] Add alignment tests for video/state/action timestamps, episode boundaries, task labels, and normalization metadata.
- [ ] Validate on one small public LeRobot dataset revision and provide an offline synthetic fixture.
- [ ] Produce a dataset-inspection artifact showing schema, episode slices, and provenance rather than model claims.
- [ ] Update evidence/ADR/changelog/artifact and gates.

## Notes / Blockers

The bridge should consume LeRobot's canonical dataset API and metadata. It must not duplicate Parquet/video decoding logic.


# Sprint 60 Plan

## Sprint Goal

Capture and intervene on SmolVLA representations while preserving LeRobot's official multimodal processors and action-generation behavior.

## Atomic Tasks

Status legend: [ ] pending / [~] in progress / [x] done

- [x] Select a public SmolVLA checkpoint/dataset pair with feasible hardware requirements and pin revisions.
- [x] Load the model and pre/post-processors through supported LeRobot APIs.
- [x] Define capture points for vision/language/state context and action-expert representations with token/modality metadata.
- [x] Verify baseline action outputs and seeds against direct LeRobot inference.
- [x] Add one bounded intervention with safe hook lifecycle, strength control, and no-change identity behavior.
- [x] Measure immediate action change, off-target dimensions, representation drift, and sensitivity to prompts/camera order.
- [x] Add lightweight structural tests and a marked GPU checkpoint benchmark.
- [x] Update evidence/ADR/changelog/artifact and gates.

## Notes / Blockers

Model size may prevent full CI execution. The sprint must still provide a reproducible hardware profile, cached revision, and smaller structural fixture.

Resolved: the pinned pair is `lerobot/smolvla_libero@31d453f7edd78c839a8bbc39744a292686daf0de` + `lerobot/libero@a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4` (`libero` / `libero_spatial`); the full CPU policy load, capture, parity, and intervention were verified locally; the marked CUDA lane runs through the remote CUDA server.


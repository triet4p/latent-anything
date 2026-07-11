# Sprint 60 Plan

## Sprint Goal

Capture and intervene on SmolVLA representations while preserving LeRobot's official multimodal processors and action-generation behavior.

## Atomic Tasks

Status legend: [ ] pending / [~] in progress / [x] done

- [ ] Select a public SmolVLA checkpoint/dataset pair with feasible hardware requirements and pin revisions.
- [ ] Load the model and pre/post-processors through supported LeRobot APIs.
- [ ] Define capture points for vision/language/state context and action-expert representations with token/modality metadata.
- [ ] Verify baseline action outputs and seeds against direct LeRobot inference.
- [ ] Add one bounded intervention with safe hook lifecycle, strength control, and no-change identity behavior.
- [ ] Measure immediate action change, off-target dimensions, representation drift, and sensitivity to prompts/camera order.
- [ ] Add lightweight structural tests and a marked GPU checkpoint benchmark.
- [ ] Update evidence/ADR/changelog/artifact and gates.

## Notes / Blockers

Model size may prevent full CI execution. The sprint must still provide a reproducible hardware profile, cached revision, and smaller structural fixture.


# Sprint 58 Plan

## Sprint Goal

Capture and analyze internal ACT policy representations through LeRobot's normal preprocessing and action-selection path.

## Atomic Tasks

Status legend: [ ] pending / [~] in progress / [x] done

- [ ] Select a public ACT checkpoint/dataset pair and pin both revisions.
- [ ] Load the policy and official pre/post-processors through supported LeRobot factories.
- [ ] Identify one semantically justified encoder/decoder/action-query capture point from source and architecture docs.
- [ ] Implement an ACT-specific adapter using the shared capture lifecycle and policy metadata.
- [ ] Prove unmodified action outputs match direct LeRobot inference.
- [ ] Run projection/probe/trajectory analysis on successful and failed episodes with controls.
- [ ] Add tiny policy unit fixtures plus a marked checkpoint integration test.
- [ ] Update evidence/ADR/changelog/artifact and gates.

## Notes / Blockers

This sprint observes ACT representations. Causal policy intervention and environment-level effect measurement are reserved for Sprint 61.


# Sprint 51 Plan

## Sprint Goal

Add SO(3)/SE(3) pose geometry with valid group operations for robotics trajectories.

## Atomic Tasks

Status legend: [ ] pending / [~] in progress / [x] done

- [ ] Define pose representation, units, frame metadata, and conversion boundaries without ambiguous Euler-angle defaults.
- [ ] Implement validation, composition, inverse, log/exp maps, distance, and interpolation for one SO(3)/SE(3) representation.
- [ ] Add property tests for group closure, identity, inverse, round-trip log/exp, and valid rotations.
- [ ] Compare group interpolation against element-wise averaging on controlled robot poses.
- [ ] Integrate pose-valued latent states with trajectory slicing and LeRobot-compatible metadata.
- [ ] Add serialization/config support and explicit frame-mismatch errors.
- [ ] Produce a trajectory artifact that visualizes motion while reporting group-distance metrics.
- [ ] Update evidence/ADR/changelog/artifact and gates.

## Notes / Blockers

This geometry is a prerequisite for meaningful manipulation of robot state/action trajectories in later LeRobot sprints.


# Sprint 75 Plan

## Sprint Goal

Add bounded-memory streaming execution for long datasets and trajectories while preserving order, state, cancellation, and profiling semantics.

## Atomic Tasks

Status legend: [ ] pending / [~] in progress / [x] done

- [x] Implement one concrete streaming execution story over long rollout action chunks using the existing `RolloutPipeline` and `LatentTransition.step` seams.
- [x] Define chunk boundary, overlap/window, state carry, backpressure, cancellation, and error propagation behavior.
- [x] Support async iteration without blocking the event loop and preserve deterministic output ordering.
- [x] Add tests for short/final chunks, variable lengths, cancellation, producer errors, consumer errors, and bounded queue memory.
- [x] Verify stateful methods/transitions either restore correct cross-chunk state or are rejected explicitly.
- [x] Measure peak memory, throughput, latency, and equivalence with eager execution.
- [x] Re-evaluate runtime/pipeline contracts with this new story and extract only proven invariants.
- [x] Update ADR/evidence/changelog/artifact and gates.
- [x] Run final Sprint 75 closure gates and record residual risks.

## Post-closure audit remediation (2026-08-25)

The closure audit found three blocking contract/test gaps and several scope
clarifications. These bounded remediation tasks and the rerun closure gates
are now complete. At this closure snapshot, Sprint 76 remained unopened and
was subsequently completed in its own plan.

- [x] Offload async iterator construction and cleanup, and strengthen blocking/cancellation/resource-leak tests.
- [x] Enforce fail-closed bounded action-chunk preflight before conversion/allocation with adversarial tests.
- [x] Strengthen the event-loop responsiveness regression so it fails when transition work runs on the loop.
- [x] Reconcile hidden-state/reset, masks/padding/seeding scope, cache/run-record behavior, profiling bounds, and benchmark wording across docs/evidence.

## Notes / Blockers

This is the first true streaming runtime story. The Sprint 24 concrete-runtime
decision remains in force because one streaming story does not prove a shared
cross-pipeline abstraction. Sprint 75 delivery and audit remediation closed on
2026-08-25; Sprint 76 was planned and unopened at that historical snapshot,
and was subsequently completed.

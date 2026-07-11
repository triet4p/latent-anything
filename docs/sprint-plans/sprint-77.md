# Sprint 77 Plan

## Sprint Goal

Run representative performance gates and make an evidence-based Rust-core go/no-go decision.

## Atomic Tasks

Status legend: [ ] pending / [~] in progress / [x] done

- [ ] Define representative workloads for geometry, trajectory alignment, capture, rollout/planning, serialization/cache, and LeRobot policy overhead.
- [ ] Add stable benchmark harnesses with warmup, repetitions, environment metadata, memory, throughput, latency, and variance.
- [ ] Profile Python, NumPy, PyTorch, I/O, and backend time to isolate framework-owned bottlenecks.
- [ ] Set product budgets for interactive analysis and policy-evaluation overhead based on measured use cases.
- [ ] Optimize low-risk Python/vectorization issues first and rerun benchmarks.
- [ ] Write a Rust go/no-go ADR naming exact kernels, expected benefit, maintenance cost, and cross-language contract.
- [ ] If no kernel meets the threshold, explicitly defer Rust; if one does, create a post-1.0 implementation plan unless stable gates require it.
- [ ] Update evidence/changelog/artifact and strict performance regression gates.

## Notes / Blockers

This sprint is complete with a justified no-go. Rust is not a milestone trophy.


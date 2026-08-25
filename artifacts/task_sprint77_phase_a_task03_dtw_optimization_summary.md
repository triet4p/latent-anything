# Sprint 77 Phase A Task 03 — measured DTW optimization

Status: complete (2026-08-25)

## Change

`compute_dtw` now uses a bounded row-wise NumPy norm path only for the proven
`euclidean` geometry. It preserves the same `point_costs`, `max_step_distance`,
window, accumulated matrix, deterministic tie-break, traceback, and result
contracts. All other geometries retain the existing `space.distance` dispatch.

## Focused checks

```text
uv run pytest -q tests/test_dtw.py tests/test_sprint77_phase_a_benchmark.py tests/test_sprint77_phase_a_profile.py
12 passed
```

The new DTW regression verifies window masking, terminal traceback, and the
expected finite cost. The two Phase-A suite tests verify all fixed workloads,
semantic digests, and repeated-suite reproducibility.

## Before/after evidence

Both reports used seed 771, the same 13 workloads, Windows 11 / Python 3.13.3,
NumPy 2.4.6, Torch 2.10.0, PyArrow 24.0.0, one Torch thread, two warmups, and
eight repetitions. The comparison requires equal workload contracts and
preserved correctness digests.

```text
trajectory_dtw median: 38,297.3 us -> 27,478.2 us (-28.25%)
trajectory_dtw correctness digest: f833ee0ae3ba0c9b037f0414394a45406d2924c3ec5da226f33cc8a73e038de5 (unchanged)
```

The other cases are retained as context in
`artifacts/sprint77_phase_a_comparison.json`; no improvement is attributed to
this source change because their differences are within host/filesystem/Torch
noise and no source in those paths changed.

## Rust evidence boundary

This low-risk optimization reduces one Python dispatch hotspot while leaving
DTW traceback dynamic programming in Python. It is evidence for a possible
future kernel investigation, not a Rust go/no-go decision or ADR. The owner
must decide that later from the complete Phase-A evidence.

## Graphify trace

```text
graphify update .
PASS; 9,991 nodes / 19,495 edges / 897 communities; 46 zero-node JSON/source warnings.
```

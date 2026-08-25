# Sprint 77 Phase A Task 02 — bottleneck attribution

Status: complete (2026-08-25)

## Delivered

`scripts/sprint77_phase_a_profile.py` runs cProfile over the representative
DTW, geodesic, activation, CEM, MPPI, portable, artifact/cache, and recorder
cases. It stores top cumulative rows with source location, calls, total time,
and cumulative time. The interpretation keeps framework-owned code separate
from NumPy, Torch, PyArrow, SQLite, filesystem, and model execution.

## Recorded findings

- `trajectory_dtw`: `compute_dtw` made 768 `LatentSpace.distance` calls; the
  largest framework/dependency rows were `dtw.py:87(compute_dtw)`,
  `latent_space.py:348(distance)`, and NumPy `linalg.norm`.
- `density_geodesic`: the bounded optimizer was dominated by the intended
  50 finite-difference iterations (`geometry.py:322(optimize_density_path)`,
  `geometry.py:288(density_path_gradient)`, and
  `geodesic.py:394(_finite_difference_gradient)`). This is a bounded numerical
  algorithm, not evidence that a Rust port is required.
- `activation_capture`: Torch linear/relu forward work dominated the fixture;
  `capture.py:102(callback)` and `_metadata` were a minority. Model forward
  time is explicitly excluded from framework attribution.
- CEM and MPPI were mostly NumPy sampling/reduction with compact framework
  wrappers; no low-risk source optimization was justified there.
- Portable decode was dominated by PyArrow IPC; artifact/cache and recorder
  were dominated by SQLite/open/close and filesystem I/O. These are backend
  costs, not Python kernels to relabel as framework bottlenecks.

## Exact validation

```text
uv run python scripts/sprint77_phase_a_profile.py --limit 12 --output artifacts/sprint77_phase_a_profile.json
PASS; selected eight cases, report schema sprint77-phase-a-profile-v1
uv run pytest -q tests/test_sprint77_phase_a_profile.py
1 passed
```

## Graphify trace

```text
graphify update .
PASS; 9,984 nodes / 19,489 edges / 868 communities; 46 zero-node JSON/source warnings.
```

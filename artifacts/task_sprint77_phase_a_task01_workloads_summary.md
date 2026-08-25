# Sprint 77 Phase A Task 01 — representative workloads and harness

Status: complete (2026-08-25)

## Delivered

- Added `scripts/sprint77_phase_a_benchmark.py` with schema `sprint77-phase-a-v1`.
- Added structural tests in `tests/test_sprint77_phase_a_benchmark.py`.
- The fixed offline suite covers geometry distance, DTW alignment, density geodesic, activation capture, rollout, CEM, MPPI, Arrow portable encode/decode, ArtifactStore plus SQLite cache, bounded streaming, local recording, metadata-only plugin listing, and the offline LeRobot NumPy boundary. It makes no real-policy or provider claim.
- The report records allowlisted environment metadata, seed 771, warmups/repetitions, median/p95/mean/stdev latency, `tracemalloc` peak, best-effort RSS, and a deterministic correctness digest. Default CI policy is semantic digest validation; latency is environment-scoped advisory evidence.
- Added `scripts/sprint77_phase_a_profile.py` and `tests/test_sprint77_phase_a_profile.py` for cumulative cProfile attribution.
- Added `scripts/sprint77_phase_a_compare.py` and `tests/test_sprint77_phase_a_compare.py` to compare identical before/after workload contracts and require preserved correctness digests.

## Exact validation and evidence

```text
uv run pytest -q tests/test_sprint77_phase_a_benchmark.py tests/test_sprint77_phase_a_profile.py tests/test_sprint77_phase_a_compare.py
2 + 1 + 1 passed (the DTW regression was run separately below)

uv run pytest -q tests/test_dtw.py tests/test_sprint77_phase_a_benchmark.py tests/test_sprint77_phase_a_profile.py
12 passed

uv run python scripts/sprint77_phase_a_benchmark.py --warmups 2 --repetitions 8 --output artifacts/sprint77_phase_a_benchmark.json
PASS; Windows 11 / Python 3.13.3 / NumPy 2.4.6 / Torch 2.10.0 / PyArrow 24.0.0 / 1 Torch thread; seed 771

uv run python scripts/sprint77_phase_a_profile.py --limit 12 --output artifacts/sprint77_phase_a_profile.json
PASS; DTW profile attributes 768 distance calls and geodesic profile attributes bounded finite-difference work

uv run python scripts/sprint77_phase_a_benchmark.py --warmups 2 --repetitions 8 --output artifacts/sprint77_phase_a_benchmark_after_dtw.json
PASS; same workload contract and all correctness digests as the before report

uv run python scripts/sprint77_phase_a_compare.py artifacts/sprint77_phase_a_benchmark.json artifacts/sprint77_phase_a_benchmark_after_dtw.json --output artifacts/sprint77_phase_a_comparison.json
PASS; trajectory_dtw median 38,297.3 us -> 27,478.2 us (-28.25%); digest preserved
```

The other case medians are retained for context, but their single before/after pair is not treated as an optimization claim because filesystem, Torch warmup, and host scheduling noise affect them. The DTW change is the only intentional source edit in this comparison.

## Graphify trace

The required graph refresh is run immediately after this atomic task and its plan update:

```text
graphify update .
```

Result: PASS; graphify watch rebuilt 9,978 nodes, 19,484 edges, and 879 communities. It warned that 46 source files produced zero nodes (including JSON benchmark artifacts); dirty graph output is expected and this warning does not change benchmark evidence.

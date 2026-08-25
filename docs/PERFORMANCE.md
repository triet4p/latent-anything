# Sprint 77 Phase-A performance evidence

Sprint 77 Phase A defines reproducible offline CPU workloads and budgets. The
owner-approved Phase B decision defers Rust/PyO3 for pre-stable work; the
bounded closure validation passed without claiming native, real-
policy, multi-environment, or Windows RSS coverage.

## Reproduce

```text
uv run python scripts/sprint77_phase_a_benchmark.py --warmups 2 --repetitions 8 --output artifacts/sprint77_phase_a_benchmark.json
uv run python scripts/sprint77_phase_a_profile.py --limit 12 --output artifacts/sprint77_phase_a_profile.json
```

The harness uses seed `771`, two warmups, eight measured repetitions, robust
median/p95/mean/stdev latency, `tracemalloc` peak, best-effort RSS, an
allowlisted environment bundle, and a correctness digest. It runs on fixed
NumPy and tiny Torch fixtures without network or provider access. LeRobot is
represented only by its existing offline captured-latent NumPy boundary; no
real policy or model-performance claim is made.

## Workload matrix

| Workload | Fixed fixture | Current after median / p95 | Attribution |
| --- | --- | ---: | --- |
| Euclidean distance | 16-D pair | 9.45 / 18.1 µs | framework dispatch |
| DTW alignment | 24 × 32, 16-D | 27,478.2 / 34,190.5 µs | framework cost matrix/traceback |
| Density geodesic | 16 points, 50 iterations | 87,838.4 / 99,269.0 µs | bounded framework optimizer |
| Activation capture | 16 × 16 tiny Torch MLP | 168.35 / 195.3 µs | hook/capture only; model forward excluded |
| Rollout | 32 actions, deterministic transition | 1,231.2 / 2,566.5 µs | pipeline/transition |
| CEM / MPPI | population 64, horizon 8, 3 iterations | 1,373.25 / 1,892.8 µs; 1,225.35 / 1,727.8 µs | planner wrapper plus NumPy |
| Portable Arrow | 32 × 16 `LatentValue` | 1,080.1 / 1,543.4 µs | PyArrow codec |
| Artifact + SQLite cache | same payload, temp local store | 78,655.9 / 119,155.6 µs | filesystem/SQLite included |
| Bounded stream | 32 actions, chunks of 8 | 1,392.3 / 1,588.7 µs | rollout stream |
| Local recorder | one run/config/metric/finish | 85,774.6 / 93,665.1 µs | filesystem lifecycle |
| Plugin listing | empty metadata provider | 27.35 / 35.0 µs | importlib metadata listing |
| LeRobot boundary | one offline NumPy capture | 5.2 / 8.9 µs | bridge-owned copy |

These values are one Windows 11 / Python 3.13.3 / 8-CPU / one-Torch-thread
run and are advisory, not portable CI limits. Native RSS was unavailable on
this Windows run; the report records `tracemalloc` and `null` RSS explicitly.

## Budgets and gates

Proposed product budgets for the declared fixtures are p95 advisory targets:

- interactive geometry distance: 100 µs;
- small DTW alignment: 50 ms;
- bounded density geodesic: 150 ms;
- activation capture overhead excluding model forward: 300 µs;
- policy-evaluation rollout/planning overhead: 5 ms for the declared small
  rollout or planner fixture;
- portable encode/decode: 3 ms;
- one bounded stream fixture: 3 ms;
- metadata-only plugin listing: 100 µs.

Filesystem recorder and artifact/cache operations remain use-case and storage
dependent; the measured values are a baseline for local SSD regression review,
not a real-time guarantee. Hard semantic gates are deterministic correctness
digests, bounded fixture sizes, no network, and focused/full test contracts.
Latency, variance, and memory are environment-sensitive advisory gates and
must not be “fixed” by shrinking workloads or loosening semantics.

## Profile and optimization boundary

The profile shows DTW's repeated Euclidean distance dispatch as the clearest
framework-owned Python hotspot. The Phase-A optimization computes bounded
row-wise NumPy Euclidean costs while preserving all other geometry dispatch,
windowing, max-step filtering, traceback, and result semantics. Identical
before/after reports show 38,297.3 → 27,478.2 µs median (-28.25%) with an
unchanged correctness digest. This is a small Python/vectorization result,
not a Rust decision.

Geodesic finite differences, Torch forward, PyArrow IPC, SQLite, and local
filesystem rows remain explicitly attributed to their existing algorithm or
dependency/backend costs. A future Rust proposal must name a kernel, call
frequency, expected benefit ceiling, NumPy/PyTorch alternative, and
cross-language maintenance/contract cost. The Phase B ADR records a deferral,
not a permanent prohibition or an implementation commitment.

Evidence files: [`sprint77_phase_a_benchmark.json`](../artifacts/sprint77_phase_a_benchmark.json),
[`sprint77_phase_a_profile.json`](../artifacts/sprint77_phase_a_profile.json),
[`sprint77_phase_a_comparison.json`](../artifacts/sprint77_phase_a_comparison.json).

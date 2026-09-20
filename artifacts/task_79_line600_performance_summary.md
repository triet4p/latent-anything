# Task Summary: Sprint 79 line 600 — performance budgets and LeRobot policy overhead vs Sprint 77 gates

**Sprint:** Sprint 79
**Task:** line 600 (Measure performance budgets and LeRobot policy overhead against Sprint 77 gates)

## Summary of Work

Reproduced the full Sprint 77 Phase-A offline CPU suite three times with the frozen protocol
(warmups 2, repetitions 8, seed 771) plus the cProfile attribution lane, and measured the
contract-defined local LeRobot overhead lanes (offline captured-latent boundary plus tiny-fixture
ACT/Diffusion/SmolVLA adapter `select_action` overhead through the real production integration
boundary). Compared every Sprint 77 p95 advisory budget one-to-one against current measurements with
environment/hardware metadata and semantic-digest checks. 9 of 10 budgeted lanes PASS; the bounded
streaming lane is environment-variance marginal (median always passes, p95 straddles the 3 ms line
across identical-protocol reruns with unchanged digest) and is reported as advisory variance, not a
regression. The pinned real-policy (SmolVLA/LIBERO CUDA) lane is concretely externally blocked and
left unmeasured without mock substitution. No gate was loosened, no sample deleted, no code changed.
Line 600 stays **unchecked** with the exact marginal/unmeasured lanes below.

## Files Modified

* [artifacts/task_79_line600_performance_summary.md](task_79_line600_performance_summary.md) — this
  reproducible summary (new file; only file changed)

## Testing (exact commands and results)

Sprint 77 authorities: `docs/PERFORMANCE.md`, `scripts/sprint77_phase_a_benchmark.py` (seed 771),
`scripts/sprint77_phase_a_profile.py`, `scripts/sprint77_phase_a_compare.py`,
`artifacts/sprint77_phase_a_benchmark.json` (baseline),
`artifacts/sprint77_phase_a_benchmark_after_dtw.json`,
`artifacts/sprint77_phase_a_comparison.json`, `artifacts/sprint77_phase_a_profile.json`.

* `uv run python scripts/sprint77_phase_a_benchmark.py --warmups 2 --repetitions 8 --output
  /tmp/line600_phase_a_rerun.json` — EXIT 0 (run 1 of 3; runs 2–3 identical protocol to
  `/tmp/line600_phase_a_rerun2.json`, `/tmp/line600_phase_a_rerun3.json`).
* `uv run python scripts/sprint77_phase_a_profile.py --limit 12 --output
  /tmp/line600_phase_a_profile.json` — EXIT 0; selected 8 cases match baseline selection.
* `uv run python scripts/sprint77_phase_a_compare.py artifacts/sprint77_phase_a_benchmark.json
  /tmp/line600_phase_a_rerun.json --output /tmp/line600_compare.json` — EXIT 0;
  `workload_contract_equal: true`, environment python/numpy/torch/seed equal, **all 13 semantic
  digests preserved** (`digest_failures: []`).
* `uv run pytest tests/test_sprint75_streaming.py tests/test_sprint77_phase_a_benchmark.py
  tests/test_sprint77_phase_a_compare.py tests/test_sprint77_phase_a_profile.py -q` — **5 passed**.
* `uv run pytest tests/test_sprint74_benchmark.py tests/test_sprint74_roundtrip.py -q` —
  **2 passed**.
* `uv run pytest tests/test_lerobot_benchmark.py -q` — **12 passed, 1 skipped**.
* `uv run python scripts/sprint75_streaming_benchmark.py` — `status: pass`, horizon 4096, eager
  0.088532 s vs stream 0.23855 s, digests equal
  (`2b103dd24ec88375fbdbe76c7bb92a00abc5e6472454a7ba867f9d3087a8bf00`).
* `uv run python scripts/act_policy_representation_benchmark.py` — EXIT 0, probe-beats-majority
  acceptance true. `diffusion_policy_representation_benchmark.py` — EXIT 0, AUROC/acceptance true.
  `smolvla_policy_representation_benchmark.py` — EXIT 0, all 8 intervention checks true.
* Local LeRobot overhead probes (2 warmups + 8 samples, controlled): offline boundary med 5.3 /
  p95 5.9 µs; ACT tiny-fixture adapter med 94.9 / p95 110.4 µs; Diffusion tiny-fixture adapter med
  227.5 / p95 234.1 µs; SmolVLA tiny-fixture adapter med 843.5 / p95 1203.1 µs (noise `(1,4,4)` per
  fixture `CHUNK=4/MAX_ACTION=4`; an earlier wrong `(1,50,6)` probe shape was an operator error,
  corrected, no product impact).
* Environment (all three reruns): Windows-11-10.0.26200-SP0, AMD64, Python 3.13.3, CPU 8, torch
  threads 1, latent-anything 0.1.0b1, numpy 2.4.6, torch 2.10.0, pyarrow 24.0.0, seed 771.
  Local CUDA: `torch.cuda.is_available() == False`, so no GPU lane was attempted locally.
* No remote CUDA run: the only GPU-requiring lane (pinned SmolVLA/LIBERO real-policy) is blocked on
  missing Linux CUDA host + model/data access + license capture (line-597 receipt), and the
  remote-cuda-test invariants forbid a substituted run; no SSH/remote command was issued.

## Metrics vs Sprint 77 gates (run 1 p95 vs budget; medians in summary table)

| Lane (fixture) | Baseline med / p95 (µs) | Current run 1 med / p95 (µs) | Budget p95 (µs) | Verdict |
|---|---|---:|---:|---|
| geometry_distance (16-D) | 13.6 / 41.2 | 9.1 / 16.9 | 100 | PASS |
| trajectory_dtw (24×32) | 38297.3 / 41001.3 | 24080.2 / 26716.6 | 50000 | PASS |
| density_geodesic (16pt/50it) | 111285.9 / 132202.4 | 78439.4 / 85337.9 | 150000 | PASS |
| activation_capture (hook only) | 176.8 / 212.6 | 138.2 / 166.0 | 300 | PASS |
| rollout (32 actions) | 1459.3 / 2973.1 | 1066.2 / 1379.2 | 5000 | PASS |
| cem_planning (pop64/h8/it3) | 1520.4 / 3123.1 | 1117.2 / 1275.9 | 5000 | PASS |
| mppi_planning (pop64/h8/it3) | 1745.2 / 3330.3 | 1057.7 / 1260.1 | 5000 | PASS |
| portable_encode_decode (32×16) | 1188.4 / 2276.3 | 849.5 / 895.2 | 3000 | PASS |
| artifact_and_disk_cache (temp store) | 100891.6 / 123258.9 | 65152.3 / 75901.5 | advisory only | BASELINE (storage-dependent, faster) |
| bounded_streaming (32 acts/ch8) | 1493.6 / 1660.9 | 1549.0 / **3023.1** | 3000 | MARGINAL — advisory variance (see below) |
| local_recorder (fs lifecycle) | 118684.4 / 148116.4 | 62032.2 / 80410.8 | advisory only | BASELINE (storage-dependent, faster) |
| plugin_listing (metadata-only) | 31.2 / 40.7 | 25.4 / 39.3 | 100 | PASS |
| lerobot_numpy_boundary (offline) | 6.0 / 10.3 | 5.0 / 8.4 | contract lane, no budget | PASS (digest match) |

Bounded-streaming variance detail (identical protocol, digest `d875632a4dca…` unchanged all runs):
run 1 med 1549.0 / p95 3023.1 (one 3023.1 first-sample outlier); run 2 med 1585.2 / p95 2456.6
(PASS); run 3 med 1423.7 / p95 1960.9 (PASS). Per `docs/PERFORMANCE.md`, latency is an
environment-sensitive advisory gate that "must not be fixed by shrinking workloads"; the workload was
not changed and no sample was dropped. Median passes in all three runs; the p95 straddle is
first-sample/JIT/filesystem noise on this Windows host, not a semantic regression.

## LeRobot policy overhead semantics (explicit)

* Measured (local, contract-defined): offline captured-latent NumPy boundary (~5 µs) — bridge-owned
  copy only, no policy/model claim; tiny-fixture adapter `select_action` overhead through the real
  `LeRobotPolicyContext → {ACT,Diffusion,SmolVLA}PolicyAdapter` boundary: ACT ~95/110 µs, Diffusion
  ~228/234 µs, SmolVLA ~844/1203 µs (med/p95). These isolate **adapter dispatch + tiny fixture
  compute**, explicitly excluding real checkpoint inference.
* Historical reference only (not re-measured, not claimed): `artifacts/smolvla_simulation_benchmark.json`
  (`evidence_status: historical_unverified_after_review_remediation`) reports per-query means ~4.8–6.1 ms
  and first-query ~140–226 ms on its original CUDA host; it predates the corrected latency aggregation
  and must not be compared against the fixture numbers above.
* Unmeasured external lane: pinned real-policy SmolVLA (`lerobot/smolvla_libero@31d453f…` +
  `lerobot/libero@a1aaacb7…`, LIBERO `libero_spatial`, Linux ~16 GB GPU, model/data access + license
  capture per the line-597 receipt). Blocked — no mock timing substituted, no remote run attempted.

## Additional Notes

* No source, gate, threshold, workload, snapshot, or empirical acceptance gate changed. The
  supported-scope ledger, map, and queue now explicitly exclude canonical BF16 OpenVLA/L19
  from active execution while retaining its D0 feasibility history; no quantized/offloaded
  substitute was introduced.
* Graphify queried first; no product code changed.
* Line 600 stays `[ ]` unchecked: 9/10 budgets PASS + 1 advisory-marginal + 1 externally-blocked
  SmolVLA real-policy lane; the plan requires the full supported-scope contract to check it.
* Lines 595/598/599 states and accepted counts (41/63 core, 41/64 scoped overall; 599 checked)
  remain honest. L21 SmolVLA remains active at its documented ~16 GiB profile.

# Task Summary: Sprint 61 Task 6 — Regression smoke tests and statistical benchmark

**Sprint:** Sprint 61
**Task:** Add regression smoke tests and a separately marked statistical benchmark.

## Summary of Work

`tests/test_lerobot_benchmark.py` adds eight offline deterministic regression tests over a compact SmolVLA-mirroring fixture (state projection, expert norm, flow-matching denoising, action queue) and a fake `SyncVectorEnv`-style environment: config validation, Wilson CI closed-form values, baseline bit-exactness + acceptance gate, intervention action changes + metrics, offline scores + disagreement rules, episode outcome recording, upstream factory wiring (no LeRobot import), and the understatement/reversal rules. The separately marked `test_smolvla_simulation_statistical_benchmark` (network + large_download, CUDA-required, `LATENT_ANYTHING_RUN_NETWORK=1`) runs the real SmolVLA + LIBERO benchmark on a small seed grid and asserts the predeclared acceptance gate.

## Files Modified

* [tests/test_lerobot_benchmark.py](tests/test_lerobot_benchmark.py) - offline suite + marked statistical lane.

## Testing

* **Test File:** [tests/test_lerobot_benchmark.py](tests/test_lerobot_benchmark.py)
* **Status:** Passed (8 passed, 1 skipped in the default offline suite)
* **Execution Command:** `uv run pytest tests/test_lerobot_benchmark.py -v`

## Additional Notes

The statistical lane is opt-in exactly like the other checkpoint lanes; conftest.py's `network`-marker skip keeps the default suite offline. The lane is also wired into the `lerobot-smolvla` CI job.

# Task Summary: Sprint 60 Task 7 — Structural tests and marked GPU benchmark

**Sprint:** Sprint 60
**Task:** Add lightweight structural tests and a marked GPU checkpoint benchmark.

## Summary of Work

`tests/test_lerobot_smolvla.py` provides 9 offline tests with a tiny deterministic fixture that mirrors LeRobot's official SmolVLA seams (SigLIP vision encoder, language embedding table, state projection, action-expert norm, flow-matching denoising, action queue): lazy-import isolation, modality/token-metadata capture, direct-path parity, queue semantics, identity-at-zero, intervention validation, exception-safe hook lifecycle, measurement bundle, and factory delegation. The marked CUDA lane (`network` + `large_download`) loads the pinned public pair, verifies direct-path parity and seed determinism, and runs the intervention measurement on the real checkpoint.

## Files Modified

* `tests/test_lerobot_smolvla.py` — offline fixture suite and marked GPU lane.
* `scripts/smolvla_policy_representation_benchmark.py` — deterministic offline benchmark with 8 acceptance criteria.
* `.github/workflows/optional-extras.yml` — `lerobot-smolvla` resolve and test lanes.

## Testing

* **Execution Command:** `uv run pytest tests/test_lerobot_smolvla.py -q`
* **Status:** `9 passed, 1 skipped` (GPU lane opt-in)
* **Benchmark:** `uv run python scripts/smolvla_policy_representation_benchmark.py` — all acceptance criteria true.

## Additional Notes

The marked GPU lane runs on the remote CUDA server (`/remote-cuda-test`) with `LATENT_ANYTHING_RUN_NETWORK=1`; the CPU policy load, capture, parity, and intervention were verified locally as part of Task 2/4/5.

# Task Summary: Sprint 60 Task 4 — Baseline action parity with direct inference

**Sprint:** Sprint 60
**Task:** Verify baseline action outputs and seeds against direct LeRobot inference.

## Summary of Work

The adapter never reimplements the policy path: it calls the official preprocessor, `policy.select_action(prepared, noise=...)`, and postprocessor while hooks only observe. Parity and seed determinism are verified both offline (fixture) and on the real model (CUDA lane):

* `test_smolvla_action_matches_direct_preprocess_select_postprocess_with_fixed_noise` — adapter action equals direct `postprocess(policy.select_action(preprocess(sample), noise=...))` bit-exactly.
* Queue semantics: a model query executes every `chunk_size` calls; queue hits return the queued action with zero captures.
* CUDA lane asserts direct-path equality and repeated-seed equality on the real checkpoint.

## Files Modified

* `tests/test_lerobot_smolvla.py` — parity, queue, and seed tests.

## Testing

* **Execution Command:** `uv run pytest tests/test_lerobot_smolvla.py -q`
* **Status:** Passed

## Additional Notes

Fixed noise is cast to float32 at the boundary, matching LeRobot's default action-chunk noise dtype and the float32 action expert.

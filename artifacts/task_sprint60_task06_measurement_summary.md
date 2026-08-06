# Task Summary: Sprint 60 Task 6 — Intervention effect measurement

**Sprint:** Sprint 60
**Task:** Measure immediate action change, off-target dimensions, representation drift, and sensitivity to prompts/camera order.

## Summary of Work

`measure_smolvla_intervention()` computes a `SmolVLAInterventionMeasurement`:

* **Immediate action change** — mean per-query `||Δaction||` plus per-dimension `mean|Δ|`.
* **On/off-target decomposition** — projects each action change onto the action-space direction induced by the expert direction through the policy's own `action_out_proj` head; reports on-target norm, off-target (orthogonal) norm, and their fraction.
* **Representation drift** — mean per-token expert displacement per denoising step (and at the first step).
* **Prompt sensitivity** — `||action(alternate task) - action(baseline)||` with the same noise.
* **Camera-order sensitivity** — `||action(swapped cameras) - action(baseline)||` with the same noise.

The offline fixture is designed so the intervention acts linearly in the expert direction: on-target fraction ≥ 0.99, per-step drift equals `|strength|·||direction||`, and prompt/camera swaps change the action. Acceptance criteria are recorded in the benchmark artifact.

## Files Modified

* `src/latent_anything/integrations/lerobot_smolvla.py` — measurement function and result types.

## Testing

* **Test:** `test_smolvla_measurement_reports_change_drift_and_sensitivity`
* **Status:** Passed
* **Benchmark:** `uv run python scripts/smolvla_policy_representation_benchmark.py` — all 8 acceptance criteria true; artifact written to `artifacts/smolvla_policy_representation_benchmark.json`.

## Additional Notes

The off-target definition is deliberately orthogonal-residual based (documented in the measurement metadata): it quantifies action change that does not follow the induced direction, including flow-matching trajectory effects beyond the linear head map.

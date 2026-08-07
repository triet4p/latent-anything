# Task Summary: Sprint 61 Task 1 — Simulation benchmark selection

**Sprint:** Sprint 61
**Task:** Select one supported simulation benchmark/task with deterministic seeds and tractable episode count.

## Summary of Work

Selected the pinned SmolVLA pair (`lerobot/smolvla_libero@31d453f7…` + `lerobot/libero@a1aaacb7f6…`) evaluated on the `libero` / `libero_spatial` LIBERO-10 simulation suite as the single benchmark. `SimulationBenchmarkConfig` (pydantic, frozen) in `src/latent_anything/integrations/lerobot_benchmark.py` carries deterministic seeds (default `(1, 2, 3)`), a tractable episode grid (one episode per seed/condition cell), task id selection, episode budget override, and observation sizes. The LIBERO environment extra is Linux-only, so the real lane is the remote CUDA server.

## Files Modified

* [src/latent_anything/integrations/lerobot_benchmark.py](src/latent_anything/integrations/lerobot_benchmark.py) - `SimulationBenchmarkConfig` + benchmark selection constants.

## Testing

* **Test File:** [tests/test_lerobot_benchmark.py](tests/test_lerobot_benchmark.py)
* **Status:** Passed
* **Execution Command:** `uv run pytest tests/test_lerobot_benchmark.py -v`

## Additional Notes

Determinism: each condition resets the same env object with the same seed; LIBERO's initial-state index is consumed per reset, so conditions stay comparable because episodes replay the same init state.

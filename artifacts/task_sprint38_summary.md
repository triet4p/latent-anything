# Task Summary: Sprint 38 — Scheduler Latent Intervention

**Sprint:** 38
**Task:** Scheduler-latent intervention with controls, metrics, and sweep

## Summary of Work

Implemented scheduler-latent intervention on the `DiffusersConditionalPipeline` through the native `callback_on_step_end` hook. Added `SchedulerIntervention` frozen dataclass to define additive direction edits (`latent ← latent + strength × direction`) at specified denoising steps. Modified `generate()` to accept an optional intervention and return modified latents through the callback kwargs. Added static helper methods `random_direction()` and `matched_norm_direction()` for the two control conditions. Predeclared the experiment design with explicit metrics, thresholds, and evaluation set. Created a comprehensive experiment script covering four-condition control comparison (no-edit, prompt-only, random-direction, matched-norm), quantitative metrics (cosine distance, SSIM, MSE, latent norm drift, trajectory cosine), and a timestep-by-strength sweep. Added 11 new deterministic offline tests and a gated real-checkpoint benchmark.

## Files Modified

- **src/latent_anything/integrations/diffusers_conditional.py** — Added `SchedulerIntervention` dataclass, `random_direction()`/`matched_norm_direction()` static helpers, optional `intervention` parameter on `generate()`, and updated callback to apply intervention and return modified latents.
- **tests/test_diffusers_conditional.py** — Added `TestSchedulerIntervention`, `TestInterventionDirectionHelpers`, and `TestFakeBackendIntervention` test classes covering validation, helper correctness, and intervention passthrough (11 new tests, 32 total).
- **tests/test_diffusers_conditional_network.py** — Added `test_intervention_produces_different_latents` gated real-checkpoint benchmark.

## Files Created

- **scripts/diffusers_conditional_intervention_experiment.py** — Comprehensive experiment script with four-condition control comparison, per-seed metrics, timestep/strength sweep, aggregate tables with uncertainty, counterexample detection, and figure outputs.
- **artifacts/task_sprint38_intervention_predeclaration.md** — Predeclared experiment design with target metric, preservation metric, quality proxy, evaluation set, and evidence-promotion thresholds.

## Documentation Updated

- **docs/sprint-plans/sprint-38.md** — All 8 atomic tasks marked as done.
- **CHANGELOG.md** — Added sprint 38 entries under Added (intervention, tests, experiment script).
- **.agents/memory/decisions.md** — Added ADR for scheduler-latent intervention design decision.
- **pyproject.toml** — Added experiment script to pyright include list.

## Testing Results

- **Offline tests:** 32 passed, 0 failed
- **Network tests:** 4 skipped (require `LATENT_ANYTHING_RUN_NETWORK=1`)
- **Lint:** ruff check and format pass on all modified files

## Evidence Status

The intervention mechanism and experiment infrastructure are D1 (implemented and tested offline). The predeclared thresholds (target change > 0.05, SSIM > 0.7, norm drift < 20%) must be verified against the real SD 1.5 checkpoint by running with `LATENT_ANYTHING_RUN_NETWORK=1`. The experiment script produces the evidence needed for seeding, with random directions establishing the lower bound — concept-specific directions may produce different targeting.

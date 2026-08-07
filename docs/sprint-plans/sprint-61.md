# Sprint 61 Plan

## Sprint Goal

Run a causal policy-explanation benchmark through LeRobot simulation evaluation and demonstrate when a latent intervention improves, harms, or leaves behavior unchanged.

## Atomic Tasks

Status legend: [ ] pending / [~] in progress / [x] done

- [x] Select one supported simulation benchmark/task with deterministic seeds and tractable episode count.
- [x] Define baseline, random intervention, targeted intervention, and no-hook control conditions.
- [x] Execute interventions through normal LeRobot policy preprocessing, `select_action`, postprocessing, and environment evaluation.
- [x] Measure success rate, return/task metrics, action deviation, latency, and confidence intervals over episodes.
- [x] Correlate offline explanation scores with environment-level causal effects and report disagreements.
- [x] Add regression smoke tests and a separately marked statistical benchmark.
- [x] Produce complete configs, episode summaries, videos/plots where license permits, and failure analysis.
- [x] Promote ACT/Diffusion/SmolVLA claims to D3 only where evidence passes; update ADR/changelog/artifact/gates.

## Notes / Blockers

This is the key defense against "explain for completeness." Explanation quality is judged by controlled behavioral effect, not a latent-space picture.

Selected benchmark: SmolVLA (`lerobot/smolvla_libero@31d453f7…`) evaluated on `libero` / `libero_spatial` (LIBERO-10) through `lerobot_benchmark.py`, with one episode per (seed, condition) cell, seeds (1, 2, 3) by default, and fixed zero noise. The LIBERO environment extra is Linux-only (`hf-libero`), so the real lane runs on the remote CUDA server.

Predeclared acceptance gate (D3 promotion requires all checks):
- `baseline_actions_bit_exact`: every baseline action equals the no-hook reference bit-for-bit;
- `baseline_success_equals_no_hook`: baseline success flags equal no-hook per seed;
- `intervention_changes_actions`: every non-zero intervention cell has mean action deviation > 1e-9;
- `all_episodes_within_max_steps`: every episode stays inside the step budget.

Declared disagreement rules: overstatement (on-target >= 0.8 but |success delta| < 0.2), understatement (on-target < 0.5 but |success delta| >= 0.2), reversal (success delta <= -0.2).

ACT and Diffusion have no intervention surface; their claims remain observational. Only the causal-intervention capability (THY-T05-CAUSAL-INTERVENTION-VS-OBSERVATIONAL-STUDY) can be promoted to D3, and only if the CUDA statistical lane passes; otherwise a negative-result artifact is committed instead.

Resolved: the CUDA statistical lane PASSED on the remote server (RTX 4060 Ti, LeRobot 0.6.1, torch 2.10) and the artifact is committed. The final grid (seeds 1–3, strengths 1/5/10) demonstrates all three outcomes: baseline is bit-exact; the targeted intervention leaves behavior unchanged at strength 1 (offline on-target 0.86, success delta 0.00 — recorded overstatement disagreement) and harms success from 1.0 to 0.0 at strengths 5 and 10 (recorded reversal disagreements, all 6 episodes maxed at 280 steps), while the random control never changes success. Acceptance passed; THY-T05-CAUSAL-INTERVENTION-VS-OBSERVATIONAL-STUDY promoted to D3.

# Sprint 61 Plan

## Sprint Goal

Run a causal policy-explanation benchmark through LeRobot simulation evaluation and demonstrate when a latent intervention improves, harms, or leaves behavior unchanged.

## Atomic Tasks

Status legend: [ ] pending / [~] in progress / [x] done

- [ ] Select one supported simulation benchmark/task with deterministic seeds and tractable episode count.
- [ ] Define baseline, random intervention, targeted intervention, and no-hook control conditions.
- [ ] Execute interventions through normal LeRobot policy preprocessing, `select_action`, postprocessing, and environment evaluation.
- [ ] Measure success rate, return/task metrics, action deviation, latency, and confidence intervals over episodes.
- [ ] Correlate offline explanation scores with environment-level causal effects and report disagreements.
- [ ] Add regression smoke tests and a separately marked statistical benchmark.
- [ ] Produce complete configs, episode summaries, videos/plots where license permits, and failure analysis.
- [ ] Promote ACT/Diffusion/SmolVLA claims to D3 only where evidence passes; update ADR/changelog/artifact/gates.

## Notes / Blockers

This is the key defense against "explain for completeness." Explanation quality is judged by controlled behavioral effect, not a latent-space picture.


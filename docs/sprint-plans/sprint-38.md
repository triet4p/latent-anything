# Sprint 38 Plan

## Sprint Goal

Demonstrate a selective intervention on Sprint 37 scheduler latent states with quantitative target, preservation, and generation-quality evidence.

## Entry Criteria

- Sprint 37 capture parity and provenance checks pass on both the tiny fixture and the selected real pipeline.

## Atomic Tasks

Status legend: [ ] pending / [~] in progress / [x] done

- [ ] Predeclare one bounded conditional generation edit, its target metric, non-target preservation metric, quality proxy, evaluation set, and evidence-promotion thresholds.
- [ ] Implement scheduler-latent intervention through `callback_on_step_end`, preserving the fixed prompt, scheduler, initial noise, and all unrelated state.
- [ ] Compare no-edit, prompt-only, random-direction, and matched-norm intervention controls under paired fixed seeds.
- [ ] Measure target change, content preservation, latent norm/cosine/nearest-neighbor or covariance-proxy drift, and decode/generation quality without depending on the later GMM sprint.
- [ ] Sweep only intervention timestep and strength to identify selective windows, saturation, reversals, and failure modes; do not mix this experiment with denoiser-layer intervention.
- [ ] Add deterministic smoke tests plus a marked reproducible real-checkpoint benchmark with immutable configuration and checkpoint provenance.
- [ ] Produce paired outputs, aggregate metric tables with uncertainty, and explicit counterexamples rather than cherry-picked successes.
- [ ] Promote relevant claims to D2/D3 only when the predeclared thresholds pass; otherwise retain D1, record the negative result, and update ADR/changelog/artifact/gates honestly.

## Notes / Blockers

This sprint isolates one causal seam: scheduler latent state. A layer-by-timestep-by-strength sweep would conflate two intervention mechanisms and is intentionally deferred. Visually plausible edits alone do not satisfy the exit criteria.


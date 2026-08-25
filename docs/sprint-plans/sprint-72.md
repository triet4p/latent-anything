# Sprint 72 Plan

## Sprint Goal

Validate tokenized world-model next-token prediction and rollout from image trajectories encoded by the fitted VQ-VAE, with codebook, temporal, and task-level metrics.

## Atomic Tasks

Status legend: [ ] pending / [~] in progress / [x] done

- [x] Compose encoded image observations, actions, and a sequence dynamics model through existing adapter/transition seams.
- [x] Implement autoregressive next-token prediction and seeded rollout for one concrete dataset/task.
- [x] Measure token likelihood/perplexity, code usage, multi-step drift, decoded consistency where a decoder exists, and task proxy metrics.
- [x] Compare teacher-forced and free-running behavior to expose compounding error.
- [x] Add invalid-token, mask, padding, horizon, and codebook-version tests.
- [x] Analyze whether tokenized dynamics fits the frozen transition contract; revise via ADR if it does not.
- [x] Produce a reproducible rollout artifact with failure horizons.
- [x] Update theory coverage, ADR/changelog/artifact and gates.

## Notes / Blockers

This sprint is an end-to-end CPU capability benchmark, not a promise to reproduce GAIA or Genie scale; arbitrary synthetic token IDs are reserved for unit tests. The regenerated artifact is D2 synthetic CPU evidence because the fitted tokenizer passes the non-trivial-token-usage gate. It still records early greedy free-running error and makes no real-checkpoint or CUDA claim.

## Post-sprint remediation closure

This bounded closure keeps the original Sprint 72 implementation history intact while resolving the known evidence blocker and stale governance state. It is not a new world-model protocol or a relaxation of the evidence contract.

- [x] **R1 — Diagnose and fix compact VQ-VAE collapse:** identify the training/root-cause failure, add a deterministic non-degenerate acceptance gate, and cover it with focused tests.
- [x] **R2 — Regenerate Sprint 70 evidence:** rerun the pinned CPU VQ-VAE benchmark and record measured reconstruction, codebook usage, and failure/acceptance fields.
- [x] **R3 — Regenerate Sprint 72 evidence:** fit tokenized dynamics from the regenerated non-collapsed tokenizer, retain teacher-forced/free-running metrics, and preserve explicit failure-horizon reporting.
- [x] **R4 — Reconcile M12 governance/docs:** synchronize `PLAN.md`, evidence ledgers, README/current capability text, M12 docs, changelog, and superseded task summaries without rewriting historical claims.
- [x] **R5 — Run closure gates:** focused tests, lint/format/type/test/docs/evidence gates, all M12 reproduction scripts, and graphify update; exact results are recorded in the remediation artifacts.

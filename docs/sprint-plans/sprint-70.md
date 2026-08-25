# Sprint 70 Plan

## Sprint Goal

Add a real VQ/discrete-latent model adapter with codebook usage, commitment, and collapse diagnostics.

## Atomic Tasks

Status legend: [ ] pending / [~] in progress / [x] done

- [x] Select a compact pretrained or trainable VQ-VAE-style model and pin dataset/checkpoint revisions.
- [x] Implement encode-to-codes, code embeddings, decode, codebook metadata, and discrete geometry integration.
- [x] Expose code sequences without silently converting them to continuous Euclidean values.
- [x] Measure reconstruction, codebook perplexity, dead-code rate, commitment distance, and code-frequency drift.
- [x] Add code replacement/interpolation policy tests that reject unsupported continuous semantics.
- [x] Compare discrete and continuous latent analysis paths on the same data where possible.
- [x] Add optional-extra/offline integration tests and reproducible artifacts.
- [x] Update evidence/ADR/changelog/artifact and gates.

## Notes / Blockers

This sprint supplies model evidence for the discrete geometry introduced in Sprint 30.

The compact CPU evidence path is intentionally diagnostic: deterministic
spread initialization prevents the first-run codebook collapse on the pinned
digits split, and the artifact requires non-trivial usage rather than merely
recording a collapsed run as healthy VQ evidence. This remains synthetic CPU
evidence, not a large pretrained VQ claim.

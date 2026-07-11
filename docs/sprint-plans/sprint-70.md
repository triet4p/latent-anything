# Sprint 70 Plan

## Sprint Goal

Add a real VQ/discrete-latent model adapter with codebook usage, commitment, and collapse diagnostics.

## Atomic Tasks

Status legend: [ ] pending / [~] in progress / [x] done

- [ ] Select a compact pretrained or trainable VQ-VAE-style model and pin dataset/checkpoint revisions.
- [ ] Implement encode-to-codes, code embeddings, decode, codebook metadata, and discrete geometry integration.
- [ ] Expose code sequences without silently converting them to continuous Euclidean values.
- [ ] Measure reconstruction, codebook perplexity, dead-code rate, commitment distance, and code-frequency drift.
- [ ] Add code replacement/interpolation policy tests that reject unsupported continuous semantics.
- [ ] Compare discrete and continuous latent analysis paths on the same data where possible.
- [ ] Add optional-extra/offline integration tests and reproducible artifacts.
- [ ] Update evidence/ADR/changelog/artifact and gates.

## Notes / Blockers

This sprint supplies model evidence for the discrete geometry introduced in Sprint 30.


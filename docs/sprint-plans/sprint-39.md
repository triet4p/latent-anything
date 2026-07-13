# Sprint 39 Plan

## Sprint Goal

Add a real, revision-pinned decoder-only transformer integration and a direct logit lens with explicit token, layer, normalization, and output-head semantics.

## Entry Criteria

- Sprint 32 capture lifecycle and cleanup invariants remain green.
- The chosen checkpoint and tokenizer can be cached explicitly for offline marked tests.

## Atomic Tasks

Status legend: [ ] pending / [~] in progress / [x] done

- [ ] Select and revision-pin a compact causal language model and tokenizer with a clear final normalization and language-model head.
- [ ] Define typed input, hidden-state, token-mask, layer-index, and lens-result values with NumPy-facing public payloads and full model/tokenizer provenance.
- [ ] Use native `output_hidden_states=True` as the canonical observation path and verify embedding/residual/final hidden-state indexing and shapes.
- [ ] Implement a direct logit lens only, with explicit final-normalization and output-head assumptions; defer learned/tuned translators to a later sprint.
- [ ] Validate native hidden states and final logits against direct backend execution, including padded-token masking and final-layer parity.
- [ ] Use the capture/hook seam only for one bounded activation intervention, then verify hook cleanup and unchanged execution outside the target layer/token.
- [ ] Measure selected-token rank/probability trajectories and their stability under predeclared prompt perturbations.
- [ ] Add tiny/offline tests, marked real-checkpoint tests, explicit cache/download behavior, and a reproducible artifact.
- [ ] Keep `HiddenStateAdapter` as a synthetic fixture, reconcile the integration ADR, and update evidence/changelog/artifact/gates.

## Notes / Blockers

Native hidden-state outputs are the parity oracle for observation; hooks are needed only for intervention. This sprint implements a direct lens, not a tuned lens, because a tuned lens introduces learned translators and checkpoint lifecycle concerns of its own.


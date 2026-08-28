# Task Summary: Sprint 79 L04.4 — Integrated Gradients handler

**Status:** implementation ready; real CUDA execution pending; no evidence
promotion.

## Summary

Added a focused M14 L04 Integrated Gradients handler for the pinned
`openai-community/gpt2` / `TransformerLMIntegration` boundary. The handler
requires `LATENT_ANYTHING_RUN_NETWORK=1`,
`LATENT_ANYTHING_NETWORK_DEVICE=cuda`, and actual CUDA availability before
constructing the lazy integration backend. It resolves the frozen one-token
` true`/` false` targets, uses transformer block 6 / native hidden-state index
7 and the last non-padding token, and records 16/64-step attribution,
zero/batch-mean baseline, randomized-target, seeded-repeat, finite and
no-mutation controls. Group-level estimates and 2,000-replicate bootstrap
intervals retain seed/group summaries without prompt text.

The envelope accepts a real-CUDA `passed_real_cuda` execution as
evidence-eligible support evidence only. `accepted_record_ids` and
`accepted_gap_ids` remain empty, and dependency-injected offline fakes remain
D0 and ineligible. Other L04 use cases retain their pending/blocked statuses.

The bounded execution plan computes deterministic zero-baseline 16/64-step
attributions and the full-batch mean baseline once per each of 24 fixture rows.
Random-target and seeded-repeat 64-step attributions still execute for every
row and each of the five declared seeds, using single-row batches. This is 312
`target_attribution` calls (624 scalar target/other IG runs): 48 calls at 16
steps and 264 at 64 steps. With the implementation's `n_steps + 4` model
forwards per scalar IG run (one capture forward, two endpoint forwards, and
`n_steps + 1` gradient-path forwards), the budget is 35,952 path-gradient
forwards plus 1,248 endpoint forwards plus 624 capture forwards = 37,824 model
forwards; only the 24 batch-mean calls use the full batch of 24.

## Verification

- Focused L04 and Integrated Gradients tests pass locally.
- Ruff and strict Pyright pass for changed Python modules.
- No model download, network execution, CUDA run, final artifact, commit, or
  push was performed in Phase A.

# Task Summary: Sprint 79 L04.4 — Integrated Gradients handler

**Status:** implementation complete; one owner-reviewed real-CUDA execution
retained as a semantic failure; no evidence promotion.

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
evidence-eligible support evidence only. The exact execution on commit
`83d1661dea0c69b8cafa8c12d0919636ca964022` did not pass the frozen
Integrated Gradients completeness gates, so it is retained as a failed,
evidence-ineligible D0 result. `accepted_record_ids` and `accepted_gap_ids`
remain empty; no ledger row or coverage count changed. Dependency-injected
offline fakes remain D0 and ineligible, and other L04 use cases retain their
pending/blocked statuses.

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

## Real-CUDA result

The run used the pinned `openai-community/gpt2@e7da7f221d5bf496a48136c0cd264e630fe9fcc8`
through `TransformerLMIntegration` on an NVIDIA GeForce RTX 4060 Ti. It made
one invocation (`invocation_count=1`, `rerun=false`) with five seeds
`[17, 29, 41, 53, 67]`, 24 fixture rows, and 120 seed/group rows. The frozen
thresholds were unchanged:

- zero baseline completeness relative error: `42.8119096032`, 95% CI
  `[30.1207528902, 57.0413396873]`, threshold `<= 0.001` — **FAIL**;
  16/64-step attribution cosine: `0.9972437907`, 95% CI
  `[0.9907480928, 0.9984744487]`, threshold `> 0.95` — **PASS**;
- batch-mean baseline completeness relative error: `0.0058147719`, 95% CI
  `[0.0005752487, 0.0155157174]`, threshold `<= 0.001` — **FAIL**;
- randomized target attribution cosine: `0.0592391281`, 95% CI
  `[-0.0802305086, 0.1991044123]`, threshold `<= 0.25` — **PASS**;
- seeded-repeat attribution cosine: `1.0`, 95% CI `[1.0, 1.0]`, threshold
  `> 0.99999999` — **PASS**;
- finite/no-mutation: finite fraction `1.0`, 95% CI `[1.0, 1.0]`, 120 rows,
  `mutated=false`, threshold `>= 1.0` — **PASS**.

The retained failure reason is `one or more frozen Integrated Gradients gates
failed` (`RuntimeError`). Remote execution elapsed `376.18669196404517` s;
peak allocated/reserved CUDA memory was `1603745792` / `1889533952` bytes.
The remote wrapper reported embedded status `1` and SSH exit `2` because a
PowerShell here-string transported CRLF status bytes to the remote Bash
`exit`; this transport discrepancy does not change the semantic failure.

## Evidence integrity and cleanup

The offline contract check and all three envelope validators pass with empty
error lists. Canonical digests recompute as follows: plan
`f3c315e356af0ee54d4196cc365ee22bd997b069d18a3e72c6b479f94e0b3e1a`, partial
artifact `880714c2a81904e4bba7b8cd854bfb707706a6a0c57f2cb348ca14b9b5db33f0`,
run record `c4d4bc4c446ce1c96d4fad38670f65096a2706d312e30b9fd2302c5ae7c36f49`,
and failure `8105314bed8288552d0f78f1842074ac8be674abf1afe8ecfe04581a671764bb`.
The execution-result digest is
`50a03c5a98de3eeb51e1434d4c28dd90652245cc65f0ae8131483abd8a19667d`;
fixture content/split/pair digests are
`f5c66f6d947c23f25d41e6aaf8982481feabc92bbff600bd929d27772fb62c0f`,
`7d788c18212bb1d7e345528c68af6f2bf3e0f745ca77e2d115d74ac3e964121b`, and
`7225e73c1238b23f6521718c8401331e59653a90499f4b2d75f32dddfe6c1c9c`.

Independent sanitization found zero credential matches and zero fixture prompt
literal matches in the retained records; no actual prompt text or disposable
path remains, and the raw SSH capture was deleted only after sanitized-record
verification. The remote disposable clone and caches were removed by the trap,
the local extract was removed, CUDA was synchronized, model gradients were
cleared, and the CUDA cache was emptied. No rerun was performed.

## Verification

- Focused L04 and Integrated Gradients tests pass locally.
- Ruff and strict Pyright pass for changed Python modules.
- The real-CUDA attempt and its sanitized SSH audit are retained above; no
  second model/network/CUDA invocation was made during closure.

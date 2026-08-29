# Task Summary: Sprint 79 L04.7 — Post-real-run correction

Implemented the post-real-run corrections without changing the frozen L04
plan or rewriting the retained attempt3 payload evidence. The sanitized audit
metadata received the explicitly scoped SHA correction documented below.

## Changes

- `scripts/_m14_l04_artifact.py` now copies the execution-level `seed` as well
  as bootstrap `seeds`, preserving the tuned-lens fit seed `79` in the
  production execution entry.
- `scripts/_m14_l04_tuned_lens.py` now masks shuffled-target fit and evaluation
  with `source_attention_mask & permuted_target_attention_mask`; source and
  tuned metrics continue to use the source mask. The policy is explicit in
  real-run provenance.
- `scripts/_m14_l04_validate_tuned_lens.py` requires the explicit shuffled
  target mask policy for independently validated accepted evidence.
- `tests/test_m14_l04_tuned_lens.py` exercises a realistic successful
  `run_tuned_logit_lens()` payload through `build_artifact()` and the complete
  artifact validator, and proves variable-length padded target positions do
  not affect shuffled fit/evaluation metrics or deterministic results.
- `docs/M14_REAL_SYSTEM_VALIDATION.md` documents the base64 PowerShell
  transport envelope: LF-normalized UTF-8 without BOM, local SHA-256,
  `ConvertToBase64String`, direct `ssh.exe target 'base64 -d | bash -s --'`,
  and remote start/digest markers. The frozen plan remains unchanged.
- `docs/sprint-plans/sprint-79.md` and `docs/EVIDENCE_GAP_PLAN.md` record the
  real CUDA computation/resource-pass result as D0/evidence-ineligible because
  the retained attempt3 artifact omitted fit seed `79`, including the
  shuffled-mask caveat and outer SSH exit `2` after cleanup.

## Historical payload evidence preserved byte-for-byte

The three retained attempt3 payload files (partial, run, and failure) were
verified byte-for-byte before and after the audit-only correction. The
sanitized audit is listed with its corrected current size/hash.

| File | Bytes | SHA-256 |
|---|---:|---|
| `artifacts/m14/l04-explanations.TunedLogitLens.attempt3.failure.json` | 10651 | `86e4fb9eea59553bd9a703c757f5c2fb04312633bc4077ca1f1aa92dd6e8fa0a` |
| `artifacts/m14/l04-explanations.TunedLogitLens.attempt3.partial.json` | 3331576 | `d00ca9bd4ab4799a8d128d1bf22b366758adaeccf9da16021953fd703e3d6cc6` |
| `artifacts/m14/l04-explanations.TunedLogitLens.attempt3.run.json` | 5452 | `8f9676e0535e1e306946aef3a28dfbd4d36fd3ebfc580bde93619a77ae7efa0e` |
| `artifacts/m14/l04-explanations.ssh.TunedLogitLens.2a6de8dbb98f824b247da23e2bc1e3cea5efea3a.recovery.audit.json` | 7638 | `0045807a6b66de4f54830ea58d60131a4e0f27a56443e7a64b91a00815ba81f3` |

The real run recorded point improvement `6.5803880806` nats, conservative
lower bound `6.5399008976` nats, all declared controls passing, and resource
budget passing on an RTX 4060 Ti (`1180.8877603` seconds,
`2167476736` allocated CUDA bytes, `2166382592` RSS bytes). The artifact is
not promoted: the current validator rejection records both missing
execution-level seed linkage (`seed=79`) and the newly required shuffled-target
mask-policy provenance. Cleanup passed; the wrapper's CRLF-sensitive status
encoding yielded outer SSH exit `2` after cleanup. No rerun was performed or
authorized. This remains immutable D0 historical evidence.

## Review correction

The sanitized recovery audit metadata now records the actual SHA-256 of the
retained attempt3 failure artifact as
`86e4fb9eea59553bd9a703c757f5c2fb04312633bc4077ca1f1aa92dd6e8fa0a`.
The audit's recorded failure size, partial/run sizes and hashes, and internal
artifact/run-record digests were rechecked against the retained files. The
partial, run, and failure artifact bytes were not rewritten; only sanitized
audit metadata is in scope for this correction.

## Verification status

Completed verification: focused L04 `120 passed`; full suite `1772 passed,
36 skipped, 39 warnings`; adversarial tuned-lens validators `39 passed`;
Ruff check and format pass; explicit strict Pyright for changed source and
test `0 errors, 0 warnings`; strict MkDocs build pass with temporary output
removed; `git diff --check` pass; and `graphify update .` pass. No SSH/CUDA
was used for these local gates. Git closure is performed in the two commits
that follow this summary update.

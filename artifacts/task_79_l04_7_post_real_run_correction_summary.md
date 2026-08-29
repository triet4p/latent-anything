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

## Final owner-authorized real CUDA run

Exactly one corrected run was executed at SHA
`278a9f76f626f8b0c6a9d9c5517c9b349f08c2d5` through the direct PowerShell
Base64 `ssh.exe` wrapper. The remote clone verified that detached SHA and ran
real GPT-2 through `TransformerLMIntegration` on the pinned WikiText-2
revision. The result is accepted D3: layer `6`, native hidden-state index `7`,
fit seed `79`, bootstrap seeds `[17, 29, 41, 53, 67]`, holdout improvement
`6.5803880806` nats, conservative lower bound `6.5399008976` nats, all
controls PASS, and resource budget PASS on an RTX 4060 Ti (`1116.8708012` s,
`2065599488` allocated CUDA bytes, `2161827840` RSS bytes). Artifact,
run-record, and failure validators all returned no errors.

The single connection returned `L04_STATUS=0` and `L04_CLEANUP=PASS`; the raw
capture was `6879774` bytes with SHA-256
`cd00dfd71812305b96c5528826868c5a3fb3fc0c74ac01e80396e6abfed593be`, and the
bundle was `2644019` bytes with SHA-256
`5fb4cca0e55fcabb94238ae9f3264d87d756a7881a4fb3b749b1c3ea74888c8e`.
The raw capture retained one observed `base64: invalid input` warning from the
transport while the decoded wrapper completed successfully; this warning is
not suppressed. The semantic/artifact evidence remains accepted at D3, but the
decoded script bytes were not independently hash-verified, so no stronger
transport provenance is claimed. The sanitized final audit is
`artifacts/m14/l04-explanations.ssh.TunedLogitLens.278a9f76f626f8b0c6a9d9c5517c9b349f08c2d5.recovery.audit.json`.
Raw deletion was performed only after audit/linkage validation and exact
size/SHA verification; post-delete absence is recorded in the audit. No retry
occurred.

Final attempt4 payload hashes:

| File | Bytes | SHA-256 |
|---|---:|---|
| `artifacts/m14/l04-explanations.TunedLogitLens.attempt4.failure.json` | 10651 | `5caaccdb19643892cc6994ceb6e0d00f2755784eb1eea98a3e8c1c849b334596` |
| `artifacts/m14/l04-explanations.TunedLogitLens.attempt4.partial.json` | 3331715 | `351e0ea131328ab87afca79a518511128d60d11011155a62193b8ab2bd5430c7` |
| `artifacts/m14/l04-explanations.TunedLogitLens.attempt4.run.json` | 5452 | `bbb12118cba87df3b4667595dc622955d251f94a159ee62ba8e06d747ab36725` |
| `artifacts/m14/l04-explanations.ssh.TunedLogitLens.278a9f76f626f8b0c6a9d9c5517c9b349f08c2d5.recovery.audit.json` | 6516 | `b6ca56bad243724d3790bfea557d182996ea261974920994704b47ce4907eeaf` |

## Verification status

Completed verification: focused L04 `120 passed`; full suite `1772 passed,
36 skipped, 39 warnings`; adversarial tuned-lens validators `39 passed`;
Ruff check and format pass; explicit strict Pyright for changed source and
test `0 errors, 0 warnings`; strict MkDocs build pass with temporary output
removed; `git diff --check` pass; and `graphify update .` pass. The final
remote CUDA validation is documented above; no additional SSH/CUDA connection
was used after that single authorized run. The direct
`base64 -d | bash -s --` recipe is NOT REUSABLE for subsequent lanes until a
future L04.8 preflight decodes into a remote temporary file, requires decoder
exit `0`, compares the decoded SHA-256 with the announced local digest, then
executes and cleans up; that replacement was not tested here.

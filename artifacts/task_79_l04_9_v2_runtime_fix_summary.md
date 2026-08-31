# Task Summary: Sprint 79 L04.9 v2 runtime failure fix

**Sprint:** Sprint 79
**Task:** L04.9 v2 Stage A real-runtime shape alignment and D0 triad recovery

## Summary of Work

Added fail-closed full-prompt shape validation to the concrete transformer
runtime, corrected v2 clean-source/corrupted-recipient position alignment for
different sequence lengths, and changed real Stage A failures from synthetic
fallbacks to truthful D0 artifacts with partial operation counters. The v2
validator and retention path now accept and independently verify incomplete
runtime-failure artifacts while preserving the strict successful D1 path.
Attempted-real D0 cleanup failures now retain counter-derived hooks remaining,
allowlisted cleanup metadata, and attested partial counters; malformed
finalizer returns are sanitized as non-promoting failures and finalizers run
once. The failed diagnostic also has a canonical sanitized no-triad sidecar
binding source/tree/raw/transport commitments and preserved-pending-owner-
exception state. The owner then verified the literal raw path, size, and hash,
deleted only that raw file under an explicit exception, and recorded verified
absence with `standard_finalize=false`; deletion is irreversible. No CUDA
rerun, holdout access, retry, standard retention finalization, or evidence
promotion was performed.

## Files Modified

* `src/latent_anything/_transformer_runtime.py` - full-prompt tensor shape
  contract and sanitized shape exception.
* `scripts/_m14_l049_v2_real_runtime.py` - independent endpoint positions,
  hidden/layer checks, recipient-shaped patch direction, canonical corrupted
  prompt, and Stage B export.
* `scripts/_m14_l049_v2_stage_a.py` - D0 failure artifact and no synthetic
  fallback after runtime exceptions.
* `scripts/m14_l049_v2_stage_a.py` - broad runtime catch and unconditional
  exact triad emission.
* `scripts/_m14_l049_v2_validate_common.py` - partial runtime attestation and
  cleanup-stage validation, counter-derived hook removal, and protocol bounds.
* `scripts/_m14_l049_v2_attestation.py` - attested cleanup hooks remaining for
  incomplete real D0 cleanup.
* `scripts/_m14_l049_v2_validate_stage_a.py` - explicit runtime-versus-semantic
  D0 discriminator and independent complete-selection checks.
* `scripts/_m14_l049_v2_schema.py` - frozen Stage A failure-kind constants.
* `tests/test_transformer_lm.py` - shortened hidden-state regression.
* `tests/test_m14_l049_v2.py` - position, sanitized failure, cleanup-finalizer,
  hook-counter, and CLI triad regressions.
* `tests/test_m14_l04_remote_postprocess.py` - D0 retain/finalize regression.
* `artifacts/m14/l04-explanations.ssh.L049V2StageA.41828c2e12e1efacb80e8cb5a0c62e4e69a688b2.sidecar.json` - canonical sanitized
  failed-attempt no-triad sidecar, now recording owner-exception deletion
  with reason `no_triad_bundle_status_66`.
* `artifacts/m14/l04-explanations.ssh.L049V2StageA.3b15627585a0fc07e28c0f8b5d0118630f3ded5d.sidecar.json` - sanitized
  sidecar for the new semantic-selection validator misclassification.
* `CHANGELOG.md`, `docs/sprint-plans/sprint-79.md`, and
  `.agents/memory/lessons-learned.md` - truthful implementation record.

## Testing

* Focused transformer/v2/postprocess suite: `99 passed`.
* Complete M14 L04 family before the final cleanup schema adjustment:
  `360 passed, 1 skipped`; the final focused run covers all modified cleanup
  paths.
* Scoped Ruff/format and Pyright: passed (`0 errors, 0 warnings, 0
  informations`).
* Graphify update and `git diff --check`: passed.

## Additional Notes

The previously captured raw diagnostic was verified at 7314 bytes with
SHA-256 `9d3682dbe0f5faa0a65881f4f79d5d946e323b5e959b29650df96355a66e2f6f`
and deleted under the explicit owner exception; the sidecar records its
verified absence and deletion is irreversible.

## Follow-up real attempt

One new exact-SHA real-CUDA Stage A attempt reached aligned scoring but
returned a semantic D0 gate failure. The CLI incorrectly rejected its complete
selection as a runtime-failure shape before writing the triad; no candidate or
metrics were retained in the raw evidence. The new discriminator fix separates
`runtime_exception`/incomplete selection from `semantic_gate`/complete
selection, including semantic no-consensus, while keeping D1 strict.

The new failed raw capture was verified before owner-exception deletion:
6008 bytes, SHA-256
`757af5cce5b4e8aa4c5b476ecc52d69ae192423179c23b3fc148510a8eafc212`.
Its sanitized sidecar records the validator misclassification, no triad or
audit, and no promotion. The sidecar now records
`deleted_by_owner_exception`, `standard_finalize=false`, the exact reason
`no_triad_bundle_status_66/semantic_gate_d0_validator_misclassification`,
the previous sidecar digest
`a6a2afe995abdaa8996e202f51851813f6a7f7ca580715ff90109885afe39fe9`, and
verified post-delete absence. The updated sidecar digest is
`02a355cd6dffe6d85e07a0cce2175c126a4f512056bccb88408cadf744ecde93`.

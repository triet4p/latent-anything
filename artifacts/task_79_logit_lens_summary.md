# Sprint 79 Logit-Lens Native-State Semantics

**Sprint:** Sprint 79
**Task:** Resolve native hidden-state index-12/direct-logit-lens parity

## Summary of Work

Confirmed from the pinned Transformers 4.57.6 GPT-2 implementation that native
hidden-state index 12 is the terminal state after `transformer.ln_f`, while
indices 0 through 11 are pre-final-normalization states. Added a private,
defaulted normalization control to the logit-lens helper and made runtime
normalization depend on the native terminal index. Added an accurate affine
LayerNorm fake, exact intermediate/final logit parity tests, capture-subset
coverage, final probability/rank checks, and numerical parity in the real
network test. No public protocol, export, result schema, L03 artifact, ledger,
queue, or L04 claim changed.

## Files Modified

* [`src/latent_anything/_transformer_analysis.py`](../src/latent_anything/_transformer_analysis.py) - Add private `apply_final_norm` control and document post-normalized terminal states.
* [`src/latent_anything/_transformer_runtime.py`](../src/latent_anything/_transformer_runtime.py) - Skip duplicate final normalization only at the native terminal index.
* [`src/latent_anything/integrations/transformer_lm.py`](../src/latent_anything/integrations/transformer_lm.py) - Update private wrapper and direct-lens semantics documentation.
* [`tests/test_transformer_lm.py`](../tests/test_transformer_lm.py) - Add semantically accurate post-`ln_f` fake and exact parity/subset/rank assertions.
* [`tests/test_transformer_lm_network.py`](../tests/test_transformer_lm_network.py) - Replace shape-only final parity with numerical allclose.
* [`CHANGELOG.md`](../CHANGELOG.md) - Record the user-visible correctness fix.
* [`docs/sprint-plans/sprint-79.md`](../docs/sprint-plans/sprint-79.md) - Close the dedicated follow-up without promoting L11.
* [`.agents/memory/lessons-learned.md`](../.agents/memory/lessons-learned.md) - Record how shape-only parity masked double normalization.

## Testing

* **Test Files:** [`tests/test_transformer_lm.py`](../tests/test_transformer_lm.py), [`tests/test_transformer_lm_network.py`](../tests/test_transformer_lm_network.py)
* **Status:** Passed
* **Execution Command:** `uv run pytest tests/test_transformer_lm.py tests/test_transformer_lm_network.py -q` (48 passed, 8 expected network skips); `uv run pyright src tests` (0 errors); `uv run pytest -q` (1632 passed, 36 skipped, 39 warnings); changed-file Ruff check/format (passed).

## Additional Notes

The L03 artifact consumes only captured hidden states, not lens results or
logits, so its feature digest and accepted metrics do not require rerunning.

## Remote CUDA verification

Attempt 1 stopped before model execution because the isolated base environment
omitted the confirmed `transformers` optional extra. Its sanitized failure
record is [`task_79_logit_lens_remote_cuda_attempt1.json`](task_79_logit_lens_remote_cuda_attempt1.json).
Attempt 2 used `uv sync --locked --extra transformers` and direct authenticated
PowerShell `ssh.exe` transport into a fresh remote clone. The exact network
command passed all 8 selected tests on the RTX 4060 Ti with CUDA 12.8,
including numerical final-layer parity, intervention, and hook cleanup. See
the final sanitized record
[`task_79_logit_lens_remote_cuda_verification_final.json`](task_79_logit_lens_remote_cuda_verification_final.json)
and its SHA-256 transcript digest. Both raw local captures were verified and
then deleted according to the repository's sanitized-audit convention; the
failure/pass metadata and digests remain. No public protocol/schema, L03
artifact, evidence count, ledger row, queue item, or L04 claim changed.

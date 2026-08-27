# Sprint 79 Transformer Hook Structured-Output Fix

## Summary

Added a private shared hook-output seam for module returns that are either a
Tensor, an exact plain tuple with the primary Tensor at position 0, or an exact
plain list with the primary Tensor at position 0. Tuple auxiliary values retain
their identities; list reconstruction copies the list without mutating the
module's original return. Mapping, empty, non-Tensor-primary, and custom
tuple/list outputs fail closed.

Migrated activation capture, Integrated Gradients, TCAV, and SAE evaluation
hooks to preserve structured outputs during intervention while retaining
observation, shape, dtype/device, gradient, and cleanup behavior.

## Changed files

- `src/latent_anything/_hook_output.py`
- `src/latent_anything/capture.py`
- `src/latent_anything/integrated_gradients.py`
- `src/latent_anything/_tcav_model.py`
- `src/latent_anything/sae_evaluation.py`
- `tests/test_latent_anything/test_capture.py`
- `tests/test_integrated_gradients.py`
- `tests/test_tcav.py`
- `tests/test_sae_evaluation.py`
- `tests/test_transformer_lm.py`
- `tests/test_transformer_lm_network.py`
- `src/latent_anything/integrations/transformer_lm.py`
- `CHANGELOG.md`
- `.agents/memory/lessons-learned.md`
- `docs/sprint-plans/sprint-79.md`
- `docs/EVIDENCE_GAP_PLAN.md`
- `docs/EVIDENCE_LEDGER.md`
- `docs/M14_REAL_SYSTEM_VALIDATION.md`
- `artifacts/task_79_l03_phase_b_summary.md`
- `artifacts/task_78.38_gap_map.json`
- `artifacts/task_79.1_execution_queue.json`
- `artifacts/task_79_transformer_hook_remote_verification_attempt1.md`
- `artifacts/task_79_transformer_hook_remote_verification_attempt1.json`
- `artifacts/task_79_transformer_hook_remote_verification.md`
- `artifacts/task_79_transformer_hook_remote_verification.json`

## Validation

- Focused transformer/network selector tests: 47 passed, 8 expected network skips.
- Ruff check and format: passed on changed source/tests.
- Strict Pyright: 0 errors, 0 warnings, 0 informations.
- Full pytest: 1631 passed, 36 skipped (39 expected warnings).
- Graphify update after evidence materialization: 11,507 nodes and 22,274
  edges (the incremental rebuild reported its expected community-label refresh
  warning).
- Remote exact-SHA strict-CUDA verification: `9ebecfa` clone guard passed;
  `LATENT_ANYTHING_RUN_NETWORK=1 LATENT_ANYTHING_NETWORK_DEVICE=cuda uv run
  pytest tests/test_transformer_lm_network.py -m network -q` produced 8 passed,
  5 deselected, including native-index-7 intervention and hook cleanup.
- Remote cleanup: disposable checkout/cache and exact `pytest`/`uv` process
  audits passed; attempt-1 and attempt-2 evidence retain their reviewed
  SHA-256 digests.
- Evidence validator remains 33/63 core and 33/65 overall; no D2 status/count
  or L03 artifact metric changed.
- Final closure commit SHA: to be recorded after the final diff review.
- No L04 or native index-12/direct-logit-lens implementation is included.

## API and architecture impact

No public API, protocol, export, or ADR changes. The helper remains in a
private underscore-prefixed module; the existing Tensor callback contract is
unchanged. Native `output_hidden_states=True` remains the transformer
observation path.

## Indexing and network-device clarification

`HiddenStateIntervention.layer` is a zero-based transformer block index mapped
to `transformer.h.<layer>`. Hugging Face GPT-2 records that block's output at
native hidden-state index `layer + 1`; the network oracle now checks native
index 7 for intervention at block 6 and confirms native index 6 is unchanged.
Real network fixtures default to CPU and accept the opt-in
`LATENT_ANYTHING_NETWORK_DEVICE=auto|cuda` selector, with explicit CUDA mode
failing when CUDA is unavailable. The initial 6/2 tuple-return failure and
intermediate 7/1 indexing-oracle failure remain preserved as history; the
final exact-SHA strict-CUDA run passed 8/8. The structured hook/output cleanup
blocker is resolved by `16db80f` + `9ebecfa`. Native hidden-state index-12 /
direct-logit-lens final-LayerNorm parity remains a separate open Sprint 79
follow-up.

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

## Validation

- Focused transformer/network selector tests: 47 passed, 8 expected network skips.
- Ruff check and format: passed on changed source/tests.
- Strict Pyright: 0 errors, 0 warnings, 0 informations.
- Full pytest: 1631 passed, 36 skipped (39 expected warnings).
- Graphify update: 11,499 nodes and 22,265 edges.
- Final commit SHA: to be recorded after the final diff review.
- No remote run, model download, L03 evidence/ledger/queue edit, or L04 work.

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
failing when CUDA is unavailable.

No remote rerun was performed in this implementation turn; the prior 7/1
remote verification remains preserved as separate uncommitted evidence.

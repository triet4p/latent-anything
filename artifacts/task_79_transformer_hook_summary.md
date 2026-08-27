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
- `.agents/memory/lessons-learned.md`

## Validation

- Focused consumers: 152 passed, 2 skipped.
- Ruff check and format: passed on all changed source/tests.
- Strict Pyright: passed on all changed source/tests.
- Full pytest: 1626 passed, 36 skipped (39 expected warnings).
- Final commit SHA: to be recorded after the final diff review.
- No remote run, model download, L03 evidence/ledger/queue edit, or L04 work.

## API and architecture impact

No public API, protocol, export, or ADR changes. The helper remains in a
private underscore-prefixed module; the existing Tensor callback contract is
unchanged. Native `output_hidden_states=True` remains the transformer
observation path.

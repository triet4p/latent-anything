# Task Summary: Sprint 14 — HiddenStateAdapter & ModelAdapter Protocol Freeze

**Sprint:** Sprint 14 (Round 11)
**Task:** HiddenStateAdapter (ModelAdapter #3, mode ii: no-explicit-latent) + freeze ModelAdapter/DecodableAdapter Protocols

## Summary of Work
Implemented `HiddenStateAdapter` — ModelAdapter #3, demonstrating mode (ii) no-explicit-latent where the hidden-state activations *are* the latent representation, with no decoder. Created frozen `ModelAdapter` and `DecodableAdapter` Protocols in `adapters/protocols.py`, splitting the adapter surface into a universal base (`encode` + `latent_space`) and a decodable extension (`+decode`). Removed the superseded `_ModelAdapterBase` (`_base.py`). Updated `ActivationPatch` to require `DecodableAdapter` with a runtime `isinstance` guard. Added 29 tests and an end-to-end demo (PCA + UMAP visualization; no decode story).

## Files Modified / Created

### New files
- `src/latent_anything/adapters/protocols.py` — Frozen `ModelAdapter` Protocol (encode + latent_space) and `DecodableAdapter` Protocol (+decode). Split reflects the core evidence: decode is NOT universal.
- `src/latent_anything/adapters/hidden_state.py` — `HiddenStateAdapter`: fixed random 2-layer ReLU MLP, encode returns hidden activations `(n_samples, hidden_dim)`, no decode method, `latent_space` with `exposure_mode="hidden_state"` metadata.
- `tests/test_latent_anything/test_hidden_state.py` — 29 tests covering construction, latent_space, encode shape/determinism/nonlinearity/validation, ModelAdapter conformance, DecodableAdapter non-conformance, ActivationPatch rejection, and reproducibility.
- `scripts/end_to_end_hidden_state_demo.py` — End-to-end demo: synthetic 8D clusters → HiddenStateAdapter encode → PCA/UMAP 2D visualization → no decode story.

### Modified files
- `src/latent_anything/adapters/__init__.py` — Export `HiddenStateAdapter`, `ModelAdapter`, `DecodableAdapter`; update docstring to reflect frozen Protocols.
- `src/latent_anything/adapters/vae.py` — Update module docstring to reference `ModelAdapter` and `DecodableAdapter` Protocols.
- `src/latent_anything/adapters/random_projection.py` — Update module docstring to reference Protocols.
- `src/latent_anything/methods/activation_patch.py` — Change adapter type from `Any` to `DecodableAdapter` with runtime `isinstance` guard. Remove pyright ignore comments.
- `.agents/memory/decisions.md` — Add Sprint 14 ADR reconciliation: ModelAdapter 3-mode ADR → **validated** (modes i and ii confirmed).

### Removed files
- `src/latent_anything/adapters/_base.py` — Superseded by frozen `ModelAdapter`/`DecodableAdapter` Protocols.

## Testing
- **Test File:** `tests/test_latent_anything/test_hidden_state.py` (29 new tests)
- **Full Suite:** 293 passed (all existing tests + 29 new)
- **Lint:** `ruff check` — clean
- **Format:** `ruff format` — clean
- **Types:** `pyright` — 0 errors (strict mode)
- **Execution:** `uv run pytest -v` — 293 passed in 72.73s
- **Demo:** `uv run python scripts/end_to_end_hidden_state_demo.py` — plot saved to `artifacts/hidden_state_demo_plot.png`

## ADR Status
| ADR | Status | Evidence |
|---|---|---|
| ModelAdapter 3-mode | **validated** (was pending) | Modes (i) and (ii) confirmed by code; mode (iii) pending Sprint 16 |
| LatentSpace geometry-keyed | validated (no change) | HiddenStateAdapter's Euclidean LatentSpace with metadata works |
| Geometry-dispatch | validated (no change) | Not exercised this sprint |

## Rule of Three Checkpoint
| Check | Status |
|---|---|
| ModelAdapter instances | VAE (#1), RandomProjection (#2), HiddenStateAdapter (#3) |
| Philosophies differ? | Yes — trained explicit latent, fixed explicit projection, no-explicit hidden activation |
| Rule branch | Instance #3 → freeze Protocol and migrate |
| ADR impact | Mode (ii) confirmed; mode (iii) still pending Sprint 16 |

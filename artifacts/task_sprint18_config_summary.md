# Task Summary: Sprint 18 — Registry-Backed Config Instantiation

**Sprint:** Sprint 18 (Round 15)
**Task:** Add pydantic config specs for registry-backed object construction

## Summary of Work

Implemented a pydantic v2 config layer on top of the in-process registry (Sprint 17). The new `config.py` module provides an `ObjectSpec` pydantic model with `kind`/`name`/`params` fields, a `build_from_config(spec)` function that resolves registry entries and instantiates them, and recursive nested spec resolution for adapter-in-method references (e.g., `ActivationPatch` with a nested `VAE` spec). Validation errors are descriptive: `KeyError` with available names for unknowns, `ValueError` for kind mismatches, `TypeError` with failing params for instantiation failures. Config instantiation is deliberately registry-local and narrow — no Hydra, no Pipeline/workflow language.

## Files Modified

- [src/latent_anything/config.py](src/latent_anything/config.py) — New file: pydantic v2 `ObjectSpec` model, `build_from_config`, `_resolve_params`, `build_from_dict` convenience wrapper.
- [src/latent_anything/__init__.py](src/latent_anything/__init__.py) — Added `ObjectSpec`, `build_from_config`, `build_from_dict` to public exports and `__all__`.
- [tests/test_latent_anything/test_config.py](tests/test_latent_anything/test_config.py) — New file: 36 tests covering ObjectSpec construction, adapter/method building, nested specs, error cases, custom registries, and all six required classes.
- [scripts/end_to_end_config_demo.py](scripts/end_to_end_config_demo.py) — New file: demo script building VAE, PCA, Lerp, ActivationPatch (with nested VAE) from config specs.
- [CHANGELOG.md](CHANGELOG.md) — Added Sprint 18 entries under Unreleased.
- [docs/PLAN.md](docs/PLAN.md) — Sprint 18 moved from Planned to Completed.
- [docs/sprint-plans/sprint-18.md](docs/sprint-plans/sprint-18.md) — All 9 tasks marked [x].
- [pyproject.toml](pyproject.toml) — Added `pydantic>=2.0,<3.0` dependency.

## Testing

- **Test File:** [tests/test_latent_anything/test_config.py](tests/test_latent_anything/test_config.py)
- **Status:** 465 passed (36 new config tests + 429 existing)
- **Execution Command:** `uv run pytest -v`
- **Lint:** `ruff check` — clean
- **Format:** `ruff format --check` — clean
- **Type Check:** `pyright strict` — 0 errors across `src/`

## Additional Notes

- Config instantiation is deliberately **instance #1**: narrow, registry-local, no generalization into Pipeline/workflow language.
- Nested specs work as both `ObjectSpec` instances and plain dicts (via `from_dict` / `build_from_dict`).
- The `_ParamsValidator` internal class was removed after review — params validation is handled by the factory call itself.
- Next planned: Sprint 19 — convert built-in adapters/methods to registry entries and prove behavior parity.

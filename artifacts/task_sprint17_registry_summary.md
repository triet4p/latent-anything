# Task Summary: Sprint 17 — In-Process Registry

**Sprint:** Sprint 17 (Round 14)
**Task:** Plugin Extraction — In-process registry for built-in adapters and methods.

## Summary of Work
Implemented an in-process `Registry` class (`OrderedDict`-backed) as registry instance #1, following the Rule of Three constraint (no Python entry points yet). Registered all 10 built-in classes (4 adapters + 3 Layer A methods + 3 Layer B methods) with deterministic insertion-order iteration. Added convenience helpers (`register`, `lookup`, `list_entries`) defaulting to a `GLOBAL_REGISTRY` singleton. Extended the public package surface in `__init__.py` with `Registry`, `RegistryEntry`, `GLOBAL_REGISTRY`, and the three convenience functions.

## Files Modified

### New files
- [src/latent_anything/registry.py](src/latent_anything/registry.py) — Core registry module with `Registry` class, `RegistryEntry` dataclass, kind constants, `GLOBAL_REGISTRY` singleton, convenience helpers, and automatic built-in class registration at import time.
- [tests/test_latent_anything/test_registry.py](tests/test_latent_anything/test_registry.py) — 48 tests covering construction, registration (including duplicate guard), lookup (including missing-name guard), kind-filtered listing, factory retrieval, RegistryEntry invariants, convenience helpers, GLOBAL_REGISTRY built-in verification, error cases, and no-breakage verification.
- [scripts/end_to_end_registry_demo.py](scripts/end_to_end_registry_demo.py) — Standalone PEP 723 demo script listing all registered entries grouped by kind, with lookup, duplicate-guard, and missing-name guard demonstrations.

### Modified files
- [src/latent_anything/__init__.py](src/latent_anything/__init__.py) — Added `Registry`, `RegistryEntry`, `GLOBAL_REGISTRY`, `list_entries`, `lookup_entry`, `register_entry` to the public package surface.
- [docs/PLAN.md](docs/PLAN.md) — Marked Sprint 17 as completed under Milestone 4; moved from Active to Completed Sprints list; updated backlog.
- [docs/sprint-plans/sprint-17.md](docs/sprint-plans/sprint-17.md) — Marked all 10 tasks as `[x]`.
- [CHANGELOG.md](CHANGELOG.md) — Added Sprint 17 entries under [Unreleased].

## Testing
- **Test File:** [tests/test_latent_anything/test_registry.py](tests/test_latent_anything/test_registry.py)
- **Status:** 48/48 passed
- **Execution Command:** `uv run pytest tests/test_latent_anything/test_registry.py -v`
- **Full suite:** 429/429 passed across all 15 test files (no regressions)
- **Lint:** `ruff check` — All checks passed
- **Format:** `ruff format` — 3 files already formatted
- **Type check:** `pyright strict` — 0 errors (registry.py, test_registry.py, demo)

## ADR Reconciliation
No ADR status changes for Sprint 17. This is a **plugin-extraction infrastructure round** that adds no new adapter, method, or geometry instances. All three validated ADRs (geometry-keyed `LatentSpace`, geometry-dispatch, `ModelAdapter` 3-mode) remain in their current validated state. The Rule of Three §4a is not applicable — this is not an instance-adding round.

## Additional Notes
- **`__bool__` lesson:** `Registry.__len__` is used by Python for truthiness by default, so `registry or GLOBAL_REGISTRY` silently fell through to the global singleton when the registry was empty. Added `__bool__` override (always returns `True`) and changed `or` to explicit `is not None` check in the convenience `register` function.
- **pyright strict on lambda:** Untyped lambdas trigger `reportUnknownLambdaType` in strict mode. Used named inner functions with typed signatures instead.
- **pyright strict on `field(default_factory=dict)`:** The `dict` factory is inferred as `dict[Unknown, Unknown]`. Changed to `field(default_factory=lambda: {})` with explicit `dict[str, Any]` annotation.
- **No entry points:** Per Sprint 17's design constraint, the registry uses only local class references. Python `importlib.metadata` entry points will be considered at Sprint 18/19 when config-driven instantiation is added.

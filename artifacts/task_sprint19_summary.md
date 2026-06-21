# Sprint 19 Summary — Registry-first built-ins

**Goal:** Convert built-in adapters and methods to registry-first built-ins and prove behavior parity against direct imports. Complete the practical plugin-extraction baseline before entry points.

**Commit:** `registry-first-builtins`

## Changes

### New files
- `src/latent_anything/_plugin_builtins.py` — Single stable import location where all built-in adapters and methods are registered into `GLOBAL_REGISTRY`. Decouples `registry.py` from concrete class dependencies. Documents the internal plugin extraction contract (5 rules).
- `tests/test_latent_anything/test_parity.py` — 22 tests proving registry constructor vs direct import constructor produce the same type, for all 10 built-in classes (4 adapters + 3 method_a + 3 method_b). Also verifies `factory` identity (factory IS the class).
- `tests/test_latent_anything/test_demo_smoke.py` — 15 tests verifying every `scripts/end_to_end_*.py` demo's core imports and helpers still work after the registry refactoring.

### Modified files
- `src/latent_anything/registry.py` — Removed all adapter/method class imports and the registration block at module bottom. Now pure infrastructure: `Registry` class, kind constants, convenience helpers, and `GLOBAL_REGISTRY` singleton. No knowledge of concrete classes.
- `src/latent_anything/__init__.py` — Added `from latent_anything import _plugin_builtins` to trigger built-in registration on package import, before any registry-dependent modules (like `config.py`).
- `docs/sprint-plans/sprint-19.md` — All 9 tasks marked `[x]`.
- `CHANGELOG.md` — Sprint 19 entries added under `[Unreleased]`.
- `docs/PLAN.md` — Sprint 19 moved to Completed Sprints; Milestone 4 marked complete; plugin extraction removed from backlog.

### No changes to
- Public API surface (`__all__`, `__init__.py` exports)
- Any adapter or method implementation
- Registry infrastructure (`Registry` class, `GLOBAL_REGISTRY`, kind constants, helpers)
- Direct import paths (`from latent_anything.adapters import VAE`, etc.)

## Verification

| Gate | Result |
|---|---|
| `ruff check` | All checks passed |
| `ruff format --check` | 43 files already formatted |
| `pyright strict` (src + new test files) | 0 errors |
| `pytest` | **502 passed** (465 existing + 22 parity + 15 demo smoke) |

## ADR reconciliation

No new ADR needed. This is an infrastructure-only round — no new adapter, method, or geometry instances. The registry + config architecture is already consistent with all validated ADRs. All three 2026-06-16 ADRs remain `validated` (unchanged).

## Entry-point decision

**Python entry points are NOT yet justified.** Per the Rule of Three principle:
- Only one registry instance exists (in-process `GLOBAL_REGISTRY`).
- No external plugins exist — all entries are built-in classes.
- Python `importlib.metadata` entry points would be warranted when a second registry instance (external plugin) demands them.

The `_plugin_builtins.py` docstring explicitly marks the module as "entry-point ready" — when external plugins arrive, a separate plugin loader will populate a different `Registry` instance. Built-in registrations stay in `_plugin_builtins.py` regardless.

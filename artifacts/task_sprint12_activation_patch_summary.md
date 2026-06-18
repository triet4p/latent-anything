# Task Summary: Sprint 12 — ActivationPatch (B-Method #3) + BMethod Protocol Freeze

**Sprint:** Sprint 12
**Task:** ActivationPatch (B-Method #3, model-mediated data→data) → freeze `BMethod` Protocol, migrate Lerp + SteeringVector

## Summary of Work

Implemented `ActivationPatch`, the third Layer B (Manipulation) method with a fundamentally different philosophy from Lerp and SteeringVector — it operates **through a ModelAdapter** (encode → patch → decode) and outputs data-space arrays, not latent-space trajectories. This triggered the **Rule of Three freeze** for the `BMethod` Protocol: the frozen Protocol was extracted to `b_protocols.py`, the UNSTABLE `_b_base.py` was removed, and both Lerp and SteeringVector were migrated (docstrings, `is_fitted` for Lerp, generic `apply_trajectory(**kwargs)` for Lerp). Three distinct B-Method patterns are now proven: stateless latent→latent (Lerp), stateful latent→latent (SteeringVector), and model-mediated data→data (ActivationPatch).

## Rule-of-Three Outcome

| Check | Status |
|---|---|
| B-Method instances | Lerp (#1, stateless latent→latent), SteeringVector (#2, stateful latent→latent), ActivationPatch (#3, model-mediated data→data) |
| Philosophies differ? | **Yes** — three genuinely different patterns |
| Rule branch | **Instance #3, different philosophy** → **Freeze `BMethod` Protocol, migrate** |
| `Method` Protocol? | **Unchanged** — remains Layer A stateful dim-reduction only. A/B/C unification disproven. |
| `_b_base.py`? | **Removed** — UNSTABLE sketch superseded by frozen `BMethod` Protocol in `b_protocols.py` |
| Lerp migrated? | Added `is_fitted` (always True) + generic `apply_trajectory(**kwargs)`. Docstring notes conformance. |
| SteeringVector migrated? | Already conforms — docstring updated. |
| Public surface? | `BMethod` and `ActivationPatch` added to `methods/__init__.py` `__all__` |

## Files Modified

- [src/latent_anything/methods/activation_patch.py](src/latent_anything/methods/activation_patch.py) — New: `ActivationPatch` class (B-Method #3)
- [src/latent_anything/methods/b_protocols.py](src/latent_anything/methods/b_protocols.py) — New: Frozen `BMethod` Protocol
- [src/latent_anything/methods/_b_base.py](src/latent_anything/methods/_b_base.py) — **Removed**: superseded by frozen `BMethod` Protocol
- [src/latent_anything/methods/lerp.py](src/latent_anything/methods/lerp.py) — Added `is_fitted` property, generic `apply_trajectory(**kwargs)` method, updated docstrings
- [src/latent_anything/methods/steering.py](src/latent_anything/methods/steering.py) — Updated docstrings to note `BMethod` conformance
- [src/latent_anything/methods/\_\_init\_\_.py](src/latent_anything/methods/__init__.py) — Added `ActivationPatch` and `BMethod` to imports and `__all__`
- [scripts/end_to_end_activation_patch_demo.py](scripts/end_to_end_activation_patch_demo.py) — New: end-to-end demo with two scenarios
- [tests/test_latent_anything/test_activation_patch.py](tests/test_latent_anything/test_activation_patch.py) — New: 28 tests for ActivationPatch + BMethod Protocol
- [tests/test_latent_anything/test_lerp.py](tests/test_latent_anything/test_lerp.py) — Added tests for `is_fitted` and `apply_trajectory`
- [docs/PLAN.md](docs/PLAN.md) — Sprint 12 → Completed, Sprint 13 → Active
- [CHANGELOG.md](CHANGELOG.md) — Updated `[Unreleased]` with Sprint 12 entries
- [.agents/memory/decisions.md](.agents/memory/decisions.md) — ADR reconciliation entry appended

## Testing

- **Test Files:**
  - [tests/test_latent_anything/test_activation_patch.py](tests/test_latent_anything/test_activation_patch.py)
  - [tests/test_latent_anything/test_lerp.py](tests/test_latent_anything/test_lerp.py)
- **Status:** 245 passed (227 existing + 28 new ActivationPatch tests + new Lerp tests)
- **Execution Command:** `uv run pytest tests/ -v`
- **Tooling Gate:** `ruff check` — clean; `ruff format` — clean; `pyright` strict — clean

## Additional Notes

- `__call__` is deliberately **NOT** in the `BMethod` Protocol — signatures genuinely differ across the three instances (Lerp: `(a, b, t)`, SteeringVector: `(latent, strength)`, ActivationPatch: `(input_data)`). Forcing a unified `__call__` would violate INCREMENTAL.md §3 ("design from imagination").
- `apply_trajectory` return type is `Trajectory | np.ndarray` in the Protocol — Lerp/SteeringVector return `Trajectory` (latent→latent), ActivationPatch returns `np.ndarray` (latent→data).
- Three distinct B-Method patterns now proven — the aspirational "unified interface for all A/B/C methods" from ARCHITECTURE.md §2 is **disproven by code**. A, B, and C layers have genuinely different method shapes.

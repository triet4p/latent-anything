# Task Summary: Sprint 11 — SteeringVector (B-Method #2, Stateful)

**Sprint:** Sprint 11
**Task:** Round 8 — SteeringVector B-Method #2 + `_BMethodBase` internal sketch

## Summary of Work
Implemented the `SteeringVector` class — B-Method #2 (stateful, fit from contrast) for Layer B (Manipulation). Unlike Lerp (#1, stateless pure function), SteeringVector has a `fit(positives, negatives)` phase that learns a unit steering direction from contrast pairs, then a `__call__(latent, strength)` phase that applies the steering. This is the first stateful B-Method.

Per Rule of Three §4a (instance #2), sketched the internal `_BMethodBase` ABC to capture the shared B-Method surface (`__call__`, `space`, `apply_trajectory`), marked **UNSTABLE**. Neither `Lerp` nor `SteeringVector` inherits from it — they conform structurally via duck-typing. The `Method` Protocol remains unchanged (still `fit`/`transform`/`fit_transform` for Layer A).

End-to-end demo script covers two scenarios: Euclidean steering with PCA visualization, and spherical (unit-norm) steering with geometry-aware normalization.

## Files Modified

### Source
- [`src/latent_anything/methods/steering.py`](../src/latent_anything/methods/steering.py) — New `SteeringVector` class (B-Method #2, stateful)
- [`src/latent_anything/methods/_b_base.py`](../src/latent_anything/methods/_b_base.py) — New `_BMethodBase` internal ABC, marked UNSTABLE
- [`src/latent_anything/methods/__init__.py`](../src/latent_anything/methods/__init__.py) — Export `SteeringVector`, add to `__all__`

### Tests
- [`tests/test_latent_anything/test_steering.py`](../tests/test_latent_anything/test_steering.py) — 32 tests covering construction, fit, direction, call, apply_trajectory, spherical normalization, edge cases, no-torch leakage

### Demo
- [`scripts/end_to_end_steering_demo.py`](../scripts/end_to_end_steering_demo.py) — End-to-end demo: Euclidean + spherical scenarios with 1×2 matplotlib visualization

### Documentation
- [`docs/PLAN.md`](../docs/PLAN.md) — Sprint 11 → Active, Sprint 10 → Completed
- [`CHANGELOG.md`](../CHANGELOG.md) — Added SteeringVector, `_BMethodBase`, demo entries
- `.agents/memory/decisions.md` — ADR reconciliation: `LatentSpace` geometry-keyed ADR validated, geometry-dispatch ADR validated, `ModelAdapter` 3-mode ADR pending

## Testing
- **Test File:** [`tests/test_latent_anything/test_steering.py`](../tests/test_latent_anything/test_steering.py)
- **Status:** 32/32 passed
- **Execution Command:** `uv run pytest tests/test_latent_anything/test_steering.py -v`
- **Tooling Gate:** `ruff check` clean, `ruff format` clean, `pyright --strict` clean

## Rule-of-Three Checkpoint

| Check | Status |
|---|---|
| B-Method instances | Lerp (#1, stateless), SteeringVector (#2, stateful) |
| Rule branch | **Instance #2** → sketch internal `_BMethodBase`, mark UNSTABLE |
| `Method` Protocol? | Unchanged — still `fit`/`transform`/`fit_transform`. Neither B-Method conforms. |
| `_BMethodBase` internal? | Sketched at `methods/_b_base.py`, marked UNSTABLE. Covers `__call__` + `space` + `apply_trajectory`. |
| B-Method freeze | At B-Method #3 (activation patching, Sprint 12) — when stateless + stateful + hook-based patterns are all proven |

## ADR Reconciliation

- **`LatentSpace` geometry-keyed ADR** — **Validated**: SteeringVector optionally accepts a `LatentSpace` and uses `space.normalize()` for geometry-aware post-steer normalization (e.g. project back to sphere).
- **Geometry-dispatch ADR** — **Validated**: Exercised when `space.geometry == "unit_norm"` triggers `space.normalize()` after steering.
- **`ModelAdapter` 3-mode ADR** — **Pending**: Not touched by this Layer B increment.

## Additional Notes
- All input/output is `numpy.ndarray` (single-point) or `Trajectory` (sequence). No torch.
- Steering direction is unit norm by default — ensures consistent behavior regardless of mean difference magnitude.
- Geometry-aware normalization is opt-in via `space` parameter.
- Simple mean-difference algorithm (not PCA-based) — follows activation engineering literature standard approach.

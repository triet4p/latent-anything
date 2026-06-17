# Task Summary: Sprint 9 — Geometry Case #2 (unit_norm/spherical)

**Sprint:** Sprint 9 (Round 6)
**Task:** Geometry case #2 — unit_norm (spherical) LatentSpace, ADR validation

## Summary of Work

Added the second geometry case (`unit_norm`/spherical) to `LatentSpace`, proving the geometry-keyed and geometry-dispatch ADR hypotheses with real code. Key changes:

1. **`LatentSpace.__init__`** — `geometry` moved from class-level `str = "euclidean"` to instance-level parameter, validated against `{"euclidean", "unit_norm"}` at construction. Default remains `"euclidean"` for full backward compatibility.
2. **`validate_point()`** — extended for `unit_norm` geometry: additionally checks `||point|| ≈ 1`.
3. **`distance(a, b) -> float`** — new method dispatching on `self.geometry`: Euclidean (`||a-b||`) or angular (`arccos(clip(a·b, -1, 1))`).
4. **`interpolate(a, b, t) -> np.ndarray`** — new method dispatching on `self.geometry`: lerp for Euclidean, proper slerp for spherical. Handles edge case `sin(ω) ≈ 0` (falls back to lerp).
5. **`normalize(point) -> np.ndarray`** — new method: Euclidean returns copy; spherical projects to unit sphere. Zero vector raises for spherical.
6. **End-to-end demo** (`scripts/end_to_end_spherical_demo.py`) — synthetic unit-norm data; demonstrates validate_point, angular distance, slerp-vs-lerp, and normalization. Saves 1×3 matplotlib figure (3D scatter + lerp path + slerp path).
7. **35 new tests** covering geometry construction, validate_point for unit_norm, distance (6 cases including known angles), interpolate/slerp (midpoint, endpoints, unit-norm invariance, edge cases), and normalize.

All geometry ops are pure numpy — zero new dependencies. Dispatch is inline `if/elif` (no `GeometryProtocol` — instance #3 needed). All 116 existing tests pass unchanged.

## Files Modified

- [src/latent_anything/latent_space.py](src/latent_anything/latent_space.py) — instance-level geometry, validate_point extension, new distance/interpolate/normalize methods
- [tests/test_latent_anything/test_latent_space.py](tests/test_latent_anything/test_latent_space.py) — 35 new tests across 7 test classes
- [scripts/end_to_end_spherical_demo.py](scripts/end_to_end_spherical_demo.py) — new end-to-end demo with matplotlib visualization

## Testing

- **Test File:** [tests/test_latent_anything/test_latent_space.py](tests/test_latent_anything/test_latent_space.py)
- **Status:** 151 total tests (116 existing + 35 new), all passed
- **Tooling:** `ruff check` clean, `ruff format` clean, `pyright strict` — 0 errors
- **Execution Command:** `uv run pytest tests/ -v`

## Additional Notes

- **Rule of Three §4a:** LatentSpace #2 (unit_norm/spherical) confirms two geometry cases coexist in the same class. Dispatch is inline `if/elif` — no `GeometryProtocol` yet (instance #3 needed for extraction).
- **Geometry enum** validated at construction via `_GEOMETRIES` frozenset.
- **Trajectory NOT modified** — geometry-aware ops live on `LatentSpace`, keeping Trajectory stable.
- **No new dependencies** — all ops are pure numpy.
- **ADR impact:** Two ADRs move from `pending` → `validated` (see decisions.md).

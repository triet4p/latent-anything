# Sprint 9 Plan

## Sprint Goal
Increment thứ sáu (Round 6): thêm **geometry case thứ hai** (`unit_norm`/spherical) vào `LatentSpace`, buộc geometry mang theo **metric** (distance, interpolation, validation phân biệt euclidean vs spherical). Đây là sprint cuối của Giai đoạn 1 — **validate hai ADR**: `LatentSpace` geometry-keyed và geometry-dispatch. Rule of Three cho `LatentSpace`: instance #2 → giữ dispatch inline, chưa extract `GeometryProtocol`.

## Đây là sprint ADR-validation, không phải sprint interface-freeze

Sprint 9 khác các sprint trước: mục tiêu chính là **xác nhận giả thuyết ADR bằng code**, không phải thêm instance mới để tiến tới freeze. Hai ADR được kiểm tra:

| ADR | Giả thuyết | Cách Sprint 9 kiểm tra |
|---|---|---|
| `LatentSpace` geometry-keyed | Geometry (không phải shape) là key của `LatentSpace`; manifold/spherical là first-class geometry. | Thêm `unit_norm` geometry → `LatentSpace` phân biệt được hai geometry khác nhau qua cùng một handle. |
| Geometry-dispatch | `Trajectory`/`LatentSpace` ops dispatch trên geometry thay vì hardcode Euclidean. | `distance()`, `interpolate()`, `validate_point()` dispatch trên `self.geometry` — slerp cho spherical, lerp cho euclidean. |

Sau Sprint 9, cả hai ADR chuyển từ `pending` → `validated` (ghi nhận trong `decisions.md`). Nhưng **interface `LatentSpace` chưa freeze** — đó là việc của instance #3 (geometry case khác triết lý: sequence/grid, Gaussian set, hoặc discrete code).

## Atomic Tasks
Status legend: [ ] pending / [~] in progress / [x] done

- [ ] Task 1: Refine `LatentSpace.__init__` — `geometry` moves from class-level `str = "euclidean"` to instance-level parameter. Accepts `"euclidean"` or `"unit_norm"`. Default remains `"euclidean"` for backward compatibility. Validates at construction. Existing code that creates `LatentSpace(dim=8)` still works unchanged.
- [ ] Task 2: Extend `LatentSpace.validate_point()` — for `geometry="unit_norm"`, additionally checks `abs(||point|| - 1.0) < 1e-10`. For euclidean, behavior unchanged (shape check only).
- [ ] Task 3: Add `LatentSpace.distance(a: np.ndarray, b: np.ndarray) -> float` — dispatches on `self.geometry`. Euclidean: `||a - b||`. Spherical: `arccos(clip(a·b, -1, 1))` (angular distance on unit sphere). Both accept 1D vectors of shape `(dim,)`.
- [ ] Task 4: Add `LatentSpace.interpolate(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray` — dispatches on `self.geometry`. Euclidean: `(1-t)*a + t*b` (lerp). Spherical: proper slerp — `sin((1-t)ω)/sin(ω)*a + sin(tω)/sin(ω)*b` where `ω = arccos(a·b)`. Handles edge case `ω ≈ 0` (returns `a`). Accepts 1D vectors.
- [ ] Task 5: Add `LatentSpace.normalize(point: np.ndarray) -> np.ndarray` — for `unit_norm`, projects to unit sphere (`point / ||point||`); for `euclidean`, returns copy unchanged. Error on zero vector.
- [ ] Task 6: Update `LatentSpace.__repr__` to reflect instance-level geometry (was already parameterized, just verify).
- [ ] Task 7: Backward compatibility — verify all existing code paths still work. `LatentSpace(dim=8)` still creates euclidean geometry. All 116 existing tests pass without modification. VAE and RandomProjection `latent_space` properties still return euclidean LatentSpaces.
- [ ] Task 8: End-to-end demo script `scripts/end_to_end_spherical_demo.py` — synthetic unit-norm data (normalized random vectors on sphere) → `LatentSpace(dim=8, geometry="unit_norm")` → demonstrate: (a) validate_point rejects non-unit vectors, (b) distance computes angular distance correctly, (c) slerp interpolates along geodesic on sphere (compare with lerp — lerp leaves the sphere, slerp stays on it), (d) normalize projects back to sphere. Side-by-side visualization: lerp path vs slerp path on 3D sphere projected to 2D.
- [ ] Task 9: Visualization — matplotlib: (1) 3D scatter of points on unit sphere colored by cluster, (2) lerp interpolation path (straight line through sphere interior), (3) slerp interpolation path (arc on sphere surface). Demonstrates the geometry-dispatch ADR: lerp is wrong for spherical; slerp preserves the manifold constraint.
- [ ] Task 10: Tests — pytest for new `LatentSpace` spherical geometry: construction with both geometries, invalid geometry raises, validate_point spherical constraint, distance correctness (known angular distances), interpolate slerp correctness (midpoint on sphere, endpoints, edge case t=0,1), normalize correctness, backward compat (default euclidean still works). Also update existing `LatentSpace` tests if any assertion on `geometry` class attribute changed. Target: ~15–18 new tests.
- [ ] Task 11: Tooling gate — `ruff check` + `ruff format` + `pyright` strict clean. All 116 existing tests + new tests pass. No torch leakage (all geometry ops are pure numpy).
- [ ] Task 12: Rule of Three §4a — ghi artifact summary: "LatentSpace #2 (unit_norm/spherical) → confirm two geometry cases coexist in same class. Dispatch is inline if/elif (no GeometryProtocol yet — instance #3 needed for extraction). Geometry enum is validated but LatentSpace interface stays concrete."
- [ ] Task 13: ADR check §4c — **major milestone**: mark `LatentSpace` geometry-keyed ADR as `validated` (2 geometry cases prove enum structure). Mark geometry-dispatch ADR as `validated` (distance/interpolation dispatch on geometry proven with 2 metrics). Append entries to `decisions.md` with pointer to code/tests.
- [ ] Task 14: Update `CHANGELOG.md` `[Unreleased]` — add spherical geometry, distance/interpolate/normalize methods, demo, and the ADR validation milestone under `Added`. Note the ADR status changes.
- [ ] Task 15: Update `docs/PLAN.md` — mark Sprint 8 complete, Sprint 9 active, remove Sprint 9 from backlog. This is the last sprint of Giai đoạn 1.

## Rule-of-Three checkpoint (to verify at end)
| Check | Status |
|---|---|
| LatentSpace instances | #1 euclidean (Sprint 4) + #2 unit_norm/spherical (Sprint 9) |
| Geometry-dispatch proven? | Yes — distance and interpolate dispatch on `self.geometry` for 2 metrics |
| Rule branch | **Instance #2** → keep dispatch inline (if/elif), do NOT extract `GeometryProtocol` |
| `GeometryProtocol`? | No — extract at instance #3 (sequence/grid, Gaussian set, or discrete code geometry) |
| ADR impact | Two ADRs move from `pending` → `validated` |

## LatentSpace Design — After Sprint 9
```python
class LatentSpace:
    """A latent space with geometry-aware metric operations.

    Parameters
    ----------
    dim : int
        Dimensionality.
    geometry : str, optional
        "euclidean" (default) or "unit_norm" (spherical).
    """

    def __init__(self, dim, geometry="euclidean", source_model="", metadata=None):
        _GEOMETRIES = {"euclidean", "unit_norm"}
        if geometry not in _GEOMETRIES:
            raise ValueError(f"Unknown geometry {geometry!r}, expected one of {_GEOMETRIES}")
        self.geometry = geometry  # instance-level (was class-level)
        self.dim = dim
        ...

    def validate_point(self, point):
        if point.shape != (self.dim,):
            raise ValueError(...)
        if self.geometry == "unit_norm":
            if abs(np.linalg.norm(point) - 1.0) > 1e-10:
                raise ValueError("unit_norm requires ||point|| = 1")

    def distance(self, a, b) -> float:
        if self.geometry == "euclidean":
            return float(np.linalg.norm(a - b))
        # unit_norm: angular distance
        cos = np.clip(np.dot(a, b), -1.0, 1.0)
        return float(np.arccos(cos))

    def interpolate(self, a, b, t) -> np.ndarray:
        if self.geometry == "euclidean":
            return (1 - t) * a + t * b
        # unit_norm: slerp
        omega = np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))
        if abs(omega) < 1e-10:
            return a.copy()
        sin_omega = np.sin(omega)
        return (np.sin((1-t)*omega)/sin_omega)*a + (np.sin(t*omega)/sin_omega)*b

    def normalize(self, point) -> np.ndarray:
        if self.geometry == "euclidean":
            return point.copy()
        norm = np.linalg.norm(point)
        if norm < 1e-15:
            raise ValueError("Cannot normalize zero vector on sphere")
        return point / norm
```

## Notes / Blockers
* **Backward compatibility is critical.** `LatentSpace(dim=8)` must still create euclidean geometry. All 116 existing tests, all adapters (VAE, RandomProjection), all methods (PCA, UMAP, SAE), and all demo scripts must pass without modification. The `geometry` parameter defaults to `"euclidean"`.
* **No new dependency.** All geometry operations are pure numpy (`np.linalg.norm`, `np.dot`, `np.arccos`, `np.sin`, `np.clip`).
* **`Trajectory` is NOT modified.** Geometry-aware ops live on `LatentSpace`, not `Trajectory`. This keeps `Trajectory` stable. If `Trajectory` needs geometry-aware ops later (Sprint 10+, lerp/steering on trajectory), they route through `LatentSpace`.
* **Dispatch is inline if/elif, not a registry.** Instance #2 keeps it simple. When instance #3 arrives (Gaussian set, sequence, or discrete code), we can extract a proper dispatch mechanism. Premature abstraction at instance #2 would bake in the wrong shape.
* **ADR validation is a project milestone.** After Sprint 9, 2 of 3 pending ADRs become `validated`. The third (ModelAdapter 3-mode) already has mode (i) confirmed and awaits modes (ii) and (iii).
* Each task one commit per Conventional Commits (`feat(core):`, `test(core):`, `chore:`).

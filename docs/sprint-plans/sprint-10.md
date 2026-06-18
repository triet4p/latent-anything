# Sprint 10 Plan

## Sprint Goal
Increment thứ bảy (Round 7): thêm **Lerp** — B-Method #1, stateless, pure transform — khởi động Giai đoạn 2 (Layer B foundation). Lerp wraps `LatentSpace.interpolate()` (đã có từ Sprint 9) thành first-class Method object, xử lý cả single-point interpolation và trajectory-level blending. Kết thúc: **giữ hardcoded** theo Rule of Three (B-Method #1 — chưa mở rộng `Method` Protocol cho stateless methods).

## Đây là B-Method đầu tiên — khác triết lý với Layer A methods

| Layer | Method pattern | Ví dụ | Interface |
|---|---|---|---|
| A (Introspection) | Stateful: fit → transform | PCA, UMAP, SAE | `fit(data)`, `transform(data)`, `fit_transform(data)` |
| **B (Manipulation)** | **Stateless: pure function** | **Lerp** (this sprint) | `__call__(a, b, t)` — không có `fit` |
| B (Manipulation) | Stateful: fit → apply | Steering (Sprint 11) | `fit(contrast_pair)`, `__call__(latent)` |

Layer B methods operate trên latent points/trajectories đã có sẵn, không phải dimensionality reduction. Lerp là case đơn giản nhất: pure function, không state, không training. Đây là stress test đầu tiên cho thấy `Method` Protocol hiện tại (`fit`/`transform`/`fit_transform`) không cover được stateless B-Methods — nhưng **chưa mở rộng Protocol ở instance #1** (Rule of Three).

## Atomic Tasks
Status legend: [ ] pending / [~] in progress / [x] done

- [x] Task 1: Implement `Lerp` concrete class in `src/latent_anything/methods/lerp.py` — stateless interpolation method.
  - Constructor: `Lerp(space: LatentSpace | None = None)` — optional `LatentSpace` for geometry-aware dispatch. If `None`, defaults to Euclidean lerp.
  - `__call__(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray` — interpolate between two 1D latent vectors `a` and `b` at parameter `t ∈ [0,1]`. Delegates to `space.interpolate(a, b, t)` if space provided, otherwise `(1-t)*a + t*b`. Returns a new array (does not mutate inputs).
  - `between(traj_a: Trajectory, traj_b: Trajectory, t: float) -> Trajectory` — pointwise interpolation between two trajectories of the same shape. Returns a new `Trajectory`. Validates that both trajectories have same `dim` and `len`.
  - `blend_sequence(trajectory: Trajectory, n_steps: int) -> Trajectory` — densely interpolate between consecutive points in a trajectory. E.g., trajectory with points [p0, p1, p2] → [p0, p0.5, p1, p1.5, p2] for `n_steps=2`. Returns new `Trajectory`.
  - All input/output is `numpy.ndarray` (single-point) or `Trajectory` (sequence). No torch.
  - `space` property → returns the optional `LatentSpace`.
- [x] Task 2: Export `Lerp` from `src/latent_anything/methods/__init__.py`. Add to `__all__`. Do NOT modify `Method` Protocol — B-Method #1 stays hardcoded, Protocol extension at instance #3.
- [x] Task 3: End-to-end demo script `scripts/end_to_end_lerp_demo.py` — two scenarios:
  - **Scenario A (Euclidean)**: Generate two random latent vectors in 8D → `Lerp()` → interpolate at t=0, 0.25, 0.5, 0.75, 1.0 → flatten to 2D via PCA → visualize the interpolation path with matplotlib.
  - **Scenario B (Spherical)**: Generate unit-norm vectors on sphere → `Lerp(space=LatentSpace(dim=8, geometry="unit_norm"))` → slerp interpolation → PCA projection → compare lerp path (straight line leaving sphere) vs slerp path (arc on sphere surface). Show trajectory blending with `blend_sequence`.
- [x] Task 4: Visualization — 1×2 matplotlib: (left) Euclidean lerp path in PCA space, (right) spherical slerp path in PCA space with sphere outline. Annotated with t values. Include trajectory blending example.
- [x] Task 5: Tests — pytest for `Lerp` class: construction with/without space, `__call__` produces correct interpolation (t=0→a, t=1→b, t=0.5→midpoint), geometry dispatch (slerp stays on sphere, lerp doesn't), `between` produces correct `Trajectory` shape, `blend_sequence` produces densified trajectory, error cases (mismatched dims, mismatched trajectory lengths). 28 tests implemented (target: ~14–16).
- [x] Task 6: Tooling gate — `ruff check` + `ruff format` + `pyright` strict clean. All 179 existing tests + 28 new tests pass. No torch leakage (pure numpy).
- [x] Task 7: Rule of Three §4a — ghi artifact summary: "B-Method #1 (Lerp, stateless, pure transform) → stay hardcoded. No Protocol modification. The existing `Method` Protocol has `fit`/`transform` — stateless methods don't fit this yet. Interface expansion happens when B-Method #3 (activation patching) reveals the full stateless+stateful spectrum."
- [x] Task 8: ADR check §4c — Lerp exercises the geometry-dispatch ADR (already `validated` from Sprint 9) but adds no new evidence. The ModelAdapter 3-mode ADR stays `pending`. Appended routine entry to `decisions.md`.
- [x] Task 9: Update `CHANGELOG.md` `[Unreleased]` — add Lerp B-Method, trajectory blending, and demo entries under `Added`. Note this is the first Layer B method.
- [x] Task 10: Update `docs/PLAN.md` — Sprint 9 already complete, Sprint 10 active. Milestone 1 already completed. Milestone 2 marked active.

## Rule-of-Three checkpoint (to verify at end)
| Check | Status |
|---|---|
| B-Method instances | Lerp (#1, stateless, pure transform) |
| Rule branch | **Instance #1** → stay hardcoded, no Protocol modification |
| `Method` Protocol? | Unchanged — still `fit`/`transform`/`fit_transform`. Stateless methods not yet integrated. |
| `Method` Protocol expansion | At B-Method #3 (activation patching) — when stateless + stateful B-patterns are both proven |

## Lerp Design Notes
```python
class Lerp:
    """Stateless interpolation between latent points or trajectories."""

    def __init__(self, space: LatentSpace | None = None):
        self._space = space

    @property
    def space(self) -> LatentSpace | None:
        return self._space

    def __call__(self, a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
        """Interpolate between two 1D latent vectors a and b at parameter t."""
        if a.shape != b.shape:
            raise ValueError(...)
        if self._space is not None:
            return self._space.interpolate(a, b, t)
        return (1 - t) * a + t * b  # default Euclidean

    def between(self, traj_a: Trajectory, traj_b: Trajectory, t: float) -> Trajectory:
        """Pointwise interpolation between two trajectories."""
        if len(traj_a) != len(traj_b):
            raise ValueError(...)
        if traj_a.dim != traj_b.dim:
            raise ValueError(...)
        data_a = traj_a.to_numpy()
        data_b = traj_b.to_numpy()
        blended = np.array([self(a, b, t) for a, b in zip(data_a, data_b)])
        return Trajectory(data=blended)

    def blend_sequence(self, trajectory: Trajectory, n_steps: int = 2) -> Trajectory:
        """Densely interpolate between consecutive points in a trajectory."""
        data = trajectory.to_numpy()
        n_points = len(trajectory)
        dense = []
        for i in range(n_points - 1):
            for step in range(n_steps):
                tt = step / n_steps
                dense.append(self(data[i], data[i+1], tt))
        dense.append(data[-1])  # final point
        return Trajectory(data=np.array(dense))
```

## Notes / Blockers
* Phụ thuộc Sprint 9 (geometry-aware `LatentSpace.interpolate()` phải xong). ✓ Đã hoàn tất.
* **No new dependency.** Lerp uses pure numpy + existing `LatentSpace` and `Trajectory`.
* **`Method` Protocol is NOT modified.** The current Protocol has `fit`/`transform`/`fit_transform` — stateless methods like Lerp don't conform. This is fine at instance #1 per Rule of Three. The Protocol will be expanded when B-Method #3 (activation patching, Sprint 12) forces us to reconcile stateful vs stateless.
* **Lerp delegates to `LatentSpace.interpolate()` for geometry dispatch.** This is good architecture — `LatentSpace` owns the metric, `Lerp` is the Method wrapper. No duplicated dispatch logic.
* **`between()` and `blend_sequence()` are trajectory-level ops.** They operate on `Trajectory` objects but return new `Trajectory` instances (immutability preserved). They're methods on `Lerp`, not on `Trajectory` — the Method owns the operation, not the data structure.
* Each task one commit per Conventional Commits (`feat(methods):`, `test(methods):`, `chore:`).

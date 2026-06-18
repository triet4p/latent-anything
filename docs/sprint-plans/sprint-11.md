# Sprint 11 Plan

## Sprint Goal
Increment thứ tám (Round 8): thêm **SteeringVector** — B-Method #2, **stateful** — method thứ hai của Layer B. Steering vector học một hướng (direction) trong latent space từ contrast pairs (positive vs negative), rồi áp dụng hướng đó để steer latent representation. Khác triết lý với Lerp (stateless pure function): SteeringVector có `fit(positives, negatives)` → `__call__(latent, strength)`. Kết thúc: **phác internal `_BMethodBase` shape tạm, đánh dấu UNSTABLE** theo Rule of Three instance #2.

## Đây là B-Method thứ hai — khác triết lý với Lerp

| Layer B Method | Pattern | Interface | Instance # |
|---|---|---|---|
| Lerp (Sprint 10) | Stateless: pure function | `__call__(a, b, t)` — no `fit` | #1 |
| **SteeringVector (this sprint)** | **Stateful: fit → apply** | **`fit(pos, neg)` → `__call__(latent, strength)`** | **#2** |
| Activation patching (Sprint 12) | Stateful: intervene in forward pass | Hook-based, different lifecycle | #3 (freeze trigger) |

SteeringVector là stateful B-Method đầu tiên. Nó có một pha `fit` (học direction từ contrast data) và một pha `__call__` (áp dụng steer lên latent). Đây là pattern khác với Lerp (stateless, không `fit`, nhận hai điểm và t). Hai pattern này sẽ thống nhất khi B-Method #3 (activation patching) buộc freeze interface ở Sprint 12.

## Atomic Tasks
Status legend: [ ] pending / [~] in progress / [x] done

- [x] Task 1: Implement `SteeringVector` concrete class in `src/latent_anything/methods/steering.py` — stateful steering method.
  - Constructor: `SteeringVector(space: LatentSpace | None = None)` — optional `LatentSpace` for geometry-aware post-steer normalization. Initially `None` means no normalization.
  - `fit(positives: np.ndarray, negatives: np.ndarray) -> None` — learn steering direction from contrast pairs. Both arrays shape `(n_samples, dim)`. Algorithm: compute mean vectors `µ_pos`, `µ_neg`; direction `v = µ_pos - µ_neg`; normalize `v = v / ||v||` (unit direction). Set `_fitted = True`.
  - `direction` property → `np.ndarray` of shape `(dim,)` — the learned unit steering direction. Raises `RuntimeError` if not fitted.
  - `__call__(latent: np.ndarray, strength: float = 1.0) -> np.ndarray` — steer a single 1D latent vector: `latent + strength * direction`. If `space` is provided, optionally `normalize` the result (e.g. project back onto sphere for `unit_norm` geometry). Returns a new array (does not mutate input).
  - `apply_trajectory(trajectory: Trajectory, strength: float = 1.0) -> Trajectory` — steer all points in a trajectory. Returns new `Trajectory`.
  - `is_fitted` property → `bool`.
  - `space` property → `LatentSpace | None`.
  - All input/output is `numpy.ndarray` (single-point) or `Trajectory` (sequence). No torch.
  - Edge cases: `fit` rejects mismatched dims between positives/negatives; `__call__` rejects latent with wrong dim; calling `__call__`/`apply_trajectory`/`direction` before `fit` raises `RuntimeError`; zero-strength returns unchanged copy.
- [x] Task 2: Sketch internal `_BMethodBase` in `src/latent_anything/methods/_b_base.py` — per Rule of Three §4a, instance #2 → sketch shared shape, mark UNSTABLE.
  - Define as a lightweight ABC or structural Protocol (TBD by implementation fit).
  - Capture common B-Method surface: `__call__` (primary operation), `space` property, `apply_trajectory`.
  - Mark docstring: **"⚠️ UNSTABLE — sketched at B-Method #2 (SteeringVector). Do not freeze. Shape may change when B-Method #3 (activation patching) reveals the full stateless+stateful spectrum."**
  - Lerp and SteeringVector both structurally conform (duck-typing). Do NOT force inheritance at this stage.
  - This mirrors the pattern from Sprint 5 (`_MethodBase` at Method #2) and Sprint 8 (`_ModelAdapterBase` at ModelAdapter #2).
- [x] Task 3: Export `SteeringVector` from `src/latent_anything/methods/__init__.py`. Add to `__all__`. Do NOT modify `Method` Protocol — B-Method Protocol expansion at instance #3 (Sprint 12).
- [x] Task 4: End-to-end demo script `scripts/end_to_end_steering_demo.py` — two scenarios:
  - **Scenario A (Euclidean, simple mean-difference)**: Generate synthetic 8D contrast dataset — "positive" cluster centered at `[+1, +1, 0, ..., 0]` with noise σ=0.3, "negative" cluster at `[-1, -1, 0, ..., 0]`. Fit `SteeringVector()`. Apply steering at strengths [0, 0.5, 1.0, 2.0] to test points from both classes. Project to 2D via PCA. Visualize steering path with matplotlib: original points and steered points connected by arrows, colored by class.
  - **Scenario B (Spherical, geometry-aware)**: Generate unit-norm contrast dataset on 3-sphere — "positive" direction `[1, 0, 0]` with small angular noise, "negative" direction `[-1, 0, 0]` with noise. Fit `SteeringVector(space=LatentSpace(dim=3, geometry="unit_norm"))`. Steer with strength=1.0, normalize back to sphere. Show that steered vectors stay on sphere (norm ≈ 1). 3D scatter plot with matplotlib (or 2D PCA projection).
- [x] Task 5: Visualization — 1×2 matplotlib: (left) Euclidean steering with arrows showing direction + magnitude at different strengths, (right) spherical steering showing points moving along great circle while staying on sphere. Annotated with strength values.
- [x] Task 6: Tests — pytest for `SteeringVector` class:
  - `test_steering_construction_default` — default constructor, not fitted, space=None
  - `test_steering_construction_with_space` — constructor with LatentSpace
  - `test_steering_fit_learns_direction` — fit on synthetic separates correctly (direction dot product with true direction > 0.9)
  - `test_steering_direction_before_fit_raises` — accessing direction before fit raises RuntimeError
  - `test_steering_call_before_fit_raises` — calling before fit raises RuntimeError
  - `test_steering_call_moves_in_direction` — `__call__` moves point along learned direction
  - `test_steering_call_zero_strength_returns_copy` — strength=0 returns unchanged copy (not same object)
  - `test_steering_call_negative_strength_reverses` — negative strength moves opposite direction
  - `test_steering_call_preserves_input` — input array not mutated
  - `test_steering_call_wrong_dim_raises` — latent dim != direction dim raises ValueError
  - `test_steering_fit_mismatched_dims_raises` — positives and negatives with different dim raises ValueError
  - `test_steering_fit_empty_raises` — empty arrays raise ValueError
  - `test_steering_direction_is_unit_norm` — learned direction has ||v|| ≈ 1
  - `test_steering_apply_trajectory_returns_new` — returns new Trajectory, not same object
  - `test_steering_apply_trajectory_preserves_shape` — output Trajectory has same (n, dim) shape
  - `test_steering_apply_trajectory_moves_all_points` — all points shifted in same direction
  - `test_steering_spherical_normalization` — with unit_norm space, steered points stay on sphere
  - `test_steering_is_fitted_flag` — is_fitted transitions False → True after fit
  - `test_steering_no_torch_leakage` — verify no torch import in module (optional: grep check)
  - Target: ~18–20 tests.
- [x] Task 7: Tooling gate — `ruff check` + `ruff format` + `pyright` strict clean. All existing tests (~211) + new tests (32) pass. No torch leakage (pure numpy SteeringVector).
- [x] Task 8: Rule of Three §4a — ghi artifact summary:
  > "B-Method #2 (SteeringVector, stateful, fit from contrast) → sketch `_BMethodBase` internal, mark UNSTABLE. Lerp (stateless, no fit) and SteeringVector (stateful, has fit) now show two distinct B-Method patterns. `_BMethodBase` captures `__call__` + `space` + `apply_trajectory`. The existing `Method` Protocol (`fit`/`transform`/`fit_transform`) was designed for stateful Layer A dimensionality-reduction methods and does NOT fit either B-Method. Wait for B-Method #3 (activation patching, Sprint 12) to freeze B-Method interface and reconcile with `Method` Protocol."
- [x] Task 9: ADR check §4c — SteeringVector exercises the two validated ADRs:
  - `LatentSpace` geometry-keyed ADR (validated): SteeringVector optionally accepts a `LatentSpace` and uses it for geometry-aware post-steer normalization (e.g. project back to sphere).
  - Geometry-dispatch ADR (validated): Exercised when `space.geometry == "unit_norm"` triggers `space.normalize()` after steering.
  - `ModelAdapter` 3-mode ADR: `pending` (no change — not touched by this Layer B increment).
  - Append routine entry to `decisions.md`.
- [x] Task 10: Update `CHANGELOG.md` `[Unreleased]` — add SteeringVector B-Method, `_BMethodBase` internal sketch, trajectory steering, and demo entries under `Added`. Note this is B-Method #2, stateful.
- [x] Task 11: Update `docs/PLAN.md` — Sprint 10 → Completed, Sprint 11 → Completed, Sprint 12 → Pending, Milestone 2 still ongoing.

## Rule-of-Three checkpoint (to verify at end)
| Check | Status |
|---|---|
| B-Method instances | Lerp (#1, stateless), SteeringVector (#2, stateful) |
| Rule branch | **Instance #2** → sketch internal `_BMethodBase`, mark UNSTABLE |
| `Method` Protocol? | Unchanged — still `fit`/`transform`/`fit_transform`. Neither B-Method conforms. |
| `_BMethodBase` internal? | Sketched at `methods/_b_base.py`, marked UNSTABLE. Covers `__call__` + `space` + `apply_trajectory`. |
| B-Method freeze | At B-Method #3 (activation patching, Sprint 12) — when stateless + stateful + hook-based patterns are all proven |

## SteeringVector Design Notes
```python
class SteeringVector:
    """Learn a steering direction from contrast pairs and apply it to latent vectors.

    SteeringVector learns a direction in latent space that separates
    positive from negative examples, then steers latent representations
    along that direction. This is the stateful counterpart to Lerp
    (stateless) — B-Method #2.

    Algorithm:
        1. fit(positives, negatives): direction = normalize(mean(pos) - mean(neg))
        2. __call__(latent, strength): latent + strength * direction
    """

    def __init__(self, space: LatentSpace | None = None) -> None:
        self._space = space
        self._direction: np.ndarray | None = None

    @property
    def space(self) -> LatentSpace | None:
        return self._space

    @property
    def is_fitted(self) -> bool:
        return self._direction is not None

    @property
    def direction(self) -> np.ndarray:
        if self._direction is None:
            raise RuntimeError("SteeringVector not fitted. Call fit() first.")
        return self._direction.copy()

    def fit(self, positives: np.ndarray, negatives: np.ndarray) -> None:
        if positives.ndim != 2 or negatives.ndim != 2:
            raise ValueError("Expected 2D arrays (n_samples, dim)")
        if positives.shape[1] != negatives.shape[1]:
            raise ValueError(
                f"Dim mismatch: positives {positives.shape[1]}, "
                f"negatives {negatives.shape[1]}"
            )
        mean_pos = positives.mean(axis=0)
        mean_neg = negatives.mean(axis=0)
        diff = mean_pos - mean_neg
        norm = np.linalg.norm(diff)
        if norm < 1e-15:
            raise ValueError("Contrast direction is zero — positives "
                             "and negatives have identical means")
        self._direction = diff / norm

    def __call__(self, latent: np.ndarray, strength: float = 1.0) -> np.ndarray:
        if self._direction is None:
            raise RuntimeError("SteeringVector not fitted. Call fit() first.")
        if latent.shape != self._direction.shape:
            raise ValueError(
                f"Latent shape {latent.shape} != direction "
                f"shape {self._direction.shape}"
            )
        result = latent + strength * self._direction
        if self._space is not None:
            result = self._space.normalize(result)
        return result

    def apply_trajectory(
        self, trajectory: Trajectory, strength: float = 1.0
    ) -> Trajectory:
        data = trajectory.to_numpy()
        steered = np.array([self(point, strength) for point in data])
        return Trajectory(data=steered)
```

## Notes / Blockers
* **No new dependency.** SteeringVector uses pure numpy + existing `LatentSpace` and `Trajectory`.
* **`Method` Protocol is NOT modified.** The current Protocol has `fit`/`transform`/`fit_transform` — stateful B-Methods have `fit` + `__call__`, which is a different interface shape. This is fine at instance #2 per Rule of Three. Reconciliation at Sprint 12.
* **`_BMethodBase` is internal, not public.** Marked UNSTABLE. Follows the pattern from `_MethodBase` (Sprint 5) and `_ModelAdapterBase` (Sprint 8).
* **Steering direction is unit norm by default.** This ensures consistent behavior regardless of the magnitude of the mean difference. Strength=1.0 shifts by exactly one unit in the learned direction.
* **Geometry-aware normalization is opt-in.** When `space` is provided and `space.geometry == "unit_norm"`, `__call__` applies `space.normalize()` after steering to keep points on the sphere. This is NOT the default — callers who want Euclidean steering with no normalization simply pass `space=None`.
* **SteeringVector does NOT use PCA for direction finding in v1.** Simple mean-difference is the standard approach in the activation engineering literature. PCA-based direction finding (e.g., finding the principal component of the difference) is a potential future enhancement but adds complexity without proven need at this stage.
* Each task one commit per Conventional Commits (`feat(methods):`, `test(methods):`, `chore:`).

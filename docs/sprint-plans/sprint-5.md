# Sprint 5 Plan

## Sprint Goal
Increment thứ hai (Round 2): thêm UMAP (Method #2 — nonlinear, stochastic, stateful), phác `Method` shape tạm *unstable*, chạy end-to-end từ latent tổng hợp đến visualize 2D. Kết thúc: **phác shared shape nội bộ `_MethodBase`** theo Rule of Three (instance #2 — được phép sketch tạm, chưa freeze, không public).

## Atomic Tasks
Status legend: [ ] pending / [~] in progress / [x] done

- [x] Task 1: Add `umap-learn` dependency to root `pyproject.toml`.
- [x] Task 2: Implement `UMAP` concrete class in `src/latent_anything/methods/umap.py` — stateful (`fit`/`transform`), wraps `umap-learn`, in/out `numpy.ndarray`, constructor accepts all standard UMAP hyperparameters (`n_neighbors`, `min_dist`, `metric`, `random_state`, etc.).
- [x] Task 3: Sketch internal `_MethodBase` in `src/latent_anything/methods/_base.py` — shared shape for stateful methods (`fit`/`transform`/`fit_transform`), docstring ghi rõ `"UNSTABLE — do not depend on this shape; will be replaced when Method #3 lands (Sprint 6)"`. Migrate PCA to use `_MethodBase`.
- [x] Task 4: Export UMAP from `src/latent_anything/methods/__init__.py` (same pattern as PCA). `_MethodBase` stays internal — NOT in `__all__`, NOT exported from top-level package.
- [x] Task 5: End-to-end script — synthetic latent → `Trajectory` → UMAP `fit` → projection 2D → visualize. Either a new `scripts/end_to_end_umap_demo.py` or extend the existing PCA demo with a UMAP comparison path.
- [x] Task 6: Visualization — 2D UMAP projection via `matplotlib` (static), optionally side-by-side with PCA for comparison.
- [x] Task 7: Tests — pytest for `UMAP` class: construction, fit, transform shape invariants, `random_state` reproducibility, error cases (1D input, empty data). Target: ~10–12 tests.
- [x] Task 8: Tooling gate — `ruff check` + `ruff format` + `pyright` strict clean across all new and changed files.
- [x] Task 9: Rule of Three §4a — ghi artifact summary: "instance #2 → sketched unstable `_MethodBase`, PCA migrated, `_MethodBase` is internal-only and NOT frozen". (skill `implement-atomic-task`).
- [x] Task 10: ADR check §4c — confirm all three pending ADRs (geometry-keyed `LatentSpace`, 3-mode `ModelAdapter`, geometry-dispatch) remain `pending`. UMAP does not touch any of them. Append entry to `decisions.md` if needed.
- [x] Task 11: Update `CHANGELOG.md` `[Unreleased]` — add UMAP, `_MethodBase` sketch, and demo entries under `Added`. Follow the changelog rules: user-facing perspective, past tense, one entry per logical change.

## Rule-of-Three checkpoint (to verify at end)
| Check | Status |
|---|---|
| Method instances | PCA (#1, linear) + UMAP (#2, nonlinear, stochastic) |
| Rule branch | **Instance #2** → sketch shared shape, mark *unstable*, NOT public |
| `_MethodBase` exposure | Internal only (`_` prefix), not in `__all__`, not exported from top-level |
| ADR impact | None — all three ADRs remain `pending` |

## Notes / Blockers
* Phụ thuộc Sprint 4 (Round 1 — PCA + `LatentSpace` + `Trajectory` phải xong). ✓ Đã hoàn tất.
* `umap-learn` là pure Python + numba dependency — xác nhận nó không kéo `torch` vào dependency tree.
* `_MethodBase` chỉ phác shape tạm cho `fit`/`transform`/`fit_transform`. Không thêm `save`/`load` hay bất kỳ abstraction nào khác ở vòng này — đó là việc của Sprint 6 khi freeze.
* Không kéo PyTorch vào public signature; UMAP dùng `umap-learn` (numba-based) là đủ.
* Mỗi task một commit riêng theo Conventional Commits (`feat(methods):`, `test(methods):`, `chore:`).
* UMAP `random_state` phải được test cho reproducibility (set seed → same output).

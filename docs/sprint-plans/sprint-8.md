# Sprint 8 Plan

## Sprint Goal
Increment thứ năm (Round 5): thêm **RandomProjection adapter** (ModelAdapter #2 — fixed-weight encoder, stateless/pretrained pattern, khác triết lý với VAE stateful/trained-from-scratch) và **phác `_ModelAdapterBase` shape tạm *unstable***. Chạy end-to-end: fixed projection → encode → `LatentSpace` → `Trajectory` → PCA/UMAP visualize. Kết thúc: **phác shared shape nội bộ** theo Rule of Three (instance #2 — sketch tạm, chưa freeze, không public).

## Why RandomProjection instead of actual VLA?

Tải OpenVLA thật (7B+ params, GPU, transformers dependency nặng) không khả thi trong sprint này và không cần thiết cho mục đích của Sprint 8. Mục đích của Sprint 8 là có **ModelAdapter #2 với triết lý khác VAE** để sketch shared shape. `RandomProjection` phục vụ đúng mục đích đó:

| Khía cạnh | VAE (ModelAdapter #1) | RandomProjection (ModelAdapter #2) |
|---|---|---|
| Cách có weights | Học từ data qua `fit()` | Fixed tại construction (pretrained pattern) |
| `fit()` | Có — gradient descent | **Không** — stateless |
| `encode()` | Learned encoder (torch MLP) | Fixed random matrix nhân |
| `decode()` | Learned decoder (torch MLP + sigmoid) | Transpose xấp xỉ (pseudo-inverse) |
| Triết lý | Stateful, trained-from-scratch | **Stateless, pretrained/fixed** |

Sự khác biệt này đủ để stress-test `ModelAdapter` shape: VAE có `fit`, RandomProjection không có. Interface thật (khi freeze ở instance #3) phải accommodate cả hai pattern.

## Atomic Tasks
Status legend: [ ] pending / [~] in progress / [x] done

- [ ] Task 1: Implement `RandomProjection` concrete class in `src/latent_anything/adapters/random_projection.py` — fixed-weight encoder using a random matrix (Johnson-Lindenstrauss style), normalized to preserve approximate distances.
  - Constructor: `input_dim`, `latent_dim`, `random_state` (optional, for reproducibility).
  - `encode(data: np.ndarray) -> np.ndarray` — multiply by stored projection matrix.
  - `decode(latent: np.ndarray) -> np.ndarray` — multiply by transpose (approximate inverse reconstruction).
  - `latent_space` property → `LatentSpace(dim=latent_dim, geometry="euclidean", source_model="random_projection")`.
  - **No `fit` method** — weights are fixed at construction. This is the key difference from VAE.
  - All in/out numpy; internal weights stored as numpy arrays (no torch needed).
- [ ] Task 2: Sketch `_ModelAdapterBase` internal shape in `src/latent_anything/adapters/_base.py` — tentative shared shape with `encode`, `decode`, `latent_space`. Docstring ghi rõ: `"UNSTABLE — do not depend on this shape; will be replaced when ModelAdapter #3 lands. Minimal shared surface: encode, decode, latent_space. Note: fit is NOT universal (VAE has it, RandomProjection does not)."` This is the internal convenience base providing `decode` stub — NOT a Protocol, NOT public.
- [ ] Task 3: Migrate VAE to note conformance to the sketched shape. Update VAE docstring to mention relationship to `_ModelAdapterBase` shared pattern. VAE does NOT inherit from `_ModelAdapterBase` (that would imply fit is universal — it's not).
- [ ] Task 4: Export `RandomProjection` from `src/latent_anything/adapters/__init__.py`. Add to `__all__`. `_ModelAdapterBase` stays internal — NOT in `__all__`.
- [ ] Task 5: End-to-end demo script `scripts/end_to_end_random_projection_demo.py` — synthetic cluster data → RandomProjection encode → `LatentSpace` → `Trajectory` → compare original data PCA vs encoded latent PCA side-by-side. Shows that random projection approximately preserves cluster structure (Johnson-Lindenstrauss lemma in action).
- [ ] Task 6: Visualization — matplotlib 1×3 grid: (1) PCA of original data, (2) PCA of random-projected latents, (3) UMAP of random-projected latents. Demonstrates structure preservation and the adapter→method pipeline.
- [ ] Task 7: Tests — pytest for `RandomProjection` class: construction defaults, encode shape, decode shape, `latent_space` property, `random_state` reproducibility (same seed → same projection matrix), approximate distance preservation (JL lemma: distances roughly preserved up to a factor), error cases (wrong input dim). Target: ~10–12 tests.
- [ ] Task 8: Tooling gate — `ruff check` + `ruff format` + `pyright` strict clean. Verify no torch needed (RandomProjection uses pure numpy).
- [ ] Task 9: Rule of Three §4a — ghi artifact summary: "ModelAdapter #2 (RandomProjection, fixed-weight/stateless, pretrained pattern) → sketched unstable `_ModelAdapterBase` with encode/decode/latent_space. fit is NOT in the shared shape (VAE-only). Shape is internal, marked UNSTABLE, not frozen. Freeze point is ModelAdapter #3." (skill `implement-atomic-task`).
- [ ] Task 10: ADR check §4c — RandomProjection does NOT touch mode (ii) no-explicit-latent or mode (iii) deterministic-renderer. The ModelAdapter 3-mode ADR stays `pending` (still only mode i confirmed). Append entry to `decisions.md`.
- [ ] Task 11: Update `CHANGELOG.md` `[Unreleased]` — add RandomProjection adapter, `_ModelAdapterBase` sketch, and demo entries under `Added`.
- [ ] Task 12: Update `docs/PLAN.md` — mark Sprint 7 complete, Sprint 8 active, remove Sprint 8 from backlog.

## Rule-of-Three checkpoint (to verify at end)
| Check | Status |
|---|---|
| ModelAdapter instances | VAE (#1, stateful, trained-from-scratch) + RandomProjection (#2, stateless, fixed-weight) |
| Khác triết lý? | **Yes** — VAE has `fit` (gradient descent training), RandomProjection has no `fit` (fixed at construction). This is the stateful vs stateless stress. |
| Rule branch | **Instance #2** → sketch shared shape, mark *unstable*, NOT public |
| `_ModelAdapterBase` exposure | Internal only (`_` prefix), not in `__all__`, not exported from top-level |
| `fit` in shared shape? | **No** — `fit` is VAE-only. Shared shape is `encode` + `decode` + `latent_space` |
| ADR impact | ModelAdapter 3-mode ADR stays `pending` (only mode i confirmed) |

## RandomProjection Design Notes
```
Construction:
  1. Generate random matrix W of shape (latent_dim, input_dim) with entries ~ N(0, 1)
  2. Normalize: W = W / sqrt(latent_dim)  (preserves approximate Euclidean distances per JL lemma)

encode(data):
  return data @ W.T   # (n_samples, input_dim) @ (input_dim, latent_dim) → (n_samples, latent_dim)

decode(latent):
  return latent @ W    # (n_samples, latent_dim) @ (latent_dim, input_dim) → (n_samples, input_dim)
                       # transpose ≈ pseudo-inverse for random Gaussian matrices
```

- Pure numpy — no torch dependency at all. This is a feature: it proves `ModelAdapter` doesn't require torch.
- Normalization factor `1/sqrt(latent_dim)` preserves approximate Euclidean distances (Johnson-Lindenstrauss lemma: if latent_dim ≥ O(log(n)/ε²), pairwise distances preserved within factor (1±ε)).
- `random_state` seeds `numpy.random.Generator` for reproducible projection matrices.
- `decode` is approximate — the transpose of a random Gaussian matrix is not a true inverse, but it's the best linear approximation. This mirrors the real-world case where some adapters have imperfect or no decode.

## Notes / Blockers
* Phụ thuộc Sprint 7 (VAE adapter phải xong). ✓ Đã hoàn tất.
* **No new dependency.** RandomProjection uses pure numpy — no torch, no sklearn, no external packages.
* `_ModelAdapterBase` is deliberately minimal. It captures only what VAE and RandomProjection both have: `encode`, `decode`, `latent_space`. The asymmetry (`fit` is VAE-only) is a feature, not a bug — it informs the eventual frozen Protocol.
* Do NOT create `ModelAdapter` Protocol. Instance #2 stays unstable internal. Freeze at instance #3.
* RandomProjection is NOT a VLA — it's a stand-in that proves the "pretrained/stateless" pattern. The real VLA adapter can be added post-freeze when infrastructure (GPU, model loading) is ready.
* Each task one commit per Conventional Commits (`feat(adapters):`, `test(adapters):`, `chore:`).
* The `_ModelAdapterBase` docstring must explicitly note: "fit is NOT part of this base — it's VAE-specific. The frozen Protocol (future) may or may not include fit depending on what instance #3 reveals."

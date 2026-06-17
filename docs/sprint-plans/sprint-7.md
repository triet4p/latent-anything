# Sprint 7 Plan

## Sprint Goal
Increment thứ tư (Round 4): thêm **VAE adapter** (ModelAdapter #1 — explicit learned latent, ADR mode i), trainable from scratch, expose `LatentSpace` từ `latent_space` property. Chạy end-to-end: train VAE trên synthetic data → encode → `LatentSpace` → `Trajectory` → PCA/UMAP visualize. Kết thúc: **giữ hardcoded** theo Rule of Three (ModelAdapter #1 — chưa extract `ModelAdapter` interface).

## Atomic Tasks
Status legend: [ ] pending / [~] in progress / [x] done

- [x] Task 1: Create `src/latent_anything/adapters/` package with `__init__.py`. This is the new adapters namespace parallel to `methods/`.
- [x] Task 2: Implement `VAE` concrete class in `src/latent_anything/adapters/vae.py` — torch-based Variational Autoencoder with:
  - Encoder: `input_dim → hidden_dim → mu (latent_dim) + logvar (latent_dim)`
  - Reparameterization trick: `z = mu + exp(0.5 * logvar) * epsilon`
  - Decoder: `latent_dim → hidden_dim → input_dim`
  - Training: `fit(data)` via gradient descent with reconstruction loss (MSE) + KL divergence
  - `encode(data: np.ndarray) -> np.ndarray` — encode to latent mean (numpy)
  - `decode(latent: np.ndarray) -> np.ndarray` — decode from latent (numpy)
  - `latent_space` property → `LatentSpace(dim=latent_dim, source_model="vae")`
  - Constructor: `input_dim`, `latent_dim`, `hidden_dim` (optional, defaults to heuristic), `learning_rate`, `n_epochs`, `random_state`
  - All public I/O is `numpy.ndarray`; torch conversion at boundary only
- [x] Task 3: Export `VAE` from `src/latent_anything/adapters/__init__.py`. Do NOT export from top-level `__init__.py` — `ModelAdapter` is not yet a frozen public primitive. The `adapters` sub-package is accessible via `from latent_anything.adapters import VAE`.
- [x] Task 4: End-to-end demo script `scripts/end_to_end_vae_demo.py` — synthetic data generation (e.g., noisy sine waves, MNIST-like digits, or structured clusters) → VAE training → encode to latent → `LatentSpace` → `Trajectory` → PCA 2D projection + UMAP 2D embedding → side-by-side visualization (original data reconstruction + latent space structure). Show: reconstruction quality, latent space clustering.
- [x] Task 5: Visualization — matplotlib 2×2 grid: (1) original data sample, (2) VAE reconstruction, (3) PCA of encoded latents colored by class, (4) UMAP of encoded latents colored by class. Demonstrates the full adapter→method pipeline.
- [x] Task 6: Tests — pytest for `VAE` class: construction defaults, encode/decode shape invariants, `latent_space` property returns correct `LatentSpace`, reconstruction sanity (loss decreases over training), encode-then-decode roundtrip produces similar output, `random_state` reproducibility, error cases (wrong input dim, unfitted encode). Target: ~12–14 tests. (Delivered: 23 tests.)
- [x] Task 7: Tooling gate — `ruff check` + `ruff format` + `pyright` strict clean across all new and changed files. Verify `torch` stays internal (no `torch.Tensor` in any public signature — `encode`, `decode`, `latent_space`, `fit`).
- [x] Task 8: Rule of Three §4a — ghi artifact summary: "ModelAdapter #1 (VAE, explicit learned latent) → stay hardcoded. No ModelAdapter Protocol/ABC. This is instance #1 of a new primitive; interface extraction happens at instance #3 per §4a." (skill `implement-atomic-task`).
- [x] Task 9: ADR check §4c — VAE touches the "ModelAdapter 3-mode" ADR (exercise mode i: explicit learned latent). The ADR remains `pending` (only 1 of 3 modes proven). Append entry to `decisions.md` noting: "VAE confirms mode (i) — explicit learned latent — is real and useful. ADR not yet `validated` because modes (ii) no-explicit-latent and (iii) deterministic-renderer remain untested."
- [x] Task 10: Update `CHANGELOG.md` `[Unreleased]` — add VAE adapter, adapters package, and demo entries under `Added`.
- [x] Task 11: Update `docs/PLAN.md` — mark Sprint 6 complete, Sprint 7 active, remove Sprint 7 from backlog.

## Rule-of-Three checkpoint (to verify at end)
| Check | Status |
|---|---|
| ModelAdapter instances | VAE (#1, explicit learned latent, own training) |
| Rule branch | **Instance #1** → stay hardcoded, no Protocol/ABC |
| `ModelAdapter` exposure | Not a frozen primitive — not in top-level `__all__`, not exported from `latent_anything` |
| ADR impact | Touches ModelAdapter 3-mode ADR (mode i confirmed), but ADR stays `pending` (2 modes untested) |

## VAE Design Notes
```
Architecture:
  data ──→ Encoder ──→ mu, logvar ──→ reparam ──→ z ──→ Decoder ──→ reconstruction

Encoder:  input_dim → Linear(hidden_dim) → ReLU → Linear(latent_dim) [mu]
                                                 → Linear(latent_dim) [logvar]
Decoder:  latent_dim → Linear(hidden_dim) → ReLU → Linear(input_dim) → Sigmoid

Loss:     MSE(reconstruction, data) + beta * KL(N(mu, sigma²) || N(0, I))

Training: torch.optim.Adam, n_epochs
```

- `hidden_dim` heuristic: `max(latent_dim * 4, input_dim)` — wider encoder gives better latent
- KL beta defaults to 1.0 (standard VAE), adjustable via `beta` parameter
- Output activation is Sigmoid (assumes [0,1] input data) — documented as a constraint
- `encode(data)` returns `mu` (the latent mean), not a sample — deterministic encoding for downstream analysis
- `decode(latent)` expects a latent vector and returns reconstruction
- `fit(data)` runs the full training loop; `loss_history_` tracked for diagnostics
- `random_state` seeds torch for reproducibility
- Data must be scaled to [0,1] before calling `fit` (documented in docstring)

## Notes / Blockers
* ~~Phụ thuộc Sprint 6 (SAE + frozen `Method` Protocol phải xong).~~ ✓ Đã hoàn tất Sprint 6.
* **New primitive, new package.** `src/latent_anything/adapters/` is a sibling to `methods/`, not a sub-package. This is intentional: `ModelAdapter` is a separate core primitive from `Method`.
* **Do NOT create a `ModelAdapter` Protocol.** Instance #1 stays hardcoded. The Protocol/ABC only appears at instance #3 (Sprint 8+). This is the same discipline we applied to `Method` (stay hardcoded at Sprint 4, sketch at Sprint 5, freeze at Sprint 6).
* `torch` is already a dependency (from Sprint 6). VAE reuses it — no new dependency needed.
* `encode` returns `mu` (deterministic), not a stochastic sample. This is the standard practice for latent analysis (downstream PCA/UMAP need deterministic points). The reparameterization trick is used during training only.
* VAE is the simplest adapter that exercises both `encode` and `decode` — it's the "hello world" of ModelAdapter, analogous to PCA being the "hello world" of Method.
* Each task one commit per Conventional Commits (`feat(adapters):`, `test(adapters):`, `chore:`).
* The end-to-end demo must show the full value chain: data → VAE → LatentSpace → Trajectory → Method (PCA/UMAP) → visualization. This is the first time ALL Layer A primitives appear together in one pipeline.

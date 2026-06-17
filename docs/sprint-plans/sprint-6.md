# Sprint 6 Plan

## Sprint Goal
Increment thứ ba (Round 3): thêm **Sparse Autoencoder** (SAE — Method #3, neural, trained iteratively — khác triết lý so với PCA linear và UMAP nonlinear-fit-once), **freeze `Method` interface**, migrate PCA+UMAP sang frozen interface, promote `Method` lên public surface. Đây là điểm verify mà ARCHITECTURE §2 đã chỉ định sẵn: PCA→UMAP→**SAE**.

## Atomic Tasks
Status legend: [ ] pending / [~] in progress / [x] done

- [ ] Task 1: Add `torch` dependency to root `pyproject.toml`. `torch` is **internal only** — must never appear in a public signature. Input/output of SAE stays `numpy.ndarray`.
- [ ] Task 2: Freeze `Method` Protocol in `src/latent_anything/methods/protocols.py` — public contract with `fit(data: np.ndarray) -> None`, `transform(data: np.ndarray) -> np.ndarray`, `fit_transform(data: np.ndarray) -> np.ndarray`. As a Protocol, it's structural: PCA/UMAP/SAE conform without explicit inheritance. Docstring ghi rõ: "Frozen at Method #3 (SAE), Sprint 6 — this is the canonical `Method` shape for stateful dimensionality-reduction methods."
- [ ] Task 3: Implement `SAE` concrete class in `src/latent_anything/methods/sae.py` — torch-based sparse autoencoder with encoder (linear + ReLU) → latent → decoder (linear), trained with reconstruction loss + L1 sparsity penalty on latent activations. Constructor accepts `n_components`, `l1_coef`, `learning_rate`, `n_epochs`, `random_state`. `fit(data)` trains via gradient descent. `transform(data)` runs encoder forward pass, returns sparse numpy features. All in/out is `numpy.ndarray`; torch conversion happens at the internal boundary only.
- [ ] Task 4: Migrate PCA and UMAP — update docstrings to note conformance to frozen `Method` Protocol. Keep `_MethodBase` as internal convenience base providing `fit_transform` default, but both classes structurally conform to `Method` without needing to import it (Python Protocol duck-typing). Remove the `# pyright: ignore[reportPrivateUsage]` on `_MethodBase` imports or update reasoning.
- [ ] Task 5: Promote `Method` and `SAE` to public surface — add both to `src/latent_anything/methods/__init__.py` `__all__`. Also export `Method` from top-level `src/latent_anything/__init__.py` (it joins `LatentSpace` and `Trajectory` as the third core primitive exposed at the top level).
- [ ] Task 6: End-to-end demo script `scripts/end_to_end_sae_demo.py` — synthetic latent → `Trajectory` → SAE training → sparse feature projection → compare with PCA 2D visualization side-by-side. Shows SAE learning a sparse decomposition of the same data.
- [ ] Task 7: Visualization — side-by-side comparison: PCA 2D projection vs SAE sparse latent features vs UMAP embedding, all on the same synthetic data, with matplotlib.
- [ ] Task 8: Tests — pytest for `SAE` class: construction, fit, transform shape invariants, `random_state` reproducibility, sparsity verification (L1 penalty produces near-zero activations), error cases (1D input, unfitted transform). Also Protocol conformance smoke tests verifying PCA and UMAP structurally satisfy `Method`. Target: ~14–16 new tests.
- [ ] Task 9: Tooling gate — `ruff check` + `ruff format` + `pyright` strict clean across all new and changed files. Verify `torch` does NOT leak into any public signature (check all function param/return types).
- [ ] Task 10: Rule of Three §4a — ghi artifact summary: "Method #3 (SAE, neural/trained) khác triết lý → froze `Method` Protocol, migrated PCA+UMAP, promoted to public surface. This is the ARCHITECTURE §2 pre-chosen verify point." (skill `implement-atomic-task`).
- [ ] Task 11: ADR check §4c — SAE does not touch any of the three pending ADRs (geometry-keyed `LatentSpace`, 3-mode `ModelAdapter`, geometry-dispatch). Append entry to `decisions.md` confirming all remain `pending`.
- [ ] Task 12: Update `CHANGELOG.md` `[Unreleased]` — add SAE method, `Method` Protocol freeze, PCA/UMAP migration, and demo entries under `Added`. Note in `Changed` that `_MethodBase` is now backed by the frozen `Method` Protocol.

## Rule-of-Three checkpoint (to verify at end)
| Check | Status |
|---|---|
| Method instances | PCA (#1, linear) + UMAP (#2, nonlinear) + SAE (#3, neural, trained) |
| SAE khác triết lý? | **Yes** — gradient-descent training, L1 sparsity, encode/decode architecture. Fundamentally different from PCA's matrix decomposition and UMAP's manifold-learning fit. |
| Rule branch | **≥3, differing philosophy** → freeze `Method` interface, migrate all prior call-sites |
| `Method` exposure | **Public** — Protocol in `protocols.py`, exported from `methods/__init__.py` and top-level `__init__.py` |
| Migration state | PCA + UMAP + SAE all conform to `Method` Protocol in same commit (§4b) |

## SAE Design Notes
```
Architecture:  data ──→ encoder (Linear + ReLU) ──→ sparse_latent ──→ decoder (Linear) ──→ reconstruction
Loss:          MSE(reconstruction, data) + l1_coef * ||sparse_latent||₁
Training:      torch.optim.Adam, n_epochs iterations over full dataset
```
- Encoder: `dim → n_components`, single hidden layer with ReLU
- Decoder: `n_components → dim`, single linear layer (no activation)
- L1 penalty on the latent activations encourages sparsity
- `transform(data)` = encoder forward pass → numpy output
- `fit(data)` = train loop, conversion numpy→torch→numpy at boundary
- No batching in v0 — full dataset per epoch (small synthetic data, <10k points). Batching is a future concern.
- Optional: track `loss_history_` for training convergence visualization.

## Notes / Blockers
* Phụ thuộc Sprint 5 (UMAP + `_MethodBase` phải xong). ✓ Đã hoàn tất.
* **`torch` enters the project for the first time.** Must be internal-only. Verify no `torch.Tensor` in any public signature (params, return types, public properties). Boundary conversion at `fit`/`transform` entry/exit points.
* `Method` Protocol uses `typing.Protocol` (not ABC) per python.md rule: "Prefer Protocol over ABC — lighter, structural, cross-language friendly."
* `_MethodBase` stays as internal convenience (provides `fit_transform` default) but its docstring is updated to explain it backs the frozen `Method` Protocol.
* Per INCREMENTAL.md §4b: migrate ALL prior call-sites in the same commit. PCA and UMAP docstrings and `_MethodBase` module-level docstring must be updated.
* The `Method` Protocol does NOT include `save`/`load` or `__call__` yet — those arrive with stateless methods in Sprint 10 (lerp). This freeze captures the stateful fit-transform pattern proven by 3 differing instances.
* Each task one commit per Conventional Commits (`feat(methods):`, `test(methods):`, `chore:`).

# Sprint 6 — SAE (Method #3) & Method Protocol Freeze — Summary

## Rule-of-Three Checkpoint

| Check | Status |
|---|---|
| Method instances | PCA (#1, linear) + UMAP (#2, nonlinear) + SAE (#3, neural, trained) |
| SAE khác triết lý? | **Yes** — gradient-descent training, L1 sparsity, encoder/decoder architecture |
| Rule branch | **≥3, differing philosophy** → freeze `Method` interface, migrate all prior call-sites |
| `Method` exposure | **Public** — Protocol in `protocols.py`, exported from `methods/__init__.py` and top-level `__init__.py` |
| Migration state | PCA + UMAP + SAE all conform to `Method` Protocol in same commit (§4b) |

## What was built

- **`Method` Protocol** (`src/latent_anything/methods/protocols.py`) — frozen structural Protocol defining `fit(data: np.ndarray) -> None`, `transform(data: np.ndarray) -> np.ndarray`. `@runtime_checkable`. Docstring: "Frozen at Method #3 (SAE), Sprint 6 — this is the canonical `Method` shape for stateful dimensionality-reduction methods."
- **SAE** (`src/latent_anything/methods/sae.py`) — sparse autoencoder (linear + ReLU encoder → sparse latent → linear decoder), torch-based with MSE + L1 sparsity loss, Adam optimizer, `fit()`/`transform()`/`fit_transform()` with all numpy surface. `loss_history_` tracked for convergence diagnostics.
- **Migrated PCA/UMAP** — docstrings updated to note conformance to frozen `Method` Protocol. `_MethodBase` docstring updated from "UNSTABLE — will be replaced" to "Internal convenience base backed by the frozen `Method` Protocol."
- **Public surface** — `Method` and `SAE` exported from `methods/__init__.py`; `Method` also exported from top-level `latent_anything/__init__.py` as the third core primitive alongside `LatentSpace` and `Trajectory`.
- **Tests** — 16 SAE tests: construction, fit, transform shape invariants, sparsity verification, `random_state` reproducibility, error cases, and Protocol conformance smoke tests. Target exceeded (≥14).
- **Demo** — `scripts/end_to_end_sae_demo.py` with side-by-side PCA vs UMAP vs SAE 2D projections + SAE loss curve.

## Tooling gate

| Tool | Result |
|---|---|
| `ruff check` | Clean |
| `ruff format --check` | Clean |
| `pyright src/` | Clean (0 errors) |
| `pytest` | 69 passed, 0 failed |

## Torch boundary

`torch` is internal-only. No `torch.Tensor` appears in any public signature. All SAE input/output is `numpy.ndarray`; torch conversion happens inside `fit()`/`transform()`. `pyproject.toml` already had `torch>=2.5,<3.0` from a prior change.

## ADR status

All three pending ADRs (geometry-keyed `LatentSpace`, 3-mode `ModelAdapter`, geometry-dispatch) remain **pending**. SAE touches no geometry-keying, no model adapter, no metric dispatch.

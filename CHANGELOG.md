# Changelog

## [Unreleased]

### Added

- SAE (Sparse Autoencoder) dimensionality reduction method — Method #3 with fundamentally different philosophy (gradient-descent training, L1 sparsity, encoder/decoder architecture), torch-based with all-numpy public surface. (#sprint-6)
- `Method` Protocol — frozen structural `typing.Protocol` defining `fit(data: np.ndarray) -> None` / `transform(data: np.ndarray) -> np.ndarray`, promoted to public surface as third core primitive. (#sprint-6)
- End-to-end demo script `scripts/end_to_end_sae_demo.py` — synthetic latent → Trajectory → SAE training → sparse feature projection → side-by-side PCA vs UMAP vs SAE 2D visualization with matplotlib. (#sprint-6)
- Test suite: 16 SAE tests covering construction, fit, transform, sparsity verification, `random_state` reproducibility, error cases, and Protocol conformance smoke tests. (#sprint-6)
- UMAP dimensionality reduction method (`fit`/`transform`/`fit_transform`) wrapping `umap-learn` with numpy public surface, stochastic and stateful. (#sprint-5)
- Internal `_MethodBase` base class sketching the shared shape for stateful methods (`fit`/`transform`/`fit_transform`), marked UNSTABLE and not part of the public API. (#sprint-5)
- PCA migrated to `_MethodBase` inheritance, removing the now-redundant `fit_transform` override. (#sprint-5)
- End-to-end demo script `scripts/end_to_end_umap_demo.py` — synthetic latent → Trajectory → UMAP fit → 2D projection → side-by-side matplotlib visualization with PCA comparison. (#sprint-5)
- Test suite: 12 UMAP tests covering construction, fit/transform shape invariants, fit_transform roundtrip, `random_state` reproducibility, and input validation errors. (#sprint-5)

### Changed

- `_MethodBase` docstring updated from "UNSTABLE — will be replaced" to "Internal convenience base backed by the frozen `Method` Protocol" following Rule of Three (Method #3 freeze trigger). (#sprint-6)
- PCA and UMAP docstrings updated to note conformance to the frozen `Method` Protocol. (#sprint-6)
- `LatentSpace` concrete class for Euclidean flat vector spaces with `dim`, `geometry`, `source_model`, metadata, and `validate_point()`. (#sprint-4)
- `Trajectory` immutable sequence class holding 2D numpy latent arrays with `len`, indexing/slice (returns new `Trajectory`), and `to_numpy()`. (#sprint-4)
- PCA dimensionality reduction method (`fit`/`transform`/`fit_transform`) wrapping scikit-learn with numpy public surface. (#sprint-4)
- End-to-end demo script `scripts/end_to_end_pca_demo.py` — synthetic latent → Trajectory → PCA → 2D matplotlib visualization. (#sprint-4)
- Test suite: 41 tests covering LatentSpace (11), Trajectory with hypothesis property-based tests (16), PCA fit/transform/roundtrip (12), and package smoke (2). (#sprint-4)
- Package scaffold: `src/latent_anything/` (src-layout) with root-level `pyproject.toml` and PEP 621 metadata.
- Development tooling: `ruff` (lint + format), `pyright` (strict type-checking), `pytest`, and `hypothesis` configured in `pyproject.toml`.
- Smoke test suite: `tests/test_latent_anything/test_package.py` verifying import and version.
- CI workflow (`.github/workflows/ci.yml`) running `ruff check`, `ruff format --check`, `pyright`, and `pytest` on Python 3.12, 3.13, and 3.14.
- README updated with package installation via `uv`, Quick Start example, and current project structure.
- Python version range adjusted: `requires-python` set to `>=3.12,<3.15`, local development targets Python 3.13.
- Deploy workflow restricted: `deploy-latent-anything-theory.yml` now triggers only on `theory-v*` tag pushes or manual `workflow_dispatch`.

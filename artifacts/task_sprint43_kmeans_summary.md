# Sprint 43 — K-means Latent Structure Discovery Summary

## Overview

Implemented a K-means clustering module (`src/latent_anything/clustering.py`) for discovering cluster structure in latent representations, with explicit geometry compatibility checks, typed results with diagnostics, multi-seed stability analysis via Hungarian label alignment, and external validity comparison against known labels.

## Files Changed

### New files
- `src/latent_anything/clustering.py` — K-means clustering module: `KMeans` class, `KMeansConfig` (pydantic), `KMeansResult` (frozen dataclass), `ClusterStabilityReport`, geometry checks, stability analysis, external validation
- `tests/test_clustering.py` — 50 tests (48 offline + 1 skipped degenerate + 2 marked network)

### Modified files
- `src/latent_anything/__init__.py` — Added clustering exports to `__all__` and import block
- `src/latent_anything/_plugin_builtins.py` — Registered `KMeans` under `KIND_ANALYSIS`
- `tests/test_api_surface.py` — Updated public-export snapshot
- `tests/test_latent_anything/test_demo_smoke.py` — Updated registry count from 6→7 method_a
- `tests/test_latent_anything/test_registry.py` — Updated registry count assertions
- `CHANGELOG.md` — Added Sprint 43 entries
- `docs/sprint-plans/sprint-43.md` — All tasks marked done

## Key Design Decisions

1. **Not a `Method` protocol.** K-means produces cluster assignments, not a dimensionality-reducing transform. It does not follow the `Method` / `AnalysisPipeline` lifecycle, similar to probes and TCAV.

2. **Full sklearn wrapper.** The heavy lifting is delegated to `sklearn.cluster.KMeans` and `sklearn.metrics`. The wrapper adds geometry checks, bootstrap stability, typed results, and provenance tracking.

3. **Geometry-aware.** Clustering is only allowed on `"euclidean"` and `"unit_norm"` LatentSpace geometries. Structured (`"gaussian_set"`) and discrete (`"discrete_code"`) spaces are rejected with a clear error.

4. **Confidence proxy.** Each sample gets a confidence score = nearest_center_distance - second_nearest_center_distance. Larger margins indicate more confident assignments.

5. **Stability analysis.** Multi-seed stability uses Hungarian algorithm to align cluster labels across runs before computing agreement. Reports per-cluster stability, mean stability, and adjusted Rand index.

6. **External validation.** `compare_with_labels()` provides ARI, AMI, homogeneity, completeness, and V-measure against known labels — without using those labels during fitting.

7. **Provenance first.** Every result carries the config, random state, and caller-supplied provenance so clustering runs are fully traceable.

## Test Coverage (offline: 48 pass, 1 skipped)

| Category | Tests |
|---|---|
| Config validation | 4 |
| Result validation and serialization | 4 |
| Fit-predict lifecycle | 6 |
| Input validation | 4 |
| Standardization / preprocessing | 3 |
| Silhouette and confidence diagnostics | 2 |
| Geometry checks | 7 |
| Cluster stability analysis | 5 |
| External validation | 3 |
| Degenerate / edge-case inputs | 5 |
| Registry construction | 5 |
| Stability report serialization | 1 |
| Real integration (marked network) | 2 |

## Verification

- **All existing tests pass** (119 affected tests)
- **ruff check** passes cleanly
- **Registry** correctly resolves `"analysis"` kind → `KMeans` factory
- All tasks documented in sprint plan marked `[x]`

# Sprint 15 Plan

## Sprint Goal
Increment thứ mười hai (Round 12): thêm **Gaussian-set latent geometry** as geometry case #3. This is the first structured, set-like latent shape and prepares the codebase for deterministic-renderer adapters grounded in theory tầng 3B.

## Why This Sprint
Theory 3B says Gaussian parameters can be the latent state of a world model. Current `LatentSpace` only supports flat vectors. Before adding a Gaussian renderer adapter, the space itself needs a structured geometry case.

## Atomic Tasks
Status legend: [ ] pending / [~] in progress / [x] done

- [x] Task 1: Extend `LatentSpace` to represent a fixed-size `gaussian_set` shape without breaking `LatentSpace(dim=...)`.
- [x] Task 2: Define minimal Gaussian parameter schema in metadata: position, scale/covariance proxy, opacity, and color channels.
- [x] Task 3: Add validation for a single Gaussian-set latent point, likely shape `(n_gaussians, param_dim)`.
- [x] Task 4: Add a simple permutation-aware distance for fixed-size sets. Keep it deterministic and tested; avoid optimal assignment complexity unless necessary.
- [x] Task 5: Add interpolation for Gaussian-set states: interpolate numeric parameters and clamp/normalize fields that need constraints.
- [x] Task 6: Refactor geometry dispatch only as much as the third geometry forces. If inline `if/elif` becomes too brittle, extract a small internal dispatch table.
- [x] Task 7: Add tests for flat-vector backward compatibility, Gaussian-set validation, distance, interpolation, and immutability.
- [x] Task 8: Add a small visualization/demo artifact showing interpolated Gaussian parameters before any renderer exists.
- [x] Task 9: Run `ruff check`, `ruff format`, `pyright`, and full pytest.
- [x] Task 10: ADR check: geometry-keyed and geometry-dispatch ADRs are already validated; this sprint exercises their first structured case.
- [x] Task 11: Update artifact summary, `CHANGELOG.md`, and `docs/PLAN.md`.

## Rule-of-Three Checkpoint
| Check | Status |
|---|---|
| Geometry instances | Euclidean (#1), unit_norm (#2), gaussian_set (#3) |
| Rule branch | Instance #3 → extract only the dispatch abstraction code actually needs |
| Public API | Preserve existing `dim` ergonomics for flat spaces |

## Notes / Blockers
* This sprint is about latent geometry, not rendering.
* Keep the Gaussian parameter schema minimal. The renderer sprint can add fields only if real code needs them.

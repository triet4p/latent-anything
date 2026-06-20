# Sprint 16 Plan

## Sprint Goal
Increment thứ mười ba (Round 13): thêm **GaussianRendererAdapter** — `ModelAdapter` mode (iii), explicit structured representation with deterministic decode. It uses Gaussian-set latents and a small numpy renderer to decode into image-like data.

## Why This Sprint
This closes the last planned `ModelAdapter` ADR mode: a latent representation whose decode path is deterministic infrastructure, not a learned neural decoder.

## Atomic Tasks
Status legend: [ ] pending / [~] in progress / [x] done

- [ ] Task 1: Implement `GaussianRendererAdapter` in `src/latent_anything/adapters/gaussian_renderer.py`.
- [ ] Task 2: Use the Sprint 15 `gaussian_set` `LatentSpace` metadata as the adapter's `latent_space`.
- [ ] Task 3: Implement deterministic `decode(latent: np.ndarray) -> np.ndarray` with a tiny 2D Gaussian splat renderer.
- [ ] Task 4: Implement `encode(data: np.ndarray) -> np.ndarray` only if a simple, honest synthetic path exists; otherwise document the adapter as latent-source-first and keep tests around decode + latent_space.
- [ ] Task 5: Make protocol typing precise: this adapter is decodable, but not learned-decoder-based.
- [ ] Task 6: Add tests for output shape, opacity/color constraints, deterministic decode, protocol conformance, and no mutation.
- [ ] Task 7: Add a demo: Gaussian-set latent → rendered image grid → interpolation sequence.
- [ ] Task 8: Run `ruff check`, `ruff format`, `pyright`, and full pytest.
- [ ] Task 9: ADR check: validate `ModelAdapter` mode (iii); if all three modes are now covered, append a final validation entry.
- [ ] Task 10: Update artifact summary, `CHANGELOG.md`, and `docs/PLAN.md`.

## Rule-of-Three Checkpoint
| Check | Status |
|---|---|
| ModelAdapter modes | explicit learned, no-explicit hidden-state, deterministic renderer |
| ADR impact | `ModelAdapter` 3-mode ADR should move to validated if implementation confirms the shape |
| Interface risk | If decode optionality from Sprint 14 is wrong, write a new ADR rather than silently bending it |

## Notes / Blockers
* Do not add `gsplat`, CUDA, or heavy rendering dependencies here.
* The goal is interface evidence, not photorealistic rendering.

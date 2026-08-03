# Sprint 54 Plan

## Sprint Goal

Integrate a real 3D Gaussian splatting renderer backend and replace the simplified 2D renderer as the headline structured-rendering evidence.

## Atomic Tasks

Status legend: [ ] pending / [~] in progress / [x] done

- [x] Select a maintained 3DGS backend/checkpoint format and pin a compatible optional-extra range.
- [x] Define 3D Gaussian latent metadata for position, rotation/covariance, scale, opacity, spherical harmonics, coordinate frame, and camera parameters.
- [x] Implement deterministic decode/render for at least one camera batch through the existing adapter/latent-value boundary.
- [x] Validate backend parity, image shape/range, camera transforms, and Gaussian parameter constraints.
- [x] Add tiny-scene unit fixtures plus a marked GPU integration test on a public scene.
- [x] Keep rendering kernels/backend glue outside the public adapter facade and preserve the 2D renderer as a lightweight fixture if useful.
- [x] Measure render quality and performance against direct backend execution.
- [x] Update evidence/ADR/changelog/artifact and gates.

## Notes / Blockers

Backend licensing, CUDA availability, and checkpoint redistribution are handled by the optional gsplat boundary; the public-scene GPU test remains opt-in.

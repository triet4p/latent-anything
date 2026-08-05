# Sprint 59 Plan

## Sprint Goal

Capture and analyze LeRobot Diffusion Policy representations across denoising timesteps and action chunks.

## Atomic Tasks

Status legend: [ ] pending / [~] in progress / [x] done

- [x] Select and revision-pin a public Diffusion Policy checkpoint, dataset, and compatible environment/task.
- [x] Use official LeRobot processors and policy construction to preserve input/output normalization.
- [x] Capture one observation-conditioning representation and one denoising/action representation with timestep metadata.
- [x] Verify unmodified action chunks match direct LeRobot inference for fixed seeds/noise.
- [x] Analyze latent trajectories across denoising timesteps and episode time without conflating the axes.
- [x] Compare successful/failure cases using probe, density, and trajectory metrics with negative controls.
- [x] Add offline fixtures and marked checkpoint integration tests.
- [x] Update evidence/ADR/changelog/artifact and gates.

## Notes / Blockers

This integration stress-tests multi-axis latent semantics: environment time, action-chunk index, and diffusion timestep must remain explicit.

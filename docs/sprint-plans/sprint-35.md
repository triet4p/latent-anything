# Sprint 35 Plan

## Sprint Goal

Add a pretrained Diffusers `AutoencoderKL` adapter and prove latent round-trips on real images with version-pinned behavior.

## Atomic Tasks

Status legend: [ ] pending / [~] in progress / [x] done

- [x] Select a public pretrained `AutoencoderKL` checkpoint and pin its model revision for tests/benchmarks.
- [x] Implement encode/decode, scaling-factor handling, stochastic-vs-mean latent selection, device/dtype control, and latent-space metadata.
- [x] Keep all Diffusers/Torch objects inside the optional integration module and NumPy or latent-value objects at the public boundary.
- [x] Add image range/layout validation and round-trip tests for batch, channel, and spatial shapes.
- [x] Compare reconstruction metrics against direct backend execution to prove adapter fidelity.
- [x] Add offline cached-checkpoint tests and a network-marked acquisition smoke test.
- [x] Produce a reproducible real-image latent interpolation artifact with density/norm diagnostics.
- [x] Update evidence, ADR/changelog/artifact, compatibility pins, and gates.

## Notes / Blockers

This sprint adapts only the autoencoder component. Denoising timesteps and pipeline-level latent capture belong to Sprint 37.

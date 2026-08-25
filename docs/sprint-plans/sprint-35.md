# Sprint 35 Plan

## Sprint Goal

Add a pretrained Diffusers `AutoencoderKL` adapter and prove latent round-trips on real images with version-pinned behavior.

## Atomic Tasks

Status legend: [ ] pending / [~] in progress / [x] done

- [x] Select a public pretrained `AutoencoderKL` checkpoint and pin its model revision for tests/benchmarks.
- [x] Implement encode/decode, scaling-factor handling, stochastic-vs-mean latent selection, device/dtype control, and latent-space metadata.
- [x] Keep all Diffusers/Torch objects inside the optional integration module and NumPy or latent-value objects at the public boundary.
- [x] Add image range/layout validation and round-trip tests for batch, channel, and spatial shapes.
- [x] Compare reconstruction metrics against direct backend execution to prove pinned-checkpoint fidelity.
- [x] Add offline cached-checkpoint tests and a network-marked acquisition smoke test.
- [x] Produce a reproducible real-image latent interpolation artifact with density/norm diagnostics.
- [x] Update evidence, ADR/changelog/artifact, compatibility pins, and gates.

## Notes / Blockers

This sprint adapts only the autoencoder component. Denoising timesteps and pipeline-level latent capture belong to Sprint 37.

## Carryover Closure (2026-08-26)

The adapter implementation, revision-pinned fidelity gate, and interpolation
gate are delivered against the already cached checkpoint. Evidence remains a
bounded local CPU D2 lane and must not be generalized to perceptual quality or
the full diffusion pipeline.

- [x] C35.1: Freeze provenance, direct-backend comparison semantics, tolerances, and bounded CPU/RAM acceptance.
- [x] C35.2: Produce and validate the local-only direct-versus-adapter fidelity evidence script, regression test, and artifact.
- [x] C35.3: Reconcile evidence ledger, plan, changelog, and closure gates after C35.2 passed.
- [x] C35.4: Produce and validate the local-only interpolation JSON/PNG artifact with endpoint, movement, reproducibility, and resource gates.

C35.1 acceptance requires the exact model ID/revision and file hashes/sizes,
MIT model-card provenance, locked Diffusers/safetensors/Torch versions,
identical deterministic NCHW inputs, explicit mean and seeded posterior-sample
semantics, scaling-factor parity, latent/decode shape/dtype/device/finiteness
checks, tolerances declared before measurement, local-only safetensors loading,
no remote code or network, and bounded CPU runtime/RAM. C35.2 and C35.4 both
pass; together they close the two Sprint 35 carryover gates within the stated
local CPU D2 scope.

## C35.4 Evidence

The canonical artifacts are `artifacts/diffusers_vae_digits_interpolation.json`
and `artifacts/diffusers_vae_digits_interpolation.png`. The seven ordered
coefficients preserve digit-0/digit-1 endpoints, exceed latent and decoded
movement thresholds, reproduce identical JSON/PNG digests in two in-process
runs, and record zero network attempts. This is not a perceptual-quality
benchmark or a claim about a complete diffusion pipeline.

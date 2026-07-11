# Sprint 37 Plan

## Sprint Goal

Add a real diffusion-pipeline adapter that captures timestep-, block-, and conditioning-aware representations without pretending they form one homogeneous latent space.

## Atomic Tasks

Status legend: [ ] pending / [~] in progress / [x] done

- [ ] Select and revision-pin a compact Diffusers text-to-image or unconditional pipeline with feasible CI smoke behavior.
- [ ] Implement one concrete capture path using Sprint 32 lifecycle for VAE latents and denoiser activations.
- [ ] Record timestep, scheduler, prompt/conditioning, block location, tensor axes, seed, and checkpoint revision in metadata.
- [ ] Define separate latent-space descriptors where VAE bottlenecks and denoiser hidden states have different semantics.
- [ ] Verify generated output and captured activations match direct backend execution within declared tolerance.
- [ ] Add CPU/tiny-checkpoint tests plus a marked full-checkpoint integration test.
- [ ] Produce a timestep trajectory artifact with norm, similarity, and reconstruction diagnostics.
- [ ] Reconcile adapter/capture abstractions and update evidence/changelog/artifact/gates.

## Notes / Blockers

The adapter must preserve the distinction between explicit compressed latents and internal activations highlighted by theory tier 14.


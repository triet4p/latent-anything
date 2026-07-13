# Sprint 37 Plan

## Sprint Goal

Add a concrete, revision-pinned conditional diffusion integration that records scheduler latent states and selected denoiser activations without forcing generative execution into the frozen `ModelAdapter.encode()` contract.

## Entry Criteria

- Sprint 35 real-checkpoint fidelity evidence is complete.
- Sprint 35 real-checkpoint interpolation artifact is complete.

## Atomic Tasks

Status legend: [ ] pending / [~] in progress / [x] done

- [ ] Select and revision-pin one compact conditional text-to-image pipeline plus one tiny/offline CI fixture; declare tested Diffusers/Transformers ranges and one complete optional installation path.
- [ ] Define concrete typed generation request/result objects with NumPy-facing public values; document why the generate lifecycle is not a `ModelAdapter.encode()` implementation and add no new framework protocol.
- [ ] Capture scheduler latent states through Diffusers' native `callback_on_step_end` under fixed prompt, scheduler, seed, and initial noise, with defensive ownership of captured values.
- [ ] Capture one semantically justified tensor-producing denoiser location through `ActivationCaptureSession`; use an integration-local output selector only if the upstream output is structured.
- [ ] Define separate `LatentSpace`/`LatentValue` descriptors for VAE bottlenecks, scheduler states, and denoiser activations, including axes, layout, timestep, conditioning, checkpoint, and scheduler provenance.
- [ ] Verify generated output, scheduler states, and selected activations against direct backend execution under fixed inputs and declared tolerances.
- [ ] Add deterministic tiny/offline CPU tests and marked real-checkpoint tests (`network`, `large_download`, or `gpu` where applicable) with no implicit checkpoint download.
- [ ] Produce a timestep-trajectory artifact with norm, cosine similarity, decoded intermediate states where supported, and explicit failed/unsupported cases.
- [ ] Reconcile the adapter/capture ADR, then update theory evidence, changelog, artifact index, and quality gates.

## Notes / Blockers

Scheduler state capture uses the backend-native callback; hooks remain reserved for internal module activations. VAE bottlenecks, scheduler states, and denoiser activations are related but non-homogeneous representation spaces. This first concrete generative integration must not create a general protocol before repeated use proves one.


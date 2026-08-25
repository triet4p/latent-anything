# Sprint 35 Carryover Task C35.1 — Fidelity Acceptance

**Status:** Complete

## Acceptance contract

- Checkpoint: `stabilityai/sd-vae-ft-mse` at revision
  `31f26fdeee1355a5c34592e401dd41e45d25a493`.
- Runtime files: `config.json` (547 bytes) and
  `diffusion_pytorch_model.safetensors` (334,643,276 bytes); the safetensors
  SHA-256 is
  `a1d993488569e928462932c8c38a0760b874d166399b14414135bd9c42df5815`.
  The pickle-based `.bin` file is explicitly excluded.
- Provenance: public, ungated, MIT model-card license; locked versions are
  Diffusers 0.39.0, safetensors 0.8.0, and Torch 2.10.0.
- Comparison: direct `diffusers.AutoencoderKL` and the public
  `DiffusersAutoencoderKLAdapter` must consume byte-identical deterministic
  NCHW arrays from the same local snapshot.
- Semantics: mean mode compares posterior means; sample mode compares seeded
  posterior samples with the same Torch seed and explicit RNG reset. Adapter
  scaling must equal direct `latent * config.scaling_factor`, and decode must
  divide by that factor exactly once.
- Hard checks: latent/decode shape agreement; declared NumPy dtype and CPU
  device; finite values; direct-versus-adapter absolute/relative tolerances
  `rtol=1e-5`, `atol=1e-6` for float32; local-only safetensors loading; no
  remote code or socket connection; bounded CPU runtime (60 seconds) and
  resident-memory peak (2 GiB for the smoke lane).
- Scope: a pass closes only the Sprint 35 fidelity gate. The interpolation
  artifact remains open and is not run here.

## Plan and validation

`docs/PLAN.md` and `docs/sprint-plans/sprint-35.md` now identify Sprint 35 as
the active carryover and mark C35.1 complete. `git diff --check` passed before
the evidence implementation. No product code, dependency declarations, or
checkpoint files were changed by this acceptance task.

## Graph refresh

- Command: `graphify update .`
- Result: exit 0; rebuilt graph with **10,088 nodes, 19,593 edges, and 904
  communities**. It reported 48 known zero-node JSON/source warnings, backed
  up semantic/curated graph files, and refreshed `graphify-out` successfully.

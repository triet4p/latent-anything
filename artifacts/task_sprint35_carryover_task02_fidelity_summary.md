# Sprint 35 Carryover Task C35.2 — Revision-Pinned Fidelity

**Status:** Complete

## Summary

Implemented a reusable local-only fidelity lane comparing direct Diffusers
`AutoencoderKL` execution with `DiffusersAutoencoderKLAdapter` on the same
verified snapshot and deterministic NCHW inputs. The lane covers mean and
seeded posterior-sample semantics, scaling-factor parity, shape/dtype/device
and finiteness checks, exact file provenance, socket-level network denial, and
CPU/RSS bounds. It writes `artifacts/diffusers_vae_fidelity.json` only after
all hard checks pass.

## Root cause and remediation

The first sample comparison consumed the global Torch RNG in the direct path
before the adapter path, and the repeat check reused an advanced RNG state.
Separately, the adapter loaded the backend in training mode, which caused an
initial mean-mode mismatch under the reusable lane. The adapter now calls
`.eval()` after loading, and `encode(..., seed=...)` creates a local CPU/device
Torch generator for reproducible posterior samples while preserving the
existing unseeded behavior. A regression test covers both eval-mode loading
and local seeded sampling. Deterministic Torch algorithms are enabled by the
evidence harness; tolerances were not loosened.

## Evidence

- Model: `stabilityai/sd-vae-ft-mse` at
  `31f26fdeee1355a5c34592e401dd41e45d25a493`.
- Safetensors: 334,643,276 bytes; SHA-256
  `a1d993488569e928462932c8c38a0760b874d166399b14414135bd9c42df5815`.
- Runtime: Diffusers 0.39.0, safetensors 0.8.0, Torch 2.10.0+cpu.
- Inputs: deterministic float32 NCHW `(2, 3, 32, 32)` gradient/sinusoid
  fixtures in `[-1, 1]`; latent shape `(2, 4, 4, 4)`.
- Mean mode seed 1234: latent and decoded max absolute error **0.0**;
  finite float32 CPU outputs.
- Seeded sample mode seed 5678: latent and decoded max absolute error
  **0.0**; repeated adapter sample with the same seed is bitwise identical.
- Scaling factor: `0.18215`; adapter latent-space dimension: 4; revision
  metadata is preserved.
- Local-only socket denial: **0** network attempts; no remote code (`auto_map`
  absent and no Python files in the snapshot).
- Runtime: **2.7973485s**; maximum observed RSS **1,446,883,328 bytes**;
  minimum available physical RAM **3,083,132,928 bytes**. Both are within
  the 60-second / 2-GiB acceptance bounds.

## Validation

- `uv run --offline --extra diffusers python scripts/diffusers_vae_fidelity.py`
  — passed and generated the JSON artifact.
- `LATENT_ANYTHING_RUN_REAL_CHECKPOINT=1 uv run --offline --extra diffusers
  pytest tests/test_diffusers_vae.py tests/test_diffusers_vae_fidelity.py -q`
  — **7 passed**.
- `uv run pytest tests/test_diffusers_vae.py::test_backend_sets_eval_mode_after_device_and_dtype -q`
  — **1 passed** during remediation.

## Files

- `scripts/diffusers_vae_fidelity.py`
- `tests/test_diffusers_vae_fidelity.py`
- `tests/test_diffusers_vae.py`
- `src/latent_anything/integrations/diffusers_vae.py`
- `artifacts/diffusers_vae_fidelity.json`

## Scope

This closes only the Sprint 35 real revision-pinned fidelity gate. It does not
claim the interpolation artifact or complete Milestone 8.

## Graph refresh

- Command: `graphify update .`
- Result: exit 0; rebuilt graph with **10,122 nodes, 19,655 edges, and 919
  communities**. It reported 49 known zero-node JSON/source warnings, backed
  up semantic/curated graph files, and refreshed `graphify-out` successfully.

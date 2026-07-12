# Task Summary: Sprint 35 — Diffusers AutoencoderKL Boundary

Implemented a lazy, pinned `AutoencoderKL` adapter with NumPy public values,
symmetric scaling-factor handling, mean/sample selection, dtype/device controls,
and structured `LatentValue` output. Offline fake-backend tests prove exact
adapter/backend scaling parity; a separately marked test can acquire the exact
checkpoint only with explicit opt-in. The interpolation script records input
contract, model revision, latent norms, and near-zero density as D1 evidence.

Rule of Three: this is ModelAdapter instance #5 but only the second optional
pretrained integration. It intentionally stays a concrete integration module;
no new public adapter Protocol is extracted. Existing protocol evidence is not
contradicted.

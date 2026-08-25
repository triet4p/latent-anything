# Diffusers AutoencoderKL Integration

The Sprint 35 adapter targets `stabilityai/sd-vae-ft-mse` at immutable revision
`31f26fdeee1355a5c34592e401dd41e45d25a493`. It is an `AutoencoderKL` with four
latent channels. Unit tests use an injected fake backend; the real fidelity and
interpolation lanes consume only the already verified local snapshot and deny
socket connections. No normal test acquires a model.

The adapter exposes only NumPy values, applies the backend scaling factor
symmetrically in encode/decode, and represents `LatentValue` outputs as NHWC so
its trailing channel axis matches `LatentSpace(dim=4)`. The reproducibility
script uses distinct sklearn-digits endpoints and writes a D2 JSON/PNG
interpolation artifact with endpoint, movement, digest, and resource checks.
The evidence is bounded local CPU validation, not perceptual-quality evidence.

The current fake-backend round trip is D1 adapter coverage; the cached
revision-pinned fidelity and interpolation artifacts are D2. Run the evidence
script only with the verified snapshot already present, using offline mode:
`uv run --offline --extra diffusers python scripts/diffusers_vae_interpolation.py`.

# Diffusers AutoencoderKL Integration

The Sprint 35 adapter targets `stabilityai/sd-vae-ft-mse` at immutable revision
`31f26fdeee1355a5c34592e401dd41e45d25a493`. It is an `AutoencoderKL` with four
latent channels. Unit tests use an injected fake backend; cache-backed model
acquisition is explicitly marked `network` and requires
`LATENT_ANYTHING_RUN_NETWORK=1`. No normal unit test may acquire a model.

The adapter exposes only NumPy values, applies the backend scaling factor
symmetrically in encode/decode, and represents `LatentValue` outputs as NHWC so
its trailing channel axis matches `LatentSpace(dim=4)`. The reproducibility
script uses real sklearn digits inputs and writes interpolation, norm, and
near-zero-density diagnostics. It is D1 integration evidence until repeated
controls promote it further.

The current fake-backend round trip is D1 fidelity coverage. Run the script
only in an environment permitted to download the pinned checkpoint:
`uv run --extra diffusers python scripts/diffusers_vae_interpolation.py`.

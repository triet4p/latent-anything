# Train a VAE or connect a pretrained AutoencoderKL

## When to use this API

Use the built-in [`VAE`](https://github.com/triet4p/latent-anything/blob/v1.0.0/src/latent_anything/adapters/vae.py) for small batches of flat, numeric feature vectors, or [`ConvVAE`](https://github.com/triet4p/latent-anything/blob/v1.0.0/src/latent_anything/adapters/conv_vae.py) for single-channel 8×8 images. Both train their own weights and expose a Euclidean latent space through the released adapter API. They are compact examples and CPU-sized models, not pretrained general-purpose image encoders.

If you already have a pretrained Diffusers `AutoencoderKL`, use the separate optional integration seam near the end of this page. It is not needed by either built-in model.

## Prerequisites

The examples use the published `latent-anything==1.0.0` package and Python `>=3.12,<3.15`:

```bash
python -m pip install "latent-anything==1.0.0"
```

The base install includes PyTorch; no extra, dataset download, or model checkpoint is needed for the two built-in examples. Add `diffusers` only for the optional pretrained route:

```bash
python -m pip install "latent-anything[diffusers]==1.0.0"
```

That extra installs Diffusers and safetensors. The pretrained example below also requires that the exact model snapshot already be available in the local Hugging Face cache; it is deliberately offline and does not acquire model files.

## Choose the built-in model by input shape

| Model | `fit` input | `encode` output | `decode` output | Input convention |
| --- | --- | --- | --- | --- |
| `VAE(input_dim, latent_dim, ...)` | `(n, input_dim)` | `(n, latent_dim)` | `(n, input_dim)` | Flatten each example to a feature vector; scale values to `[0, 1]` |
| `ConvVAE(latent_dim, ...)` | `(n, 1, 8, 8)` | `(n, latent_dim)` | `(n, 1, 8, 8)` | One grayscale channel in NCHW order; scale pixels to `[0, 1]` |

Both models are trainable: construct a model, call `fit()` on the training data, and then call `encode()` and/or `decode()`. The `VAE` rejects encode/decode calls before fitting; use fitted weights for `ConvVAE` too. Keep held-out evaluation examples out of `fit()`.

The latent sampling step is part of training: each fit pass samples from the approximate posterior for its reconstruction loss. After fitting, `encode()` returns the posterior **mean** (`μ`) as a deterministic NumPy array for analysis. It does not return a posterior sample or expose a `latent_mode` switch. `decode(z)` deterministically maps the caller-provided latent batch back to the model's input shape; the built-in classes do not provide a separate prior-sampling helper.

## Runnable CPU example: fit, encode, and decode

This small example uses seeded synthetic values so it needs no files or network. Save it as `vae_paths_example.py` and run it after installing the base package. It verifies both input/output contracts and that repeated `encode()` calls return the same means. The synthetic data is for exercising the API only, not a reconstruction-quality benchmark.

```python
from __future__ import annotations

import numpy as np

from latent_anything.adapters import ConvVAE, VAE

rng = np.random.default_rng(7)

flat = rng.uniform(0.0, 1.0, size=(6, 3)).astype(np.float32)
vae = VAE(input_dim=3, latent_dim=2, n_epochs=4, random_state=7)
vae.fit(flat)
flat_mean = vae.encode(flat)
flat_mean_again = vae.encode(flat)
flat_reconstruction = vae.decode(flat_mean)
assert flat_mean.shape == (6, 2)
assert flat_reconstruction.shape == flat.shape
np.testing.assert_array_equal(flat_mean, flat_mean_again)
print(
    f"VAE: input={flat.shape}, latent_mean={flat_mean.shape}, "
    f"decoded={flat_reconstruction.shape}; encode deterministic"
)

images = rng.uniform(0.0, 1.0, size=(4, 1, 8, 8)).astype(np.float32)
conv_vae = ConvVAE(latent_dim=3, random_state=7, n_epochs=2)
conv_vae.fit(images)
image_mean = conv_vae.encode(images)
image_mean_again = conv_vae.encode(images)
image_reconstruction = conv_vae.decode(image_mean)
assert image_mean.shape == (4, 3)
assert image_reconstruction.shape == images.shape
np.testing.assert_array_equal(image_mean, image_mean_again)
print(
    f"ConvVAE: input={images.shape}, latent_mean={image_mean.shape}, "
    f"decoded={image_reconstruction.shape}; encode deterministic"
)
```

Expected output:

```text
VAE: input=(6, 3), latent_mean=(6, 2), decoded=(6, 3); encode deterministic
ConvVAE: input=(4, 1, 8, 8), latent_mean=(4, 3), decoded=(4, 1, 8, 8); encode deterministic
```

For your own data, scale features or pixels into `[0, 1]` before fitting, preserve the model's required sample axes, and evaluate quality on held-out data against declared baselines. `VAE` is for flat vectors; do not pass an image tensor to it without flattening and choosing the matching `input_dim`. `ConvVAE` is specifically the built-in 8×8, one-channel image model, not a general-resolution RGB architecture.

## Optional seam: pretrained Diffusers AutoencoderKL

The optional integration is imported from `latent_anything.integrations.diffusers_vae`; it is separate from the frozen `latent_anything.adapters` re-export list. Its established evidence target is `stabilityai/sd-vae-ft-mse` at immutable revision `31f26fdeee1355a5c34592e401dd41e45d25a493` (four latent channels). The cached `diffusion_pytorch_model.safetensors` used for the local evidence has SHA-256 `a1d993488569e928462932c8c38a0760b874d166399b14414135bd9c42df5815` and size `334643276` bytes.

**Do not run this example until the exact revision is already cached locally.** The adapter loads the model lazily by model ID and revision. The code below sets `HF_HUB_OFFLINE=1` before importing the integration; you can also set it before starting Python (for example, PowerShell: `$env:HF_HUB_OFFLINE = "1"`). On macOS/Linux, prefix the launch command with `HF_HUB_OFFLINE=1`. This makes a missing snapshot fail locally instead of triggering acquisition. It is important because the adapter's `from_pretrained()` call does not itself pass `local_files_only=True`.

```python
from __future__ import annotations

import os

# Set this before importing/using Diffusers or Hugging Face Hub.
os.environ["HF_HUB_OFFLINE"] = "1"

import numpy as np

from latent_anything.integrations.diffusers_vae import DiffusersAutoencoderKLAdapter

adapter = DiffusersAutoencoderKLAdapter(
    "stabilityai/sd-vae-ft-mse",
    "31f26fdeee1355a5c34592e401dd41e45d25a493",
    device="cpu",
    latent_mode="mean",
)
images_nchw = np.zeros((1, 3, 32, 64), dtype=np.float32)  # RGB, [-1, 1], channels first
latent_nchw = adapter.encode(images_nchw)                 # scaled NCHW: (1, 4, 4, 8)
latent_value = adapter.encode_value(images_nchw)           # LatentValue data is NHWC: (1, 4, 8, 4)
latent_nhwc = latent_value.to_numpy()
np.testing.assert_allclose(latent_nhwc, np.moveaxis(latent_nchw, 1, -1))
reconstruction_nchw = adapter.decode(latent_nchw)          # decode takes scaled NCHW
assert reconstruction_nchw.shape == images_nchw.shape
```

`encode()` accepts finite NCHW image batches in `[-1, 1]` and returns scaled NCHW latent arrays. The adapter applies the checkpoint's `scaling_factor` during encode and divides by the same factor before decode. `encode_value()` instead moves the channel axis to the end (NHWC), so the channel count matches `LatentSpace(dim=4)`; transpose back to NCHW before passing that value to `decode()`.

The default `latent_mode="mean"` uses the posterior mean. To use a posterior draw, construct with `latent_mode="sample"` and pass a fixed `seed` to `encode(images_nchw, seed=17)` when reproducibility matters. Sampling differs from the deterministic built-in `encode()` methods.

This optional snippet is intentionally **not executed** as part of this page's CPU example: it requires the pinned snapshot, Diffusers/safetensors extras, and a matching local cache. This task does not acquire the model files. Keep `HF_HUB_OFFLINE=1`: the adapter's ordinary `from_pretrained()` path can contact the Hub if it is run without that offline guard. For the already-verified local evidence path and its artifacts, see the [Diffusers integration guide](https://github.com/triet4p/latent-anything/blob/main/docs/DIFFUSERS_INTEGRATION.md).

## Evidence boundaries

A successful small example is a smoke check of calls and shapes; it does not show that learned features are useful or reconstructions are good. D1 means an implementation and focused tests exist. D2 means a separately defined end-to-end evidence lane passed its quantitative checks; neither label implies general model quality.

- The project records a D2 `ConvVAE` held-out sklearn-digits lane: fit on 1,437 training images, evaluate 360 held-out images, and compare held-out reconstruction MSE `0.1717` with an all-zero baseline `0.2359`. The acceptance gate passed, but the train-pixel-mean diagnostic is stronger (`0.0731` MSE). This is bounded evidence for that compact 8×8 grayscale model, data split, and CPU run—not for arbitrary images, architectures, or pretrained image generation.
- The optional Diffusers adapter's fake-backend round trip is D1 coverage. Its pinned, cached CPU fidelity and interpolation artifacts are D2 evidence; they establish the named adapter/checkpoint path, not perceptual quality or a general diffusion-pipeline claim.
- The synthetic snippet above and your own unbenchmarked data remain usage demonstrations, not D2 evidence. No D3 or general pretrained quality claim is made.

For the release's canonical import surface, signatures, and optional-profile boundaries, use the [frozen API reference](https://github.com/triet4p/latent-anything/blob/v1.0.0/docs/API_REFERENCE.md) and [1.0.0 API freeze snapshot](https://github.com/triet4p/latent-anything/blob/v1.0.0/artifacts/api_freeze_snapshot_1.0.0.json). For controlled latent-explanation claims specifically, see the [VAE explanation benchmark](https://github.com/triet4p/latent-anything/blob/main/docs/VAE_EXPLANATION_BENCHMARK.md); passing a representation-fit smoke test alone does not meet that benchmark's intervention and control requirements. Return to the [API guide home](index.md) for other entry paths.

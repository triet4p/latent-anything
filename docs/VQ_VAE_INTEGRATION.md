# VQ-VAE discrete latent integration

Sprint 70 adds `latent_anything.adapters.VQVAE`, a compact trainable VQ-VAE
for offline 8×8 grayscale evidence. It is intentionally a small model rather
than a claim about a large pretrained VQGAN or tokenized world model.

## Representation contract

`encode(images)` returns an integer NumPy array with shape `(n_samples, 16)`.
The 16 values are categorical IDs for a 4×4 latent grid. The adapter declares
`LatentSpace(geometry="discrete_code")`, records the codebook size, and keeps
the dataset/model revision in metadata. `encode_value()` preserves the integer
dtype in an immutable `LatentValue`.

Continuous codebook vectors are available only through the explicitly named
`code_embeddings(codes)` method. `decode()` accepts integer sequences, while
`interpolate_codes()` and `latent_space.interpolate()` reject continuous
interpolation. Code edits must use `replace_codes()` with an integer-to-integer
replacement map.

## Diagnostics and evidence

The adapter reports reconstruction MSE, codebook loss, commitment loss and
distance, codebook perplexity, dead-code rate, and train/test code-frequency
drift. The reproducible comparison also trains the existing continuous
`ConvVAE` on the same pinned split so the two analysis paths are not conflated.

Run the offline evidence path with:

```text
uv run python scripts/vq_vae_digits_evidence.py
```

It writes `artifacts/vq_vae_digits_evidence.json` and
`artifacts/vq_vae_digits_evidence_config.json`. The compact run now uses
deterministic spread initialization for the codebook, and the artifact requires
perplexity above `1.0` plus a dead-code rate below `1.0`; it remains diagnostic
synthetic CPU evidence rather than a claim about a large pretrained VQGAN. The
dataset is `sklearn.datasets.load_digits` from the locked
`scikit-learn==1.9.0` profile, and the model revision is `compact-vq-vae-v1`.

The focused tests are offline and run in the optional-extras workflow's base
profile:

```text
uv run pytest tests/test_latent_anything/test_vq_vae.py tests/test_vq_vae_benchmark.py -q
```

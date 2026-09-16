"""Run the M14 L09 real Diffusers AutoencoderKL parity checks."""
from __future__ import annotations

import importlib.metadata
import platform

import numpy as np
import psutil

from latent_anything.integrations.diffusers_vae import DiffusersAutoencoderKLAdapter

MODEL_ID = "stabilityai/sd-vae-ft-mse"
MODEL_REVISION = "31f26fdeee1355a5c34592e401dd41e45d25a493"


def main() -> None:
    process = psutil.Process()
    rss_peak = [process.memory_info().rss]
    def sample_rss() -> None:
        rss_peak[0] = max(rss_peak[0], process.memory_info().rss)
    rng = np.random.default_rng(7)
    images = rng.uniform(-1, 1, size=(2, 3, 32, 32)).astype(np.float32)
    mean_adapter = DiffusersAutoencoderKLAdapter(MODEL_ID, MODEL_REVISION, latent_mode="mean")
    latent = mean_adapter.encode(images)
    sample_rss()
    decoded = mean_adapter.decode(latent)
    sample_rss()
    backend = mean_adapter._backend()  # type: ignore[reportPrivateUsage]
    sample_rss()
    import torch

    with torch.no_grad():
        direct = backend.encode(torch.from_numpy(images)).latent_dist.mean
        direct = (direct * backend.config.scaling_factor).cpu().numpy()
    sample_rss()
    sampled = DiffusersAutoencoderKLAdapter(MODEL_ID, MODEL_REVISION, latent_mode="sample")
    sample_a = sampled.encode(images, seed=123)
    sample_rss()
    sample_b = sampled.encode(images, seed=123)
    sample_rss()
    sample_c = sampled.encode(images, seed=124)
    sample_rss()
    print(
        {
            "direct_adapter_max_abs_error": float(np.max(np.abs(latent - direct))),
            "sample_seed_123_reproducible": bool(np.array_equal(sample_a, sample_b)),
            "different_seed_changes_sample": bool(not np.array_equal(sample_a, sample_c)),
            "latent_shape": latent.shape,
            "decoded_shape": decoded.shape,
            "finite": bool(np.isfinite(latent).all() and np.isfinite(decoded).all()),
            "rss_peak_bytes": rss_peak[0],
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": importlib.metadata.version("torch"),
            "diffusers": importlib.metadata.version("diffusers"),
            "huggingface_hub": importlib.metadata.version("huggingface-hub"),
        }
    )


if __name__ == "__main__":
    main()

"""Run the M14 L09 real Diffusers AutoencoderKL parity checks."""

from __future__ import annotations

import importlib.metadata
import json
import platform
from pathlib import Path

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
    payload = {
        "schema_version": "m14-l09-run-v1",
        "model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "weights_sha256": "a1d993488569e928462932c8c38a0760b874d166399b14414135bd9c42df5815",
            "license": "MIT",
        },
        "inputs": {"shape": list(images.shape), "dtype": str(images.dtype), "input_rng_seed": 7},
        "sampling_seeds": {"seed_a": 123, "seed_b": 123, "seed_c": 124},
        "metrics": {
            "direct_adapter_max_abs_error": float(np.max(np.abs(latent - direct))),
            "sample_seed_123_reproducible": bool(np.array_equal(sample_a, sample_b)),
            "different_seed_changes_sample": bool(not np.array_equal(sample_a, sample_c)),
            "latent_shape": list(latent.shape),
            "decoded_shape": list(decoded.shape),
            "finite": bool(np.isfinite(latent).all() and np.isfinite(decoded).all()),
        },
        "environment": {
            "rss_peak_bytes": rss_peak[0],
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": importlib.metadata.version("torch"),
            "diffusers": importlib.metadata.version("diffusers"),
            "huggingface_hub": importlib.metadata.version("huggingface-hub"),
            "network_policy": (
                "HF cache-only after initial pinned acquisition; no model download during final measured rerun"
            ),
        },
    }
    artifact_path = Path("artifacts/m14/l09-diffusers-vae-run.json")
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()

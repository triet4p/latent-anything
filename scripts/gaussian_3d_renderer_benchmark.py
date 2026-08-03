"""Reproducible CPU reference benchmark for the Sprint 54 renderer seam."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from latent_anything.adapters import Gaussian3DRendererAdapter, GaussianCamera
from latent_anything.integrations.gsplat_renderer import ReferenceGaussianBackend


def main() -> None:
    camera = GaussianCamera(64, 48, np.array([[45.0, 0.0, 32.0], [0.0, 45.0, 24.0], [0.0, 0.0, 1.0]]), np.eye(4))
    latent = np.zeros((32, 14), dtype=np.float64)
    latent[:, 2] = 2.0
    latent[:, 6] = 1.0
    latent[:, 7:10] = 0.08
    latent[:, 10] = 0.7
    latent[:, 11:14] = 0.5
    adapter = Gaussian3DRendererAdapter(32, camera, backend=ReferenceGaussianBackend())
    start = time.perf_counter()
    image = adapter.decode(latent)
    elapsed = time.perf_counter() - start
    result = {
        "backend": "reference",
        "width": 64,
        "height": 48,
        "n_gaussians": 32,
        "seconds": elapsed,
        "image_min": float(image.min()),
        "image_max": float(image.max()),
        "shape": list(image.shape),
        "gpu_backend": "not measured: CUDA/gsplat unavailable",
    }
    Path("artifacts/gaussian_3d_renderer_benchmark.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

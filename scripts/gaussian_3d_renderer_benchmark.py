"""Reproducible multi-view CPU benchmark for Sprint 55 Gaussian edits."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from latent_anything.adapters import Gaussian3DRendererAdapter, GaussianCamera
from latent_anything.gaussian_3d import edit_opacity, naive_parameter_arithmetic, rigid_transform
from latent_anything.gaussian_3d_evaluation import evaluate_multiview
from latent_anything.integrations.gsplat_renderer import ReferenceGaussianBackend
from latent_anything.pose import SE3, SO3


def main() -> None:
    intrinsics = np.array([[45.0, 0.0, 32.0], [0.0, 45.0, 24.0], [0.0, 0.0, 1.0]])
    camera = GaussianCamera(64, 48, intrinsics, np.eye(4))
    latent = np.zeros((8, 14), dtype=np.float64)
    latent[:, 0] = np.linspace(-0.5, 0.5, len(latent))
    latent[:, 2] = 2.0 + np.linspace(0.0, 0.7, len(latent))
    latent[:, 6] = 1.0
    latent[:, 7:10] = 0.08
    latent[:, 10] = 0.7
    latent[:, 11:14] = 0.5
    backend = ReferenceGaussianBackend()
    adapter = Gaussian3DRendererAdapter(len(latent), camera, backend=backend)
    edited = rigid_transform(
        latent,
        SE3(SO3.exp(np.array([0.0, 0.0, 0.15])), np.array([0.05, 0.0, 0.0])),
        indices=[0, 1],
    )
    edited = edit_opacity(edited, [0, 1], value=0.2)
    held_out = [
        camera,
        GaussianCamera(
            64,
            48,
            intrinsics,
            np.array([[1.0, 0.0, 0.0, 0.15], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]),
        ),
    ]

    def render(value: np.ndarray, view: GaussianCamera) -> np.ndarray:
        return Gaussian3DRendererAdapter(len(value), view, backend=backend).decode(value)

    start = time.perf_counter()
    image = adapter.decode(latent)
    elapsed = time.perf_counter() - start
    result = {
        "backend": "reference",
        "width": 64,
        "height": 48,
        "n_gaussians": len(latent),
        "seconds": elapsed,
        "image_min": float(image.min()),
        "image_max": float(image.max()),
        "shape": list(image.shape),
        "gpu_backend": "not measured: CUDA/gsplat unavailable",
        "intervention": "rigid_transform + bounded opacity edit",
        "held_out_views": len(held_out),
        "metrics": evaluate_multiview(latent, edited, target_indices=[0, 1], cameras=held_out, render=render).__dict__,
        "naive_arithmetic": {
            "status": "rejected by schema"
            if np.any(naive_parameter_arithmetic(latent, np.full_like(latent, 0.4))[:, 10] > 1.0)
            else "not-invalid",
            "reason": "unconstrained opacity arithmetic exceeds [0, 1]",
        },
    }
    Path("artifacts/gaussian_3d_renderer_benchmark.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

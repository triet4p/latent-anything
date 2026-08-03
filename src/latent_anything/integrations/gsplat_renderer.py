"""Internal 3D Gaussian rasterizer backends."""

from __future__ import annotations

from typing import Any, Protocol, cast

import numpy as np


class GaussianCamera(Protocol):
    @property
    def width(self) -> int: ...

    @property
    def height(self) -> int: ...

    @property
    def intrinsics(self) -> np.ndarray: ...

    @property
    def world_to_camera(self) -> np.ndarray: ...


class GaussianRasterizerBackend(Protocol):
    name: str

    def render(
        self,
        means: np.ndarray,
        quaternions: np.ndarray,
        scales: np.ndarray,
        opacities: np.ndarray,
        colors: np.ndarray,
        camera: GaussianCamera,
    ) -> np.ndarray: ...


class GsplatBackend:
    """Lazy wrapper around gsplat, so the base package remains import-clean."""

    name = "gsplat"

    def render(
        self,
        means: np.ndarray,
        quaternions: np.ndarray,
        scales: np.ndarray,
        opacities: np.ndarray,
        colors: np.ndarray,
        camera: GaussianCamera,
    ) -> np.ndarray:
        try:
            import torch
            from gsplat import rasterization  # type: ignore[reportMissingImports]
        except ImportError as exc:
            raise RuntimeError("3D rendering requires the optional '3d' extra: uv sync --extra 3d") from exc
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tensors = [
            torch.as_tensor(value, dtype=torch.float32, device=device)
            for value in (means, quaternions, scales, opacities, colors)
        ]
        viewmats = torch.as_tensor(camera.world_to_camera, dtype=torch.float32, device=device)[None]
        intrinsics = torch.as_tensor(camera.intrinsics, dtype=torch.float32, device=device)[None]
        renders: Any
        renders, _, _ = rasterization(  # type: ignore[reportUnknownVariableType]
            means=tensors[0],
            quats=tensors[1],
            scales=tensors[2],
            opacities=tensors[3],
            colors=tensors[4],
            viewmats=viewmats,
            Ks=intrinsics,
            width=camera.width,
            height=camera.height,
            sh_degree=None,
        )
        # gsplat has no strict type stubs; this optional boundary is dynamic.
        rendered: Any = cast(Any, renders[0])
        rendered_array = rendered.detach().cpu().numpy()  # type: ignore[reportUnknownMemberType]
        return np.asarray(rendered_array, dtype=np.float64).clip(0.0, 1.0)


class ReferenceGaussianBackend:
    """Deterministic CPU backend for tiny fixtures and backend parity tests."""

    name = "reference"

    def render(
        self,
        means: np.ndarray,
        quaternions: np.ndarray,
        scales: np.ndarray,
        opacities: np.ndarray,
        colors: np.ndarray,
        camera: GaussianCamera,
    ) -> np.ndarray:
        del quaternions
        homogeneous = np.concatenate((means, np.ones((means.shape[0], 1))), axis=1)
        points = (camera.world_to_camera @ homogeneous.T).T[:, :3]
        projected = (camera.intrinsics @ points.T).T
        projected = projected[:, :2] / np.maximum(
            projected[
                :,
                2:,
            ],
            1e-8,
        )
        yy, xx = np.mgrid[0 : camera.height, 0 : camera.width]
        image = np.zeros((camera.height, camera.width, 3), dtype=np.float64)
        for point, scale, opacity, color in zip(projected, scales, opacities, colors, strict=True):
            weight = opacity * np.exp(
                -0.5
                * (
                    ((xx - point[0]) / max(scale[0] * 20, 1e-6)) ** 2
                    + ((yy - point[1]) / max(scale[1] * 20, 1e-6)) ** 2
                )
            )
            image += weight[..., None] * color
        return image.clip(0.0, 1.0)

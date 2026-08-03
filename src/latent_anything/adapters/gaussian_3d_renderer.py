"""Public facade for deterministic 3D Gaussian splat decoding."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from latent_anything.integrations.gsplat_renderer import GaussianRasterizerBackend, GsplatBackend
from latent_anything.latent_space import LatentSpace


@dataclass(frozen=True)
class GaussianCamera:
    """Pinhole camera using metres/world units and pixel intrinsics."""

    width: int
    height: int
    intrinsics: np.ndarray
    world_to_camera: np.ndarray

    def __post_init__(self) -> None:
        if self.width < 1 or self.height < 1:
            raise ValueError("camera width and height must be positive")
        if self.intrinsics.shape != (3, 3) or self.world_to_camera.shape != (4, 4):
            raise ValueError("camera intrinsics must be (3, 3) and world_to_camera must be (4, 4)")
        if not np.isfinite(self.intrinsics).all() or not np.isfinite(self.world_to_camera).all():
            raise ValueError("camera matrices must be finite")


class Gaussian3DRendererAdapter:
    """Decode one fixed-size 3D Gaussian latent set into an RGB image."""

    PARAM_DIM = 14

    def __init__(
        self, n_gaussians: int, camera: GaussianCamera, backend: GaussianRasterizerBackend | None = None
    ) -> None:
        if n_gaussians < 1:
            raise ValueError("n_gaussians must be >= 1")
        self._n_gaussians = n_gaussians
        self._camera = camera
        self._backend = backend if backend is not None else GsplatBackend()

    @property
    def latent_space(self) -> LatentSpace:
        return LatentSpace(
            dim=self.PARAM_DIM,
            geometry="gaussian_3d",
            source_model="gsplat",
            n_gaussians=self._n_gaussians,
            metadata={
                "backend": self._backend.name,
                "coordinate_frame": "world_right_handed",
                "position_unit": "m",
                "rotation_parameterization": "xyzw_quaternion",
                "scale_parameterization": "standard_deviation_m",
                "opacity_parameterization": "linear_bounded",
                "spherical_harmonics_degree": 0,
                "spherical_harmonics_channels": 3,
                "camera": {
                    "width": self._camera.width,
                    "height": self._camera.height,
                    "intrinsics": self._camera.intrinsics.tolist(),
                    "world_to_camera": self._camera.world_to_camera.tolist(),
                },
                "parameter_layout": {
                    "position": (0, 3),
                    "rotation": (3, 4),
                    "scale": (7, 3),
                    "opacity": (10, 1),
                    "spherical_harmonics": (11, 3),
                },
            },
        )

    def decode(self, latent: np.ndarray) -> np.ndarray:
        self._validate(latent)
        return self._backend.render(
            latent[:, :3], latent[:, 3:7], latent[:, 7:10], latent[:, 10], latent[:, 11:14], self._camera
        )

    def _validate(self, latent: np.ndarray) -> None:
        if latent.shape != (self._n_gaussians, self.PARAM_DIM):
            raise ValueError(f"expected latent shape {(self._n_gaussians, self.PARAM_DIM)}, got {latent.shape}")
        if not np.isfinite(latent).all():
            raise ValueError("3D Gaussian latent must be finite")
        if np.any(latent[:, 7:10] <= 0):
            raise ValueError("3D Gaussian scales must be positive")
        if np.any(np.linalg.norm(latent[:, 3:7], axis=1) < 1e-8):
            raise ValueError("3D Gaussian rotations require non-zero quaternions")
        if np.any((latent[:, 10] < 0) | (latent[:, 10] > 1)):
            raise ValueError("3D Gaussian opacity must be in [0, 1]")

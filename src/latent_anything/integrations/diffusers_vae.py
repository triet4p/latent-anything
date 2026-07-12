"""Lazy Diffusers AutoencoderKL adapter with NumPy public boundaries."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from latent_anything.integrations import require_optional
from latent_anything.latent_space import LatentSpace
from latent_anything.latent_value import LatentValue


class DiffusersAutoencoderKLAdapter:
    """Pinned AutoencoderKL integration; backend objects remain private."""

    def __init__(
        self,
        model_id: str,
        revision: str,
        *,
        device: str = "cpu",
        latent_mode: Literal["mean", "sample"] = "mean",
        dtype: Any = np.float32,
    ) -> None:
        if latent_mode not in {"mean", "sample"}:
            raise ValueError("latent_mode must be 'mean' or 'sample'")
        self.model_id = model_id
        self.revision = revision
        self.device = device
        self.latent_mode = latent_mode
        self.dtype = np.dtype(dtype)
        if self.dtype.kind != "f":
            raise TypeError("dtype must be a floating NumPy dtype")
        self._model: Any | None = None

    def _backend(self) -> Any:
        if self._model is None:
            diffusers = require_optional("diffusers", extra="diffusers")
            autoencoder = diffusers.AutoencoderKL
            self._model = autoencoder.from_pretrained(self.model_id, revision=self.revision).to(self.device)
        return self._model

    @property
    def latent_space(self) -> LatentSpace:
        return LatentSpace(dim=4, source_model=self.model_id, metadata={"revision": self.revision})

    def encode(self, images: np.ndarray) -> np.ndarray:
        if (
            images.ndim != 4
            or images.shape[1] not in {1, 3}
            or images.shape[2] < 1
            or images.shape[3] < 1
            or not np.isfinite(images).all()
            or np.any((images < -1) | (images > 1))
        ):
            raise ValueError("Expected NCHW images in [-1, 1] with one or three channels")
        import torch

        model = self._backend()
        with torch.no_grad():
            tensor = torch.from_numpy(images.astype(self.dtype, copy=False)).to(self.device)  # pyright: ignore[reportUnknownMemberType]
            distribution = model.encode(tensor).latent_dist
            latent = distribution.mean if self.latent_mode == "mean" else distribution.sample()
            return (latent * model.config.scaling_factor).detach().cpu().numpy()

    def encode_value(self, images: np.ndarray) -> LatentValue:
        """Return an NHWC structured value so the channel dimension matches the space."""
        return LatentValue(
            np.moveaxis(self.encode(images), 1, -1),
            self.latent_space,
            metadata={"layout": "NHWC", "scaled": True},
        )

    def decode(self, latent: np.ndarray) -> np.ndarray:
        if latent.ndim != 4 or not np.isfinite(latent).all():
            raise ValueError("Expected NCHW latent batch")
        import torch

        model = self._backend()
        with torch.no_grad():
            tensor = torch.from_numpy(latent.astype(self.dtype, copy=False)).to(self.device)  # pyright: ignore[reportUnknownMemberType]
            output = model.decode(tensor / model.config.scaling_factor).sample
            return output.detach().cpu().numpy()

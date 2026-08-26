"""Lazy Diffusers AutoencoderKL adapter with NumPy public boundaries."""

from __future__ import annotations

from typing import Any, Literal, cast

import numpy as np
from numpy.typing import DTypeLike

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
        dtype: DTypeLike = np.float32,
    ) -> None:
        if latent_mode not in {"mean", "sample"}:
            raise ValueError("latent_mode must be 'mean' or 'sample'")
        self.model_id = model_id
        self.revision = revision
        self.device = device
        self.latent_mode = latent_mode
        self.dtype = np.dtype(dtype)
        if self.dtype not in {np.dtype(np.float16), np.dtype(np.float32)}:
            raise TypeError("dtype must be float16 or float32 to match supported Diffusers checkpoints")
        self._model: Any | None = None

    def _torch_dtype(self) -> Any:
        """Map the supported NumPy boundary dtypes to the backend dtypes."""
        import torch

        return torch.float16 if self.dtype == np.dtype(np.float16) else torch.float32

    def _backend(self) -> Any:
        if self._model is None:
            diffusers = require_optional("diffusers", extra="diffusers")
            autoencoder = diffusers.AutoencoderKL
            self._model = (
                autoencoder.from_pretrained(self.model_id, revision=self.revision)
                .to(device=self.device, dtype=self._torch_dtype())
                .eval()
            )
        return self._model

    @property
    def latent_space(self) -> LatentSpace:
        """Load the optional backend and return its latent-channel contract."""
        model = self._backend()
        latent_channels = cast(int, model.config.latent_channels)
        return LatentSpace(dim=latent_channels, source_model=self.model_id, metadata={"revision": self.revision})

    def encode(self, data: np.ndarray, *, seed: int | None = None) -> np.ndarray:
        """Encode finite NCHW images in ``[-1, 1]`` into scaled latent arrays.

        Sampling uses ``seed`` when ``latent_mode`` is ``"sample"``; the
        optional Diffusers backend is loaded lazily at this boundary.
        """
        if (
            data.ndim != 4
            or data.shape[1] not in {1, 3}
            or data.shape[2] < 1
            or data.shape[3] < 1
            or not np.isfinite(data).all()
            or np.any((data < -1) | (data > 1))
        ):
            raise ValueError("Expected NCHW images in [-1, 1] with one or three channels")
        import torch

        model = self._backend()
        with torch.no_grad():
            tensor = torch.from_numpy(data.astype(self.dtype, copy=False)).to(  # pyright: ignore[reportUnknownMemberType] # torch's NumPy boundary is untyped
                device=self.device, dtype=self._torch_dtype()
            )  # pyright: ignore[reportUnknownMemberType] # third-party torch stub boundary
            distribution = model.encode(tensor).latent_dist
            if self.latent_mode == "mean":
                latent = distribution.mean
            elif seed is None:
                latent = distribution.sample()
            else:
                generator = torch.Generator(device=self.device)
                generator.manual_seed(seed)
                latent = distribution.sample(generator=generator)
            return (latent * model.config.scaling_factor).detach().cpu().numpy()

    def encode_value(self, images: np.ndarray) -> LatentValue:
        """Return an NHWC structured value so the channel dimension matches the space."""
        return LatentValue(
            np.moveaxis(self.encode(images), 1, -1),
            self.latent_space,
            metadata={"layout": "NHWC", "scaled": True},
        )

    def decode(self, latent: np.ndarray) -> np.ndarray:
        """Decode a finite NCHW latent batch through the lazy Diffusers backend."""
        if latent.ndim != 4 or not np.isfinite(latent).all():
            raise ValueError("Expected NCHW latent batch")
        import torch

        model = self._backend()
        with torch.no_grad():
            tensor = torch.from_numpy(latent.astype(self.dtype, copy=False)).to(  # pyright: ignore[reportUnknownMemberType] # torch's NumPy boundary is untyped
                device=self.device, dtype=self._torch_dtype()
            )  # pyright: ignore[reportUnknownMemberType] # third-party torch stub boundary
            output = model.decode(tensor / model.config.scaling_factor).sample
            return output.detach().cpu().numpy()

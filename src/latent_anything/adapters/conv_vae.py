"""Small convolutional VAE integration with NumPy-only public I/O."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn, optim

from latent_anything.latent_space import LatentSpace
from latent_anything.latent_value import LatentValue


class ConvVAE:
    """CPU-friendly 8×8 grayscale convolutional VAE for reproducible image smoke data."""

    def __init__(self, latent_dim: int = 4, random_state: int = 0, n_epochs: int = 5) -> None:
        if latent_dim < 1 or n_epochs < 1:
            raise ValueError("latent_dim and n_epochs must be positive")
        self.latent_dim = latent_dim
        self.random_state = random_state
        self.n_epochs = n_epochs
        torch.manual_seed(random_state)  # pyright: ignore[reportUnknownMemberType]
        self._encoder = nn.Sequential(nn.Conv2d(1, 4, 3, padding=1), nn.ReLU(), nn.Flatten())
        self._mu = nn.Linear(256, latent_dim)
        self._logvar = nn.Linear(256, latent_dim)
        self._decoder = nn.Sequential(nn.Linear(latent_dim, 64), nn.Sigmoid())
        self.metrics_: dict[str, float] = {}

    @property
    def latent_space(self) -> LatentSpace:
        return LatentSpace(dim=self.latent_dim, source_model="conv_vae_8x8")

    def encode(self, images: np.ndarray) -> np.ndarray:
        if images.ndim != 4 or images.shape[1:] != (1, 8, 8):
            raise ValueError(f"Expected images shaped (n, 1, 8, 8), got {images.shape}")
        with torch.no_grad():
            values = self._mu(self._encoder(torch.from_numpy(images.astype(np.float32))))  # pyright: ignore[reportUnknownMemberType]
        return values.detach().cpu().numpy().astype(np.float64)

    def fit(self, images: np.ndarray) -> None:
        """Train on [0, 1] 8×8 images and record reconstruction/KL metrics."""

        if images.ndim != 4 or images.shape[1:] != (1, 8, 8) or np.any((images < 0) | (images > 1)):
            raise ValueError("fit expects [0, 1] images shaped (n, 1, 8, 8)")
        torch.manual_seed(self.random_state)  # pyright: ignore[reportUnknownMemberType]
        data = torch.from_numpy(images.astype(np.float32))  # pyright: ignore[reportUnknownMemberType]
        parameters = (
            list(self._encoder.parameters())
            + list(self._mu.parameters())
            + list(self._logvar.parameters())
            + list(self._decoder.parameters())
        )
        optimizer = optim.Adam(parameters, lr=1e-3)
        for _ in range(self.n_epochs):
            features = self._encoder(data)
            mu, logvar = self._mu(features), self._logvar(features)
            latent = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
            reconstruction = self._decoder(latent).reshape_as(data)
            reconstruction_loss = nn.functional.mse_loss(reconstruction, data)
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = reconstruction_loss + kl_loss
            optimizer.zero_grad()
            loss.backward()  # pyright: ignore[reportUnknownMemberType]
            optimizer.step()  # pyright: ignore[reportUnknownMemberType]
        self.metrics_ = {
            "reconstruction_mse": float(reconstruction_loss.detach()),  # pyright: ignore[reportPossiblyUnboundVariable]
            "posterior_kl": float(kl_loss.detach()),  # pyright: ignore[reportPossiblyUnboundVariable]
            "latent_utilization": float(torch.var(mu, dim=0).mean().detach()),  # pyright: ignore[reportPossiblyUnboundVariable]
        }

    def encode_value(self, images: np.ndarray) -> LatentValue:
        return LatentValue(self.encode(images), self.latent_space)

    def decode(self, latent: np.ndarray) -> np.ndarray:
        if latent.ndim != 2 or latent.shape[1] != self.latent_dim:
            raise ValueError(f"Expected latent shape (n, {self.latent_dim}), got {latent.shape}")
        with torch.no_grad():
            values = self._decoder(torch.from_numpy(latent.astype(np.float32)))  # pyright: ignore[reportUnknownMemberType]
        return values.detach().cpu().numpy().astype(np.float64).reshape(-1, 1, 8, 8)

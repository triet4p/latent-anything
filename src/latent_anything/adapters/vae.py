"""Variational Autoencoder (VAE) — ModelAdapter #1 (explicit learned latent).

A concrete VAE implementation that trains from scratch and exposes its
latent space via the ``latent_space`` property. This is the first
ModelAdapter instance in the latent-anything framework, demonstrating
mode (i): explicit learned latent, with a learned encoder and decoder.

Conforms to ``ModelAdapter``, ``DecodableAdapter``, and the narrower
``FlatBatchDecodableAdapter`` Protocol because its public encode/decode
contract is batch-matrix based. ``fit`` is VAE-specific and deliberately
not part of these Protocols — it's not universal across adapters.

All public input/output is ``numpy.ndarray``. PyTorch is used internally
but never leaked to callers.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from latent_anything.latent_space import LatentSpace


class VAE:
    """Variational Autoencoder with a Gaussian latent prior.

    Trains a standard VAE with reconstruction loss (MSE) and KL divergence
    regularization. After training, ``encode`` returns the latent mean
    (deterministic) for downstream analysis with PCA/UMAP/etc.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input data.
    latent_dim : int
        Dimensionality of the latent space.
    hidden_dim : int, optional
        Hidden layer dimensionality. Defaults to ``max(latent_dim * 4, input_dim)``.
    learning_rate : float, optional
        Learning rate for the Adam optimizer.
    n_epochs : int, optional
        Number of training epochs.
    beta : float, optional
        Weight of the KL divergence term. Default 1.0 (standard VAE).
    random_state : int, optional
        Seed for PyTorch reproducibility.

    Notes
    -----
    Input data must be scaled to [0, 1] before calling ``fit``.
    The decoder output activation is sigmoid, which assumes [0, 1]-range inputs.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dim: int | None = None,
        learning_rate: float = 0.001,
        n_epochs: int = 200,
        beta: float = 1.0,
        random_state: int | None = None,
    ) -> None:
        if input_dim < 1:
            msg = f"input_dim must be >= 1, got {input_dim}"
            raise ValueError(msg)
        if latent_dim < 1:
            msg = f"latent_dim must be >= 1, got {latent_dim}"
            raise ValueError(msg)

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim if hidden_dim is not None else max(latent_dim * 4, input_dim)
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.beta = beta
        self.random_state = random_state

        self._encoder: _VAEEncoder | None = None
        self._decoder: _VAEDecoder | None = None
        self._fitted: bool = False
        self.loss_history_: list[float] = []

    @property
    def latent_space(self) -> LatentSpace:
        """Return a ``LatentSpace`` describing this VAE's latent manifold.

        Returns
        -------
        LatentSpace
            A Euclidean flat latent space of dimension ``latent_dim``.
        """
        return LatentSpace(dim=self.latent_dim, source_model="vae")

    @property
    def supports_flat_batch(self) -> bool:
        """Return ``True`` because VAE encode/decode use flat sample batches."""
        return True

    def fit(self, data: np.ndarray) -> None:
        """Train the VAE on the given data via gradient descent.

        Parameters
        ----------
        data : np.ndarray
            2D array of shape ``(n_samples, input_dim)`` with values in [0, 1].

        Raises
        ------
        ValueError
            If ``data`` is not 2D or has wrong number of features.
        """
        if data.ndim != 2:
            msg = f"Expected 2D array, got {data.ndim}D"
            raise ValueError(msg)
        if data.shape[0] < 1:
            msg = "Data must have at least 1 sample"
            raise ValueError(msg)
        if data.shape[1] != self.input_dim:
            msg = f"Expected input_dim={self.input_dim}, got data with {data.shape[1]} features"
            raise ValueError(msg)

        # Set random seed for reproducibility
        if self.random_state is not None:
            torch.manual_seed(self.random_state)  # pyright: ignore[reportUnknownMemberType]

        # Build model
        self._encoder = _VAEEncoder(self.input_dim, self.hidden_dim, self.latent_dim)
        self._decoder = _VAEDecoder(self.latent_dim, self.hidden_dim, self.input_dim)

        optimizer = optim.Adam(
            list(self._encoder.parameters()) + list(self._decoder.parameters()),
            lr=self.learning_rate,
        )

        data_tensor = torch.from_numpy(data.astype(np.float32))  # pyright: ignore[reportUnknownMemberType]

        self.loss_history_ = []
        self._encoder.train()
        self._decoder.train()

        for _epoch in range(self.n_epochs):
            optimizer.zero_grad()

            # Forward pass
            mu, logvar = self._encoder(data_tensor)
            z = _reparameterize(mu, logvar)
            reconstruction = self._decoder(z)

            # Loss
            recon_loss = nn.functional.mse_loss(reconstruction, data_tensor, reduction="mean")
            kl_loss = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + self.beta * kl_loss

            loss.backward()  # pyright: ignore[reportUnknownMemberType]
            optimizer.step()  # pyright: ignore[reportUnknownMemberType]

            self.loss_history_.append(float(loss.detach().cpu().numpy()))

        self._fitted = True

    def encode(self, data: np.ndarray) -> np.ndarray:
        """Encode data to the latent mean (deterministic).

        Parameters
        ----------
        data : np.ndarray
            2D array of shape ``(n_samples, input_dim)``.

        Returns
        -------
        np.ndarray
            Latent mean vectors of shape ``(n_samples, latent_dim)``.

        Raises
        ------
        RuntimeError
            If the VAE has not been fitted.
        """
        if not self._fitted:
            msg = "VAE must be fitted before encode"
            raise RuntimeError(msg)
        if data.ndim != 2 or data.shape[1] != self.input_dim:
            msg = f"Expected data shape (n, {self.input_dim}), got {data.shape}"
            raise ValueError(msg)

        self._encoder.eval()  # type: ignore[union-attr]
        with torch.no_grad():
            data_tensor = torch.from_numpy(data.astype(np.float32))  # pyright: ignore[reportUnknownMemberType]
            mu, _logvar = self._encoder(data_tensor)  # type: ignore[union-attr]
            result = mu.detach().cpu().numpy()
        return np.asarray(result, dtype=np.float64)

    def decode(self, latent: np.ndarray) -> np.ndarray:
        """Decode latent vectors back to data space.

        Parameters
        ----------
        latent : np.ndarray
            2D array of shape ``(n_samples, latent_dim)``.

        Returns
        -------
        np.ndarray
            Reconstructed data of shape ``(n_samples, input_dim)``.

        Raises
        ------
        RuntimeError
            If the VAE has not been fitted.
        """
        if not self._fitted:
            msg = "VAE must be fitted before decode"
            raise RuntimeError(msg)
        if latent.ndim != 2 or latent.shape[1] != self.latent_dim:
            msg = f"Expected latent shape (n, {self.latent_dim}), got {latent.shape}"
            raise ValueError(msg)

        self._decoder.eval()  # type: ignore[union-attr]
        with torch.no_grad():
            latent_tensor = torch.from_numpy(latent.astype(np.float32))  # pyright: ignore[reportUnknownMemberType]
            reconstruction = self._decoder(latent_tensor)  # type: ignore[union-attr]
            result = reconstruction.detach().cpu().numpy()
        return np.asarray(result, dtype=np.float64)


# ---------------------------------------------------------------------------
# Internal PyTorch modules (not exported)
# ---------------------------------------------------------------------------


class _VAEEncoder(nn.Module):
    """Encoder: input_dim → hidden_dim → mu (latent_dim) + logvar (latent_dim)."""

    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class _VAEDecoder(nn.Module):
    """Decoder: latent_dim → hidden_dim → input_dim → Sigmoid."""

    def __init__(self, latent_dim: int, hidden_dim: int, input_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.fc1(z))
        return torch.sigmoid(self.fc2(h))


def _reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Reparameterization trick: z = mu + exp(0.5 * logvar) * epsilon."""
    std = torch.exp(0.5 * logvar)
    epsilon = torch.randn_like(std)
    return mu + std * epsilon

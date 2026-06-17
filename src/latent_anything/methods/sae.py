"""Sparse Autoencoder (SAE) dimensionality reduction method, torch-based.

Method #3 — the third instance of the ``Method`` shape, with a fundamentally
different philosophy: gradient-descent training with L1 sparsity penalty,
encoder/decoder architecture, rather than PCA's matrix decomposition or
UMAP's manifold-learning fit.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class SAE:
    """Sparse Autoencoder dimensionality reduction method.

    A neural-network-based method that learns a sparse latent representation
    via an encoder (linear + ReLU) → latent → decoder (linear) architecture,
    trained with reconstruction loss (MSE) + L1 sparsity penalty on latent
    activations.

    Conforms to the ``Method`` Protocol (structural duck-typing).

    Parameters
    ----------
    n_components : int
        Dimensionality of the sparse latent space.
    l1_coef : float, optional
        Coefficient for L1 sparsity penalty on latent activations.
    learning_rate : float, optional
        Learning rate for Adam optimizer.
    n_epochs : int, optional
        Number of training epochs over the full dataset.
    random_state : int, optional
        Seed for PyTorch reproducibility.
    """

    def __init__(
        self,
        n_components: int,
        l1_coef: float = 0.01,
        learning_rate: float = 0.01,
        n_epochs: int = 500,
        random_state: int | None = None,
    ) -> None:
        self.n_components = n_components
        self.l1_coef = l1_coef
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.random_state = random_state

        self._encoder: nn.Linear | None = None
        self._decoder: nn.Linear | None = None
        self._fitted: bool = False
        self.loss_history_: list[float] = []

    def fit(self, data: np.ndarray) -> None:
        """Fit the SAE to the data via gradient descent training.

        Parameters
        ----------
        data : np.ndarray
            2D array of shape ``(n_samples, n_features)``.
        """
        if data.ndim != 2:
            msg = f"Expected 2D array, got {data.ndim}D"
            raise ValueError(msg)
        if data.shape[0] < 1 or data.shape[1] < 1:
            msg = "Data must have at least 1 sample and 1 feature"
            raise ValueError(msg)

        n_features = data.shape[1]

        # Set random seed for reproducibility
        if self.random_state is not None:
            torch.manual_seed(self.random_state)  # pyright: ignore[reportUnknownMemberType]

        # Build encoder / decoder
        self._encoder = nn.Linear(n_features, self.n_components)
        self._decoder = nn.Linear(self.n_components, n_features)

        # Convert data to torch tensor
        data_t = torch.from_numpy(data).float()  # pyright: ignore[reportUnknownMemberType]

        # Training loop
        optimizer = optim.Adam(
            list(self._encoder.parameters()) + list(self._decoder.parameters()),
            lr=self.learning_rate,
        )
        self.loss_history_ = []

        for _ in range(self.n_epochs):
            optimizer.zero_grad()

            # Forward: encode → ReLU → decode
            latent = torch.relu(self._encoder(data_t))
            reconstruction = self._decoder(latent)

            # Loss: MSE reconstruction + L1 sparsity on latent activations
            recon_loss = nn.functional.mse_loss(reconstruction, data_t)
            l1_penalty = self.l1_coef * torch.sum(torch.abs(latent))
            loss = recon_loss + l1_penalty

            loss.backward()  # pyright: ignore[reportUnknownMemberType]
            optimizer.step()  # pyright: ignore[reportUnknownMemberType]

            self.loss_history_.append(float(loss.item()))

        self._fitted = True

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Transform data to the sparse latent representation.

        Parameters
        ----------
        data : np.ndarray
            2D array of shape ``(n_samples, n_features)``.

        Returns
        -------
        np.ndarray
            Sparse latent activations of shape ``(n_samples, n_components)``.
        """
        if not self._fitted:
            msg = "SAE must be fitted before transform"
            raise RuntimeError(msg)
        if self._encoder is None:
            msg = "SAE encoder is not initialised"
            raise RuntimeError(msg)

        data_t = torch.from_numpy(data).float()  # pyright: ignore[reportUnknownMemberType]
        with torch.no_grad():
            latent = torch.relu(self._encoder(data_t))
        return latent.numpy()

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """Fit and transform in one step.

        Parameters
        ----------
        data : np.ndarray
            2D array of shape ``(n_samples, n_features)``.

        Returns
        -------
        np.ndarray
            Transformed array of shape ``(n_samples, n_components)``.
        """
        self.fit(data)
        return self.transform(data)

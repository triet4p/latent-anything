"""Sparse Autoencoder (SAE) dimensionality reduction method, torch-based.

Method #3 — the third instance of the ``Method`` shape, with a fundamentally
different philosophy: gradient-descent training with L1 sparsity penalty,
encoder/decoder architecture, rather than PCA's matrix decomposition or
UMAP's manifold-learning fit.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from typing import Any

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

            # Loss: MSE reconstruction + per-element L1 sparsity on latent activations.
            # Normalising by the number of elements keeps the L1 gradient
            # comparable to the reconstruction gradient; an unnormalised
            # ``sum(|latent|)`` collapses every feature to dead even at
            # ``l1_coef = 1e-4``.
            recon_loss = nn.functional.mse_loss(reconstruction, data_t)
            l1_penalty = self.l1_coef * torch.mean(torch.abs(latent))
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

    def reconstruct(self, data: np.ndarray) -> np.ndarray:
        """Decode sparse latent activations back to the input space.

        Parameters
        ----------
        data : np.ndarray
            2D array of shape ``(n_samples, n_features)``.

        Returns
        -------
        np.ndarray
            Reconstruction of shape ``(n_samples, n_features)``.
        """
        if not self._fitted:
            msg = "SAE must be fitted before reconstruct"
            raise RuntimeError(msg)
        if self._decoder is None:
            msg = "SAE decoder is not initialised"
            raise RuntimeError(msg)
        latent = self.transform(data)
        data_t = torch.from_numpy(latent).float()  # pyright: ignore[reportUnknownMemberType]
        with torch.no_grad():
            return self._decoder(data_t).numpy()

    # -- Checkpoint serialization (fitted state only) ----------------------

    def state_dict(self) -> dict[str, np.ndarray]:
        """Return the fitted encoder/decoder weights as portable NumPy arrays."""
        if not self._fitted or self._encoder is None or self._decoder is None:
            msg = "SAE must be fitted before exporting state"
            raise RuntimeError(msg)
        return {
            "encoder_weight": self._encoder.weight.detach().cpu().numpy().copy(),
            "encoder_bias": self._encoder.bias.detach().cpu().numpy().copy(),
            "decoder_weight": self._decoder.weight.detach().cpu().numpy().copy(),
            "decoder_bias": self._decoder.bias.detach().cpu().numpy().copy(),
        }

    def load_state_dict(self, state: Mapping[str, np.ndarray]) -> None:
        """Restore fitted weights from a state dict returned by :meth:`state_dict`.

        The component counts are inferred from the array shapes, so a checkpoint
        may be loaded into a fresh ``SAE()`` with any hyperparameters.
        """
        encoder_weight = np.asarray(state["encoder_weight"], dtype=np.float64)
        encoder_bias = np.asarray(state["encoder_bias"], dtype=np.float64)
        decoder_weight = np.asarray(state["decoder_weight"], dtype=np.float64)
        decoder_bias = np.asarray(state["decoder_bias"], dtype=np.float64)
        if encoder_weight.ndim != 2 or decoder_weight.ndim != 2:
            raise ValueError("encoder/decoder weights must be 2D arrays")
        if encoder_weight.shape[0] != decoder_weight.shape[1]:
            raise ValueError("encoder output dim must match decoder input dim")
        if encoder_bias.shape != (encoder_weight.shape[0],):
            raise ValueError("encoder bias shape must match encoder output dim")
        if decoder_bias.shape != (decoder_weight.shape[0],):
            raise ValueError("decoder bias shape must match decoder output dim")
        self.n_components = encoder_weight.shape[0]
        self._encoder = nn.Linear(encoder_weight.shape[1], encoder_weight.shape[0])
        self._decoder = nn.Linear(decoder_weight.shape[1], decoder_weight.shape[0])
        with torch.no_grad():
            self._encoder.weight.copy_(torch.as_tensor(encoder_weight, dtype=torch.float32))
            self._encoder.bias.copy_(torch.as_tensor(encoder_bias, dtype=torch.float32))
            self._decoder.weight.copy_(torch.as_tensor(decoder_weight, dtype=torch.float32))
            self._decoder.bias.copy_(torch.as_tensor(decoder_bias, dtype=torch.float32))
        self._fitted = True

    def save_checkpoint(self, path: str | os.PathLike[str]) -> None:
        """Serialize the fitted SAE to a portable ``.npz`` checkpoint.

        The checkpoint stores encoder/decoder weights plus a JSON-encoded copy
        of the constructor hyperparameters, so :meth:`load_checkpoint` can
        rebuild the same ``SAE`` without the original config object.
        """
        state = self.state_dict()
        config: dict[str, Any] = {
            "n_components": self.n_components,
            "l1_coef": self.l1_coef,
            "learning_rate": self.learning_rate,
            "n_epochs": self.n_epochs,
            "random_state": self.random_state,
        }
        np.savez(
            path,
            encoder_weight=state["encoder_weight"],
            encoder_bias=state["encoder_bias"],
            decoder_weight=state["decoder_weight"],
            decoder_bias=state["decoder_bias"],
            config_json=json.dumps(config),
        )

    @staticmethod
    def load_checkpoint(path: str | os.PathLike[str]) -> SAE:
        """Load a fitted SAE from a checkpoint written by :meth:`save_checkpoint`.

        Parameters
        ----------
        path :
            Path to the ``.npz`` checkpoint file.

        Returns
        -------
        SAE
            A fitted ``SAE`` with restored weights and hyperparameters.
        """
        with np.load(path, allow_pickle=False) as data:  # pyright: ignore[reportUnknownMemberType]
            config_raw = data["config_json"].item()
            if not isinstance(config_raw, str):
                raise ValueError(f"checkpoint {path} has no config_json string")
            config = json.loads(config_raw)
            sae = SAE(
                n_components=int(config["n_components"]),
                l1_coef=float(config["l1_coef"]),
                learning_rate=float(config["learning_rate"]),
                n_epochs=int(config["n_epochs"]),
                random_state=config["random_state"],
            )
            sae.load_state_dict(
                {
                    "encoder_weight": data["encoder_weight"],
                    "encoder_bias": data["encoder_bias"],
                    "decoder_weight": data["decoder_weight"],
                    "decoder_bias": data["decoder_bias"],
                }
            )
        return sae

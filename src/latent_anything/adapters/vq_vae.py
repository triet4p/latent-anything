"""Compact vector-quantized VAE with an integer code-sequence API."""

from __future__ import annotations

import json
from collections.abc import Mapping
from copy import deepcopy
from hashlib import sha256
from typing import Any

import numpy as np
import torch
from torch import nn, optim

from latent_anything.latent_space import LatentSpace
from latent_anything.latent_value import LatentValue


class VQVAE:
    """CPU-friendly VQ-VAE for reproducible 8x8 grayscale image evidence.

    The public latent representation is a two-dimensional integer array of
    code IDs with shape ``(n_samples, 16)``. Codebook vectors are available
    only through the explicit :meth:`code_embeddings` method; ``encode``
    never silently changes categorical IDs into continuous coordinates.
    """

    input_shape = (1, 8, 8)
    dataset_revision = "sklearn-digits-8x8@scikit-learn==1.9.0"
    model_revision = "compact-vq-vae-v1"

    def __init__(
        self,
        codebook_size: int = 16,
        embedding_dim: int = 8,
        commitment_cost: float = 0.25,
        random_state: int = 0,
        n_epochs: int = 5,
        learning_rate: float = 1e-3,
    ) -> None:
        if codebook_size < 2:
            raise ValueError("codebook_size must be at least 2")
        if embedding_dim < 1 or n_epochs < 1 or learning_rate <= 0:
            raise ValueError("embedding_dim and n_epochs must be positive; learning_rate must be positive")
        if commitment_cost <= 0.0:
            raise ValueError("commitment_cost must be positive")
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.random_state = random_state
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        torch.manual_seed(random_state)  # pyright: ignore[reportUnknownMemberType]
        self._encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, embedding_dim, kernel_size=4, stride=2, padding=1),
        )
        self._codebook = nn.Embedding(codebook_size, embedding_dim)
        self._decoder = nn.Sequential(
            nn.ConvTranspose2d(embedding_dim, 8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )
        self.codebook_counts_ = np.zeros(codebook_size, dtype=np.int64)
        self.metrics_: dict[str, float] = {}

    @property
    def sequence_length(self) -> int:
        """Number of categorical code positions emitted for one image."""

        return 16

    @property
    def codebook_version(self) -> str:
        """Return a digest binding the tokenizer checkpoint and code schema."""

        digest = sha256()
        schema = {
            "codebook_size": self.codebook_size,
            "embedding_dim": self.embedding_dim,
            "input_shape": self.input_shape,
            "sequence_length": self.sequence_length,
            "model_revision": self.model_revision,
        }
        digest.update(json.dumps(schema, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        for name, value in sorted(self._state_dict_numpy().items()):
            digest.update(name.encode("utf-8"))
            digest.update(str(value.dtype).encode("utf-8"))
            digest.update(str(value.shape).encode("utf-8"))
            digest.update(np.ascontiguousarray(value).tobytes())
        return f"{self.model_revision}@{digest.hexdigest()}"

    def _state_dict_numpy(self) -> dict[str, np.ndarray]:
        return {name: value.detach().cpu().numpy() for name, value in self._full_state_dict().items()}

    def _full_state_dict(self) -> dict[str, torch.Tensor]:
        state: dict[str, torch.Tensor] = {}
        for prefix, module in (("encoder", self._encoder), ("codebook", self._codebook), ("decoder", self._decoder)):
            state.update({f"{prefix}.{name}": value for name, value in module.state_dict().items()})
        return state

    @property
    def latent_space(self) -> LatentSpace:
        """Return the declared discrete code-sequence geometry."""

        return LatentSpace(
            dim=self.sequence_length,
            geometry="discrete_code",
            source_model=self.model_revision,
            codebook_size=self.codebook_size,
            metadata={
                "representation": "integer_code_sequence",
                "code_sequence_shape": (4, 4),
                "codebook_embedding_dim": self.embedding_dim,
                "dataset_revision": self.dataset_revision,
                "model_revision": self.model_revision,
                "codebook_version": self.codebook_version,
                "interpolation": "unsupported",
            },
        )

    def _validate_images(self, images: np.ndarray) -> None:
        if images.ndim != 4 or tuple(images.shape[1:]) != self.input_shape:
            raise ValueError(f"Expected images shaped (n, 1, 8, 8), got {images.shape}")
        if images.shape[0] < 1 or not np.isfinite(images).all() or np.any((images < 0) | (images > 1)):
            raise ValueError("images must be a non-empty finite batch in [0, 1]")

    def _validate_codes(self, codes: np.ndarray) -> None:
        expected = (self.sequence_length,)
        if codes.ndim != 2 or tuple(codes.shape[1:]) != expected:
            raise ValueError(f"Expected integer codes shaped (n, {self.sequence_length}), got {codes.shape}")
        if codes.shape[0] < 1:
            raise ValueError("codes must contain at least one sequence")
        if not np.issubdtype(codes.dtype, np.integer):
            raise TypeError("VQVAE codes must preserve an integer NumPy dtype")
        if np.any(codes < 0) or np.any(codes >= self.codebook_size):
            raise ValueError(f"codes must be in [0, {self.codebook_size - 1}]")

    def _quantize(self, encoded: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        flat = encoded.permute(0, 2, 3, 1).reshape(-1, self.embedding_dim)
        weights = self._codebook.weight
        distances = (
            flat.pow(2).sum(dim=1, keepdim=True) + weights.pow(2).sum(dim=1).unsqueeze(0) - 2.0 * flat @ weights.t()
        )
        indices = torch.argmin(distances, dim=1)
        quantized = self._codebook(indices).reshape(encoded.shape[0], 4, 4, self.embedding_dim)
        return indices.reshape(encoded.shape[0], self.sequence_length), quantized.permute(0, 3, 1, 2)

    def _encode_tensors(self, images: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self._validate_images(images)
        tensor = torch.from_numpy(images.astype(np.float32))  # pyright: ignore[reportUnknownMemberType]
        encoded = self._encoder(tensor)
        indices, quantized = self._quantize(encoded)
        return encoded, indices, quantized

    def fit(self, images: np.ndarray) -> None:
        """Train the VQ-VAE and record reconstruction/codebook diagnostics."""

        self._validate_images(images)
        torch.manual_seed(self.random_state)  # pyright: ignore[reportUnknownMemberType]
        data = torch.from_numpy(images.astype(np.float32))  # pyright: ignore[reportUnknownMemberType]
        parameters = (
            list(self._encoder.parameters()) + list(self._codebook.parameters()) + list(self._decoder.parameters())
        )
        optimizer = optim.Adam(parameters, lr=self.learning_rate)
        reconstruction_mse = 0.0
        codebook_loss_value = 0.0
        commitment_loss_value = 0.0
        commitment_distance_value = 0.0
        for _ in range(self.n_epochs):
            encoded = self._encoder(data)
            indices, quantized = self._quantize(encoded)
            straight_through = encoded + (quantized - encoded).detach()
            reconstruction = self._decoder(straight_through)
            reconstruction_loss = nn.functional.mse_loss(reconstruction, data)
            codebook_loss = nn.functional.mse_loss(quantized, encoded.detach())
            commitment_distance = nn.functional.mse_loss(encoded, quantized.detach())
            commitment_loss = self.commitment_cost * commitment_distance
            loss = reconstruction_loss + codebook_loss + commitment_loss
            optimizer.zero_grad()
            loss.backward()  # pyright: ignore[reportUnknownMemberType]
            optimizer.step()  # pyright: ignore[reportUnknownMemberType]
            reconstruction_mse = float(reconstruction_loss.detach())  # pyright: ignore[reportUnknownMemberType]
            codebook_loss_value = float(codebook_loss.detach())  # pyright: ignore[reportUnknownMemberType]
            commitment_loss_value = float(commitment_loss.detach())  # pyright: ignore[reportUnknownMemberType]
            commitment_distance_value = float(commitment_distance.detach())  # pyright: ignore[reportUnknownMemberType]

        with torch.no_grad():
            _, indices, _ = self._encode_tensors(images)
        counts = torch.bincount(indices.reshape(-1), minlength=self.codebook_size).detach().cpu().numpy()
        self.codebook_counts_ = counts.astype(np.int64)
        diagnostics = self.codebook_diagnostics()
        self.metrics_ = {
            "reconstruction_mse": reconstruction_mse,
            "codebook_loss": codebook_loss_value,
            "commitment_loss": commitment_loss_value,
            "commitment_distance": commitment_distance_value,
            "codebook_perplexity": diagnostics["codebook_perplexity"],
            "dead_code_rate": diagnostics["dead_code_rate"],
        }

    def encode(self, data: np.ndarray) -> np.ndarray:
        """Encode images to integer code IDs, preserving categorical semantics."""

        with torch.no_grad():
            _, indices, _ = self._encode_tensors(data)
        return indices.detach().cpu().numpy().astype(np.int64)

    def encode_value(self, images: np.ndarray) -> LatentValue:
        """Return immutable discrete codes with model and dataset provenance."""

        return LatentValue(
            self.encode(images),
            self.latent_space,
            metadata={"dataset_revision": self.dataset_revision, "model_revision": self.model_revision},
        )

    def code_embeddings(self, codes: np.ndarray) -> np.ndarray:
        """Explicitly map integer codes to codebook vectors for decoding/inspection."""

        self._validate_codes(codes)
        with torch.no_grad():
            values = self._codebook(torch.from_numpy(codes.astype(np.int64)))  # pyright: ignore[reportUnknownMemberType]
        return values.detach().cpu().numpy().astype(np.float64)

    def decode(self, latent: np.ndarray) -> np.ndarray:
        """Decode integer code sequences; continuous latent vectors are rejected."""

        self._validate_codes(latent)
        with torch.no_grad():
            embeddings = self._codebook(torch.from_numpy(latent.astype(np.int64)))  # pyright: ignore[reportUnknownMemberType]
            quantized = embeddings.reshape(-1, 4, 4, self.embedding_dim).permute(0, 3, 1, 2)
            values = self._decoder(quantized)
        return values.detach().cpu().numpy().astype(np.float64)

    def replace_codes(self, codes: np.ndarray, replacements: Mapping[int, int]) -> np.ndarray:
        """Apply an explicit categorical code replacement map without interpolation."""

        self._validate_codes(codes)
        for source, target in replacements.items():
            if type(source) is not int or type(target) is not int:
                raise TypeError("code replacements must map integer IDs to integer IDs")
            if not 0 <= source < self.codebook_size or not 0 <= target < self.codebook_size:
                raise ValueError("code replacement IDs must be within the codebook")
        result = codes.copy()
        for source, target in replacements.items():
            result[result == source] = target
        return result

    def interpolate_codes(self, a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
        """Reject continuous interpolation; categorical edits need a policy."""

        del a, b, t
        raise ValueError("VQVAE code sequences have no continuous interpolation; use replace_codes")

    def codebook_diagnostics(self, codes: np.ndarray | None = None) -> dict[str, float]:
        """Measure perplexity and dead-code rate for stored or supplied codes."""

        counts = self._counts(codes)
        probabilities = counts / max(float(counts.sum()), 1.0)
        active = probabilities > 0
        entropy = -float(np.sum(probabilities[active] * np.log(probabilities[active])))
        return {
            "codebook_perplexity": float(np.exp(entropy)),
            "dead_code_rate": float(np.mean(counts == 0)),
        }

    def codebook_metadata(self) -> dict[str, Any]:
        """Return serializable codebook geometry, provenance, and usage metadata."""

        metadata: dict[str, Any] = {
            "codebook_size": self.codebook_size,
            "embedding_dim": self.embedding_dim,
            "sequence_length": self.sequence_length,
            "grid_shape": [4, 4],
            "dataset_revision": self.dataset_revision,
            "model_revision": self.model_revision,
            "codebook_version": self.codebook_version,
            "counts": self.codebook_counts_.tolist(),
        }
        metadata.update(self.codebook_diagnostics())
        return deepcopy(metadata)

    def code_frequency_drift(self, reference: np.ndarray, comparison: np.ndarray) -> float:
        """Return total-variation drift between two code-frequency distributions."""

        reference_counts = self._counts(reference)
        comparison_counts = self._counts(comparison)
        reference_probabilities = reference_counts / max(float(reference_counts.sum()), 1.0)
        comparison_probabilities = comparison_counts / max(float(comparison_counts.sum()), 1.0)
        return float(0.5 * np.abs(reference_probabilities - comparison_probabilities).sum())

    def _counts(self, codes: np.ndarray | None) -> np.ndarray:
        if codes is None:
            return self.codebook_counts_.copy()
        self._validate_codes(codes)
        return np.bincount(codes.reshape(-1), minlength=self.codebook_size).astype(np.int64)

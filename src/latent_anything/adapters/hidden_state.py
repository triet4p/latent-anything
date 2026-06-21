"""HiddenStateAdapter — ModelAdapter #3 (mode ii: no-explicit-latent).

A concrete adapter that treats hidden-state activations as the latent
representation, without assuming the existence of a decoder. This is
the third ModelAdapter instance, demonstrating mode (ii) of the
3-mode ModelAdapter ADR: **no-explicit-latent** — where the latent
*is* the hidden activation, not a bottleneck with a learned decoder.

There is **no ``decode`` method.** This is intentional: the latent
representation is just a hidden activation from a fixed/random
feature stack, and there is no decoder to map back to data space.
Methods that require ``decode`` (e.g. ``ActivationPatch``) cannot
work with this adapter.

All input/output is ``numpy.ndarray``. No PyTorch dependency, no
heavy transformer imports.

Philosophical difference from VAE (#1) and RandomProjection (#2):
- VAE: stateful, trained-from-scratch, explicit learned encoder + decoder.
- RandomProjection: stateless, fixed random projection, explicit
  encode + decode via linear algebra.
- HiddenStateAdapter: stateless, fixed random nonlinear feature
  stack, encode-only — no decode, no explicit bottleneck.
"""

from __future__ import annotations

import numpy as np

from latent_anything.latent_space import LatentSpace


class HiddenStateAdapter:
    """Fixed random MLP feature stack — mode (ii) no-explicit-latent.

    At construction, creates a deterministic 2-layer ReLU MLP with
    fixed random weights. The "hidden state" after the first layer
    is treated as the latent representation. There is **no decoder**
    — this adapter cannot reconstruct input from latent.

    This simulates the pattern of real no-explicit-latent models
    (JiT, ViT, LLM intermediate activations) where the hidden state
    *is* the latent, but without requiring a heavy transformer
    dependency.

    Parameters
    ----------
    input_dim : int
        Dimensionality of input data.
    hidden_dim : int
        Dimensionality of the hidden-state activations (latent dim).
    random_state : int, optional
        Seed for the numpy random generator. Ensures reproducible
        weight matrices.

    Attributes
    ----------
    weight1_ : np.ndarray
        First-layer weight matrix of shape ``(input_dim, hidden_dim)``,
        drawn from ``N(0, 1/sqrt(input_dim))`` at construction.
    bias1_ : np.ndarray
        First-layer bias vector of shape ``(hidden_dim,)``, initialised
        to zeros.
    weight2_ : np.ndarray
        Second-layer weight matrix of shape ``(hidden_dim, hidden_dim)``,
        drawn from ``N(0, 1/sqrt(hidden_dim))`` at construction.
    bias2_ : np.ndarray
        Second-layer bias vector of shape ``(hidden_dim,)``, initialised
        to zeros.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        random_state: int | None = None,
    ) -> None:
        if input_dim < 1:
            msg = f"input_dim must be >= 1, got {input_dim}"
            raise ValueError(msg)
        if hidden_dim < 1:
            msg = f"hidden_dim must be >= 1, got {hidden_dim}"
            raise ValueError(msg)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.random_state = random_state

        rng = np.random.default_rng(random_state)

        # He-like initialisation: N(0, sqrt(2/fan_in))
        self.weight1_: np.ndarray = (
            rng.normal(
                size=(input_dim, hidden_dim),
            ).astype(np.float64)
            * (2.0 / input_dim) ** 0.5
        )
        self.bias1_: np.ndarray = np.zeros(hidden_dim, dtype=np.float64)

        self.weight2_: np.ndarray = (
            rng.normal(
                size=(hidden_dim, hidden_dim),
            ).astype(np.float64)
            * (2.0 / hidden_dim) ** 0.5
        )
        self.bias2_: np.ndarray = np.zeros(hidden_dim, dtype=np.float64)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def latent_space(self) -> LatentSpace:
        """Return a ``LatentSpace`` describing the hidden-state manifold.

        Returns
        -------
        LatentSpace
            A Euclidean flat latent space of dimension ``hidden_dim``,
            with metadata marking the exposure mode as ``"hidden_state"``.
        """
        return LatentSpace(
            dim=self.hidden_dim,
            source_model="hidden_state",
            metadata={"exposure_mode": "hidden_state"},
        )

    # ------------------------------------------------------------------
    # Encode
    # ------------------------------------------------------------------

    def encode(self, data: np.ndarray) -> np.ndarray:
        """Encode data to hidden-state activations.

        Computes ``ReLU(data @ W1 + b1)`` as the returned hidden
        representation, simulating an intermediate activation capture
        in a deep network. The second-layer weights are kept on the
        adapter as deeper-network context, but ``encode`` intentionally
        exposes the first hidden layer only.

        Parameters
        ----------
        data : np.ndarray
            2D array of shape ``(n_samples, input_dim)``.

        Returns
        -------
        np.ndarray
            Hidden activation vectors of shape ``(n_samples, hidden_dim)``.

        Raises
        ------
        ValueError
            If ``data`` is not 2D or has wrong number of features.
        """
        if data.ndim != 2:
            msg = f"Expected 2D array, got {data.ndim}D"
            raise ValueError(msg)
        if data.shape[1] != self.input_dim:
            msg = f"Expected input_dim={self.input_dim}, got data with {data.shape[1]} features"
            raise ValueError(msg)

        hidden = np.maximum(0.0, data @ self.weight1_ + self.bias1_)
        return np.asarray(hidden, dtype=np.float64)

    # ------------------------------------------------------------------
    # No decode
    # ------------------------------------------------------------------

    # ``decode`` is deliberately absent. This adapter has no decoder —
    # the hidden-state activations are the latent representation, and
    # there is no learned or deterministic mapping back to data space.

"""Frozen ``ModelAdapter`` and ``DecodableAdapter`` Protocols.

Frozen at ModelAdapter #3 (``HiddenStateAdapter``, Sprint 14). Validated
by 3 instances with differing philosophies:

- **VAE (#1)** — explicit learned latent (mode i): ``encode`` + ``decode``,
  both learned from data via gradient descent.
- **RandomProjection (#2)** — fixed explicit projection (mode i-like):
  ``encode`` + ``decode``, both deterministic linear transforms with
  fixed random weights. No training needed.
- **HiddenStateAdapter (#3)** — no-explicit-latent (mode ii):
  ``encode`` only, returns hidden-state activations. No ``decode``
  because there is no decoder — the latent *is* the hidden activation.

The split into two Protocols reflects the core evidence this sprint
produces:

- ``ModelAdapter`` — the universal surface: ``encode`` + ``latent_space``.
  Every adapter must provide this.
- ``DecodableAdapter`` — the decodable surface: extends ``ModelAdapter``
  with ``decode``. Only adapters with a decoder (learned or deterministic)
  implement this.

Methods like ``ActivationPatch`` that work through ``encode → patch → decode``
require ``DecodableAdapter``, not just ``ModelAdapter``. Methods that only
need latent-space access work with ``ModelAdapter`` alone.

.. note::

    These are structural (duck-typed) Protocols. Classes conform by
    providing the required properties and methods with matching
    signatures — they do **not** need to inherit from these Protocols
    or import them.

Frozen at ModelAdapter #3 (HiddenStateAdapter, Sprint 14). Validated by
3 instances with differing philosophies. The 3-mode ModelAdapter ADR
(2026-06-16) now has modes (i) and (ii) confirmed; mode (iii)
(deterministic renderer) remains pending until Sprint 16.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from latent_anything.latent_space import LatentSpace


@runtime_checkable
class ModelAdapter(Protocol):
    """Base protocol for all model adapters.

    Every adapter must provide ``encode`` to transform input data into
    a latent representation, and ``latent_space`` to describe the geometry
    of that representation.

    ``decode`` is deliberately **not** part of this Protocol — not all
    adapters have a decoder (see ``DecodableAdapter``).
    """

    @property
    def latent_space(self) -> LatentSpace:
        """Return a ``LatentSpace`` describing this adapter's latent manifold."""
        ...

    def encode(self, data: np.ndarray) -> np.ndarray:
        """Encode input data to latent space.

        Parameters
        ----------
        data : np.ndarray
            2D array of shape ``(n_samples, input_dim)``.

        Returns
        -------
        np.ndarray
            Latent vectors of shape ``(n_samples, latent_dim)``.
        """
        ...


@runtime_checkable
class DecodableAdapter(ModelAdapter, Protocol):
    """Protocol for adapters that support decoding back to data space.

    Extends ``ModelAdapter`` with a ``decode`` method. Adapters that
    have a decoder (learned, as in VAE, or deterministic, as in
    RandomProjection) conform to this Protocol.

    Methods like ``ActivationPatch`` that work through
    ``encode → patch → decode`` require ``DecodableAdapter``.
    """

    def decode(self, latent: np.ndarray) -> np.ndarray:
        """Decode latent vectors back to data space.

        Parameters
        ----------
        latent : np.ndarray
            2D array of shape ``(n_samples, latent_dim)``.

        Returns
        -------
        np.ndarray
            Reconstructed data of shape ``(n_samples, output_dim)``.
        """
        ...

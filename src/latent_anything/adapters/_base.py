"""UNSTABLE — internal shared shape sketched for ModelAdapter instances.

.. warning::

    **This module is UNSTABLE.** Do not depend on this shape in public
    code or plugin implementations. It captures only what the first two
    ``ModelAdapter`` instances (VAE, RandomProjection) happen to share,
    and **will be replaced** when ModelAdapter #3 lands.

    Minimal shared surface discovered so far:
    - ``encode(data: np.ndarray) -> np.ndarray``
    - ``decode(latent: np.ndarray) -> np.ndarray``
    - ``latent_space`` property → ``LatentSpace``

    Note: ``fit`` is **NOT** part of this shared shape — it is
    VAE-specific. The frozen ``ModelAdapter`` Protocol (future, when
    instance #3 of differing philosophy appears) may or may not
    include ``fit`` depending on what the third instance reveals.

This module is internal (``_``-prefixed), not exported from the
``adapters`` package public surface, and not in ``__all__``.
The class is deliberately not consumed yet — it is a sketch.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from latent_anything.latent_space import LatentSpace


class _ModelAdapterBase(ABC):  # pyright: ignore[reportUnusedClass]
    """UNSTABLE internal base for ModelAdapter instances.

    DO NOT depend on this class externally. It is a convenience base
    that captures the minimal surface shared by VAE and RandomProjection.

    The frozen ``ModelAdapter`` Protocol will be extracted when a third
    instance of differing philosophy arrives (Rule of Three, see
    ``docs/INCREMENTAL.md`` §4a).
    """

    @property
    @abstractmethod
    def latent_space(self) -> LatentSpace:
        """Return a ``LatentSpace`` describing this adapter's latent manifold."""

    @abstractmethod
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

    @abstractmethod
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
        """

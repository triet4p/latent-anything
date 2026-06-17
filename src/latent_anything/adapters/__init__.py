"""Model adapters — expose latent spaces from concrete model implementations.

This is the adapters sub-package of latent_anything, a sibling to
``methods/``. It provides concrete model implementations that produce
a ``LatentSpace`` via their ``latent_space`` property, supporting
``encode`` and ``decode`` operations.

The ``ModelAdapter`` primitive is NOT yet frozen (Rule of Three: instance #2).
An internal ``_ModelAdapterBase`` shape has been sketched (marked UNSTABLE)
but is NOT part of the public surface. Freeze point is ModelAdapter #3.
"""

from __future__ import annotations

from latent_anything.adapters.random_projection import RandomProjection as RandomProjection
from latent_anything.adapters.vae import VAE as VAE

__all__ = ["RandomProjection", "VAE"]

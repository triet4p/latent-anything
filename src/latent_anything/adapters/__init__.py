"""Model adapters — expose latent spaces from concrete model implementations.

This is the adapters sub-package of latent_anything, a sibling to
``methods/``. It provides concrete model implementations that produce
a ``LatentSpace`` via their ``latent_space`` property, supporting
``encode`` and ``decode`` operations.

The ``ModelAdapter`` primitive is NOT yet frozen (Rule of Three: instance #1).
No Protocol or ABC exists — each adapter is a standalone hardcoded class.
"""

from __future__ import annotations

from latent_anything.adapters.vae import VAE as VAE

__all__ = ["VAE"]

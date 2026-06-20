"""Model adapters — expose latent spaces from concrete model implementations.

This is the adapters sub-package of latent_anything, a sibling to
``methods/``. It provides concrete model implementations that produce
a ``LatentSpace`` via their ``latent_space`` property.

The ``ModelAdapter`` and ``DecodableAdapter`` Protocols are now frozen
(Rule of Three: instance #3, Sprint 14). Three instances with differing
philosophies validate the split:

- ``VAE`` (#1) — explicit learned latent (mode i), conforms to both
  ``ModelAdapter`` and ``DecodableAdapter``.
- ``RandomProjection`` (#2) — fixed explicit projection (mode i-like),
  conforms to both ``ModelAdapter`` and ``DecodableAdapter``.
- ``HiddenStateAdapter`` (#3) — no-explicit-latent (mode ii), conforms
  to ``ModelAdapter`` only (no ``decode``).
"""

from __future__ import annotations

from latent_anything.adapters.hidden_state import HiddenStateAdapter as HiddenStateAdapter
from latent_anything.adapters.protocols import DecodableAdapter as DecodableAdapter
from latent_anything.adapters.protocols import ModelAdapter as ModelAdapter
from latent_anything.adapters.random_projection import RandomProjection as RandomProjection
from latent_anything.adapters.vae import VAE as VAE

__all__ = [
    "DecodableAdapter",
    "HiddenStateAdapter",
    "ModelAdapter",
    "RandomProjection",
    "VAE",
]

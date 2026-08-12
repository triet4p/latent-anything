"""Model adapters — expose latent spaces from concrete model implementations.

This is the adapters sub-package of latent_anything, a sibling to
``methods/``. It provides concrete model implementations that produce
a ``LatentSpace`` via their ``latent_space`` property.

The adapter Protocols are now frozen (Rule of Three: instance #3,
Sprint 14). Three instances with differing philosophies validate the
``ModelAdapter``/``DecodableAdapter`` split:

- ``VAE`` (#1) — explicit learned latent (mode i), conforms to both
  ``ModelAdapter`` and ``DecodableAdapter``.
- ``RandomProjection`` (#2) — fixed explicit projection (mode i-like),
  conforms to both ``ModelAdapter`` and ``DecodableAdapter``.
- ``HiddenStateAdapter`` (#3) — no-explicit-latent (mode ii), conforms
  to ``ModelAdapter`` only (no ``decode``).
- ``GaussianRendererAdapter`` (#4) — deterministic renderer (mode iii),
  conforms to ``DecodableAdapter`` but not ``FlatBatchDecodableAdapter``.
- ``VQVAE`` (#5) — explicit learned discrete latent (mode i), conforms to
  ``ModelAdapter`` and ``DecodableAdapter`` but preserves integer code IDs.
"""

from __future__ import annotations

from latent_anything.adapters.conv_vae import ConvVAE as ConvVAE
from latent_anything.adapters.gaussian_3d_renderer import Gaussian3DRendererAdapter as Gaussian3DRendererAdapter
from latent_anything.adapters.gaussian_3d_renderer import GaussianCamera as GaussianCamera
from latent_anything.adapters.gaussian_renderer import GaussianRendererAdapter as GaussianRendererAdapter
from latent_anything.adapters.hidden_state import HiddenStateAdapter as HiddenStateAdapter
from latent_anything.adapters.jepa import JEPAEvaluationReport as JEPAEvaluationReport
from latent_anything.adapters.jepa import JEPALatentHealth as JEPALatentHealth
from latent_anything.adapters.jepa import JEPAPrediction as JEPAPrediction
from latent_anything.adapters.jepa import JEPAPredictionMetrics as JEPAPredictionMetrics
from latent_anything.adapters.jepa import JEPARolloutMetrics as JEPARolloutMetrics
from latent_anything.adapters.jepa import JEPAWorldModelAdapter as JEPAWorldModelAdapter
from latent_anything.adapters.jepa import JEPAWorldModelConfig as JEPAWorldModelConfig
from latent_anything.adapters.protocols import DecodableAdapter as DecodableAdapter
from latent_anything.adapters.protocols import FlatBatchDecodableAdapter as FlatBatchDecodableAdapter
from latent_anything.adapters.protocols import ModelAdapter as ModelAdapter
from latent_anything.adapters.random_projection import RandomProjection as RandomProjection
from latent_anything.adapters.vae import VAE as VAE
from latent_anything.adapters.vq_vae import VQVAE as VQVAE
from latent_anything.tokenized_world_model import (
    TokenizedEvaluationReport as TokenizedEvaluationReport,
)
from latent_anything.tokenized_world_model import (
    TokenizedWorldModel as TokenizedWorldModel,
)
from latent_anything.tokenized_world_model import (
    TokenizedWorldModelConfig as TokenizedWorldModelConfig,
)
from latent_anything.tokenized_world_model import (
    TokenPrediction as TokenPrediction,
)
from latent_anything.tokenized_world_model import (
    TokenPredictionMetrics as TokenPredictionMetrics,
)
from latent_anything.tokenized_world_model import (
    TokenRolloutMetrics as TokenRolloutMetrics,
)

__all__ = [
    "ConvVAE",
    "DecodableAdapter",
    "FlatBatchDecodableAdapter",
    "GaussianRendererAdapter",
    "Gaussian3DRendererAdapter",
    "GaussianCamera",
    "HiddenStateAdapter",
    "JEPAWorldModelAdapter",
    "JEPAWorldModelConfig",
    "JEPALatentHealth",
    "JEPAEvaluationReport",
    "JEPAPrediction",
    "JEPAPredictionMetrics",
    "JEPARolloutMetrics",
    "ModelAdapter",
    "RandomProjection",
    "VAE",
    "VQVAE",
    "TokenPrediction",
    "TokenPredictionMetrics",
    "TokenRolloutMetrics",
    "TokenizedEvaluationReport",
    "TokenizedWorldModel",
    "TokenizedWorldModelConfig",
]

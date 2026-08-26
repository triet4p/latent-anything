"""Compatibility facade for concrete deterministic and stochastic transitions.

The public module remains the stable import path.  Concrete lifecycles live in
focused private modules, while result dataclasses live in a cycle-free types
module.  No shared lifecycle abstraction is introduced here.
"""

from __future__ import annotations

from latent_anything._transition_deterministic import DeterministicLatentTransition
from latent_anything._transition_stochastic import StochasticGaussianLatentTransition
from latent_anything._transition_types import (
    GaussianPrediction,
    OneStepMetrics,
    RolloutMetrics,
    StochasticOneStepMetrics,
    StochasticRollout,
    StochasticRolloutMetrics,
)

# Keep these historical module attributes available for callers that imported
# the foundational value types from this module incidentally.
from latent_anything.latent_space import LatentSpace as LatentSpace
from latent_anything.trajectory import Trajectory as Trajectory

_PUBLIC_TYPES = (
    DeterministicLatentTransition,
    GaussianPrediction,
    OneStepMetrics,
    RolloutMetrics,
    StochasticGaussianLatentTransition,
    StochasticOneStepMetrics,
    StochasticRollout,
    StochasticRolloutMetrics,
)
for _public_type in _PUBLIC_TYPES:
    _public_type.__module__ = __name__

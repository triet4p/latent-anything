"""Typed result models shared by pipeline modules."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from latent_anything.latent_space import LatentSpace
from latent_anything.trajectory import Trajectory


@dataclass(frozen=True)
class PipelineResult:
    """Result from :class:`AnalysisPipeline`."""

    latents: np.ndarray
    transformed: np.ndarray
    latent_space: LatentSpace


@dataclass(frozen=True)
class RolloutResult:
    """Result from one mean latent rollout.

    ``trajectory`` contains the initial state followed by one predictive state
    for every action.  Arrays are owned by the result and callers should use
    ``to_numpy`` when they need a defensive copy of the trajectory.
    """

    initial_state: np.ndarray
    actions: np.ndarray
    trajectory: Trajectory
    latent_space: LatentSpace
    cache_hit: bool = False

    @property
    def states(self) -> Trajectory:
        """Compatibility alias for callers that call rollout states directly."""

        return self.trajectory

    def to_numpy(self) -> np.ndarray:
        """Return a defensive copy of the predicted state sequence."""

        return self.trajectory.to_numpy()

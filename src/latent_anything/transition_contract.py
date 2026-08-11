"""The minimal transition surface proven by the first three instances."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from latent_anything.trajectory import Trajectory


@runtime_checkable
class LatentTransition(Protocol):
    """Common mean-transition behavior shared by deterministic and stochastic models.

    Fitting, distribution-valued prediction, and recurrent reset semantics are
    intentionally outside this contract because their shapes and lifecycles
    differ across the three concrete implementations.
    """

    @property
    def state_dim(self) -> int:
        """Return the flat latent state width."""

        ...

    @property
    def action_dim(self) -> int:
        """Return the action width."""

        ...

    @property
    def source_space_identity(self) -> str:
        """Return the transition's source-space identity."""

        ...

    def step(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Return the predictive mean for one state/action pair."""

        ...

    def mean_rollout(self, initial_state: np.ndarray, actions: np.ndarray) -> Trajectory:
        """Return a recursive predictive-mean trajectory."""

        ...

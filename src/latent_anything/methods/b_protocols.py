"""Frozen ``BMethod`` Protocol for Layer B (Manipulation) methods.

Frozen at B-Method #3 (``ActivationPatch``, Sprint 12). Validated by
3 instances with differing philosophies:

- **Lerp (#1)** — stateless latent→latent, pure function, no ``fit``.
- **SteeringVector (#2)** — stateful latent→latent, ``fit`` from contrast pairs.
- **ActivationPatch (#3)** — model-mediated data→data, ``fit`` via adapter encode→patch→decode.

The Protocol captures only the **invariant surface** that all three
instances share: ``space``, ``is_fitted``, and ``apply_trajectory``.
``__call__`` is deliberately excluded — its signature genuinely differs
across the three instances (Lerp: ``(a, b, t)``, SteeringVector:
``(latent, strength)``, ActivationPatch: ``(input_data)``).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from latent_anything.latent_space import LatentSpace
from latent_anything.trajectory import Trajectory


@runtime_checkable
class BMethod(Protocol):
    """Structural protocol for Layer B manipulation methods.

    A ``BMethod`` represents a learnable transformation on latent
    representations — either directly (latent → latent) or mediated
    through a ModelAdapter (data → encode → patch → decode → data).
    It follows an optional two-phase lifecycle:

    1. **Fit** (optional): ``fit(...)`` learns the transformation
       parameters. Stateless methods like ``Lerp`` skip this phase
       and are always ``is_fitted == True``.
    2. **Apply**: ``apply_trajectory(trajectory, **kwargs)`` applies
       the learned transformation to every point in a trajectory.

    .. note::

        ``__call__`` is **not** part of this Protocol — signatures
        genuinely differ across the three validated instances.
        Callers use instance-specific ``__call__`` with duck-typing.

    .. note::

        This is a structural (duck-typed) Protocol. Classes conform by
        providing the required properties and methods with matching
        signatures — they do **not** need to inherit from ``BMethod``
        or import it.

    Frozen at B-Method #3 (``ActivationPatch``, Sprint 12). Validated
    by 3 instances with differing philosophies: stateless latent→latent
    (Lerp), stateful latent→latent (SteeringVector), model-mediated
    data→data (ActivationPatch).
    """

    @property
    def space(self) -> LatentSpace | None:
        """Return the ``LatentSpace`` for geometry-aware dispatch, or ``None``."""
        ...

    @property
    def is_fitted(self) -> bool:
        """Return ``True`` if the method is ready to apply."""
        ...

    def apply_trajectory(self, trajectory: Trajectory, **kwargs: float) -> Trajectory | np.ndarray:
        """Apply the B-Method operation to every point in a trajectory.

        Parameters
        ----------
        trajectory : Trajectory
            Input trajectory.
        **kwargs : float
            Method-specific keyword arguments (e.g. ``t`` for Lerp,
            ``strength`` for SteeringVector).

        Returns
        -------
        Trajectory | np.ndarray
            For latent→latent methods (Lerp, SteeringVector), returns a
            ``Trajectory``. For model-mediated data→data methods
            (ActivationPatch), returns ``np.ndarray`` in data space.
        """
        ...

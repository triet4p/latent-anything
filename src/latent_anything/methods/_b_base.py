"""⚠️ UNSTABLE — internal shared shape sketched for B-Method instances.

.. warning::

    **This module is UNSTABLE.** Do not depend on this shape in public
    code or plugin implementations. It captures only what the first two
    B-Method instances (Lerp, SteeringVector) happen to share, and
    **will be replaced** when B-Method #3 (activation patching) lands.

    Minimal shared surface discovered so far:
    - ``__call__(...) -> np.ndarray`` — primary operation (signature
      varies between instances: Lerp takes ``(a, b, t)``, SteeringVector
      takes ``(latent, strength)``).
    - ``space`` property → ``LatentSpace | None``.
    - ``apply_trajectory(...) -> Trajectory`` — trajectory-level
      operation.

    Note: ``fit`` is **NOT** part of this shared shape — it is
    SteeringVector-specific. Lerp (stateless) has no ``fit``. The frozen
    B-Method interface (future, when instance #3 of differing philosophy
    appears) may or may not include ``fit`` depending on what the third
    instance reveals.

This module is internal (``_``-prefixed), not exported from the
``methods`` package public surface, and not in ``__all__``.
The class is deliberately not consumed yet — it is a sketch.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from latent_anything.latent_space import LatentSpace
from latent_anything.trajectory import Trajectory


class _BMethodBase(ABC):  # pyright: ignore[reportUnusedClass]
    """⚠️ UNSTABLE internal base for B-Method instances.

    DO NOT depend on this class externally. It is a convenience base
    that captures the minimal surface shared by Lerp (#1, stateless)
    and SteeringVector (#2, stateful).

    Shape may change when B-Method #3 (activation patching) reveals
    the full stateless+stateful spectrum. The frozen B-Method Protocol
    will be extracted at that point (Rule of Three, see
    ``docs/INCREMENTAL.md`` §4a).
    """

    @property
    @abstractmethod
    def space(self) -> LatentSpace | None:
        """Return the optional ``LatentSpace`` for geometry-aware dispatch."""

    @abstractmethod
    def apply_trajectory(self, trajectory: Trajectory, **kwargs: float) -> Trajectory:
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
        Trajectory
            A new ``Trajectory`` with the operation applied.
        """

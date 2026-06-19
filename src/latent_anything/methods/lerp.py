"""A stateless interpolation method for latent points and trajectories.

This is B-Method #1 — the first Layer B (Manipulation) method in the
latent-anything framework. Unlike Layer A methods (PCA, UMAP, SAE) which
are stateful (fit → transform), Lerp is a pure function with no internal
state. It delegates to ``LatentSpace.interpolate()`` for geometry-aware
dispatch, or defaults to Euclidean lerp when no space is provided.

Conforms to the ``BMethod`` Protocol (structural, duck-typed) after the
Rule of Three freeze at B-Method #3 (``ActivationPatch``, Sprint 12).
"""

from __future__ import annotations

from numbers import Integral, Real

import numpy as np

from latent_anything.latent_space import LatentSpace
from latent_anything.trajectory import Trajectory


class Lerp:
    """Stateless interpolation between latent points or trajectories.

    ``Lerp`` wraps ``LatentSpace.interpolate()`` as a first-class
    Method object. All operations are pure functions — no ``fit``,
    no internal state.

    Conforms to the ``BMethod`` Protocol (structural, duck-typed).
    ``is_fitted`` always returns ``True`` because Lerp is stateless —
    there is no ``fit`` phase.

    Parameters
    ----------
    space : LatentSpace | None, optional
        Optional ``LatentSpace`` for geometry-aware dispatch. If
        ``None``, defaults to Euclidean lerp ``(1-t)*a + t*b``.
    """

    def __init__(self, space: LatentSpace | None = None) -> None:
        self._space = space

    @property
    def space(self) -> LatentSpace | None:
        """Return the optional ``LatentSpace`` used for geometry dispatch."""
        return self._space

    @property
    def is_fitted(self) -> bool:
        """Return ``True`` — Lerp is stateless and always ready.

        Stateless methods have no ``fit`` phase and are always
        considered fitted. This satisfies the ``BMethod`` Protocol
        requirement.
        """
        return True

    def __call__(self, a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
        """Interpolate between two 1D latent vectors ``a`` and ``b``.

        If a ``LatentSpace`` was provided at construction, delegates to
        ``space.interpolate(a, b, t)`` for geometry-aware dispatch (e.g.
        slerp for unit_norm). Otherwise, performs Euclidean lerp.

        Parameters
        ----------
        a : np.ndarray
            1-D array of shape ``(dim,)``.
        b : np.ndarray
            1-D array of shape ``(dim,)``.
        t : float
            Interpolation parameter in ``[0, 1]``.

        Returns
        -------
        np.ndarray
            Interpolated point — a new array, not a view.

        Raises
        ------
        ValueError
            If ``a`` and ``b`` have different shapes.
        """
        if a.shape != b.shape:
            msg = f"a and b must have the same shape, got {a.shape} and {b.shape}"
            raise ValueError(msg)
        if self._space is not None:
            return self._space.interpolate(a, b, t)
        return (1.0 - t) * a + t * b

    def between(self, traj_a: Trajectory, traj_b: Trajectory, t: float) -> Trajectory:
        """Pointwise interpolation between two trajectories.

        Produces a new ``Trajectory`` where each point is the interpolation
        of corresponding points in ``traj_a`` and ``traj_b`` at parameter
        ``t``.

        Parameters
        ----------
        traj_a : Trajectory
            First trajectory.
        traj_b : Trajectory
            Second trajectory (must have same ``dim`` and ``len`` as
            ``traj_a``).
        t : float
            Interpolation parameter in ``[0, 1]``.

        Returns
        -------
        Trajectory
            A new ``Trajectory`` with the same shape as the inputs.

        Raises
        ------
        ValueError
            If the trajectories have different lengths or dimensions.
        """
        if len(traj_a) != len(traj_b):
            msg = f"Trajectories must have the same length, got {len(traj_a)} and {len(traj_b)}"
            raise ValueError(msg)
        if traj_a.dim != traj_b.dim:
            msg = f"Trajectories must have the same dim, got {traj_a.dim} and {traj_b.dim}"
            raise ValueError(msg)

        data_a = traj_a.to_numpy()
        data_b = traj_b.to_numpy()
        blended = np.array([self(a_row, b_row, t) for a_row, b_row in zip(data_a, data_b)])
        return Trajectory(data=blended)

    def blend_sequence(self, trajectory: Trajectory, n_steps: int = 2) -> Trajectory:
        """Densely interpolate between consecutive points in a trajectory.

        For a trajectory with points ``[p0, p1, p2]`` and ``n_steps=2``,
        produces ``[p0, p0.5, p1, p1.5, p2]`` where ``p0.5`` is the
        interpolation at ``t=0.5`` between ``p0`` and ``p1``, and ``p1.5``
        is between ``p1`` and ``p2``.

        Parameters
        ----------
        trajectory : Trajectory
            Input trajectory to densify.
        n_steps : int, optional
            Number of interpolation steps between each consecutive pair
            of points. Defaults to 2.

        Returns
        -------
        Trajectory
            A new densified ``Trajectory``.

        Raises
        ------
        ValueError
            If ``n_steps < 1``.
        """
        if n_steps < 1:
            msg = f"n_steps must be >= 1, got {n_steps}"
            raise ValueError(msg)

        data = trajectory.to_numpy()
        n_points = len(trajectory)
        dense: list[np.ndarray] = []

        for i in range(n_points - 1):
            for step in range(n_steps):
                tt = step / n_steps
                dense.append(self(data[i], data[i + 1], tt))
        dense.append(data[-1])  # final point

        return Trajectory(data=np.array(dense))

    def apply_trajectory(self, trajectory: Trajectory, **kwargs: object) -> Trajectory:
        """Apply interpolation to a trajectory (generic ``BMethod`` interface).

        Supports two modes via ``**kwargs``:

        - ``other`` (Trajectory) + ``t`` (float): delegates to
          ``between(trajectory, other, t)``.
        - ``n_steps`` (int): delegates to ``blend_sequence(trajectory, n_steps)``.

        Parameters
        ----------
        trajectory : Trajectory
            Input trajectory.
        **kwargs : object
            Keyword arguments — must contain either ``other`` + ``t``
            or ``n_steps``.

        Returns
        -------
        Trajectory
            A new ``Trajectory`` with the operation applied.

        Raises
        ------
        ValueError
            If the keyword arguments do not match the expected patterns.
        """
        if "other" in kwargs and "t" in kwargs:
            other_value = kwargs.pop("other")
            t_value = kwargs.pop("t")
            if not isinstance(other_value, Trajectory):
                msg = f"'other' must be a Trajectory, got {type(other_value).__name__}"
                raise ValueError(msg)
            if not isinstance(t_value, Real):
                msg = f"'t' must be a real number, got {type(t_value).__name__}"
                raise ValueError(msg)
            return self.between(trajectory, other_value, float(t_value))
        if "n_steps" in kwargs:
            n_steps_value = kwargs.pop("n_steps")
            if not isinstance(n_steps_value, Integral):
                msg = f"'n_steps' must be an integer, got {type(n_steps_value).__name__}"
                raise ValueError(msg)
            n_steps_val = int(n_steps_value)
            return self.blend_sequence(trajectory, n_steps=n_steps_val)
        msg = (
            "Lerp.apply_trajectory requires either "
            "'other' (Trajectory) + 't' (float) or 'n_steps' (int). "
            f"Got kwargs={kwargs!r}"
        )
        raise ValueError(msg)

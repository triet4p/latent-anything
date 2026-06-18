"""A stateful steering method that learns a direction from contrast pairs.

This is B-Method #2 — the second Layer B (Manipulation) method, and the
first *stateful* B-Method. Unlike Lerp (stateless, pure function,
no ``fit``), ``SteeringVector`` has a ``fit(positives, negatives)``
phase that learns a unit steering direction from contrast data, and a
``__call__(latent, strength)`` phase that applies the steering.

SteeringVector constitutes instance #2 of the B-Method pattern.
Per the Rule of Three (§4a in INCREMENTAL.md), the shared B-Method shape
is sketched in ``_b_base.py`` as an internal UNSTABLE base. Public
interface freeze happens at B-Method #3 (activation patching, Sprint 12).
"""

from __future__ import annotations

import numpy as np

from latent_anything.latent_space import LatentSpace
from latent_anything.trajectory import Trajectory


class SteeringVector:
    """Learn a steering direction from contrast pairs and apply it to latent vectors.

    ``SteeringVector`` learns a direction in latent space that separates
    positive from negative examples, then steers latent representations
    along that direction. This is the stateful counterpart to ``Lerp``
    (stateless) — B-Method #2.

    Algorithm:
        1. ``fit(positives, negatives)``: compute ``direction = normalize(mean(pos) - mean(neg))``.
        2. ``__call__(latent, strength)``: ``latent + strength * direction``.

    Parameters
    ----------
    space : LatentSpace | None, optional
        Optional ``LatentSpace`` for geometry-aware post-steer
        normalization. If ``None``, no normalization is applied.
        When provided with ``geometry="unit_norm"``, steered points
        are projected back onto the unit sphere.
    """

    def __init__(self, space: LatentSpace | None = None) -> None:
        self._space = space
        self._direction: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def space(self) -> LatentSpace | None:
        """Return the optional ``LatentSpace`` used for geometry-aware normalization."""
        return self._space

    @property
    def is_fitted(self) -> bool:
        """Return ``True`` if ``fit()`` has been called successfully."""
        return self._direction is not None

    @property
    def direction(self) -> np.ndarray:
        """Return the learned unit steering direction.

        Returns
        -------
        np.ndarray
            1-D array of shape ``(dim,)`` — the unit steering direction.

        Raises
        ------
        RuntimeError
            If ``fit()`` has not been called yet.
        """
        if self._direction is None:
            msg = "SteeringVector not fitted. Call fit() first."
            raise RuntimeError(msg)
        return self._direction.copy()

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, positives: np.ndarray, negatives: np.ndarray) -> None:
        """Learn the steering direction from contrast pairs.

        Computes ``direction = normalize(mean(positives) - mean(negatives))``
        — a unit vector pointing from the negative region toward the positive
        region in latent space.

        Parameters
        ----------
        positives : np.ndarray
            2-D array of shape ``(n_pos, dim)`` — positive (desired) examples.
        negatives : np.ndarray
            2-D array of shape ``(n_neg, dim)`` — negative (undesired) examples.

        Raises
        ------
        ValueError
            If either array is not 2-D, or if they have different
            feature dimensions, or if the contrast direction is zero.
        """
        if positives.ndim != 2:
            msg = f"positives must be a 2D array (n_samples, dim), got {positives.ndim}D"
            raise ValueError(msg)
        if negatives.ndim != 2:
            msg = f"negatives must be a 2D array (n_samples, dim), got {negatives.ndim}D"
            raise ValueError(msg)
        if positives.shape[0] == 0:
            msg = "positives array is empty"
            raise ValueError(msg)
        if negatives.shape[0] == 0:
            msg = "negatives array is empty"
            raise ValueError(msg)
        if positives.shape[1] != negatives.shape[1]:
            msg = (
                f"Dimension mismatch: positives have dim {positives.shape[1]}, negatives have dim {negatives.shape[1]}"
            )
            raise ValueError(msg)

        mean_pos = positives.mean(axis=0)
        mean_neg = negatives.mean(axis=0)
        diff = mean_pos - mean_neg
        norm = np.linalg.norm(diff)
        if norm < 1e-15:
            msg = "Contrast direction is zero — positives and negatives have identical means"
            raise ValueError(msg)

        self._direction = diff / norm

    # ------------------------------------------------------------------
    # Apply
    # ------------------------------------------------------------------

    def __call__(self, latent: np.ndarray, strength: float = 1.0) -> np.ndarray:
        """Steer a single 1-D latent vector along the learned direction.

        Computes ``latent + strength * direction``, then optionally
        normalises the result if a ``LatentSpace`` was provided at
        construction.

        Parameters
        ----------
        latent : np.ndarray
            1-D array of shape ``(dim,)`` — the latent vector to steer.
        strength : float, optional
            Steering strength. ``strength=0`` returns an unchanged copy.
            Negative values steer opposite the learned direction.
            Defaults to ``1.0``.

        Returns
        -------
        np.ndarray
            A new array — the steered latent vector.

        Raises
        ------
        RuntimeError
            If ``fit()`` has not been called yet.
        ValueError
            If ``latent`` shape does not match the learned direction shape.
        """
        if self._direction is None:
            msg = "SteeringVector not fitted. Call fit() first."
            raise RuntimeError(msg)
        if latent.shape != self._direction.shape:
            msg = f"Latent shape {latent.shape} does not match learned direction shape {self._direction.shape}"
            raise ValueError(msg)

        result = latent + strength * self._direction
        if self._space is not None:
            result = self._space.normalize(result)
        return result

    def apply_trajectory(self, trajectory: Trajectory, strength: float = 1.0) -> Trajectory:
        """Steer all points in a trajectory along the learned direction.

        Parameters
        ----------
        trajectory : Trajectory
            The input trajectory to steer.
        strength : float, optional
            Steering strength applied to every point. Defaults to ``1.0``.

        Returns
        -------
        Trajectory
            A new ``Trajectory`` with all points steered.

        Raises
        ------
        RuntimeError
            If ``fit()`` has not been called yet.
        """
        data = trajectory.to_numpy()
        steered = np.array([self(point, strength) for point in data])
        return Trajectory(data=steered)

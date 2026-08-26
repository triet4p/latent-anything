"""Shared leakage-guarded split helpers for labeled probes.

This module intentionally exposes no public protocol or user-facing API.  The
split algorithm is shared by the linear probe, nonlinear probe, and TCAV so
that every labeled analysis uses the same deterministic partition semantics.
"""

from __future__ import annotations

import numpy as np


def stratified_split(
    labels: np.ndarray,
    *,
    test_size: float,
    val_size: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return deterministic, per-class train/validation/test masks."""
    n = len(labels)
    classes = np.unique(labels)
    rng = np.random.default_rng(random_state)

    train = np.zeros(n, dtype=bool)
    val = np.zeros(n, dtype=bool)
    test = np.zeros(n, dtype=bool)

    for label in classes:
        idx = np.flatnonzero(labels == label)
        perm = rng.permutation(idx)
        n_cls = len(perm)

        n_test = max(1, int(round(n_cls * test_size)))
        n_val = max(1, int(round((n_cls - n_test) * val_size))) if val_size > 0 and n_cls - n_test >= 2 else 0
        n_train = n_cls - n_test - n_val

        # Ensure every split has at least one sample per class when possible.
        if n_train < 1 and n_cls >= 2:
            n_train = 1
            remaining = n_cls - n_train
            n_test = max(1, int(round(remaining * test_size / (test_size + max(val_size, 1e-9)))))
            n_val = remaining - n_test

        train[perm[:n_train]] = True
        val[perm[n_train : n_train + n_val]] = True
        test[perm[n_train + n_val :]] = True

    return train, val, test

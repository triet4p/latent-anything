"""Small reproducible dispatch-overhead benchmark for the four geometry cases."""

from __future__ import annotations

from time import perf_counter

import numpy as np

from latent_anything import LatentSpace


def _measure(space: LatentSpace, a: np.ndarray, b: np.ndarray, iterations: int = 20_000) -> float:
    start = perf_counter()
    for _ in range(iterations):
        space.distance(a, b)
    return (perf_counter() - start) / iterations * 1_000_000


def main() -> None:
    """Print microseconds per distance dispatch; this benchmark performs no writes."""

    cases = {
        "euclidean": (LatentSpace(dim=8), np.zeros(8), np.ones(8)),
        "unit_norm": (LatentSpace(dim=2, geometry="unit_norm"), np.array([1.0, 0.0]), np.array([0.0, 1.0])),
        "gaussian_set": (
            LatentSpace(dim=8, geometry="gaussian_set", n_gaussians=1, position_dim=2, scale_dim=2, color_dim=3),
            np.array([[0.0, 0.0, 1.0, 1.0, 1.0, 0.2, 0.3, 0.4]]),
            np.array([[1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.3, 0.4]]),
        ),
        "discrete_code": (
            LatentSpace(dim=8, geometry="discrete_code", codebook_size=16),
            np.zeros(8, dtype=int),
            np.ones(8, dtype=int),
        ),
    }
    for name, (space, a, b) in cases.items():
        print(f"{name}: {_measure(space, a, b):.2f} us/distance")


if __name__ == "__main__":
    main()

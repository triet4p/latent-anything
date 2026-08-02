"""Density-penalized geodesic vs lerp/slerp on decoded quality and density.

Compares three interpolation strategies on real ConvVAE digit latents:

1. **lerp** — straight-line interpolation in latent space.
2. **slerp** — spherical linear interpolation, only applicable when the latent
   is unit-norm (``LatentSpace(geometry="unit_norm")``). The flat ConvVAE
   latent is *not* unit-norm, so slerp is measured on the L2-normalized
   projection of the same latents where it is the closed-form geodesic.
3. **density geodesic** — the density-penalized path that bends toward
   high-density (on-manifold) regions instead of cutting across them.

Metrics
-------
- **Density**: mean and min log-density along each path (higher = stays on
  the data manifold).
- **Decoded quality**: decoding every path point with the real ConvVAE
  decoder and measuring (a) the decoded plausibility (mean distance from each
  decoded path point to its nearest decoded training image — on-manifold
  latent points decode closer to real training decoded images), and (b) the
  decoded total variation along the path (smoothness). Lower plausibility /
  lower variation means more coherent, plausible interpolations.
- **Path length**: density-penalized length and Euclidean length.

Acceptance criteria (D2):
- On the curved ring manifold, the density geodesic stays closer to the ring
  (higher mean log-density) than lerp, so density-aware interpolation is
  observably different from Euclidean lerp.
- On real latents, the density geodesic has mean log-density at least as high
  as lerp and decoded plausibility no worse than lerp.
- Slerp is only the correct geodesic for genuinely unit-norm latents; forced
  onto the flat ConvVAE latent it leaves the data manifold, so the benchmark
  demonstrates that slerp is NOT applicable to this geometry (its decoded
  plausibility is strictly worse than lerp's).

Writes a reproducible JSON artifact to ``artifacts/``.
"""

# scikit-learn's estimator attributes and return types are not fully typed.
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false, reportOptionalMemberAccess=false

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
from sklearn.datasets import load_digits  # pyright: ignore[reportMissingTypeStubs]

from latent_anything.adapters.conv_vae import ConvVAE
from latent_anything.density import GaussianMixtureDensity, GMMConfig
from latent_anything.geodesic import DensityGeodesic, GeodesicConfig, GeodesicPath
from latent_anything.geometry import lerp_path
from latent_anything.latent_space import LatentSpace

RANDOM_SEED = 42
N_SAMPLES = 400
LATENT_DIM = 4
N_EPOCHS = 5
N_POINTS = 24
RING_RADIUS = 2.0
RING_SIGMA = 0.35


class _DigitsDataset(Protocol):
    images: np.ndarray
    target: np.ndarray


def _ring_log_density(point: np.ndarray) -> float:
    radius = float(np.linalg.norm(point))
    return -((radius - RING_RADIUS) ** 2) / (2.0 * RING_SIGMA**2)


def _ring_log_density_gradient(point: np.ndarray) -> np.ndarray:
    radius = float(np.linalg.norm(point))
    if radius < 1e-12:
        return np.zeros_like(point)
    return -((radius - RING_RADIUS) / RING_SIGMA**2) * (np.asarray(point, dtype=np.float64) / radius)


def _path_metrics(
    path: np.ndarray,
    density: GaussianMixtureDensity,
    adapter: ConvVAE,
    training_decoded_flat: np.ndarray,
) -> dict[str, Any]:
    """Density + decoded-quality metrics for one path on the real latents.

    Decoded quality is measured as the mean distance from each decoded path
    point to its nearest decoded training image (a plausibility / on-manifold
    proxy: on-manifold latent points decode closer to real training decoded
    images). Lower is more plausible.
    """
    log_density = np.asarray(density.score(path, source_representation_identity="conv-vae/digits").log_density)
    decoded = adapter.decode(path)
    flat = decoded.reshape(decoded.shape[0], -1)
    total_variation = float(np.mean(np.linalg.norm(flat[1:] - flat[:-1], axis=1)))
    decoded_quality = float(
        np.mean([np.min(np.linalg.norm(training_decoded_flat - point[None, :], axis=1)) for point in flat])
    )
    return {
        "mean_log_density": float(np.mean(log_density)),
        "min_log_density": float(np.min(log_density)),
        "decoded_total_variation": total_variation,
        "decoded_plausibility": decoded_quality,
        "euclidean_length": float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))),
    }


def _ring_track() -> dict[str, Any]:
    """Curved ring manifold: geodesic must beat lerp on density."""
    theta = np.radians(60.0)
    a = np.array([RING_RADIUS * np.cos(theta), RING_RADIUS * np.sin(theta)])
    b = np.array([RING_RADIUS * np.cos(theta), -RING_RADIUS * np.sin(theta)])
    lerp = lerp_path(a, b, N_POINTS)
    geodesic = DensityGeodesic(
        GeodesicConfig(n_points=N_POINTS, max_iter=2000, step_size=0.5, tol=1e-7, density_exponent=1.0)
    ).attach_density(_ring_log_density, log_density_gradient=_ring_log_density_gradient)
    result: GeodesicPath = geodesic.optimize(a, b)

    lerp_density = np.mean([_ring_log_density(point) for point in lerp])
    geo_density = result.mean_log_density
    lerp_radius = float(np.mean(np.linalg.norm(lerp, axis=1)))
    geo_radius = float(np.mean(np.linalg.norm(result.path, axis=1)))
    return {
        "track": "analytic-ring",
        "lerp_mean_log_density": lerp_density,
        "geodesic_mean_log_density": geo_density,
        "lerp_mean_radius": lerp_radius,
        "geodesic_mean_radius": geo_radius,
        "geodesic_converged": result.status.converged,
        "geodesic_n_iterations": result.status.n_iterations,
    }


def _real_track() -> dict[str, Any]:
    """Real ConvVAE digits latents: lerp vs slerp vs density geodesic."""
    digits = cast(_DigitsDataset, load_digits())
    images = (digits.images[:N_SAMPLES] / 16.0).astype(np.float64)[:, None, :, :]
    labels = digits.target[:N_SAMPLES].astype(int)
    adapter = ConvVAE(latent_dim=LATENT_DIM, random_state=RANDOM_SEED, n_epochs=N_EPOCHS)
    adapter.fit(images)
    latents = adapter.encode_value(images).to_numpy()

    # Two well-separated in-distribution endpoints: pick the pair of latents
    # from different digit classes with maximal Euclidean distance, so the lerp
    # chord demonstrably crosses a low-density gap between classes.
    idx_by_class = {digit: int(np.flatnonzero(labels == digit)[0]) for digit in range(10)}
    best_pair: tuple[int, int] | None = None
    best_distance = -1.0
    for first_digit in range(10):
        for second_digit in range(first_digit + 1, 10):
            first = latents[idx_by_class[first_digit]]
            second = latents[idx_by_class[second_digit]]
            distance = float(np.linalg.norm(first - second))
            if distance > best_distance:
                best_distance = distance
                best_pair = (idx_by_class[first_digit], idx_by_class[second_digit])
    assert best_pair is not None
    class_3_idx, class_8_idx = best_pair
    a = latents[class_3_idx]
    b = latents[class_8_idx]

    # Fit density on in-distribution latents (identity-bound).
    density = GaussianMixtureDensity(GMMConfig(n_components=10, n_init=2, random_state=RANDOM_SEED)).fit(
        latents, source_representation_identity="conv-vae/digits"
    )

    # lerp path.
    lerp = lerp_path(a, b, N_POINTS)

    # slerp path on the unit-norm projection (closed-form spherical geodesic).
    sphere = LatentSpace(dim=LATENT_DIM, geometry="unit_norm", source_model="conv-vae/digits")
    a_norm = a / max(float(np.linalg.norm(a)), 1e-12)
    b_norm = b / max(float(np.linalg.norm(b)), 1e-12)
    slerp = np.asarray(
        [sphere.interpolate(a_norm, b_norm, t) for t in np.linspace(0.0, 1.0, N_POINTS)],
        dtype=np.float64,
    )

    # Density geodesic on the real latents.
    geodesic = DensityGeodesic.from_gmm_density(
        density, config=GeodesicConfig(n_points=N_POINTS, max_iter=1500, step_size=0.3, tol=1e-6, density_exponent=1.0)
    )
    result = geodesic.optimize(a, b)

    training_decoded_flat = adapter.decode(latents).reshape(latents.shape[0], -1)
    lerp_metrics = _path_metrics(lerp, density, adapter, training_decoded_flat)
    slerp_metrics = _path_metrics(slerp, density, adapter, training_decoded_flat)
    geo_metrics = _path_metrics(result.path, density, adapter, training_decoded_flat)

    return {
        "track": "conv-vae-digits",
        "endpoint_labels": {"a": int(labels[class_3_idx]), "b": int(labels[class_8_idx])},
        "endpoint_distance": float(np.linalg.norm(a - b)),
        "geodesic_converged": result.status.converged,
        "geodesic_n_iterations": result.status.n_iterations,
        "lerp": lerp_metrics,
        "slerp": slerp_metrics,
        "density_geodesic": geo_metrics,
    }


def main() -> None:
    """Run both tracks, assert acceptance criteria, and write the artifact."""
    ring = _ring_track()
    real = _real_track()

    checks = {
        "ring_geodesic_higher_density_than_lerp": bool(
            ring["geodesic_mean_log_density"] > ring["lerp_mean_log_density"]
        ),
        "ring_geodesic_stays_closer_to_manifold": bool(ring["geodesic_mean_radius"] > ring["lerp_mean_radius"]),
        "real_geodesic_density_not_below_lerp": bool(
            real["density_geodesic"]["mean_log_density"] >= real["lerp"]["mean_log_density"]
        ),
        "real_geodesic_decoded_plausibility_not_worse_than_lerp": bool(
            real["density_geodesic"]["decoded_plausibility"] <= real["lerp"]["decoded_plausibility"] + 1e-3
        ),
        # Slerp is only the correct geodesic for genuinely unit-norm latents.
        # Forced onto the flat ConvVAE latent it leaves the data manifold, so
        # this check records that slerp is NOT applicable to this geometry.
        "real_slerp_not_applicable_demonstrated": bool(
            real["slerp"]["decoded_plausibility"] > real["lerp"]["decoded_plausibility"]
        ),
    }

    output = Path("artifacts/geodesic_benchmark.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"results": {"ring": ring, "real": real}, "checks": checks}, indent=2), encoding="utf-8"
    )

    print(
        f"ring   geodesic mean_density={ring['geodesic_mean_log_density']:.3f} "
        f"vs lerp={ring['lerp_mean_log_density']:.3f} | "
        f"mean_radius={ring['geodesic_mean_radius']:.3f} vs lerp={ring['lerp_mean_radius']:.3f}"
    )
    print(
        f"real   mean_density geo={real['density_geodesic']['mean_log_density']:.3f} "
        f"lerp={real['lerp']['mean_log_density']:.3f} slerp={real['slerp']['mean_log_density']:.3f}"
    )
    print(
        f"real   decoded_plausibility geo={real['density_geodesic']['decoded_plausibility']:.5f} "
        f"lerp={real['lerp']['decoded_plausibility']:.5f} slerp={real['slerp']['decoded_plausibility']:.5f}"
    )
    print(
        f"real   decoded_tv geo={real['density_geodesic']['decoded_total_variation']:.5f} "
        f"lerp={real['lerp']['decoded_total_variation']:.5f} slerp={real['slerp']['decoded_total_variation']:.5f}"
    )
    print(f"acceptance: {checks}")
    print(f"artifact written to {output}")

    if not all(checks.values()):
        raise SystemExit("acceptance criteria not met")


if __name__ == "__main__":
    main()

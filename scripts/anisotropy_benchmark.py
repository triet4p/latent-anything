"""Euclidean vs Mahalanobis neighbors and OOD scores under anisotropy.

Compares the two distance notions on (a) a controlled anisotropic Gaussian
dataset with known covariance and (b) real latents from a ConvVAE trained on
sklearn digits. Reports neighbour overlap (Jaccard) and OOD AUROC for both
metrics, and writes a reproducible JSON artifact to ``artifacts/``.

Acceptance criteria (see D2 evidence ledger):
- On the controlled dataset, Mahalanobis OOD AUROC is strictly higher than
  Euclidean OOD AUROC because the score respects the anisotropic metric.
- On both datasets, the Euclidean and Mahalanobis neighbour sets differ, so
  the metric choice is observable.
"""

# scikit-learn's estimator attributes and return types are not fully typed.
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false, reportOptionalMemberAccess=false

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
from sklearn.datasets import load_digits  # pyright: ignore[reportMissingTypeStubs]
from sklearn.metrics import roc_auc_score  # pyright: ignore[reportMissingTypeStubs]

from latent_anything.covariance import fit_covariance_state
from latent_anything.geometry import mahalanobis_distance

DistanceFn = Callable[[np.ndarray, np.ndarray], float]


class _DigitsDataset(Protocol):
    images: np.ndarray
    target: np.ndarray


def _rows(matrix: np.ndarray) -> list[np.ndarray]:
    """Return each row of a 2D array as an explicitly-typed 1D array."""
    return [np.asarray(matrix[index], dtype=np.float64) for index in range(matrix.shape[0])]


def _euclidean(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _metric_mahalanobis(covariance: np.ndarray) -> DistanceFn:
    def distance(a: np.ndarray, b: np.ndarray) -> float:
        return mahalanobis_distance(a, b, covariance)

    return distance


def _neighbors(train: np.ndarray, queries: np.ndarray, k: int, *, distance: DistanceFn) -> list[np.ndarray]:
    """Return the k nearest training indices per query under a distance callable."""
    train_rows = _rows(train)
    result: list[np.ndarray] = []
    for query in _rows(queries):
        distances = np.asarray([distance(query, point) for point in train_rows])
        result.append(np.argsort(distances, kind="stable")[:k])
    return result


def _jaccard(a: np.ndarray, b: np.ndarray) -> float:
    intersection = len(set(a.tolist()) & set(b.tolist()))
    union = len(set(a.tolist()) | set(b.tolist()))
    return float(intersection / union) if union else 0.0


def _mean_neighbor_overlap(train: np.ndarray, queries: np.ndarray, k: int, covariance: np.ndarray) -> dict[str, Any]:
    euclidean = _neighbors(train, queries, k, distance=_euclidean)
    mahalanobis = _neighbors(train, queries, k, distance=_metric_mahalanobis(covariance))
    overlaps = [jaccard for jaccard in (_jaccard(a, b) for a, b in zip(euclidean, mahalanobis, strict=True))]
    return {
        "k": k,
        "mean_jaccard_overlap": float(np.mean(overlaps)),
        "jaccard_per_query": [float(j) for j in overlaps],
    }


def _ood_auroc(train: np.ndarray, ood: np.ndarray, mean: np.ndarray, covariance: np.ndarray) -> dict[str, float]:
    """AUROC separating train (ID) from ood (OOD) by each distance to the mean."""
    y_true = np.concatenate([np.zeros(len(train)), np.ones(len(ood))])
    train_rows = _rows(train)
    ood_rows = _rows(ood)
    euclidean_scores = np.concatenate(
        [np.asarray([_euclidean(p, mean) for p in train_rows])] + [np.asarray([_euclidean(p, mean) for p in ood_rows])]
    )
    distance = _metric_mahalanobis(covariance)
    mahalanobis_scores = np.concatenate(
        [np.asarray([distance(p, mean) for p in train_rows])] + [np.asarray([distance(p, mean) for p in ood_rows])]
    )
    return {
        "euclidean_auroc": float(roc_auc_score(y_true, euclidean_scores)),
        "mahalanobis_auroc": float(roc_auc_score(y_true, mahalanobis_scores)),
    }


def _controlled_anisotropic() -> dict[str, Any]:
    """Elongated 2D Gaussian: direction of least variance is the OOD axis."""
    rng = np.random.default_rng(42)
    covariance = np.diag([25.0, 0.25])  # strong anisotropy
    mean = np.array([0.0, 0.0])
    n_train = 600
    n_ood = 300
    train = rng.multivariate_normal(mean, covariance, size=n_train)
    ood = rng.multivariate_normal(np.array([0.0, 3.0]), np.diag([25.0, 0.25]), size=n_ood)

    state = fit_covariance_state(train, source_representation_identity="controlled-anisotropic")
    queries = rng.multivariate_normal(mean, covariance, size=40)
    return {
        "dataset": "controlled-anisotropic-2d",
        "n_train": n_train,
        "n_ood": n_ood,
        "true_covariance_diag": [25.0, 0.25],
        "neighbor_overlap": _mean_neighbor_overlap(train, queries, k=10, covariance=state.covariance),
        "ood": _ood_auroc(train, ood, state.mean, state.covariance),
        "fitted_covariance": state.covariance.tolist(),
    }


def _conv_vae_latents() -> dict[str, Any]:
    """Real latents from a compact ConvVAE trained on sklearn digits."""
    from latent_anything.adapters.conv_vae import ConvVAE

    digits = cast(_DigitsDataset, load_digits())
    images = (digits.images[:300] / 16.0).astype(np.float64)[:, None, :, :]
    labels = digits.target[:300].astype(int)
    adapter = ConvVAE(latent_dim=4, random_state=42, n_epochs=4)
    adapter.fit(images)
    latent = adapter.encode_value(images).to_numpy()

    train_mask = labels < 5
    ood_mask = labels >= 5
    train = latent[train_mask]
    ood = latent[ood_mask]

    state = fit_covariance_state(train, source_representation_identity="conv-vae/digits")
    queries = train[:40]
    return {
        "dataset": "conv-vae-digits",
        "n_train": int(train.shape[0]),
        "n_ood": int(ood.shape[0]),
        "latent_dim": int(train.shape[1]),
        "neighbor_overlap": _mean_neighbor_overlap(train, queries, k=10, covariance=state.covariance),
        "ood": _ood_auroc(train, ood, state.mean, state.covariance),
        "mean_eigenvalue_ratio": float(
            np.max(np.linalg.eigvalsh(state.covariance))
            / np.maximum(np.min(np.linalg.eigvalsh(state.covariance)), 1e-12)
        ),
    }


def main() -> None:
    """Run both benchmark tracks, assert acceptance criteria, and write the artifact."""
    results = {
        "controlled": _controlled_anisotropic(),
        "real_latents": _conv_vae_latents(),
    }

    controlled_ood = results["controlled"]["ood"]
    real_ood = results["real_latents"]["ood"]
    checks = {
        "controlled_mahalanobis_beats_euclidean": controlled_ood["mahalanobis_auroc"]
        > controlled_ood["euclidean_auroc"],
        "controlled_neighbor_sets_differ": results["controlled"]["neighbor_overlap"]["mean_jaccard_overlap"] < 0.95,
        "real_neighbor_sets_differ": results["real_latents"]["neighbor_overlap"]["mean_jaccard_overlap"] < 0.95,
        "real_metrics_recorded": real_ood["mahalanobis_auroc"] > 0.0,
    }

    output = Path("artifacts/anisotropy_benchmark.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"results": results, "checks": checks}, indent=2), encoding="utf-8")

    print(
        f"controlled   euclidean_auroc={controlled_ood['euclidean_auroc']:.3f} "
        f"mahalanobis_auroc={controlled_ood['mahalanobis_auroc']:.3f}"
    )
    print(
        f"real_latents euclidean_auroc={real_ood['euclidean_auroc']:.3f} "
        f"mahalanobis_auroc={real_ood['mahalanobis_auroc']:.3f}"
    )
    print(
        f"neighbor jaccard: controlled={results['controlled']['neighbor_overlap']['mean_jaccard_overlap']:.3f} "
        f"real={results['real_latents']['neighbor_overlap']['mean_jaccard_overlap']:.3f}"
    )
    print(f"acceptance: {checks}")
    print(f"artifact written to {output}")

    if not all(checks.values()):
        raise SystemExit("acceptance criteria not met")


if __name__ == "__main__":
    main()

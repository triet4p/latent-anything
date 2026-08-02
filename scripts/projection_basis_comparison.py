"""Projection-basis comparison: PCA, probe coefficients, and concept directions.

Builds three orthonormal subspaces from the *same* real ConvVAE digit latents
using three different derivation families, then compares them without treating
them as interchangeable:

- **PCA** (``origin="pca"``) — top principal component; captures maximum
  variance, which is *not* the same as semantics.
- **Probe coefficients** (``origin="probe"``) — supervised logistic-regression
  direction for the target concept (diagonal digits {1, 4, 7}).
- **Concept direction** (``origin="concept"``) — normalized mean-difference
  between concept-positive and concept-negative latents.

Each basis is recorded with its origin in an ``OrthonormalSubspace`` (the
framework never swaps one family for another silently), and the comparison
reports pairwise subspace alignment and the target-removal effect of each.

Acceptance criteria (see D2 evidence ledger):
- The bases are *not* interchangeable: at least one pairwise alignment is far
  below 1 (here all three pairs are reported).
- Variance is not semantics: removing the PCA direction suppresses the target
  concept far less than removing the supervised probe or concept direction.
- Both supervised directions (probe and concept) suppress the target below
  chance-ish, and they agree qualitatively despite not being identical.

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
from sklearn.decomposition import PCA  # pyright: ignore[reportMissingTypeStubs]

from latent_anything.adapters.conv_vae import ConvVAE
from latent_anything.geometry import subspace_alignment
from latent_anything.latent_value import LatentValue
from latent_anything.probes import LinearProbe
from latent_anything.projection import OrthonormalSubspace, SubspaceProjection

# Acceptance criteria (predeclared, D2).
MAX_ALIGNMENT_FOR_DIFFERENT_BASES = 0.9
VARIANCE_NON_SEMANTIC_MARGIN = 0.2
SUPERVISED_SUPPRESSION_MAX = 0.62

TARGET_CONCEPT_DIGITS = (1, 4, 7)
RANDOM_SEED = 42
N_SAMPLES = 600
LATENT_DIM = 6
N_EPOCHS = 6
N_PCA_COMPONENTS = 1


class _DigitsDataset(Protocol):
    images: np.ndarray
    target: np.ndarray


def _load_images() -> tuple[np.ndarray, np.ndarray]:
    digits = cast(_DigitsDataset, load_digits())
    images = (digits.images[:N_SAMPLES] / 16.0).astype(np.float64)[:, None, :, :]
    labels = digits.target[:N_SAMPLES].astype(int)
    return images, labels


def _fit_target_probe(features: np.ndarray, labels: np.ndarray) -> LinearProbe:
    probe = LinearProbe()
    probe.fit(features, labels)
    return probe


def _accuracy(probe: LinearProbe, features: np.ndarray, labels: np.ndarray) -> float:
    predictions = probe.predict(features)
    return float(np.mean(predictions == labels))


def _mean_diff_direction(positive: np.ndarray, negative: np.ndarray) -> np.ndarray:
    direction = positive.mean(axis=0) - negative.mean(axis=0)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-15:
        raise ValueError("concept direction is zero")
    return direction / norm


def main() -> None:
    """Run the basis comparison and write the artifact."""
    images, labels = _load_images()
    adapter = ConvVAE(latent_dim=LATENT_DIM, random_state=RANDOM_SEED, n_epochs=N_EPOCHS)
    adapter.fit(images)

    latent = adapter.encode_value(images).to_numpy()
    identity = adapter.latent_space.source_model

    split = int(latent.shape[0] * 0.7)
    train = latent[:split]
    test = latent[split:]

    target_labels = np.isin(labels, TARGET_CONCEPT_DIGITS).astype(int)
    target_probe = _fit_target_probe(train, target_labels[:split])
    target_before = _accuracy(target_probe, test, target_labels[split:])

    # ── Derive the three basis families ─────────────────────────────
    pca = PCA(n_components=N_PCA_COMPONENTS)
    pca.fit(latent)  # pyright: ignore[reportUnknownMemberType]
    pca_subspace = OrthonormalSubspace.from_pca(
        pca,  # pyright: ignore[reportUnknownArgumentType]
        source_representation_identity=identity,
        n_components=N_PCA_COMPONENTS,
        provenance={"family": "pca", "n_components": N_PCA_COMPONENTS},
    )

    probe_subspace = OrthonormalSubspace.from_probe_coefficients(
        np.asarray(target_probe.result.coefficients, dtype=np.float64),
        source_representation_identity=identity,
        provenance={"family": "probe", "concept": "diagonal digits {1,4,7}"},
    )

    positive = latent[target_labels == 1]
    negative = latent[target_labels == 0]
    concept_subspace = OrthonormalSubspace.from_concept_direction(
        _mean_diff_direction(positive, negative),
        source_representation_identity=identity,
        provenance={"family": "concept", "method": "mean_diff"},
    )

    subspaces: dict[str, OrthonormalSubspace] = {
        "pca": pca_subspace,
        "probe": probe_subspace,
        "concept": concept_subspace,
    }

    # ── Pairwise alignment (principal angles mean squared cosine) ──
    alignments: dict[str, float] = {}
    for first, second in (("pca", "probe"), ("pca", "concept"), ("probe", "concept")):
        alignments[f"{first}_vs_{second}"] = float(subspace_alignment(subspaces[first].basis, subspaces[second].basis))

    # ── Removal effect per basis ────────────────────────────────────
    test_value = LatentValue(test, adapter.latent_space)
    removal_effects: dict[str, dict[str, Any]] = {}
    for name, subspace in subspaces.items():
        removal = SubspaceProjection.from_subspace(subspace)
        removed = removal.remove(test_value)
        removal_effects[name] = {
            "origin": subspace.origin,
            "target_accuracy_after": _accuracy(target_probe, removed.to_numpy(), target_labels[split:]),
            "mean_coverage": float(np.mean(removal.coverage(test_value))),
        }

    checks = {
        "bases_are_not_interchangeable": min(alignments.values()) < MAX_ALIGNMENT_FOR_DIFFERENT_BASES,
        "variance_is_not_semantics": (
            removal_effects["pca"]["target_accuracy_after"] - removal_effects["probe"]["target_accuracy_after"]
            >= VARIANCE_NON_SEMANTIC_MARGIN
        ),
        "probe_suppresses_target": removal_effects["probe"]["target_accuracy_after"] <= SUPERVISED_SUPPRESSION_MAX,
        "concept_suppresses_target": removal_effects["concept"]["target_accuracy_after"] <= SUPERVISED_SUPPRESSION_MAX,
        "supervised_bases_agree_qualitatively": (
            abs(removal_effects["probe"]["target_accuracy_after"] - removal_effects["concept"]["target_accuracy_after"])
            <= 0.2
        ),
    }

    results: dict[str, Any] = {
        "dataset": "digits-conv-vae",
        "n_samples": N_SAMPLES,
        "latent_dim": LATENT_DIM,
        "identity": identity,
        "target_concept": f"diagonal digits {list(TARGET_CONCEPT_DIGITS)}",
        "target_accuracy_before_removal": target_before,
        "subspaces": {
            name: {"origin": subspace.origin, "n_basis": subspace.n_basis, "dim": subspace.dim}
            for name, subspace in subspaces.items()
        },
        "pairwise_alignment": alignments,
        "removal_effects": removal_effects,
        "acceptance": checks,
    }

    output = Path("artifacts/projection_basis_comparison.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"target before removal: {target_before:.3f}")
    for name, effect in removal_effects.items():
        print(
            f"{name:7s} ({effect['origin']:7s}) target_after={effect['target_accuracy_after']:.3f} "
            f"coverage={effect['mean_coverage']:.3f}"
        )
    print(f"pairwise alignment: {alignments}")
    print(f"acceptance: {checks}")
    print(f"artifact written to {output}")

    if not all(checks.values()):
        raise SystemExit("acceptance criteria not met")


if __name__ == "__main__":
    main()

"""Concept removal benchmark: target suppression, off-target preservation, decode degradation.

Removes a learned concept subspace from real ConvVAE digit latents and
measures three effects:

1. **Target suppression** — the removed latents should no longer linearly
   separate the target concept (the "diagonal digits" class {1, 4, 7}): the
   binary probe accuracy should collapse toward chance.
2. **Off-target preservation** — an orthogonal property (digit parity) that
   the removal was *not* aimed at should remain readable after removal.
3. **Decode degradation** — decoding the removed latents should still
   reconstruct the original images with bounded extra error versus the
   un-removed reconstruction.

A random-subspace control removal of the same dimensionality shows that the
suppression is specific to the learned concept direction: it must suppress the
target substantially *less* than the concept removal, even though any
dimension removal damages the probe signal.

Writes a reproducible JSON artifact to ``artifacts/`` and asserts the
acceptance criteria below (see D2 evidence ledger).
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
from latent_anything.latent_value import LatentValue
from latent_anything.probes import LinearProbe
from latent_anything.projection import OrthonormalSubspace, SubspaceProjection

# Acceptance criteria (predeclared, D2).
TARGET_BEFORE_MIN = 0.75
TARGET_AFTER_MAX = 0.62  # chance is 0.5; collapse toward chance
OFF_TARGET_DROP_MAX = 0.2
DECODE_RATIO_MAX = 3.0
CONCEPT_STRONGER_THAN_RANDOM_MIN = 0.1

TARGET_CONCEPT_DIGITS = (1, 4, 7)
RANDOM_SEED = 42
N_SAMPLES = 600
LATENT_DIM = 6
N_EPOCHS = 6


class _DigitsDataset(Protocol):
    images: np.ndarray
    target: np.ndarray


def _load_images() -> tuple[np.ndarray, np.ndarray]:
    digits = cast(_DigitsDataset, load_digits())
    images = (digits.images[:N_SAMPLES] / 16.0).astype(np.float64)[:, None, :, :]
    labels = digits.target[:N_SAMPLES].astype(int)
    return images, labels


def _fit_probe(features: np.ndarray, labels: np.ndarray) -> LinearProbe:
    probe = LinearProbe()
    probe.fit(features, labels)
    return probe


def _accuracy(probe: LinearProbe, features: np.ndarray, labels: np.ndarray) -> float:
    predictions = probe.predict(features)
    return float(np.mean(predictions == labels))


def _decode_mse(adapter: ConvVAE, latent: np.ndarray, images: np.ndarray) -> float:
    reconstruction = adapter.decode(np.asarray(latent, dtype=np.float64))
    return float(np.mean(np.square(reconstruction - images)))


def main() -> None:
    """Run the benchmark and write the artifact."""
    images, labels = _load_images()
    adapter = ConvVAE(latent_dim=LATENT_DIM, random_state=RANDOM_SEED, n_epochs=N_EPOCHS)
    adapter.fit(images)

    latent = adapter.encode_value(images).to_numpy()
    identity = adapter.latent_space.source_model

    # Deterministic 70/30 split by index.
    split = int(latent.shape[0] * 0.7)
    train_idx = np.arange(split)
    test_idx = np.arange(split, latent.shape[0])

    train_latent = latent[train_idx]
    test_latent = latent[test_idx]
    test_images = images[test_idx]

    # ── Target concept: diagonal digits {1, 4, 7} ──────────────────
    target_labels = np.isin(labels, TARGET_CONCEPT_DIGITS).astype(int)
    train_target = target_labels[train_idx]
    test_target = target_labels[test_idx]

    # ── Off-target property: digit parity (even/odd) ───────────────
    parity_labels = (labels % 2).astype(int)
    train_parity = parity_labels[train_idx]
    test_parity = parity_labels[test_idx]

    target_probe = _fit_probe(train_latent, train_target)
    parity_probe = _fit_probe(train_latent, train_parity)

    target_before = _accuracy(target_probe, test_latent, test_target)
    parity_before = _accuracy(parity_probe, test_latent, test_parity)

    # ── Concept removal via a supervised probe-coefficient basis ──
    test_value = LatentValue(test_latent, adapter.latent_space)
    concept_subspace = OrthonormalSubspace.from_probe_coefficients(
        np.asarray(target_probe.result.coefficients, dtype=np.float64),
        source_representation_identity=identity,
        provenance={"concept": "diagonal digits {1,4,7}", "basis_family": "probe"},
    )
    removal = SubspaceProjection.from_subspace(concept_subspace)
    removed = removal.remove(test_value)

    target_after = _accuracy(target_probe, removed.to_numpy(), test_target)
    parity_after = _accuracy(parity_probe, removed.to_numpy(), test_parity)
    mean_coverage = float(np.mean(removal.coverage(test_value)))

    # ── Decode degradation ─────────────────────────────────────────
    decode_mse_baseline = _decode_mse(adapter, test_latent, test_images)
    decode_mse_removed = _decode_mse(adapter, removed.to_numpy(), test_images)
    decode_ratio = decode_mse_removed / max(decode_mse_baseline, 1e-12)

    # ── Random-subspace control (same dimensionality) ─────────────
    rng = np.random.default_rng(RANDOM_SEED)
    random_basis = np.linalg.qr(rng.normal(size=(LATENT_DIM, LATENT_DIM)))[0][:, :1]
    random_subspace = OrthonormalSubspace.from_basis(
        random_basis,
        source_representation_identity=identity,
        origin="explicit",
        provenance={"control": "random-subspace"},
    )
    random_removal = SubspaceProjection.from_subspace(random_subspace)
    random_removed = random_removal.remove(test_value)
    random_target_after = _accuracy(target_probe, random_removed.to_numpy(), test_target)

    checks = {
        "target_suppressed_toward_chance": target_after <= TARGET_AFTER_MAX,
        "target_separable_before": target_before >= TARGET_BEFORE_MIN,
        "off_target_preserved": (parity_before - parity_after) <= OFF_TARGET_DROP_MAX,
        "decode_degradation_bounded": decode_ratio <= DECODE_RATIO_MAX,
        "concept_removal_stronger_than_random": (
            random_target_after - target_after >= CONCEPT_STRONGER_THAN_RANDOM_MIN
        ),
    }

    results: dict[str, Any] = {
        "dataset": "digits-conv-vae",
        "n_samples": N_SAMPLES,
        "latent_dim": LATENT_DIM,
        "n_epochs": N_EPOCHS,
        "identity": identity,
        "basis_origin": "probe",
        "target_concept": f"diagonal digits {list(TARGET_CONCEPT_DIGITS)}",
        "off_target_property": "digit parity",
        "target_accuracy": {"before": target_before, "after_removal": target_after},
        "parity_accuracy": {"before": parity_before, "after_removal": parity_after},
        "mean_subspace_coverage": mean_coverage,
        "decode_mse": {"baseline": decode_mse_baseline, "after_removal": decode_mse_removed, "ratio": decode_ratio},
        "random_control": {"target_accuracy_after": random_target_after},
        "acceptance": checks,
    }

    output = Path("artifacts/concept_removal_benchmark.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(
        f"target  accuracy before={target_before:.3f} after_removal={target_after:.3f} "
        f"(random_control={random_target_after:.3f})"
    )
    print(f"parity  accuracy before={parity_before:.3f} after_removal={parity_after:.3f}")
    print(f"decode  mse baseline={decode_mse_baseline:.4f} removed={decode_mse_removed:.4f} ratio={decode_ratio:.2f}")
    print(f"acceptance: {checks}")
    print(f"artifact written to {output}")

    if not all(checks.values()):
        raise SystemExit("acceptance criteria not met")


if __name__ == "__main__":
    main()

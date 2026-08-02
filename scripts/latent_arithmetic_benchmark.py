"""Latent arithmetic benchmark: coordinate-system binding and controlled steering.

Demonstrates the latent-arithmetic contract on real ConvVAE digit latents:

1. **Controlled semantic effect** — steering a test batch toward the "diagonal
   digits" concept direction with ``LatentValue.add_scaled`` (the steering form
   of latent arithmetic) monotonically increases the fraction classified as the
   concept. The arithmetic is meaningful because every operand shares the same
   coordinate-system identity.
2. **Cross-system rejection** — arithmetic between values that declare
   different coordinate-system identities raises ``ValueError`` instead of
   silently returning a plausible-looking array. The classic ``z_a - z_b + z_c``
   composition only exists inside one coordinate system.

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
from latent_anything.latent_space import LatentSpace
from latent_anything.latent_value import LatentValue
from latent_anything.probes import LinearProbe

# Acceptance criteria (predeclared, D2).
STEERING_STRENGTHS = (0.0, 0.5, 1.0, 1.5, 2.0)
MAX_STRENGTH_FRACTION_MIN = 0.65  # steering must materially raise the concept fraction

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


def _mean_diff_direction(positive: np.ndarray, negative: np.ndarray) -> np.ndarray:
    direction = positive.mean(axis=0) - negative.mean(axis=0)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-15:
        raise ValueError("concept direction is zero")
    return direction / norm


def _concept_fraction(probe: LinearProbe, features: np.ndarray) -> float:
    predictions = probe.predict(features)
    return float(np.mean(predictions == 1))


def main() -> None:
    """Run the arithmetic benchmark and write the artifact."""
    images, labels = _load_images()
    adapter = ConvVAE(latent_dim=LATENT_DIM, random_state=RANDOM_SEED, n_epochs=N_EPOCHS)
    adapter.fit(images)

    latent = adapter.encode_value(images).to_numpy()
    identity = adapter.latent_space.source_model
    space: LatentSpace = adapter.latent_space

    split = int(latent.shape[0] * 0.7)
    train = latent[:split]
    test = latent[split:]

    target_labels = np.isin(labels, TARGET_CONCEPT_DIGITS).astype(int)
    probe = LinearProbe()
    probe.fit(train, target_labels[:split])

    # ── Concept direction and steering arithmetic ────────────────────
    direction = _mean_diff_direction(
        latent[target_labels == 1],
        latent[target_labels == 0],
    )

    # ── Steering arithmetic within one coordinate system ────────────
    test_value = LatentValue(test, space)
    delta_value = LatentValue(np.tile(direction, (test.shape[0], 1)), space)
    fractions: dict[str, float] = {}
    for strength in STEERING_STRENGTHS:
        steered = test_value.add_scaled(delta_value, float(strength))
        fractions[f"strength_{strength:g}"] = _concept_fraction(probe, steered.to_numpy())

    fraction_series = [fractions[f"strength_{strength:g}"] for strength in STEERING_STRENGTHS]
    monotone = all(later >= earlier for earlier, later in zip(fraction_series, fraction_series[1:]))

    # ── Cross-system arithmetic is rejected, not silently computed ──
    other_identity = LatentValue(
        np.array([1.0] * LATENT_DIM),
        LatentSpace(dim=LATENT_DIM, source_model="different_model@v2"),
    )
    rejected = False
    try:
        _ = test_value + other_identity
    except ValueError:
        rejected = True

    # Analogy composition inside the coordinate system stays well-defined.
    composed = test_value.subtract(delta_value).add_scaled(delta_value, 2.0)
    composed_finite = bool(np.isfinite(composed.to_numpy()).all())

    checks = {
        "steering_raises_concept_fraction": fraction_series[-1] >= MAX_STRENGTH_FRACTION_MIN,
        "steering_is_monotone": monotone,
        "cross_identity_arithmetic_rejected": rejected,
        "same_identity_arithmetic_defined": composed_finite,
    }

    results: dict[str, Any] = {
        "dataset": "digits-conv-vae",
        "n_samples": N_SAMPLES,
        "latent_dim": LATENT_DIM,
        "identity": identity,
        "target_concept": f"diagonal digits {list(TARGET_CONCEPT_DIGITS)}",
        "steering_strengths": list(STEERING_STRENGTHS),
        "concept_fraction_by_strength": fractions,
        "cross_identity_arithmetic_rejected": rejected,
        "same_identity_analogy_arithmetic_finite": composed_finite,
        "acceptance": checks,
    }

    output = Path("artifacts/latent_arithmetic_benchmark.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"identity: {identity}")
    print("concept fraction by strength:", {k: round(v, 3) for k, v in fractions.items()})
    print(f"monotone: {monotone}  cross-identity rejected: {rejected}  composed finite: {composed_finite}")
    print(f"acceptance: {checks}")
    print(f"artifact written to {output}")

    if not all(checks.values()):
        raise SystemExit("acceptance criteria not met")


if __name__ == "__main__":
    main()

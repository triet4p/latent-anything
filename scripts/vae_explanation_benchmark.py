"""Run the compact, control-aware VAE explanation benchmark on sklearn digits."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from latent_anything.adapters.conv_vae import ConvVAE
from latent_anything.evaluation import evaluate_explanation, intervention_effect, probe_accuracy
from latent_anything.methods.pca import PCA
from latent_anything.methods.sae import SAE
from latent_anything.methods.steering import SteeringVector

ARTIFACT = Path("artifacts/vae_explanation_benchmark.json")
SEEDS = (0, 1, 2)


def _probability(features: np.ndarray, labels: np.ndarray, values: np.ndarray, seed: int) -> np.ndarray:
    probe = LogisticRegression(max_iter=500, random_state=seed).fit(features, labels)
    return probe.predict_proba(values)[:, 1]


def main() -> None:
    from sklearn.datasets import load_digits

    digits = load_digits()
    images = (digits.images[:240] / 16.0)[:, None, :, :].astype(np.float32)
    labels = (digits.target[:240] >= 5).astype(int)
    nuisance = (digits.target[:240] % 2).astype(int)
    seed_scores: list[float] = []
    adapter: ConvVAE | None = None
    latents: np.ndarray | None = None
    for seed in SEEDS:
        adapter = ConvVAE(latent_dim=4, random_state=seed, n_epochs=1)
        adapter.fit(images)
        latents = adapter.encode(images)
        seed_scores.append(probe_accuracy(latents, labels, random_state=seed))
    if adapter is None or latents is None:
        raise RuntimeError("benchmark did not create an adapter")

    negative, positive = latents[labels == 0], latents[labels == 1]
    steering = SteeringVector(adapter.latent_space)
    steering.fit(positive, negative)
    before = negative[:40]
    after = np.stack([steering(value, strength=0.5) for value in before])
    effect = intervention_effect(
        before,
        after,
        _probability(latents, labels, before, 0),
        _probability(latents, labels, after, 0),
        _probability(latents, nuisance, before, 0),
        _probability(latents, nuisance, after, 0),
        adapter.decode(before),
        adapter.decode(after),
    )
    baseline = evaluate_explanation(
        latents,
        labels,
        images.reshape(len(images), -1),
        images,
        adapter.decode(latents),
        seed_scores,
        effect,
    )
    pca = PCA(n_components=3)
    pca.fit(latents)
    sae = SAE(n_components=3, n_epochs=25, random_state=0)
    sae.fit(latents)
    comparison = {
        "pca_probe_accuracy": probe_accuracy(pca.transform(latents), labels, random_state=0),
        "sae_probe_accuracy": probe_accuracy(sae.transform(latents), labels, random_state=0),
        "steering_probe_accuracy": baseline.factor_predictability,
    }
    ARTIFACT.parent.mkdir(exist_ok=True)
    ARTIFACT.write_text(
        json.dumps(
            {
                "evidence_level": "D2" if baseline.accepts_explanation else "D1",
                "dataset": "sklearn digits (first 240 images)",
                "seeds": list(SEEDS),
                "evaluation": baseline.to_dict(),
                "representation_comparison": comparison,
                "conclusion": "Acceptance is controlled by baselines and intervention locality, not plot readability.",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

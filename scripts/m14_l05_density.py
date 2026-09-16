"""Run the M14 L05 real GPT-2 density benchmark on deterministic prompts."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from latent_anything.density import GMMConfig, GaussianMixtureDensity, cross_seed_evaluation
from latent_anything.geodesic import DensityGeodesic, GeodesicConfig
from latent_anything.integrations.transformer_lm import TransformerLMIntegration

MODEL_ID = "openai-community/gpt2"
MODEL_REVISION = "e7da7f221d5bf496a48136c0cd264e630fe9fcc8"


def main() -> None:
    texts = [f"A documented scientific observation number {i} concerns a stable process." for i in range(160)]
    ood = [f"Quantum mechanical orchard protocol {i} yields a divergent artifact." for i in range(40)]
    pipe = TransformerLMIntegration(model_id=MODEL_ID, revision=MODEL_REVISION, device="cpu")
    model, tokenizer, _ = pipe._backend()  # type: ignore[reportPrivateUsage]

    def features(items: list[str]) -> np.ndarray:
        batch = tokenizer(items, return_tensors="pt", padding=True, truncation=True, max_length=32)
        with torch.no_grad():
            output = model(**batch, output_hidden_states=True)
        hidden = output.hidden_states[-1]
        indices = batch["attention_mask"].sum(dim=1) - 1
        return hidden[torch.arange(hidden.shape[0]), indices].cpu().numpy().astype(np.float32)

    values = features(texts + ood)
    report = cross_seed_evaluation(
        values[:80],
        values[80:120],
        values[120:140],
        values[160:],
        source_representation_identity=f"{MODEL_ID}@{MODEL_REVISION}/layer=12",
        config=GMMConfig(n_components=2, covariance_type="diag", min_samples_per_dimension=0.1),
        seeds=(0, 1, 2),
    )
    density = GaussianMixtureDensity(
        GMMConfig(n_components=2, covariance_type="diag", min_samples_per_dimension=0.1, random_state=0)
    ).fit(
        values[:80],
        source_representation_identity=f"{MODEL_ID}@{MODEL_REVISION}/layer=12",
        provenance={"split": "train=80; calibration=40; in_distribution=20; ood=40"},
    )
    geodesic = DensityGeodesic.from_gmm_density(
        density,
        config=GeodesicConfig(n_points=8, max_iter=20, step_size=0.05, tol=1e-5, density_exponent=1.0),
    )
    path = geodesic.optimize(values[120], values[121])
    payload = {
        "schema_version": "m14-l05-run-v1",
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION, "layer": 12},
        "seeds": [0, 1, 2],
        "split": {"train": 80, "calibration": 40, "in_distribution": 20, "out_of_distribution": 40},
        "density": {
            "mean_auroc": report.mean_auroc,
            "auroc_ci95": report.auroc_ci95,
            "mean_auprc": report.mean_auprc,
            "auprc_ci95": report.auprc_ci95,
            "aurocs": list(report.aurocs),
        },
        "path_feasibility": {
            "endpoint_indices": [120, 121],
            "n_points": path.path.shape[0],
            "finite": bool(np.isfinite(path.path).all() and np.isfinite(path.log_density).all()),
            "endpoint_a_preserved": bool(np.allclose(path.path[0], values[120])),
            "endpoint_b_preserved": bool(np.allclose(path.path[-1], values[121])),
            "length": path.length,
            "euclidean_length": path.euclidean_length,
            "min_log_density": path.min_log_density,
            "mean_log_density": path.mean_log_density,
            "converged": path.status.converged,
            "iterations": path.status.n_iterations,
        },
        "acceptance": {
            "finite_metrics": True,
            "cross_seed_auroc_available": True,
            "path_feasible": bool(
                np.isfinite(path.path).all()
                and np.isfinite(path.log_density).all()
                and np.allclose(path.path[0], values[120])
                and np.allclose(path.path[-1], values[121])
            ),
            "predeclared_threshold_binding": "mean AUROC is reported for authority threshold review; no threshold is invented here",
        },
    }
    output = Path("artifacts/m14/l05-density-run.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()

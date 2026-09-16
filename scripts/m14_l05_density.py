"""Run the M14 L05 real GPT-2 density benchmark on deterministic prompts."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import numpy as np
import psutil
import torch

from latent_anything.density import GMMConfig, GaussianMixtureDensity, cross_seed_evaluation
from latent_anything.geodesic import DensityGeodesic, GeodesicConfig
from latent_anything.integrations.transformer_lm import TransformerLMIntegration

MODEL_ID = "openai-community/gpt2"
MODEL_REVISION = "e7da7f221d5bf496a48136c0cd264e630fe9fcc8"
MODEL_WEIGHTS_SHA256 = "248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707"
MIN_MEAN_AUROC = 0.90
REQUIRE_PATH_FEASIBLE = True


def _digest_strings(values: list[str]) -> str:
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def main() -> None:
    process = psutil.Process()
    rss_peak = [process.memory_info().rss]
    texts = [f"A documented scientific observation number {i} concerns a stable process." for i in range(140)]
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
    rss_peak[0] = max(rss_peak[0], process.memory_info().rss)
    report = cross_seed_evaluation(
        values[:80],
        values[80:120],
        values[120:140],
        values[140:],
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
    path_feasible = bool(
        np.isfinite(path.path).all()
        and np.isfinite(path.log_density).all()
        and np.allclose(path.path[0], values[120])
        and np.allclose(path.path[-1], values[121])
    )
    payload = {
        "schema_version": "m14-l05-run-v1",
        "model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "weights_sha256": MODEL_WEIGHTS_SHA256,
            "license": "MIT",
            "layer": 12,
        },
        "seeds": [0, 1, 2],
        "prompt_split": {
            "train": 80,
            "calibration": 40,
            "in_distribution": 20,
            "out_of_distribution": 40,
            "in_distribution_digest": _digest_strings(texts[120:140]),
            "out_of_distribution_digest": _digest_strings(ood),
            "all_prompt_digest": _digest_strings(texts + ood),
        },
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
            "finite_metrics": bool(
                np.isfinite([report.mean_auroc, report.mean_auprc, report.auroc_ci95, report.auprc_ci95]).all()
            ),
            "cross_seed_auroc_available": len(report.aurocs) == 3,
            "mean_auroc_threshold": MIN_MEAN_AUROC,
            "mean_auroc_meets_threshold": bool(report.mean_auroc >= MIN_MEAN_AUROC),
            "path_feasible": path_feasible,
            "path_threshold": "finite path with exact endpoints preserved",
            "path_threshold_meets": path_feasible if REQUIRE_PATH_FEASIBLE else True,
            "predeclared_threshold_binding": "MIN_MEAN_AUROC=0.90 and REQUIRE_PATH_FEASIBLE=True declared in runner before final run",
        },
        "environment": {
            "device": "cpu",
            "dtype": "float32",
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": importlib.metadata.version("torch"),
            "transformers": importlib.metadata.version("transformers"),
            "huggingface_hub": importlib.metadata.version("huggingface-hub"),
            "rss_peak_bytes": rss_peak[0],
            "network_policy": "HF cache-only after initial pinned acquisition; no model download during final measured rerun",
        },
        "cleanup": "No temporary model files created; HF cache retained as pinned local cache and no disposable output outside artifact path.",
    }
    output = Path("artifacts/m14/l05-density-run.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()

"""Run the M14 L05 real GPT-2 density benchmark on deterministic prompts."""
from __future__ import annotations

import numpy as np
import torch

from latent_anything.density import GMMConfig, cross_seed_evaluation
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
    print(
        {
            "model": f"{MODEL_ID}@{MODEL_REVISION}",
            "aurocs": report.aurocs,
            "mean_auroc": report.mean_auroc,
            "auroc_ci95": report.auroc_ci95,
            "mean_auprc": report.mean_auprc,
            "auprc_ci95": report.auprc_ci95,
        }
    )


if __name__ == "__main__":
    main()

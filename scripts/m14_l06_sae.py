"""Persist the M14 L06 real GPT-2 SAE/FeatureAtlas evidence."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path

import numpy as np
import psutil

from latent_anything.integrations.transformer_lm import TransformerGenerationRequest, TransformerLMIntegration
from latent_anything.sae_evaluation import (
    SAEConfig,
    SAEFeatureEvaluation,
    build_feature_atlas,
    cross_seed_sae_stability,
    rank_feature_examples,
    save_feature_atlas,
)

MODEL_ID = "openai-community/gpt2"
MODEL_REVISION = "e7da7f221d5bf496a48136c0cd264e630fe9fcc8"
MODEL_WEIGHTS_SHA256 = "248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707"
PROMPTS = (
    "The cat sat on the mat", "The dog barked at the stranger", "A cup of coffee sat on the desk",
    "The train arrived at the station", "Children played in the park", "The chef prepared a delicious meal",
    "Scientists studied the distant galaxy", "The musician composed a new symphony",
    "The engineer reviewed the careful experiment notes", "The painter mixed a bright blue color",
    "The researcher measured the changing temperature", "The gardener watered the flowers before sunset",
    "The airplane crossed the clouds above the ocean", "The programmer fixed a subtle parser bug",
    "The doctor explained the treatment to the patient", "The artist displayed a sculpture in the gallery",
    "The student solved a difficult equation during class", "The photographer captured a landscape during sunrise",
    "The librarian organized the books by subject", "The scientist recorded observations in a notebook",
    "The musician practiced scales before the concert", "The team discussed the schedule for the project",
    "The farmer harvested vegetables from the field", "The architect designed a bridge across the river",
    "The child opened a colorful present on her birthday", "The teacher demonstrated the lesson with an example",
    "The pilot checked the instruments before takeoff", "The baker placed fresh bread into the oven",
    "The journalist interviewed the author about the novel", "The mechanic repaired the engine in the workshop",
    "The family visited a museum during the holiday", "The swimmer trained every morning at the pool",
)


def main() -> None:
    process = psutil.Process()
    rss_peak = [process.memory_info().rss]
    pipe = TransformerLMIntegration(model_id=MODEL_ID, revision=MODEL_REVISION, device="cpu")
    request = TransformerGenerationRequest(
        prompt=PROMPTS, max_length=24, capture_hidden_states=True, capture_layers=(6,), top_k_logit_lens=0
    )
    result = pipe.generate(request)
    hidden = result.hidden_states[0].values
    real = result.attention_mask == 1
    activations = hidden[real]
    input_ids = result.input_ids
    token_labels = [str(token) for token in pipe.decode_tokens(input_ids[real])]
    rss_peak[0] = max(rss_peak[0], process.memory_info().rss)
    mean = activations.mean(axis=0)
    std = activations.std(axis=0) + 1e-8
    standardised = (activations - mean) / std
    config = SAEConfig(n_components=64, l1_coef=0.005, learning_rate=1e-2, n_epochs=1500)
    evaluation = SAEFeatureEvaluation(config).fit(
        standardised,
        source_representation_identity=f"{pipe.provenance}_layer_6",
        provenance={"dataset": "prompt-batch-v1", "n_tokens": int(activations.shape[0])},
    )
    rss_peak[0] = max(rss_peak[0], process.memory_info().rss)
    stability = cross_seed_sae_stability(standardised, config=config, seeds=(0, 1, 2), source_representation_identity=f"{pipe.provenance}_layer_6")
    permutation = np.random.default_rng(config.random_state).permutation(len(token_labels))
    validation_labels = [token_labels[index] for index in permutation[evaluation.n_train:]]
    most_active = int(np.argmax(evaluation.activation_frequencies))
    ranking = rank_feature_examples(evaluation, most_active, k=5, example_labels=validation_labels)
    atlas = build_feature_atlas(evaluation, k_examples=3, k_decoder_dims=5, example_labels=validation_labels)
    atlas_path = Path("artifacts/m14/l06-feature-atlas.json")
    atlas_path.parent.mkdir(parents=True, exist_ok=True)
    save_feature_atlas(atlas, atlas_path)
    atlas_hash = hashlib.sha256(atlas_path.read_bytes()).hexdigest()
    payload = {
        "schema_version": "m14-l06-run-v1",
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION, "weights_sha256": MODEL_WEIGHTS_SHA256, "license": "MIT", "layer": 6},
        "seeds": [0, 1, 2],
        "corpus": {"revision": "prompt-batch-v1", "license": "original deterministic prompts", "prompt_count": len(PROMPTS), "prompt_digest": hashlib.sha256("\n".join(PROMPTS).encode()).hexdigest()},
        "config": config.model_dump(mode="json"),
        "metrics": {
            "n_tokens": int(activations.shape[0]), "reconstruction_mse": evaluation.reconstruction_mse,
            "n_dead_features": evaluation.n_dead_features, "dead_fraction": evaluation.dead_fraction,
            "mean_l0": evaluation.mean_l0, "mean_l1": evaluation.mean_l1,
            "stability_mean_matched_cosine": stability.mean_matched_cosine,
            "stability_min_matched_cosine": stability.min_matched_cosine,
            "stability_alignment_quality": stability.alignment_quality,
            "most_active_feature": most_active, "top_labels": ranking.top_labels, "bottom_labels": ranking.bottom_labels,
        },
        "atlas": {"path": str(atlas_path).replace("\\", "/"), "sha256": atlas_hash, "entries": len(atlas.entries)},
        "environment": {"device": "cpu", "python": platform.python_version(), "platform": platform.platform(), "numpy": np.__version__, "torch": importlib.metadata.version("torch"), "transformers": importlib.metadata.version("transformers"), "huggingface_hub": importlib.metadata.version("huggingface-hub"), "rss_peak_bytes": rss_peak[0], "network_policy": "HF cache-only after initial pinned acquisition; no model download during final measured rerun"},
        "acceptance": {
            "reconstruction_finite": bool(np.isfinite(evaluation.reconstruction_mse)),
            "dead_features_bounded": evaluation.n_dead_features < config.n_components,
            "cross_seed_stability": stability.min_matched_cosine > 0.85 and stability.alignment_quality > 0.7,
            "cross_seed_min_matched_cosine_threshold": 0.85,
            "cross_seed_alignment_quality_threshold": 0.7,
            "atlas_hash_recorded": bool(atlas_hash),
            "real_hidden_states": True,
        },
        "cleanup": "No disposable corpus/checkpoint files; pinned HF cache retained and atlas is the immutable output artifact.",
    }
    output = Path("artifacts/m14/l06-sae-run.json")
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()

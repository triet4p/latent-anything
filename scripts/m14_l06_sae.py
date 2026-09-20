"""Persist the M14 L06 real GPT-2 SAE/FeatureAtlas evidence."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
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
RUN_COMMAND = (
    "env LATENT_ANYTHING_RUN_NETWORK=1 uv run --locked --extra transformers "
    "--with huggingface-hub==0.35.3 python scripts/m14_l06_sae.py"
)
MODEL_WEIGHTS_SHA256 = "248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707"
_PROMPT_SUBJECTS = (
    "The scientist",
    "The engineer",
    "The teacher",
    "The artist",
    "The musician",
    "The doctor",
    "The farmer",
    "The pilot",
)
_PROMPT_VERBS = ("studied", "designed", "explained", "observed", "recorded", "tested", "reviewed", "measured")
_PROMPT_OBJECTS = (
    "the changing climate",
    "a careful experiment",
    "the bright landscape",
    "a complex machine",
    "the historical archive",
    "the distant planet",
    "a new hypothesis",
    "the difficult problem",
)
_PROMPT_CONTEXTS = ("during the morning", "for the annual report")
PROMPTS = tuple(
    f"{subject} {verb} {obj} {context}."
    for subject in _PROMPT_SUBJECTS
    for verb in _PROMPT_VERBS
    for obj in _PROMPT_OBJECTS
    for context in _PROMPT_CONTEXTS
)
CORPUS_REVISION = "prompt-grid-v2-1024"


def main() -> None:
    source_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    process = psutil.Process()
    rss_peak = [process.memory_info().rss]
    pipe = TransformerLMIntegration(model_id=MODEL_ID, revision=MODEL_REVISION, device="cpu")
    activations_parts: list[np.ndarray] = []
    token_labels: list[str] = []
    batch_size = 128
    for start in range(0, len(PROMPTS), batch_size):
        request = TransformerGenerationRequest(
            prompt=PROMPTS[start : start + batch_size],
            max_length=24,
            capture_hidden_states=True,
            capture_layers=(6,),
            top_k_logit_lens=0,
        )
        result = pipe.generate(request)
        hidden = result.hidden_states[0].values
        real = result.attention_mask == 1
        activations_parts.append(hidden[real])
        token_labels.extend(str(token) for token in pipe.decode_tokens(result.input_ids[real]))
        rss_peak[0] = max(rss_peak[0], process.memory_info().rss)
    activations = np.concatenate(activations_parts, axis=0)
    mean = activations.mean(axis=0)
    std = activations.std(axis=0) + 1e-8
    standardised = (activations - mean) / std
    config = SAEConfig(n_components=4, l1_coef=0.001, learning_rate=1e-2, n_epochs=1000)
    evaluation = SAEFeatureEvaluation(config).fit(
        standardised,
        source_representation_identity=f"{pipe.provenance}_layer_6",
        provenance={"dataset": CORPUS_REVISION, "n_tokens": int(activations.shape[0])},
    )
    rss_peak[0] = max(rss_peak[0], process.memory_info().rss)
    stability = cross_seed_sae_stability(
        standardised,
        config=config,
        seeds=(0, 1, 2),
        source_representation_identity=f"{pipe.provenance}_layer_6",
    )
    permutation = np.random.default_rng(config.random_state).permutation(len(token_labels))
    validation_labels = [token_labels[index] for index in permutation[evaluation.n_train :]]
    most_active = int(np.argmax(evaluation.activation_frequencies))
    ranking = rank_feature_examples(evaluation, most_active, k=5, example_labels=validation_labels)
    atlas = build_feature_atlas(evaluation, k_examples=3, k_decoder_dims=5, example_labels=validation_labels)
    atlas_path = Path("artifacts/m14/l06-feature-atlas.json")
    atlas_path.parent.mkdir(parents=True, exist_ok=True)
    save_feature_atlas(atlas, atlas_path)
    atlas_hash = hashlib.sha256(atlas_path.read_bytes()).hexdigest()
    payload = {
        "source_sha": source_sha,
        "command": RUN_COMMAND,
        "schema_version": "m14-l06-run-v1",
        "model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "weights_sha256": MODEL_WEIGHTS_SHA256,
            "license": "MIT",
            "layer": 6,
        },
        "seeds": [0, 1, 2],
        "batch_size": batch_size,
        "corpus": {
            "revision": CORPUS_REVISION,
            "license": "original deterministic prompts",
            "prompt_count": len(PROMPTS),
            "prompt_digest": hashlib.sha256("\n".join(PROMPTS).encode()).hexdigest(),
        },
        "config": config.model_dump(mode="json"),
        "metrics": {
            "n_tokens": int(activations.shape[0]),
            "reconstruction_mse": evaluation.reconstruction_mse,
            "n_dead_features": evaluation.n_dead_features,
            "dead_fraction": evaluation.dead_fraction,
            "mean_l0": evaluation.mean_l0,
            "mean_l1": evaluation.mean_l1,
            "stability_mean_matched_cosine": stability.mean_matched_cosine,
            "stability_min_matched_cosine": stability.min_matched_cosine,
            "stability_alignment_quality": stability.alignment_quality,
            "most_active_feature": most_active,
            "top_labels": ranking.top_labels,
            "bottom_labels": ranking.bottom_labels,
        },
        "diagnosis": {
            "prior_failed_attempt": {
                "corpus_revision": "prompt-batch-v1",
                "n_tokens": 234,
                "n_components": 64,
                "stability_mean_matched_cosine": 0.0,
                "stability_min_matched_cosine": 0.0,
                "stability_alignment_quality": 0.0,
            },
            "source_fix": (
                "cross_seed_sae_stability now holds the train/validation split fixed so "
                "seed comparisons isolate optimization initialization"
            ),
            "finding": (
                "The 234-token corpus and split confounding contributed to the zero score. "
                "The expanded 1024-prompt run removes the confound, but independent SAE "
                "seeds remain below the predeclared 0.85 cosine and 0.7 alignment thresholds."
            ),
        },
        "atlas": {"path": str(atlas_path).replace("\\", "/"), "sha256": atlas_hash, "entries": len(atlas.entries)},
        "environment": {
            "device": "cpu",
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": importlib.metadata.version("torch"),
            "transformers": importlib.metadata.version("transformers"),
            "huggingface_hub": importlib.metadata.version("huggingface-hub"),
            "rss_peak_bytes": rss_peak[0],
            "network_policy": (
                "HF cache-only after initial pinned acquisition; no model download during final measured rerun"
            ),
        },
        "acceptance": {
            "reconstruction_finite": bool(np.isfinite(evaluation.reconstruction_mse)),
            "dead_features_bounded": evaluation.n_dead_features < config.n_components,
            "cross_seed_stability": stability.min_matched_cosine > 0.85 and stability.alignment_quality > 0.7,
            "cross_seed_min_matched_cosine_threshold": 0.85,
            "cross_seed_alignment_quality_threshold": 0.7,
            "atlas_hash_recorded": bool(atlas_hash),
            "real_hidden_states": True,
        },
        "cleanup": (
            "No disposable corpus/checkpoint files; pinned HF cache retained and atlas "
            "is the immutable output artifact."
        ),
    }
    output = Path("artifacts/m14/l06-sae-run.json")
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()

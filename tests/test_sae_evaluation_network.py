"""Marked real-checkpoint SAE feature evaluation evidence for the pinned GPT-2 seam.

Runs the full evaluation pipeline on layer-6 activations of the pinned
revision GPT-2 checkpoint: fit a sparse autoencoder, measure reconstruction/
sparsity/stability, rank top-activating and bottom-activating token examples,
cross-check one feature with concept sensitivity and causal steering, and
write a portable feature-atlas JSON artifact.
"""

from __future__ import annotations

import os
import pathlib

import numpy as np
import pytest

from latent_anything.integrations.transformer_lm import (
    TransformerGenerationRequest,
    TransformerLMIntegration,
)
from latent_anything.sae_evaluation import (
    SAEConfig,
    SAEFeatureEvaluation,
    build_feature_atlas,
    cross_check_feature,
    load_feature_atlas,
    rank_feature_examples,
    save_feature_atlas,
)
from latent_anything.tcav import TransformerLogitTarget


def _layer_6_activation_batch(
    pipe: TransformerLMIntegration,
    prompts: tuple[str, ...],
    *,
    layer: int = 6,
    max_length: int = 24,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return (activations, input_ids, token_labels) from a real forward pass."""
    request = TransformerGenerationRequest(
        prompt=prompts,
        max_length=max_length,
        capture_hidden_states=True,
        capture_layers=(layer,),
        top_k_logit_lens=0,
    )
    result = pipe.generate(request)
    hidden = result.hidden_states[0].values  # (batch, seq, hidden)
    input_ids = result.input_ids  # (batch, seq)
    mask = result.attention_mask  # (batch, seq)
    real = mask == 1
    activations = hidden[real]
    real_ids = input_ids[real]
    labels = [str(token) for token in pipe.decode_tokens(real_ids)]
    return activations, input_ids, labels


@pytest.mark.network
@pytest.mark.large_download
@pytest.mark.skipif(
    os.environ.get("LATENT_ANYTHING_RUN_NETWORK") != "1",
    reason="set LATENT_ANYTHING_RUN_NETWORK=1 to acquire or validate the pinned checkpoint",
)
def test_pinned_transformer_sae_feature_atlas_and_cross_check(tmp_path: pathlib.Path) -> None:
    pipe = TransformerLMIntegration(device="cpu")
    prompts = (
        "The cat sat on the mat",
        "The dog barked at the stranger",
        "A cup of coffee sat on the desk",
        "The train arrived at the station",
        "Children played in the park",
        "The chef prepared a delicious meal",
        "Scientists studied the distant galaxy",
        "The musician composed a new symphony",
    )
    activations, input_ids, token_labels = _layer_6_activation_batch(pipe, prompts, layer=6, max_length=24)
    assert activations.shape[0] > 0, "no real-token activations captured"
    assert activations.shape[1] == pipe.hidden_dim == 768

    # Standardise the residual-stream activations before fitting.
    mean = activations.mean(axis=0)
    std = activations.std(axis=0) + 1e-8
    standardised = (activations - mean) / std

    config = SAEConfig(n_components=32, l1_coef=0.3, learning_rate=1e-2, n_epochs=400)
    evaluation = SAEFeatureEvaluation(config).fit(
        standardised,
        source_representation_identity=f"{pipe.provenance}_layer_6",
        provenance={"dataset": "prompt-batch-v1", "n_tokens": int(activations.shape[0])},
    )

    # Regression sanity: reconstruction better than the trivial mean baseline,
    # not every feature dead, and L0 strictly below the component count.
    assert np.isfinite(evaluation.reconstruction_mse)
    assert evaluation.reconstruction_mse < 1.0, f"reconstruction_mse too high: {evaluation.reconstruction_mse}"
    assert evaluation.n_dead_features < evaluation.config.n_components
    assert evaluation.mean_l0 < evaluation.config.n_components
    assert float(evaluation.activation_frequencies.max()) > 0.1
    assert evaluation.provenance["dataset"] == "prompt-batch-v1"

    # Rank real-token examples and counterexamples for the most-active feature.
    most_active = int(np.argmax(evaluation.activation_frequencies))
    ranking = rank_feature_examples(evaluation, most_active, k=5, example_labels=token_labels)
    assert len(ranking.top_labels) == 5 and len(ranking.bottom_labels) == 5
    assert max(ranking.top_activations) > min(ranking.bottom_activations)
    top_text = " ".join(label for label in ranking.top_labels if label is not None)
    assert top_text.strip(), "top-activating examples decoded to empty text"

    # Cross-check one feature with concept sensitivity and causal steering.
    target = TransformerLogitTarget(token_id=int(input_ids[0, -1]), position=-1)
    check = cross_check_feature(
        evaluation,
        most_active,
        model=pipe._backend()[0],  # type: ignore[reportPrivateUsage]
        input_ids=input_ids[:4],
        attention_mask=np.ones((4, input_ids.shape[1]), dtype=np.int64),
        target=target,
        layer=6,
        intervention_strength=1.0,
    )
    assert check.has_causal_evidence
    assert check.n_examples_checked == 4
    assert check.concept_sensitivity is not None and np.isfinite(check.concept_sensitivity)
    assert check.intervention_effect is not None and np.isfinite(check.intervention_effect)
    assert check.intervention_agreement is not None

    # Build and persist a portable feature-atlas artifact.
    atlas = build_feature_atlas(evaluation, k_examples=3, k_decoder_dims=5, example_labels=token_labels)
    path = tmp_path / "sprint46_gpt2_layer6_feature_atlas.json"
    save_feature_atlas(atlas, path)
    loaded = load_feature_atlas(path)
    assert loaded.entry(most_active).top_examples == atlas.entry(most_active).top_examples
    assert loaded.provenance["dataset"] == "prompt-batch-v1"

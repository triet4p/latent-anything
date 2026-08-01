"""Marked real-checkpoint attribution evidence for the pinned GPT-2 seam."""

from __future__ import annotations

import os

import numpy as np
import pytest

from latent_anything.integrated_gradients import (
    IntegratedGradients,
    IntegratedGradientsConfig,
    evaluate_sensitivity,
)
from latent_anything.integrations.transformer_lm import TransformerLMIntegration
from latent_anything.tcav import TransformerLogitTarget


@pytest.mark.network
@pytest.mark.large_download
@pytest.mark.skipif(
    os.environ.get("LATENT_ANYTHING_RUN_NETWORK") != "1",
    reason="set LATENT_ANYTHING_RUN_NETWORK=1 to acquire or validate the pinned checkpoint",
)
def test_pinned_transformer_integrated_gradients_examples() -> None:
    """Record positive, negative, and step-unstable target examples."""
    pipe = TransformerLMIntegration(device="cpu")
    encoded = pipe.tokenize(("The cat sat", "The cat flew"), max_length=8)
    input_ids = encoded["input_ids"].cpu().numpy()
    attention_mask = encoded["attention_mask"].cpu().numpy()
    model, _, _ = pipe._backend()  # type: ignore[reportPrivateUsage]

    positive = IntegratedGradients(
        IntegratedGradientsConfig(target_layer=6, activation_position=-1, n_steps=64)
    ).compute(
        model,
        input_ids[:1],
        attention_mask[:1],
        TransformerLogitTarget(token_id=int(input_ids[0, -1]), position=-1),
        source_model_version=pipe.provenance,
    )
    negative = IntegratedGradients(
        IntegratedGradientsConfig(target_layer=6, activation_position=-1, n_steps=64)
    ).compute(
        model,
        input_ids[:1],
        attention_mask[:1],
        TransformerLogitTarget(token_id=int(input_ids[0, -2]), position=-1),
        source_model_version=pipe.provenance,
    )
    unstable_coarse = IntegratedGradients(
        IntegratedGradientsConfig(target_layer=6, activation_position=-1, n_steps=8)
    ).compute(
        model,
        input_ids[1:2],
        attention_mask[1:2],
        TransformerLogitTarget(token_id=int(input_ids[1, -1]), position=-1),
        source_model_version=pipe.provenance,
    )
    unstable_fine = IntegratedGradients(
        IntegratedGradientsConfig(target_layer=6, activation_position=-1, n_steps=64)
    ).compute(
        model,
        input_ids[1:2],
        attention_mask[1:2],
        TransformerLogitTarget(token_id=int(input_ids[1, -1]), position=-1),
        source_model_version=pipe.provenance,
    )
    report = evaluate_sensitivity((unstable_coarse, unstable_fine))

    for result in (positive, negative, unstable_coarse, unstable_fine):
        assert result.provenance["source_model_version"] == pipe.provenance
        assert np.all(np.isfinite(result.attributions))
        assert np.isfinite(result.completeness_error)
    assert report.step_counts == (8, 64)

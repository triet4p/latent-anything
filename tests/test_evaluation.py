"""Regression tests for explanation-validity controls."""

from __future__ import annotations

import numpy as np
import pytest

from latent_anything.evaluation import evaluate_explanation, intervention_effect, probe_accuracy


def _fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    labels = np.repeat([0, 1], 40)
    latents = np.column_stack((labels * 3 + rng.normal(0, 0.1, 80), rng.normal(0, 1, 80)))
    inputs = rng.normal(0, 1, (80, 4))
    return latents, labels, inputs


def test_evaluation_accepts_local_direction_over_negative_controls() -> None:
    latents, labels, inputs = _fixture()
    before = latents[:10]
    after = before + np.array([1.0, 0.0])
    effect = intervention_effect(
        before, after, np.zeros(10), np.ones(10), np.zeros(10), np.full(10, 0.02), before, after
    )
    result = evaluate_explanation(
        latents, labels, inputs, inputs, inputs + 0.1, [0.96, 0.95, 0.97], effect, random_state=2
    )
    assert result.accepts_explanation
    assert result.factor_predictability > result.shuffled_label_predictability
    assert result.to_dict()["intervention_effect"]["target_factor_change"] == 1.0


def test_evaluation_rejects_nonlocal_effect_and_invalid_probe_shape() -> None:
    latents, labels, inputs = _fixture()
    effect = intervention_effect(
        latents[:2], latents[:2], np.zeros(2), np.ones(2), np.zeros(2), np.ones(2), inputs[:2], inputs[:2]
    )
    result = evaluate_explanation(latents, labels, inputs, inputs, inputs, [0.5], effect, random_state=2)
    assert not result.accepts_explanation
    with pytest.raises(ValueError, match="matching 1D"):
        probe_accuracy(latents, labels[:, None], random_state=0)

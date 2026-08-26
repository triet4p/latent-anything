"""Sprint 72 tests for the compact tokenized world model."""

from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch

from latent_anything import (
    LatentTransition,
    RolloutPipeline,
    TokenizedWorldModel,
)
from latent_anything.adapters import VQVAE, DecodableAdapter, ModelAdapter
from latent_anything.registry import GLOBAL_REGISTRY, KIND_RUNTIME
from latent_anything.tokenized_world_model import (
    TokenizedWorldModelConfig,
    TokenPrediction,
    TokenPredictionMetrics,
    TokenRolloutMetrics,
)


def _mutate_codebook_for_checkpoint_test(tokenizer: VQVAE) -> None:
    """Mutate the implementation checkpoint through a narrow typed test seam."""

    codebook = getattr(tokenizer, "_codebook", None)
    if not isinstance(codebook, torch.nn.Embedding):
        raise AssertionError("VQVAE test fixture does not expose an embedding codebook")
    with torch.no_grad():
        codebook.weight[0, 0].add_(1.0)


def _tokens(episodes: int = 6, horizon: int = 4) -> tuple[np.ndarray, np.ndarray]:
    values = np.zeros((episodes, horizon + 1, 16), dtype=np.int64)
    for episode in range(episodes):
        values[episode, 0] = (np.arange(16) + episode) % 4
        for step in range(horizon):
            values[episode, step + 1] = (values[episode, step] + 1) % 4
    actions = np.zeros((episodes, horizon, 1), dtype=np.float64)
    actions[:, :, 0] = np.arange(horizon, dtype=np.float64)
    return values, actions


def _model(*, epochs: int = 3) -> TokenizedWorldModel:
    tokenizer = VQVAE(codebook_size=4, embedding_dim=3, n_epochs=1, random_state=72)
    model = TokenizedWorldModel(tokenizer, action_dim=1, hidden_dim=8, epochs=epochs, seed=72)
    token_sequences, actions = _tokens()
    model.fit_tokens(token_sequences, actions)
    return model


def test_tokenized_world_model_composes_adapter_and_transition_seams() -> None:
    model = _model()

    assert isinstance(model, ModelAdapter)
    assert isinstance(model, DecodableAdapter)
    assert isinstance(model, LatentTransition)
    assert GLOBAL_REGISTRY.lookup("tokenized_world_model").kind == KIND_RUNTIME
    assert GLOBAL_REGISTRY.lookup("tokenized_world_model").factory is TokenizedWorldModel
    assert model.latent_space.geometry == "discrete_code"
    assert model.latent_space.metadata["codebook_version"] == model.codebook_version
    assert model.is_fitted


def test_tokenized_world_model_public_api_and_result_schema_snapshot() -> None:
    assert tuple(inspect.signature(TokenizedWorldModel.fit).parameters) == (
        "self",
        "observations",
        "actions",
        "sequence_mask",
        "codebook_version",
    )
    assert tuple(inspect.signature(TokenizedWorldModel.predict_next).parameters) == (
        "self",
        "tokens",
        "action",
        "sampling",
        "temperature",
        "top_k",
        "seed",
    )
    assert tuple(TokenizedWorldModelConfig.model_fields) == (
        "action_dim",
        "hidden_dim",
        "epochs",
        "learning_rate",
        "seed",
        "model_revision",
        "codebook_version",
    )
    assert tuple(TokenPrediction.__dataclass_fields__) == (
        "tokens",
        "token_log_likelihood",
        "sampling",
    )
    assert tuple(TokenPredictionMetrics.__dataclass_fields__) == (
        "cross_entropy",
        "perplexity",
        "token_accuracy",
        "n_tokens",
    )
    assert tuple(TokenRolloutMetrics.__dataclass_fields__) == (
        "token_error_by_horizon",
        "exact_frame_accuracy_by_horizon",
        "decoded_mse_by_horizon",
        "task_proxy_accuracy_by_horizon",
        "failure_horizon",
    )


def test_tokenized_world_model_adapter_keywords_follow_the_public_protocol() -> None:
    model = _model()
    images = np.zeros((2, 1, 8, 8), dtype=np.float64)

    tokens = model.encode(data=images)
    decoded = model.decode(latent=tokens)

    assert tokens.shape == (2, model.tokens_per_frame)
    assert decoded.shape == images.shape


def test_tokenized_world_model_predicts_integer_tokens_with_seeded_sampling() -> None:
    model = _model()
    token_sequences, actions = _tokens()
    first = model.predict_next(token_sequences[:2, 0], actions[:2, 0], sampling="sample", seed=19)
    second = model.predict_next(token_sequences[:2, 0], actions[:2, 0], sampling="sample", seed=19)

    np.testing.assert_array_equal(first.tokens, second.tokens)
    assert first.tokens.dtype == np.int64
    assert first.tokens.shape == (2, model.tokens_per_frame)
    assert np.isfinite(first.token_log_likelihood).all()


def test_tokenized_world_model_fit_and_teacher_free_metrics_are_seed_reproducible() -> None:
    token_sequences, actions = _tokens(episodes=4, horizon=3)
    first = TokenizedWorldModel(
        VQVAE(codebook_size=4, embedding_dim=3, n_epochs=1, random_state=72),
        1,
        hidden_dim=8,
        epochs=2,
        seed=72,
    )
    second = TokenizedWorldModel(
        VQVAE(codebook_size=4, embedding_dim=3, n_epochs=1, random_state=72),
        1,
        hidden_dim=8,
        epochs=2,
        seed=72,
    )
    first.fit_tokens(token_sequences, actions)
    second.fit_tokens(token_sequences, actions)

    first_prediction = first.predict_next(token_sequences[:2, 0], actions[:2, 0])
    second_prediction = second.predict_next(token_sequences[:2, 0], actions[:2, 0])
    np.testing.assert_array_equal(first_prediction.tokens, second_prediction.tokens)
    np.testing.assert_allclose(first_prediction.token_log_likelihood, second_prediction.token_log_likelihood)
    assert first.fit_metadata == second.fit_metadata


def test_tokenized_world_model_rejects_invalid_padding_and_codebook_versions() -> None:
    tokenizer = VQVAE(codebook_size=4, embedding_dim=3, n_epochs=1)
    with pytest.raises(ValueError, match="does not match tokenizer"):
        TokenizedWorldModel(tokenizer, action_dim=1, codebook_version="wrong-codebook")

    model = _model()
    token_sequences, actions = _tokens()
    invalid = token_sequences.copy()
    invalid[0, 0, 0] = model.pad_token_id + 1
    with pytest.raises(ValueError, match="invalid token IDs"):
        model.predict_next(invalid[0, 0], actions[0, 0])

    padded = token_sequences.copy()
    padded[-1, -1] = model.pad_token_id
    mask = np.ones(actions.shape[:2], dtype=np.int64)
    mask[-1, -1] = 0
    model.fit_tokens(padded, actions, sequence_mask=mask)
    assert model.fit_metadata["masked_transitions"] == 1


def test_tokenized_world_model_rejects_tokenizer_mutation_before_fit_and_rollout() -> None:
    model = _model()
    _mutate_codebook_for_checkpoint_test(model.tokenizer)
    token_sequences, actions = _tokens()

    with pytest.raises(ValueError, match="checkpoint changed"):
        model.step(token_sequences[0, 0], actions[0, 0])
    with pytest.raises(ValueError, match="checkpoint changed"):
        model.rollout(token_sequences[0, 0], actions[0])

    tokenizer = VQVAE(codebook_size=4, embedding_dim=3, n_epochs=1)
    unfitted = TokenizedWorldModel(tokenizer, action_dim=1, hidden_dim=8, epochs=1, seed=72)
    _mutate_codebook_for_checkpoint_test(tokenizer)
    with pytest.raises(ValueError, match="checkpoint changed"):
        unfitted.fit_tokens(token_sequences, actions)


def test_tokenized_world_model_rollout_preserves_discrete_states_and_pipeline_contract() -> None:
    model = _model()
    token_sequences, actions = _tokens()

    rollout = model.rollout(token_sequences[0, 0], actions[0], sampling="sample", seed=7)
    pipeline_result = RolloutPipeline(model).run(token_sequences[0, 0], actions[0])

    assert rollout.to_numpy().dtype == np.int64
    assert rollout.shape == (actions.shape[1] + 1, model.tokens_per_frame)
    assert pipeline_result.trajectory.metadata["source_space_identity"] == model.source_space_identity
    assert pipeline_result.to_numpy().dtype == np.int64
    assert pipeline_result.to_numpy().shape == rollout.shape


def test_tokenized_world_model_reports_teacher_forcing_drift_and_task_proxy() -> None:
    model = _model(epochs=5)
    token_sequences, actions = _tokens()
    observations = np.zeros((token_sequences.shape[0], token_sequences.shape[1], 1, 8, 8), dtype=np.float64)
    observations[:, :, 0, 0, 0] = token_sequences[:, :, 0] / 3.0

    report = model.evaluate(observations, actions, task_proxy=lambda values: values[:, 0, 0, 0] > 0.5)

    assert report.teacher_forced.n_tokens == token_sequences.shape[0] * actions.shape[1] * 16
    assert np.isfinite(report.teacher_forced.perplexity)
    assert report.free_running.horizon == actions.shape[1]
    assert report.free_running.decoded_mse_by_horizon is not None
    assert report.free_running.task_proxy_accuracy_by_horizon is not None
    assert report.provenance["codebook_version"] == model.codebook_version

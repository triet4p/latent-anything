"""Offline tests for sparse-autoencoder feature evaluation (Sprint 46).

The tests use a controlled synthetic dictionary dataset with known sparse
structure so reconstruction, sparsity, dead-feature detection, cross-seed
stability, ranking, and the feature atlas can be checked against regression
thresholds. The cross-check tests use a tiny linear transformer whose layer-0
residual is the SAE input space.
"""

from __future__ import annotations

import inspect
import json
from typing import Any

import numpy as np
import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st
from torch import nn

# torch has incomplete type stubs — these warnings are noise.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false, reportMissingTypeStubs=false
from latent_anything.config import build_from_dict
from latent_anything.sae_evaluation import (
    SAEConfig,
    SAEEvaluationResult,
    SAEFeatureEvaluation,
    build_feature_atlas,
    cross_check_feature,
    cross_seed_sae_stability,
    evaluate_sae_features,
    load_feature_atlas,
    rank_feature_examples,
    save_feature_atlas,
)
from latent_anything.tcav import TransformerLogitTarget

# Regression-threshold configs tuned for the synthetic dictionary data
# (standardised activations).  Higher L1 is needed to force the unused
# dictionary column's feature dead.
_RECOVERY_CONFIG = SAEConfig(n_components=6, l1_coef=0.3, learning_rate=1e-2, n_epochs=1000)
_DEAD_CONFIG = SAEConfig(n_components=6, l1_coef=0.5, learning_rate=1e-2, n_epochs=1000)
_CROSS_CHECK_CONFIG = SAEConfig(n_components=6, l1_coef=0.3, learning_rate=1e-2, n_epochs=1000)


def _sparse_dictionary_data(
    n_samples: int = 500,
    n_features: int = 16,
    n_components_true: int = 6,
    k_active: int = 2,
    *,
    seed: int = 7,
    noise: float = 0.005,
    used_columns: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate data = codes @ D.T with a known sparse code matrix.

    Returns ``(data, codes, dictionary)`` where ``codes[i, j] > 0`` means
    sample ``i`` was generated from dictionary column ``j``.
    """
    rng = np.random.default_rng(seed)
    dictionary = rng.normal(size=(n_features, n_components_true))
    dictionary /= np.linalg.norm(dictionary, axis=0, keepdims=True)
    pool = list(range(n_components_true)) if used_columns is None else used_columns
    codes = np.zeros((n_samples, n_components_true), dtype=np.float64)
    for i in range(n_samples):
        active = rng.choice(pool, size=min(k_active, len(pool)), replace=False)
        codes[i, active] = rng.uniform(0.3, 1.0, size=len(active))
    data = codes @ dictionary.T + noise * rng.normal(size=(n_samples, n_features))
    return data, codes, dictionary


def _standardize(data: np.ndarray) -> np.ndarray:
    mean = data.mean(axis=0)
    std = data.std(axis=0) + 1e-8
    return (data - mean) / std


def _decoder_alignment(
    evaluation: SAEEvaluationResult,
    dictionary: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    """Best absolute cosine of each decoder column with any dictionary column.

    The decoder operates in standardised input space; ``std`` maps it back to
    the original data space where ``dictionary`` lives.
    """
    decoder = np.asarray(evaluation.decoder_weights, dtype=np.float64)
    decoder_real = decoder * std[:, None]
    denominator = np.linalg.norm(decoder_real, axis=0) * np.linalg.norm(dictionary, axis=0)
    cosine_matrix = np.abs(decoder_real.T @ dictionary) / denominator[None, :]
    return np.max(cosine_matrix, axis=1)


def _val_split_indices(n_samples: int, config: SAEConfig) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(config.random_state)
    perm = rng.permutation(n_samples)
    split = max(1, int(n_samples * (1.0 - config.val_fraction)))
    return perm[:split], perm[split:]


# ---------------------------------------------------------------------------
# Task 1 — reconstruction, sparsity, activity, decoder norms
# ---------------------------------------------------------------------------


class TestReconstructionAndSparsity:
    def test_dictionary_recovery_reconstruction_below_variance(self) -> None:
        data, _codes, _dictionary = _sparse_dictionary_data()
        std = data.std(axis=0) + 1e-8
        evaluation = evaluate_sae_features(
            _standardize(data),
            config=_RECOVERY_CONFIG,
            source_representation_identity="synth-dictionary",
        )
        # Reconstruction must explain most of the variance of the standardised data.
        assert evaluation.reconstruction_mse < 0.25
        # Decoder columns must align with the source dictionary directions.
        alignment = _decoder_alignment(evaluation, _dictionary, std)
        assert float(alignment.mean()) > 0.9, f"decoder alignment too low: {alignment.mean():.3f}"

    def test_metrics_are_sane_and_typed(self) -> None:
        data, _codes, _dictionary = _sparse_dictionary_data()
        evaluation = evaluate_sae_features(
            _standardize(data), config=_RECOVERY_CONFIG, source_representation_identity="synth-dictionary"
        )
        assert evaluation.n_train + evaluation.n_val == data.shape[0]
        assert evaluation.decoder_weights.shape == (data.shape[1], evaluation.config.n_components)
        assert np.all((evaluation.activation_frequencies >= 0) & (evaluation.activation_frequencies <= 1))
        assert np.all(evaluation.decoder_norms > 0)
        assert len(evaluation.features) == evaluation.config.n_components
        # Arrays are read-only.
        with pytest.raises(ValueError):
            evaluation.activation_frequencies[0] = 1.0  # type: ignore[index]
        assert not evaluation.decoder_weights.flags.writeable
        assert evaluation.provenance["split"] == "train_validation"
        summary = evaluation.to_dict()
        json.dumps(summary)

    def test_l0_is_sparse_relative_to_component_count(self) -> None:
        data, _codes, _dictionary = _sparse_dictionary_data()
        evaluation = evaluate_sae_features(
            _standardize(data), config=_RECOVERY_CONFIG, source_representation_identity="synth-dictionary"
        )
        # k=2 source structure should keep the mean L0 well below the 6 components.
        assert evaluation.mean_l0 < 5
        assert evaluation.mean_l0 >= 1
        assert evaluation.mean_l1 >= 0

    def test_train_validation_separation_holds_out_validation(self) -> None:
        data, _codes, _dictionary = _sparse_dictionary_data()
        evaluation = evaluate_sae_features(
            _standardize(data), config=_RECOVERY_CONFIG, source_representation_identity="synth-dictionary"
        )
        assert evaluation.n_train == int(data.shape[0] * (1 - _RECOVERY_CONFIG.val_fraction))
        assert evaluation.n_val == int(data.shape[0] * _RECOVERY_CONFIG.val_fraction)


class TestDeadFeatureDetection:
    def test_unused_dictionary_column_produces_dead_feature(self) -> None:
        data, _codes, _dictionary = _sparse_dictionary_data(used_columns=[0, 1, 2, 3, 4], seed=11)
        evaluation = evaluate_sae_features(
            _standardize(data), config=_DEAD_CONFIG, source_representation_identity="synth-dictionary"
        )
        assert evaluation.n_dead_features >= 1, "expected at least one dead feature"
        assert float(evaluation.activation_frequencies.min()) < 0.01
        assert evaluation.dead_fraction >= 1 / evaluation.config.n_components


# ---------------------------------------------------------------------------
# Task 2 — checkpoint serialization
# ---------------------------------------------------------------------------


class TestCheckpointSerialization:
    def test_checkpoint_roundtrip_preserves_transforms(self, tmp_path: Any) -> None:
        data, _codes, _dictionary = _sparse_dictionary_data()
        path = str(tmp_path / "sae.npz")
        evaluator = SAEFeatureEvaluation(_RECOVERY_CONFIG)
        evaluator.fit(_standardize(data), source_representation_identity="synth-dictionary")
        evaluator.save_checkpoint(path)

        loaded = SAEFeatureEvaluation.load_checkpoint(path)
        loaded_sae = loaded.sae
        fitted_sae = evaluator.sae
        assert loaded_sae is not None and fitted_sae is not None
        np.testing.assert_allclose(
            loaded_sae.transform(_standardize(data)[:20]),
            fitted_sae.transform(_standardize(data)[:20]),
            atol=1e-5,
        )

    def test_save_checkpoint_requires_fit(self, tmp_path: Any) -> None:
        path = str(tmp_path / "unfitted.npz")
        evaluator = SAEFeatureEvaluation(_RECOVERY_CONFIG)
        with pytest.raises(RuntimeError, match="no fitted SAE"):
            evaluator.save_checkpoint(path)


# ---------------------------------------------------------------------------
# Task 3 — cross-seed stability with direction matching
# ---------------------------------------------------------------------------


class TestCrossSeedStability:
    @given(st.permutations(tuple(range(6))))
    def test_decoder_matching_is_invariant_to_feature_permutation(self, permutation: tuple[int, ...]) -> None:
        from latent_anything._sae_metrics import match_by_decoder_cosine

        reference = np.eye(6, dtype=np.float64)
        matched = match_by_decoder_cosine(reference, reference[:, permutation], threshold=0.99)
        assert len(matched) == 6
        assert all(abs(cosine - 1.0) < 1e-12 for _, cosine in matched)

    def test_features_match_across_seeds_by_direction(self) -> None:
        data, _codes, _dictionary = _sparse_dictionary_data()
        report = cross_seed_sae_stability(
            _standardize(data),
            config=_RECOVERY_CONFIG,
            seeds=(0, 1, 2),
            source_representation_identity="synth-dictionary",
        )
        assert report.method == "decoder_matching"
        assert report.seeds == (0, 1, 2)
        assert report.n_components == _RECOVERY_CONFIG.n_components
        # If arbitrary feature indices were compared directly (rather than
        # matched), the mean cosine across seeds would be near zero.
        assert report.mean_matched_cosine > 0.9, f"matched cosine too low: {report.mean_matched_cosine:.3f}"
        assert report.min_matched_cosine > 0.85
        assert report.alignment_quality > 0.7
        assert len(report.reconstruction_mses) == 3

    def test_stability_requires_two_seeds(self) -> None:
        data, _codes, _dictionary = _sparse_dictionary_data()
        with pytest.raises(ValueError, match="at least two seeds"):
            cross_seed_sae_stability(_standardize(data), config=_RECOVERY_CONFIG, seeds=(0,))


# ---------------------------------------------------------------------------
# Task 4 — ranking feature examples and counterexamples
# ---------------------------------------------------------------------------


class TestFeatureRanking:
    def test_top_examples_activate_stronger_than_counterexamples(self) -> None:
        data, _codes, _dictionary = _sparse_dictionary_data()
        evaluation = evaluate_sae_features(
            _standardize(data), config=_RECOVERY_CONFIG, source_representation_identity="synth-dictionary"
        )
        ranking = rank_feature_examples(evaluation, 0, k=8)
        assert max(ranking.top_activations) > min(ranking.bottom_activations)
        assert set(ranking.top_example_indices).isdisjoint(ranking.bottom_example_indices)
        assert len(ranking.top_example_indices) == 8

    def test_top_examples_correspond_to_true_source_feature(self) -> None:
        data, codes, dictionary = _sparse_dictionary_data()
        standardised = _standardize(data)
        config = _RECOVERY_CONFIG
        evaluation = evaluate_sae_features(
            standardised, config=config, source_representation_identity="synth-dictionary"
        )
        # Which SAE feature best matches dictionary column 2?
        decoder = np.asarray(evaluation.decoder_weights, dtype=np.float64)
        column_2 = dictionary[:, 2]
        dot = np.abs(decoder.T @ column_2) / (np.linalg.norm(decoder, axis=0) * np.linalg.norm(column_2))
        feature_index = int(np.argmax(dot))
        assert float(dot[feature_index]) > 0.9

        # Align labels with the deterministic validation split.
        _train_idx, val_idx = _val_split_indices(standardised.shape[0], config)
        labels = codes[:, 2] > 0
        val_labels = [str(bool(labels[i])) for i in val_idx]
        ranking = rank_feature_examples(evaluation, feature_index, k=10, example_labels=val_labels)
        val_codes = codes[val_idx]
        top_active = float(np.mean(val_codes[ranking.top_example_indices, 2] > 0))
        bottom_active = float(np.mean(val_codes[ranking.bottom_example_indices, 2] > 0))
        assert top_active > 0.8, f"top examples rarely use source feature 2: {top_active}"
        assert bottom_active < 0.3, f"counterexamples still use source feature 2: {bottom_active}"
        assert all(label in {"True", "False"} for label in ranking.top_labels)
        assert all(label in {"True", "False"} for label in ranking.bottom_labels)


# ---------------------------------------------------------------------------
# Task 5 — cross-check with probes, concepts, and causal steering
# ---------------------------------------------------------------------------


class _TransformerBody(nn.Module):
    """Container that mirrors GPT-2's ``transformer.h`` module-list seam."""

    h: nn.ModuleList

    def __init__(self) -> None:
        super().__init__()
        self.h = nn.ModuleList([nn.Identity()])


class _TinyTargetTransformer(nn.Module):
    """Linear transformer whose layer-0 residual is a dictionary embedding.

    Token ``i`` embeds to dictionary column ``i`` for ``i < n_components_true``.
    The logit of the last vocabulary token reads dictionary column 5, so a SAE
    feature aligned with that column should show positive causal influence.
    """

    transformer: _TransformerBody

    def __init__(self, vocab: int, hidden: int, dictionary: np.ndarray) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab, hidden)
        self.transformer = _TransformerBody()
        self.lm_head = nn.Linear(hidden, vocab, bias=False)
        with torch.no_grad():
            weights = self.embedding.weight.clone()
            for i in range(dictionary.shape[1]):
                weights[i] = torch.as_tensor(dictionary[:, i], dtype=torch.float32)
            weights[dictionary.shape[1] :] = 0.0
            self.embedding.weight.copy_(weights)
            self.lm_head.weight.zero_()
            self.lm_head.weight[vocab - 1].copy_(torch.as_tensor(dictionary[:, 5], dtype=torch.float32))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        output_hidden_states: bool = False,
    ) -> object:
        del attention_mask, output_hidden_states
        hidden = self.embedding(input_ids)
        hidden = self.transformer.h[0](hidden)
        return type("Output", (), {"logits": self.lm_head(hidden)})


def _tiny_transformer_fixture(
    n_repeats: int = 40,
) -> tuple[_TinyTargetTransformer, np.ndarray, np.ndarray]:
    """Return ``(model, residuals, token_ids)`` for the tiny transformer."""
    torch.manual_seed(0)
    n_components_true = 6
    hidden = 8
    vocab = 8
    rng = np.random.default_rng(3)
    dictionary = rng.normal(size=(hidden, n_components_true))
    dictionary /= np.linalg.norm(dictionary, axis=0, keepdims=True)
    model = _TinyTargetTransformer(vocab, hidden, dictionary)
    tokens = np.asarray([[t] for t in list(range(n_components_true)) * n_repeats], dtype=np.int64)
    token_ids = tokens.reshape(-1)
    ids_t = torch.as_tensor(tokens)
    mask_t = torch.ones_like(ids_t)
    captured: dict[str, Any] = {}
    handle = model.transformer.h[0].register_forward_hook(lambda _m, _i, output: captured.update(x=output))
    try:
        with torch.no_grad():
            model(ids_t, attention_mask=mask_t)
    finally:
        handle.remove()
    residuals = captured["x"].detach().numpy().reshape(-1, hidden)
    return model, residuals, token_ids


class TestCrossCheck:
    def _evaluation_and_target_feature(
        self,
    ) -> tuple[object, SAEEvaluationResult, int, int]:
        """Fit an SAE on the tiny transformer residuals and find the feature
        best aligned with the target token's readout direction."""
        model, residuals, _token_ids = _tiny_transformer_fixture()
        n_residuals = residuals.shape[0]
        evaluation = evaluate_sae_features(
            residuals,
            config=_CROSS_CHECK_CONFIG,
            source_representation_identity="tiny-transformer-layer-0",
        )
        target_direction = np.asarray(model.lm_head.weight[7].detach().numpy(), dtype=np.float64)
        decoder = np.asarray(evaluation.decoder_weights, dtype=np.float64)
        dot = np.abs(decoder.T @ target_direction) / (
            np.linalg.norm(decoder, axis=0) * np.linalg.norm(target_direction)
        )
        feature_index = int(np.argmax(dot))
        return model, evaluation, feature_index, n_residuals

    def test_probe_improves_over_shuffled_label_baseline(self) -> None:
        _model, evaluation, feature_index, n_residuals = self._evaluation_and_target_feature()
        n_val = evaluation.val_activations.shape[0]
        _train_idx, val_idx = _val_split_indices(n_residuals, _CROSS_CHECK_CONFIG)
        labels = np.asarray(val_idx % 6 == 5, dtype=np.int64)
        assert labels.shape[0] == n_val
        check = cross_check_feature(evaluation, feature_index, labels=labels, probe_random_state=0)
        assert check.probe_accuracy is not None
        assert check.shuffled_label_accuracy is not None
        assert check.probe_accuracy > check.shuffled_label_accuracy
        assert not check.has_causal_evidence
        assert check.n_examples_checked == 0

    def test_causal_steering_agrees_with_gradient_sensitivity(self) -> None:
        model, evaluation, feature_index, n_residuals = self._evaluation_and_target_feature()
        target = TransformerLogitTarget(token_id=7, position=-1)
        n_total = n_residuals
        token_pattern = np.tile(np.arange(6), n_total // 6 + 1)[:n_total]
        input_ids = np.asarray([[t] for t in token_pattern], dtype=np.int64)
        attention_mask = np.ones_like(input_ids)
        check = cross_check_feature(
            evaluation,
            feature_index,
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            target=target,
            layer=0,
            intervention_strength=1.0,
        )
        assert check.has_causal_evidence
        assert check.n_examples_checked == n_total
        assert check.concept_sensitivity is not None
        assert check.intervention_effect is not None
        assert check.intervention_agreement is not None
        assert check.intervention_effect > 0.0, "steering should increase the target logit"
        assert check.concept_sensitivity > 0.0, "target direction should align with the feature"
        assert check.intervention_agreement is not None
        assert check.intervention_agreement > 0.9, (
            f"steering/gradient agreement too low: {check.intervention_agreement}"
        )


# ---------------------------------------------------------------------------
# Task 6 — portable feature-atlas artifact
# ---------------------------------------------------------------------------


class TestFeatureAtlas:
    def test_public_signature_and_result_schema_snapshot(self) -> None:
        assert str(inspect.signature(evaluate_sae_features)) == (
            "(data: 'np.ndarray', *, config: 'SAEConfig | None' = None, "
            "val_data: 'np.ndarray | None' = None, source_representation_identity: 'str' = '', "
            "provenance: 'dict[str, Any] | None' = None) -> 'SAEEvaluationResult'"
        )
        expected = {
            "n_train",
            "n_val",
            "reconstruction_mse",
            "train_reconstruction_mse",
            "mean_l0",
            "mean_l1",
            "n_dead_features",
            "dead_fraction",
            "activation_frequencies",
            "decoder_norms",
            "feature_summaries",
            "source_representation_identity",
            "provenance",
        }
        data, _codes, _dictionary = _sparse_dictionary_data(80)
        evaluation = evaluate_sae_features(_standardize(data), config=SAEConfig(n_components=4, n_epochs=30))
        assert set(evaluation.to_dict()) == expected

    def test_atlas_is_json_portable_and_queryable(self) -> None:
        data, _codes, _dictionary = _sparse_dictionary_data()
        evaluation = evaluate_sae_features(
            _standardize(data), config=_RECOVERY_CONFIG, source_representation_identity="synth-dictionary"
        )
        atlas = build_feature_atlas(evaluation, k_examples=4, k_decoder_dims=3)
        assert len(atlas.entries) == evaluation.config.n_components
        entry = atlas.entry(2)
        assert entry.feature_index == 2
        assert len(entry.top_examples) == 4
        assert len(entry.bottom_examples) == 4
        assert len(entry.top_decoder_dims) == 3
        assert "schema" in atlas.to_dict()
        payload = json.dumps(atlas.to_dict())
        assert '"feature_index"' in payload

    def test_atlas_roundtrips_through_save_load(self, tmp_path: Any) -> None:
        data, _codes, _dictionary = _sparse_dictionary_data()
        evaluation = evaluate_sae_features(
            _standardize(data), config=_RECOVERY_CONFIG, source_representation_identity="synth-dictionary"
        )
        atlas = build_feature_atlas(evaluation, k_examples=3, k_decoder_dims=2)
        path = str(tmp_path / "feature_atlas.json")
        save_feature_atlas(atlas, path)
        loaded = load_feature_atlas(path)
        assert loaded.n_components == atlas.n_components
        assert loaded.entry(0).top_examples == atlas.entry(0).top_examples
        assert loaded.entry(0).top_decoder_dims == atlas.entry(0).top_decoder_dims
        assert loaded.source_representation_identity == atlas.source_representation_identity

    def test_atlas_tamper_or_truncation_fails_closed(self, tmp_path: Any) -> None:
        path = tmp_path / "tampered.json"
        path.write_text(json.dumps({"schema": "latent-anything/feature-atlas-v1", "entries": []}), encoding="utf-8")
        with pytest.raises(KeyError):
            load_feature_atlas(path)


# ---------------------------------------------------------------------------
# Task 8 — construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_direct_and_config_construction(self) -> None:
        direct = SAEFeatureEvaluation(SAEConfig(n_components=4))
        configured = build_from_dict({"kind": "analysis", "name": "sae_evaluation", "params": {"n_components": 4}})
        assert isinstance(direct, SAEFeatureEvaluation)
        assert isinstance(configured, SAEFeatureEvaluation)
        assert configured.config.n_components == 4

    def test_invalid_feature_index_raises(self) -> None:
        data, _codes, _dictionary = _sparse_dictionary_data()
        evaluation = evaluate_sae_features(
            _standardize(data), config=_RECOVERY_CONFIG, source_representation_identity="synth-dictionary"
        )
        with pytest.raises(ValueError, match="outside"):
            rank_feature_examples(evaluation, evaluation.config.n_components)

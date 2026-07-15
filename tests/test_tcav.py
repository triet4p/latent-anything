"""Comprehensive tests for the TCAV module.

Covers:
- Unit tests for ConceptDataset validation
- Unit tests for ConceptDirectionResult
- Direction learning (mean diff + linear separator)
- Direction stability and separability
- TCAVScore validation
- TCAVResult and serialization
- TCAVConfig and TCAV class
- Config construction via registry
- Gradient computation (synthetic model)
- Full compute_tcav pipeline (synthetic)
- Intervention agreement (synthetic)
- Edge cases and degenerate inputs
- Marked real-integration tests (network)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from pydantic import ValidationError

from latent_anything.tcav import (
    TCAV,
    ConceptDataset,
    ConceptDirectionResult,
    TCAVConfig,
    TCAVResult,
    TCAVScore,
    TransformerLogitTarget,
    compute_tcav,
    intervention_agreement,
    learn_linear_separator_direction,
    learn_mean_diff_direction,
)

# ---------------------------------------------------------------------------
#  Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def well_separated_concept() -> ConceptDataset:
    """Two well-separated clusters in 4D (concept at origin, reference shifted)."""
    rng = np.random.default_rng(42)
    concept = rng.normal(0, 0.3, (30, 4))
    reference = rng.normal(2.0, 0.3, (30, 4))
    return ConceptDataset(
        concept_examples=concept,
        reference_examples=reference,
        concept_name="test_concept",
        source="test_src",
        representation_space="test_4d",
        model_version="test@v1",
    )


@pytest.fixture
def weak_separation_concept() -> ConceptDataset:
    """Two weakly-separated clusters (overlapping)."""
    rng = np.random.default_rng(42)
    concept = rng.normal(0, 1.0, (50, 8))
    reference = rng.normal(0.5, 1.0, (50, 8))
    return ConceptDataset(
        concept_examples=concept,
        reference_examples=reference,
        concept_name="weak",
    )


@pytest.fixture
def tiny_concept() -> ConceptDataset:
    """Absolute minimum viable dataset (2 samples each)."""
    return ConceptDataset(
        concept_examples=np.array([[0.0, 0.0], [0.1, 0.1]]),
        reference_examples=np.array([[2.0, 2.0], [2.1, 2.1]]),
        concept_name="tiny",
    )


@pytest.fixture
def simple_transformer_target() -> TransformerLogitTarget:
    return TransformerLogitTarget(token_id=42, position=-1)


# ---------------------------------------------------------------------------
#  ConceptDataset  (Task 1)
# ---------------------------------------------------------------------------


class TestConceptDataset:
    def test_valid_dataset(self, well_separated_concept: ConceptDataset) -> None:
        ds = well_separated_concept
        assert ds.n_features == 4
        assert ds.n_concept == 30
        assert ds.n_reference == 30
        assert ds.concept_name == "test_concept"
        assert ds.source == "test_src"

    def test_feature_dim_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="feature dimension mismatch"):
            ConceptDataset(
                concept_examples=np.ones((5, 3)),
                reference_examples=np.ones((5, 4)),
                concept_name="bad",
            )

    def test_1d_concept_raises(self) -> None:
        with pytest.raises(ValueError, match="concept_examples must be 2D"):
            ConceptDataset(
                concept_examples=np.array([1.0, 2.0]),
                reference_examples=np.ones((3, 2)),
                concept_name="bad",
            )

    def test_too_few_samples_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 2 concept examples"):
            ConceptDataset(
                concept_examples=np.ones((1, 3)),
                reference_examples=np.ones((5, 3)),
                concept_name="bad",
            )
        with pytest.raises(ValueError, match="at least 2 reference examples"):
            ConceptDataset(
                concept_examples=np.ones((5, 3)),
                reference_examples=np.ones((1, 3)),
                concept_name="bad",
            )

    def test_empty_name_raises(self) -> None:
        with pytest.raises(ValueError, match="concept_name must not be empty"):
            ConceptDataset(
                concept_examples=np.ones((5, 2)),
                reference_examples=np.ones((5, 2)),
                concept_name="",
            )


# ---------------------------------------------------------------------------
#  ConceptDirectionResult  (Task 2)
# ---------------------------------------------------------------------------


class TestConceptDirectionResult:
    def test_valid_result(self) -> None:
        r = ConceptDirectionResult(
            direction=np.array([1.0, 0.0, 0.0]),
            method="mean_diff",
            stability=0.95,
            stability_ci95=0.02,
            separability_accuracy=0.9,
            n_concept=30,
            n_reference=30,
        )
        assert r.method == "mean_diff"
        assert_array_equal(r.direction, np.array([1.0, 0.0, 0.0]))

    def test_2d_direction_raises(self) -> None:
        with pytest.raises(ValueError, match="direction must be 1D"):
            ConceptDirectionResult(
                direction=np.ones((2, 3)),
                method="mean_diff",
                stability=0.9,
                stability_ci95=0.0,
                separability_accuracy=0.9,
                n_concept=5,
                n_reference=5,
            )

    def test_unknown_method_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown method"):
            ConceptDirectionResult(
                direction=np.array([1.0, 0.0]),
                method="svm",
                stability=0.9,
                stability_ci95=0.0,
                separability_accuracy=0.9,
                n_concept=5,
                n_reference=5,
            )

    def test_stability_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError, match="stability must be in"):
            ConceptDirectionResult(
                direction=np.array([1.0, 0.0]),
                method="mean_diff",
                stability=1.5,
                stability_ci95=0.0,
                separability_accuracy=0.9,
                n_concept=5,
                n_reference=5,
            )


# ---------------------------------------------------------------------------
#  Direction learning: mean diff  (Task 2)
# ---------------------------------------------------------------------------


class TestLearnMeanDiffDirection:
    def test_direction_separates_concepts(self, well_separated_concept: ConceptDataset) -> None:
        result = learn_mean_diff_direction(well_separated_concept)
        assert result.method == "mean_diff"
        assert result.direction.shape == (4,)
        # Direction should be near unit norm
        assert abs(np.linalg.norm(result.direction) - 1.0) < 1e-6

    def test_separability_above_chance(self, well_separated_concept: ConceptDataset) -> None:
        result = learn_mean_diff_direction(well_separated_concept)
        assert result.separability_accuracy > 0.7

    def test_stability_is_high_on_clean_data(self, well_separated_concept: ConceptDataset) -> None:
        result = learn_mean_diff_direction(well_separated_concept, n_bootstrap=50)
        assert result.stability > 0.8
        assert result.stability_ci95 > 0.0

    def test_tiny_dataset(self, tiny_concept: ConceptDataset) -> None:
        result = learn_mean_diff_direction(tiny_concept, n_bootstrap=10)
        assert result.direction.shape == (2,)
        assert result.n_concept == 2
        assert result.n_reference == 2

    def test_weak_separation_lower_stability(
        self,
        well_separated_concept: ConceptDataset,
        weak_separation_concept: ConceptDataset,
    ) -> None:
        strong_result = learn_mean_diff_direction(well_separated_concept, n_bootstrap=30)
        weak_result = learn_mean_diff_direction(weak_separation_concept, n_bootstrap=30)

        # Well-separated should have higher stability than weak
        assert strong_result.stability >= weak_result.stability - 0.1

    def test_direction_is_consistent_across_seeds(self, well_separated_concept: ConceptDataset) -> None:
        r1 = learn_mean_diff_direction(well_separated_concept, bootstrap_seed=0)
        r2 = learn_mean_diff_direction(well_separated_concept, bootstrap_seed=1)
        # Directions should point in similar directions (cosine > 0.9)
        cosine = float(np.dot(r1.direction, r2.direction))
        assert cosine > 0.9


# ---------------------------------------------------------------------------
#  Direction learning: linear separator  (Task 2)
# ---------------------------------------------------------------------------


class TestLearnLinearSeparatorDirection:
    def test_direction_separates_concepts(self, well_separated_concept: ConceptDataset) -> None:
        result = learn_linear_separator_direction(well_separated_concept)
        assert result.method == "linear_separator"
        assert result.direction.shape == (4,)
        assert abs(np.linalg.norm(result.direction) - 1.0) < 1e-6

    def test_separability_above_chance(self, well_separated_concept: ConceptDataset) -> None:
        result = learn_linear_separator_direction(well_separated_concept)
        assert result.separability_accuracy > 0.7

    def test_stability_is_high_on_clean_data(self, well_separated_concept: ConceptDataset) -> None:
        result = learn_linear_separator_direction(well_separated_concept, n_bootstrap=30)
        assert result.stability > 0.8

    def test_tiny_dataset(self, tiny_concept: ConceptDataset) -> None:
        result = learn_linear_separator_direction(tiny_concept, n_bootstrap=10)
        assert result.direction.shape == (2,)

    def test_different_c_values(self, well_separated_concept: ConceptDataset) -> None:
        r_strong = learn_linear_separator_direction(well_separated_concept, c_value=0.01)
        r_weak = learn_linear_separator_direction(well_separated_concept, c_value=10.0)
        # Both should produce valid unit directions
        assert abs(np.linalg.norm(r_strong.direction) - 1.0) < 1e-6
        assert abs(np.linalg.norm(r_weak.direction) - 1.0) < 1e-6


# ---------------------------------------------------------------------------
#  Direction consistency: mean_diff vs linear_separator  (Task 2)
# ---------------------------------------------------------------------------


class TestDirectionConsistency:
    def test_both_methods_agree_on_clean_data(self, well_separated_concept: ConceptDataset) -> None:
        mean_result = learn_mean_diff_direction(well_separated_concept)
        lr_result = learn_linear_separator_direction(well_separated_concept)
        # Both should point in roughly the same direction
        cosine = float(np.dot(mean_result.direction, lr_result.direction))
        assert cosine > 0.7, f"Directions disagree: cosine={cosine:.3f}"


# ---------------------------------------------------------------------------
#  TransformerLogitTarget  (Task 3)
# ---------------------------------------------------------------------------


class TestTransformerLogitTarget:
    def test_default_target(self) -> None:
        t = TransformerLogitTarget(token_id=100)
        assert t.target_type == "transformer_logit"
        assert t.token_id == 100
        assert t.position == -1
        assert t.batch_index == 0

    def test_negative_token_id_raises(self) -> None:
        with pytest.raises(ValidationError):
            TransformerLogitTarget(token_id=-1)

    def test_negative_batch_raises(self) -> None:
        with pytest.raises(ValidationError):
            TransformerLogitTarget(token_id=10, batch_index=-1)

    def test_config_serialization_roundtrip(self) -> None:
        t = TransformerLogitTarget(token_id=42, position=5, batch_index=1)
        data = t.model_dump()
        restored = TransformerLogitTarget(**data)
        assert restored.token_id == 42
        assert restored.position == 5
        assert restored.batch_index == 1


# ---------------------------------------------------------------------------
#  TCAVScore  (Task 4)
# ---------------------------------------------------------------------------


class TestTCAVScore:
    def test_valid_score(self, well_separated_concept: ConceptDataset) -> None:
        cav = learn_mean_diff_direction(well_separated_concept)
        target = TransformerLogitTarget(token_id=42)
        score = TCAVScore(
            concept_name="test",
            layer=8,
            target=target,
            cav_direction=cav,
            sensitivity=0.8,
            n_examples=20,
            n_positive=16,
            p_value=0.01,
            per_example_sensitivities=np.random.default_rng(0).normal(0, 1, 20),
        )
        assert score.sensitivity == 0.8
        assert score.n_positive == 16
        assert score.concept_name == "test"

    def test_sensitivity_out_of_range_raises(self, well_separated_concept: ConceptDataset) -> None:
        cav = learn_mean_diff_direction(well_separated_concept)
        target = TransformerLogitTarget(token_id=42)
        with pytest.raises(ValueError, match="sensitivity must be in"):
            TCAVScore(
                concept_name="test",
                layer=8,
                target=target,
                cav_direction=cav,
                sensitivity=1.2,
                n_examples=10,
                n_positive=5,
                p_value=0.5,
                per_example_sensitivities=np.zeros(10),
            )

    def test_n_positive_exceeds_n_examples_raises(self, well_separated_concept: ConceptDataset) -> None:
        cav = learn_mean_diff_direction(well_separated_concept)
        target = TransformerLogitTarget(token_id=42)
        with pytest.raises(ValueError, match="n_positive"):
            TCAVScore(
                concept_name="test",
                layer=8,
                target=target,
                cav_direction=cav,
                sensitivity=0.5,
                n_examples=10,
                n_positive=15,
                p_value=0.5,
                per_example_sensitivities=np.zeros(10),
            )

    def test_sensitivities_shape_mismatch_raises(self, well_separated_concept: ConceptDataset) -> None:
        cav = learn_mean_diff_direction(well_separated_concept)
        target = TransformerLogitTarget(token_id=42)
        with pytest.raises(ValueError, match="per_example_sensitivities shape"):
            TCAVScore(
                concept_name="test",
                layer=8,
                target=target,
                cav_direction=cav,
                sensitivity=0.5,
                n_examples=10,
                n_positive=5,
                p_value=0.5,
                per_example_sensitivities=np.zeros(5),
            )


# ---------------------------------------------------------------------------
#  TCAVResult  (Task 4)
# ---------------------------------------------------------------------------


class TestTCAVResult:
    def test_to_dict(self, well_separated_concept: ConceptDataset) -> None:
        cav = learn_mean_diff_direction(well_separated_concept)
        target = TransformerLogitTarget(token_id=42)
        score = TCAVScore(
            concept_name="test",
            layer=8,
            target=target,
            cav_direction=cav,
            sensitivity=0.8,
            n_examples=20,
            n_positive=16,
            p_value=0.01,
            per_example_sensitivities=np.ones(20),
        )
        result = TCAVResult(
            scores=(score,),
            aggregate_score=0.8,
            aggregate_ci95=0.05,
            significance="significant",
            corrected_p_value=0.005,
            n_random_concepts=50,
            random_baseline_scores=np.random.default_rng(0).uniform(0.3, 0.7, 50),
            random_baseline_mean=0.5,
            random_baseline_std=0.1,
            empirical_p_value=0.01,
            intervention_agreement=0.85,
        )
        d = result.to_dict()
        assert isinstance(d["aggregate_score"], float)
        assert isinstance(d["random_baseline_scores"], list)
        assert isinstance(d["scores"], list)
        assert d["significance"] == "significant"
        assert d["intervention_agreement"] == 0.85


# ---------------------------------------------------------------------------
#  TCAVConfig and TCAV class  (Task 8)
# ---------------------------------------------------------------------------


class TestTCAVConfig:
    def test_default_config(self) -> None:
        cfg = TCAVConfig()
        assert cfg.target_layer == 8
        assert cfg.direction_method == "mean_diff"
        assert cfg.n_bootstrap == 50
        assert cfg.n_random_concepts == 50
        assert cfg.n_seeds == 5
        assert cfg.alpha == 0.05

    def test_invalid_direction_method_raises(self) -> None:
        with pytest.raises(ValidationError):
            TCAVConfig(direction_method="svm")

    def test_invalid_alpha_raises(self) -> None:
        with pytest.raises(ValidationError):
            TCAVConfig(alpha=0.0)
        with pytest.raises(ValidationError):
            TCAVConfig(alpha=1.0)


class TestTCAVClass:
    def test_default_construction(self) -> None:
        tcav = TCAV()
        assert isinstance(tcav.config, TCAVConfig)
        assert tcav.config.target_layer == 8

    def test_custom_config(self) -> None:
        cfg = TCAVConfig(target_layer=6, direction_method="linear_separator")
        tcav = TCAV(cfg)
        assert tcav.config.target_layer == 6
        assert tcav.config.direction_method == "linear_separator"

    def test_config_property(self) -> None:
        tcav = TCAV()
        assert tcav.config is tcav._config


# ---------------------------------------------------------------------------
#  Config construction via registry  (Task 8)
# ---------------------------------------------------------------------------


class TestRegistryConstruction:
    def test_build_from_object_spec(self) -> None:
        from latent_anything.config import ObjectSpec, build_from_config

        spec = ObjectSpec(kind="analysis", name="tcav")
        tcav = build_from_config(spec)
        assert isinstance(tcav, TCAV)
        assert isinstance(tcav.config, TCAVConfig)

    def test_build_from_object_spec_with_params(self) -> None:
        from latent_anything.config import ObjectSpec, build_from_config

        spec = ObjectSpec(
            kind="analysis",
            name="tcav",
            params={"target_layer": 6, "direction_method": "linear_separator"},
        )
        tcav = build_from_config(spec)
        assert isinstance(tcav, TCAV)
        assert tcav.config.target_layer == 6
        assert tcav.config.direction_method == "linear_separator"

    def test_build_from_dict(self) -> None:
        from latent_anything.config import build_from_dict

        tcav = build_from_dict({"kind": "analysis", "name": "tcav"})
        assert isinstance(tcav, TCAV)

    def test_registry_entry_exists(self) -> None:
        from latent_anything.registry import GLOBAL_REGISTRY

        entry = GLOBAL_REGISTRY.lookup("tcav")
        assert entry.kind == "analysis"
        assert entry.factory is TCAV


# ---------------------------------------------------------------------------
#  Gradient computation with a minimal synthetic PyTorch model  (Task 3)
# ---------------------------------------------------------------------------

import torch  # noqa: E402


class _SyntheticTransformer(torch.nn.Module):
    """Minimal transformer-like model for gradient tests.

    Has embedding + 2 transformer-like blocks + lm_head.
    """

    def __init__(self, hidden_dim: int = 16, vocab_size: int = 100) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        self.transformer = torch.nn.ModuleDict()
        self.transformer["wte"] = torch.nn.Embedding(vocab_size, hidden_dim)
        self.transformer["h"] = torch.nn.ModuleList([_SyntheticBlock(hidden_dim) for _ in range(3)])
        self.transformer["ln_f"] = torch.nn.LayerNorm(hidden_dim)
        self.lm_head = torch.nn.Linear(hidden_dim, vocab_size, bias=False)
        # Tie weights
        self.lm_head.weight = self.transformer["wte"].weight

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        output_hidden_states: bool = False,
    ) -> Any:
        del attention_mask
        x = self.transformer["wte"](input_ids)
        all_hidden = [x]
        for block in self.transformer["h"]:
            x = block(x)
            all_hidden.append(x)
        x = self.transformer["ln_f"](x)
        logits = self.lm_head(x)

        if output_hidden_states:
            return type("Output", (), {"logits": logits, "hidden_states": tuple(all_hidden)})()
        return type("Output", (), {"logits": logits})()


class _SyntheticBlock(torch.nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.ln1 = torch.nn.LayerNorm(hidden_dim)
        self.attn = _SyntheticAttention(hidden_dim)
        self.ln2 = torch.nn.LayerNorm(hidden_dim)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim * 4),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class _SyntheticAttention(torch.nn.Module):
    def __init__(self, hidden_dim: int, n_heads: int = 2):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.qkv = torch.nn.Linear(hidden_dim, hidden_dim * 3)
        self.proj = torch.nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)  # simplified


@pytest.fixture
def synthetic_model():
    """A small synthetic transformer for gradient tests."""
    torch.manual_seed(42)
    model = _SyntheticTransformer(hidden_dim=8, vocab_size=32)
    model.eval()
    return model


class TestGradientComputation:
    """Tests for internal gradient computation via compute_tcav (synthetic model)."""

    def test_gradient_returns_correct_shape(self, synthetic_model) -> None:
        """Gradient should have shape matching hidden_dim."""
        from latent_anything.tcav import _compute_transformer_layer_gradient

        model = synthetic_model
        ids = np.array([[1, 2, 3, 4, 5]], dtype=np.int64)
        mask = np.array([[1, 1, 1, 1, 1]], dtype=np.int64)
        target = TransformerLogitTarget(token_id=10, position=-1)

        grad = _compute_transformer_layer_gradient(model, ids, mask, layer=1, target=target)
        assert grad.shape == (8,), f"Expected (8,), got {grad.shape}"
        assert grad.dtype == np.float64

    def test_gradient_is_finite(self, synthetic_model) -> None:
        from latent_anything.tcav import _compute_transformer_layer_gradient

        model = synthetic_model
        ids = np.array([[1, 2, 3, 4, 5]], dtype=np.int64)
        mask = np.array([[1, 1, 1, 1, 1]], dtype=np.int64)
        target = TransformerLogitTarget(token_id=10, position=-1)

        grad = _compute_transformer_layer_gradient(model, ids, mask, layer=1, target=target)
        assert np.all(np.isfinite(grad))

    def test_gradient_differs_per_position(self, synthetic_model) -> None:
        from latent_anything.tcav import _compute_transformer_layer_gradient

        model = synthetic_model
        ids = np.array([[1, 2, 3, 4, 5]], dtype=np.int64)
        mask = np.array([[1, 1, 1, 1, 1]], dtype=np.int64)

        grad_last = _compute_transformer_layer_gradient(
            model, ids, mask, layer=1, target=TransformerLogitTarget(token_id=10, position=-1)
        )
        grad_first = _compute_transformer_layer_gradient(
            model, ids, mask, layer=1, target=TransformerLogitTarget(token_id=10, position=0)
        )
        # Different positions should give different gradients
        assert not np.allclose(grad_last, grad_first)

    def test_unknown_layer_raises(self, synthetic_model) -> None:
        from latent_anything.tcav import _compute_transformer_layer_gradient

        ids = np.array([[1, 2, 3]], dtype=np.int64)
        mask = np.array([[1, 1, 1]], dtype=np.int64)
        target = TransformerLogitTarget(token_id=10)

        with pytest.raises(ValueError, match="not found in model"):
            _compute_transformer_layer_gradient(synthetic_model, ids, mask, layer=99, target=target)


# ---------------------------------------------------------------------------
#  Layer activation extraction (synthetic model)
# ---------------------------------------------------------------------------


class TestLayerActivationExtraction:
    def test_extract_returns_correct_shape(self, synthetic_model) -> None:
        from latent_anything.tcav import _extract_layer_activation

        ids = np.array([[1, 2, 3, 4, 5]], dtype=np.int64)
        mask = np.array([[1, 1, 1, 1, 1]], dtype=np.int64)

        act = _extract_layer_activation(synthetic_model, ids, mask, layer=1)
        assert act.shape == (8,), f"Expected (8,), got {act.shape}"

    def test_extract_last_position(self, synthetic_model) -> None:
        from latent_anything.tcav import _extract_layer_activation

        ids = np.array([[1, 2, 3, 4, 5]], dtype=np.int64)
        mask = np.array([[1, 1, 1, 1, 1]], dtype=np.int64)

        act = _extract_layer_activation(synthetic_model, ids, mask, layer=1, position=-1)
        assert act.shape == (8,)

    def test_unknown_layer_raises(self, synthetic_model) -> None:
        from latent_anything.tcav import _extract_layer_activation

        ids = np.array([[1, 2, 3]], dtype=np.int64)
        mask = np.array([[1, 1, 1]], dtype=np.int64)

        with pytest.raises(ValueError, match="not found in model"):
            _extract_layer_activation(synthetic_model, ids, mask, layer=99)


# ---------------------------------------------------------------------------
#  Full compute_tcav pipeline (synthetic model)  (Tasks 5–6)
# ---------------------------------------------------------------------------


class TestComputeTCAV:
    def test_compute_tcav_returns_result(self, synthetic_model) -> None:
        """End-to-end test with synthetic model and concept."""
        model = synthetic_model

        # Build a concept dataset from synthetic activations
        rng = np.random.default_rng(42)
        concept_acts = rng.normal(0, 0.5, (20, 8))
        reference_acts = rng.normal(1.0, 0.5, (20, 8))
        ds = ConceptDataset(
            concept_examples=concept_acts,
            reference_examples=reference_acts,
            concept_name="test_concept",
            source="synthetic",
            representation_space="synth_layer_1",
            model_version="synth@v0",
        )

        # Inputs for gradient computation
        ids = np.array([[1, 2, 3, 4, 5]], dtype=np.int64)
        mask = np.array([[1, 1, 1, 1, 1]], dtype=np.int64)
        target = TransformerLogitTarget(token_id=10, position=-1)

        result = compute_tcav(
            model=model,
            target_layer=1,
            concept_dataset=ds,
            input_ids_batch=ids,
            attention_mask_batch=mask,
            target=target,
            direction_method="mean_diff",
            n_bootstrap=10,
            n_random_concepts=10,
            n_seeds=3,
            alpha=0.05,
            n_concepts_family=1,
        )

        assert isinstance(result, TCAVResult)
        assert isinstance(result.aggregate_score, float)
        assert 0.0 <= result.aggregate_score <= 1.0
        assert len(result.scores) == 3  # n_seeds
        assert isinstance(result.random_baseline_scores, np.ndarray)
        assert result.n_random_concepts == 10
        assert result.significance in ("significant", "not_significant")

    def test_compute_tcav_linear_separator(self, synthetic_model) -> None:
        """Same test but with linear_separator method."""
        model = synthetic_model
        rng = np.random.default_rng(42)
        concept_acts = rng.normal(0, 0.5, (20, 8))
        reference_acts = rng.normal(1.0, 0.5, (20, 8))
        ds = ConceptDataset(
            concept_examples=concept_acts,
            reference_examples=reference_acts,
            concept_name="test_concept",
        )
        ids = np.array([[1, 2, 3, 4, 5]], dtype=np.int64)
        mask = np.array([[1, 1, 1, 1, 1]], dtype=np.int64)
        target = TransformerLogitTarget(token_id=10, position=-1)

        result = compute_tcav(
            model=model,
            target_layer=1,
            concept_dataset=ds,
            input_ids_batch=ids,
            attention_mask_batch=mask,
            target=target,
            direction_method="linear_separator",
            n_bootstrap=10,
            n_random_concepts=5,
            n_seeds=2,
        )
        assert isinstance(result, TCAVResult)
        assert isinstance(result.aggregate_score, float)

    def test_tcav_class_compute(self, synthetic_model) -> None:
        """TCAV class should produce the same shape as compute_tcav."""
        rng = np.random.default_rng(42)
        ds = ConceptDataset(
            concept_examples=rng.normal(0, 0.5, (15, 8)),
            reference_examples=rng.normal(1.0, 0.5, (15, 8)),
            concept_name="test",
        )
        ids = np.array([[1, 2, 3, 4, 5]], dtype=np.int64)
        mask = np.array([[1, 1, 1, 1, 1]], dtype=np.int64)
        target = TransformerLogitTarget(token_id=10, position=-1)

        tcav = TCAV(TCAVConfig(target_layer=1, n_bootstrap=10, n_random_concepts=10, n_seeds=2))
        result = tcav.compute(
            model=synthetic_model,
            concept_dataset=ds,
            input_ids_batch=ids,
            attention_mask_batch=mask,
            target=target,
        )
        assert isinstance(result, TCAVResult)
        assert isinstance(result.aggregate_score, float)

    def test_batch_processing(self, synthetic_model) -> None:
        """Multiple examples in batch should produce valid scores."""
        model = synthetic_model
        rng = np.random.default_rng(42)
        ds = ConceptDataset(
            concept_examples=rng.normal(0, 0.5, (15, 8)),
            reference_examples=rng.normal(1.0, 0.5, (15, 8)),
            concept_name="test",
        )
        # Batch of 3 examples
        ids = np.array([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1], [1, 3, 5, 7, 9]], dtype=np.int64)
        mask = np.ones_like(ids)
        target = TransformerLogitTarget(token_id=10, position=-1)

        result = compute_tcav(
            model=model,
            target_layer=1,
            concept_dataset=ds,
            input_ids_batch=ids,
            attention_mask_batch=mask,
            target=target,
            n_bootstrap=10,
            n_random_concepts=5,
            n_seeds=2,
        )
        assert result.scores[0].n_examples == 3
        assert result.scores[0].per_example_sensitivities.shape == (3,)


# ---------------------------------------------------------------------------
#  Intervention agreement (synthetic model)  (Task 7)
# ---------------------------------------------------------------------------


class TestInterventionAgreement:
    def test_intervention_agreement_returns_float(self, synthetic_model) -> None:
        model = synthetic_model
        direction = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        ids = np.array([[1, 2, 3, 4, 5]], dtype=np.int64)
        mask = np.array([[1, 1, 1, 1, 1]], dtype=np.int64)
        target = TransformerLogitTarget(token_id=10, position=-1)

        agreement = intervention_agreement(
            model=model,
            target_layer=1,
            concept_direction=direction,
            input_ids_batch=ids,
            attention_mask_batch=mask,
            target=target,
            strength=1.0,
        )
        assert isinstance(agreement, float)
        assert 0.0 <= agreement <= 1.0

    def test_intervention_multiple_examples(self, synthetic_model) -> None:
        model = synthetic_model
        direction = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        ids = np.array([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1], [1, 3, 5, 7, 9]], dtype=np.int64)
        mask = np.ones_like(ids)
        target = TransformerLogitTarget(token_id=10, position=-1)

        agreement = intervention_agreement(
            model=model,
            target_layer=1,
            concept_direction=direction,
            input_ids_batch=ids,
            attention_mask_batch=mask,
            target=target,
            strength=0.5,
        )
        assert 0.0 <= agreement <= 1.0


# ---------------------------------------------------------------------------
#  Build concept dataset from text (requires transformer integration)
# ---------------------------------------------------------------------------


class TestBuildFromText:
    """Tests for build_concept_dataset_from_text with synthetic integration."""

    def test_requires_transformer_dependency(self) -> None:
        """The function should require a real integration, so just check import."""
        from latent_anything.tcav import build_concept_dataset_from_text

        assert callable(build_concept_dataset_from_text)


# ---------------------------------------------------------------------------
#  Edge cases and degenerate inputs
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_single_feature_dimension(self) -> None:
        """1D features (n, 1) should work."""
        ds = ConceptDataset(
            concept_examples=np.array([[0.0], [0.1], [0.2]]),
            reference_examples=np.array([[2.0], [2.1], [2.2]]),
            concept_name="1d_test",
        )
        result = learn_mean_diff_direction(ds, n_bootstrap=10)
        assert result.direction.shape == (1,)
        assert abs(np.linalg.norm(result.direction) - 1.0) < 1e-6

    def test_high_dimensional_features(self) -> None:
        """Many features (100) should still work."""
        rng = np.random.default_rng(42)
        ds = ConceptDataset(
            concept_examples=rng.normal(0, 0.3, (10, 100)),
            reference_examples=rng.normal(1.0, 0.3, (10, 100)),
            concept_name="high_dim",
        )
        result = learn_mean_diff_direction(ds, n_bootstrap=10)
        assert result.direction.shape == (100,)

    def test_strong_c_regularization(self, well_separated_concept: ConceptDataset) -> None:
        result = learn_linear_separator_direction(well_separated_concept, c_value=1e-6)
        assert result.direction.shape == (4,)
        assert abs(np.linalg.norm(result.direction) - 1.0) < 1e-6

    def test_imbalanced_dataset(self) -> None:
        """More reference than concept samples."""
        ds = ConceptDataset(
            concept_examples=np.random.default_rng(0).normal(0, 0.3, (10, 4)),
            reference_examples=np.random.default_rng(1).normal(2.0, 0.3, (50, 4)),
            concept_name="imbalanced",
        )
        result = learn_mean_diff_direction(ds)
        assert result.n_concept == 10
        assert result.n_reference == 50


# ---------------------------------------------------------------------------
#  Real-integration tests (marked network)  --  Task 6
# ---------------------------------------------------------------------------


@pytest.mark.network
class TestRealIntegration:
    """Integration tests that download real models (off by default).

    Run with::

        uv run pytest tests/test_tcav.py -m network -v
    """

    def test_on_transformer_hidden_states(self) -> None:
        """Learn a concept direction from transformer hidden states."""
        from latent_anything.integrations.transformer_lm import (
            TransformerGenerationRequest,
            TransformerLMIntegration,
        )

        integration = TransformerLMIntegration()
        request = TransformerGenerationRequest(
            prompt="The cat sat on the",
            max_length=64,
        )
        gen_result = integration.generate(request)

        # Extract layer-8 activations for concept/reference
        hidden_states = gen_result.hidden_states
        layer_8 = [hs for hs in hidden_states if hs.layer == 8]
        if not layer_8:
            pytest.skip("Layer 8 not captured")
        acts = layer_8[0].values  # (batch, seq, hidden)

        # Use first token vs last token as synthetic "concept"
        seq_len = acts.shape[1]
        if seq_len < 4:
            pytest.skip("Sequence too short")

        concept_acts = acts[0, : seq_len // 2, :]  # first half = concept
        reference_acts = acts[0, seq_len // 2 :, :]  # second half = reference

        ds = ConceptDataset(
            concept_examples=concept_acts,
            reference_examples=reference_acts,
            concept_name="position_concept",
            source="transformer_test",
            representation_space="gpt2_layer_8",
            model_version=integration.provenance,
        )
        result = learn_mean_diff_direction(ds, n_bootstrap=20)
        assert result.direction.shape == (acts.shape[2],)
        assert abs(np.linalg.norm(result.direction) - 1.0) < 1e-6
        assert result.stability > 0.5

    def test_on_vae_latents(self) -> None:
        """Learn a concept direction from VAE latent representations."""
        from latent_anything.integrations.diffusers_vae import DiffusersAutoencoderKLAdapter

        adapter = DiffusersAutoencoderKLAdapter("CompVis/stable-diffusion-v1-4", "7460a6f", latent_mode="mean")
        rng = np.random.default_rng(42)
        images = rng.uniform(-1, 1, (32, 3, 64, 64)).astype(np.float32)
        latents = adapter.encode(images)

        n_samples, n_channels = latents.shape[0], latents.shape[1]
        latents_flat = latents.reshape(n_samples, n_channels, -1).mean(axis=2)

        # Create concept/reference based on spatial variance
        spatial_var = latents.reshape(n_samples, n_channels, -1).var(axis=2).mean(axis=1)
        median_var = np.median(spatial_var)
        concept_mask = spatial_var > median_var
        reference_mask = ~concept_mask

        ds = ConceptDataset(
            concept_examples=latents_flat[concept_mask],
            reference_examples=latents_flat[reference_mask],
            concept_name="variance_concept",
            source="vae_test",
            representation_space="vae_mean_pooled",
            model_version="sd-v1-4",
        )
        result = learn_mean_diff_direction(ds, n_bootstrap=20)
        assert result.direction.shape == (n_channels,)
        assert abs(np.linalg.norm(result.direction) - 1.0) < 1e-6

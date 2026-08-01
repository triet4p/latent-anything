"""Offline tests for activation-space Integrated Gradients."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest
import torch
from torch import nn

from latent_anything.config import build_from_dict
from latent_anything.integrated_gradients import (
    IntegratedGradients,
    IntegratedGradientsConfig,
    compute_integrated_gradients,
    evaluate_sensitivity,
)
from latent_anything.tcav import TransformerLogitTarget


class _LinearBlock(nn.Module):
    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values


class _TupleBlock(nn.Module):
    def forward(self, values: torch.Tensor) -> tuple[torch.Tensor, None]:
        return values, None


class _AnalyticTransformer(nn.Module):
    def __init__(self, *, nonlinear: bool = False, tuple_output: bool = False) -> None:
        super().__init__()
        self.embedding = nn.Embedding(8, 3)
        block: nn.Module = _TupleBlock() if tuple_output else _LinearBlock()
        self.transformer = nn.Module()
        self.transformer.h = nn.ModuleList([block])
        self.lm_head = nn.Linear(3, 4, bias=False)
        with torch.no_grad():
            self.embedding.weight.copy_(torch.arange(24, dtype=torch.float32).reshape(8, 3) / 10.0)
            self.lm_head.weight.zero_()
            self.lm_head.weight[2].copy_(torch.tensor([1.0, -2.0, 0.5]))
        self.nonlinear = nonlinear

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        output_hidden_states: bool = False,
    ) -> object:
        del attention_mask, output_hidden_states
        hidden = self.embedding(input_ids)
        block = cast(_TupleBlock | _LinearBlock, cast(Any, self.transformer.h)[0])
        block_output: torch.Tensor | tuple[torch.Tensor, None] = block(hidden)
        hidden: torch.Tensor = block_output[0] if isinstance(block_output, tuple) else block_output
        if self.nonlinear:
            hidden = hidden.pow(3)
        return type("Output", (), {"logits": self.lm_head(hidden), "hidden_states": None})()


def _inputs() -> tuple[np.ndarray, np.ndarray]:
    return np.array([[3, 4]], dtype=np.int64), np.ones((1, 2), dtype=np.int64)


class TestIntegratedGradients:
    def test_linear_model_has_exact_completeness(self) -> None:
        model = _AnalyticTransformer()
        ids, mask = _inputs()
        result = compute_integrated_gradients(
            model,
            ids,
            mask,
            TransformerLogitTarget(token_id=2, position=-1),
            config=IntegratedGradientsConfig(target_layer=0, activation_position=-1, n_steps=8),
            source_model_version="analytic-linear-v1",
        )

        np.testing.assert_allclose(result.attributions.sum(), result.completeness_delta, atol=1e-6)
        assert abs(result.completeness_error) < 1e-6
        assert result.provenance["source_model_version"] == "analytic-linear-v1"
        assert result.target_layer == 0

    def test_nonlinear_completeness_improves_with_steps(self) -> None:
        model = _AnalyticTransformer(nonlinear=True)
        ids, mask = _inputs()
        target = TransformerLogitTarget(token_id=2, position=-1)
        coarse = IntegratedGradients(
            IntegratedGradientsConfig(target_layer=0, activation_position=-1, n_steps=2)
        ).compute(model, ids, mask, target)
        fine = IntegratedGradients(
            IntegratedGradientsConfig(target_layer=0, activation_position=-1, n_steps=64)
        ).compute(model, ids, mask, target)

        assert abs(fine.completeness_error) < abs(coarse.completeness_error)
        assert fine.convergence_delta == abs(fine.completeness_error)

    def test_tuple_output_hook_is_replaced_and_cleaned_up(self) -> None:
        model = _AnalyticTransformer(tuple_output=True)
        ids, mask = _inputs()
        result = IntegratedGradients(
            IntegratedGradientsConfig(target_layer=0, activation_position=-1, n_steps=8)
        ).compute(model, ids, mask, TransformerLogitTarget(token_id=2))

        assert result.attributions.shape == (3,)
        assert all(len(cast(Any, module)._forward_hooks) == 0 for module in model.modules())
        assert all(parameter.grad is None for parameter in model.parameters())
        assert torch.is_grad_enabled()

    def test_explicit_baseline_and_readonly_arrays(self) -> None:
        model = _AnalyticTransformer()
        ids, mask = _inputs()
        result = IntegratedGradients(
            IntegratedGradientsConfig(target_layer=0, activation_position=-1, n_steps=8)
        ).compute(
            model,
            ids,
            mask,
            TransformerLogitTarget(token_id=2),
            baseline_activation=np.array([0.1, 0.2, 0.3]),
        )

        assert result.baseline_kind == "explicit"
        assert not result.attributions.flags.writeable
        with pytest.raises(ValueError):
            result.attributions[0] = 0.0

    def test_sensitivity_report_records_steps_baselines_and_targets(self) -> None:
        model = _AnalyticTransformer()
        ids, mask = _inputs()
        target = TransformerLogitTarget(token_id=2)
        low = IntegratedGradients(IntegratedGradientsConfig(target_layer=0, activation_position=-1, n_steps=4)).compute(
            model, ids, mask, target
        )
        high = IntegratedGradients(
            IntegratedGradientsConfig(target_layer=0, activation_position=-1, n_steps=16)
        ).compute(model, ids, mask, target)
        explicit_baseline = IntegratedGradients(
            IntegratedGradientsConfig(target_layer=0, activation_position=-1, n_steps=16)
        ).compute(model, ids, mask, TransformerLogitTarget(token_id=1), baseline_activation=np.zeros(3))
        randomized_model = _AnalyticTransformer()
        with torch.no_grad():
            randomized_model.lm_head.weight.copy_(torch.randn_like(randomized_model.lm_head.weight))
        randomized = IntegratedGradients(
            IntegratedGradientsConfig(target_layer=0, activation_position=-1, n_steps=16)
        ).compute(randomized_model, ids, mask, target)
        report = evaluate_sensitivity((low, high, explicit_baseline), randomized=randomized)

        assert report.step_counts == (4, 16, 16)
        assert report.baseline_kinds == ("zero", "zero", "explicit")
        assert report.target_token_ids == (2, 2, 1)
        assert report.randomization_cosine < 0.999

    def test_direct_and_config_construction(self) -> None:
        direct = IntegratedGradients(IntegratedGradientsConfig(target_layer=0))
        configured = build_from_dict(
            {"kind": "analysis", "name": "integrated_gradients", "config": {"target_layer": 0}}
        )

        assert isinstance(direct, IntegratedGradients)
        assert isinstance(configured, IntegratedGradients)

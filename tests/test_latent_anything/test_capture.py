"""Lifecycle tests for internal PyTorch activation capture."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from latent_anything.capture import ActivationCaptureSession


class TinySequential(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.first = nn.Linear(3, 4)
        self.second = nn.ReLU()

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.second(self.first(values))


class TinyNested(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block = nn.Sequential(nn.Linear(3, 3), nn.Tanh())

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.block(values)


def test_capture_records_declared_locations_numpy_metadata_and_no_hooks_leak() -> None:
    model = TinySequential()
    values = torch.ones((2, 3))
    with ActivationCaptureSession(model, ["first", "second"], source_model_version="tiny-v1") as session:
        model(values)
    assert [capture.metadata.location for capture in session.captures] == ["first", "second"]
    assert session.captures[0].values.shape == (2, 4)
    assert session.captures[0].metadata.batch_axis == 0
    assert session.captures[0].metadata.dtype == "float32"
    assert session.captures[0].metadata.source_model_version == "tiny-v1"
    assert len(model.first._forward_hooks) == 0
    with ActivationCaptureSession(model, ["first"]) as repeated:
        model(values)
    assert len(repeated.captures) == 1


def test_capture_supports_nested_modules_and_intervention_without_mutating_input() -> None:
    model = TinyNested()
    values = torch.ones((2, 3), requires_grad=True)
    original = values.detach().clone()
    with ActivationCaptureSession(
        model,
        ["block.0"],
        intervention=lambda output, _metadata: output * 0,
        gradient_mode="disabled",
    ) as session:
        result = model(values)
    assert torch.allclose(values.detach(), original)
    assert torch.allclose(result, torch.zeros_like(result))
    assert len(session.captures) == 1


def test_capture_selection_and_shape_errors_remove_hooks_on_exception() -> None:
    model = TinySequential()
    with pytest.raises(KeyError, match="Unknown"):
        ActivationCaptureSession(model, ["missing"])
    with pytest.raises(ValueError, match="duplicates"):
        ActivationCaptureSession(model, ["first", "first"])
    with pytest.raises(RuntimeError), ActivationCaptureSession(model, ["first"]):
        raise RuntimeError("boom")
    assert len(model.first._forward_hooks) == 0

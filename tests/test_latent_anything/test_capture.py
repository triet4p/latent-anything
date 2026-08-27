"""Lifecycle tests for internal PyTorch activation capture."""

from __future__ import annotations

from typing import Any, cast

import pytest
import torch
from torch import nn

from latent_anything._hook_output import extract_primary_tensor, replace_primary_tensor
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


class TupleBlock(nn.Module):
    def forward(self, values: torch.Tensor) -> tuple[torch.Tensor, object]:
        return values, self


class ListBlock(nn.Module):
    def forward(self, values: torch.Tensor) -> list[object]:
        return [values, self]


class CustomTuple(tuple[object, ...]):
    pass


class CustomList(list[object]):
    pass


@pytest.mark.parametrize(
    ("output", "expected_type"),
    [
        (torch.ones(2), torch.Tensor),
        ((torch.ones(2), object()), tuple),
        ([torch.ones(2), object()], list),
    ],
)
def test_hook_output_helper_extracts_supported_primary_tensor(output: object, expected_type: type[object]) -> None:
    primary = extract_primary_tensor(output)
    assert isinstance(primary, torch.Tensor)
    assert type(output) is expected_type or isinstance(output, torch.Tensor)


def test_hook_output_helper_reconstructs_tuple_and_list_without_losing_auxiliary_values() -> None:
    primary = torch.ones(2)
    replacement = torch.zeros(2)
    aux = object()
    tuple_output = (primary, aux, "cache")
    list_output: list[object] = [primary, aux, "cache"]

    tuple_result = replace_primary_tensor(tuple_output, replacement)
    list_result = replace_primary_tensor(list_output, replacement)

    assert type(tuple_result) is tuple
    assert tuple_result[0] is replacement
    assert tuple_result[1] is aux
    assert tuple_result[2] == "cache"
    assert type(list_result) is list
    assert list_result[0] is replacement
    assert list_result[1] is aux
    assert list_result[2] == "cache"
    assert list_output[0] is primary


@pytest.mark.parametrize(
    "output",
    [
        (),
        [],
        ("not a tensor", object()),
        ["not a tensor", object()],
        {"hidden_states": torch.ones(2)},
        CustomTuple((torch.ones(2), object())),
        CustomList((torch.ones(2), object())),
    ],
)
def test_hook_output_helper_rejects_ambiguous_or_invalid_outputs(output: object) -> None:
    with pytest.raises(
        TypeError, match="(empty|position 0|mapping|custom|expected Tensor|expected Tensor, tuple, or list)"
    ):
        extract_primary_tensor(output)


def test_capture_reconstructs_tuple_and_list_outputs_and_cleans_up() -> None:
    for block in (TupleBlock(), ListBlock()):
        model = nn.Sequential(block)
        aux = block
        values = torch.ones(2)
        with ActivationCaptureSession(
            model,
            ["0"],
            intervention=lambda tensor, _metadata: tensor + 1,
        ) as session:
            result = model(values)
        assert len(session.captures) == 1
        assert not model[0]._forward_hooks  # type: ignore[reportPrivateUsage]
        if isinstance(result, tuple):
            assert result[0].tolist() == [2.0, 2.0]
            assert result[1] is aux
        else:
            assert type(result) is list
            assert result[0].tolist() == [2.0, 2.0]
            assert result[1] is aux


def test_capture_rejects_invalid_intervention_and_cleans_up() -> None:
    model = nn.Sequential(TupleBlock())
    session = ActivationCaptureSession(
        model,
        ["0"],
        intervention=cast(Any, lambda _tensor, _metadata: "bad"),
    )
    with pytest.raises(TypeError, match="intervention must return a Tensor"), session:
        model(torch.ones(2))
    assert not model[0]._forward_hooks  # type: ignore[reportPrivateUsage]


def test_capture_preserves_gradient_flow_for_structured_output() -> None:
    model = nn.Sequential(TupleBlock())
    values = torch.ones(2, requires_grad=True)
    with ActivationCaptureSession(
        model,
        ["0"],
        intervention=lambda tensor, _metadata: tensor * 2,
        gradient_mode="preserve",
    ) as session:
        result = cast(tuple[torch.Tensor, object], model(values))
        result[0].sum().backward()

    assert values.grad is not None
    assert torch.equal(values.grad, torch.full_like(values, 2.0))
    assert len(session.captures) == 1


def test_capture_rejects_wrong_shape_intervention_and_cleans_up() -> None:
    model = nn.Sequential(TupleBlock())
    session = ActivationCaptureSession(model, ["0"], intervention=lambda tensor, _metadata: tensor[:-1])
    with pytest.raises(ValueError, match="original shape"), session:
        model(torch.ones(2))
    assert not model[0]._forward_hooks  # type: ignore[reportPrivateUsage]


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
    assert not session.is_active
    capture_count = len(session.captures)
    model(values)
    assert len(session.captures) == capture_count
    with ActivationCaptureSession(model, ["first"]) as repeated:
        model(values)
    assert len(repeated.captures) == 1
    assert not repeated.is_active


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
    failed_session = ActivationCaptureSession(model, ["first"])
    with pytest.raises(RuntimeError), failed_session:
        raise RuntimeError("boom")
    model(torch.ones((1, 3)))
    assert failed_session.captures == ()

"""Internal PyTorch activation capture and intervention lifecycle."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from torch import nn

GradientMode = Literal["disabled", "preserve"]


@dataclass(frozen=True)
class CaptureMetadata:
    """Provenance and axes for one captured module output."""

    location: str
    call_index: int
    shape: tuple[int, ...]
    batch_axis: int | None
    sequence_axis: int | None
    device: str
    dtype: str
    source_model_version: str


@dataclass(frozen=True)
class CapturedActivation:
    """A detached NumPy activation and its immutable provenance."""

    values: np.ndarray
    metadata: CaptureMetadata


Intervention = Callable[[torch.Tensor, CaptureMetadata], torch.Tensor]


class ActivationCaptureSession(AbstractContextManager["ActivationCaptureSession"]):
    """Exception-safe session for captures in model forward execution order.

    Axis semantics are adapter-provided; this hook layer records only the
    universally meaningful batch axis.
    """

    def __init__(
        self,
        model: nn.Module,
        locations: Sequence[str],
        *,
        source_model_version: str = "",
        intervention: Intervention | None = None,
        gradient_mode: GradientMode = "disabled",
    ) -> None:
        if not locations:
            raise ValueError("locations must not be empty")
        if len(set(locations)) != len(locations):
            raise ValueError("locations must not contain duplicates")
        if gradient_mode not in {"disabled", "preserve"}:
            raise ValueError("gradient_mode must be 'disabled' or 'preserve'")
        modules: dict[str, nn.Module] = dict(model.named_modules())  # pyright: ignore[reportUnknownArgumentType]
        missing = [location for location in locations if location not in modules or location == ""]
        if missing:
            raise KeyError(f"Unknown capture location(s): {missing}")
        self._locations = tuple(locations)
        self._modules = {location: modules[location] for location in locations}
        self._source_model_version = source_model_version
        self._intervention = intervention
        self._gradient_mode = gradient_mode
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._captures: list[CapturedActivation] = []
        self._shapes: dict[str, tuple[int, ...]] = {}
        self._calls: dict[str, int] = {location: 0 for location in locations}

    @property
    def captures(self) -> tuple[CapturedActivation, ...]:
        """Return captured records in forward execution order."""

        return tuple(self._captures)

    def __enter__(self) -> ActivationCaptureSession:
        for location in self._locations:
            self._handles.append(self._modules[location].register_forward_hook(self._hook(location)))
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        del exc_type, exc_value, traceback
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    @property
    def is_active(self) -> bool:
        """Whether this session currently owns registered hooks."""

        return bool(self._handles)

    def _hook(self, location: str) -> Callable[[nn.Module, tuple[object, ...], object], object]:
        def callback(module: nn.Module, inputs: tuple[object, ...], output: object) -> object:
            """Capture one tensor activation and optionally apply intervention."""
            del module, inputs
            if not isinstance(output, torch.Tensor):
                raise TypeError(f"Capture location {location!r} returned {type(output).__name__}, expected Tensor")
            self._calls[location] += 1
            metadata = self._metadata(location, output, self._calls[location] - 1)
            previous_shape = self._shapes.setdefault(location, metadata.shape)
            if previous_shape != metadata.shape:
                msg = f"Capture location {location!r} changed shape from {previous_shape} to {metadata.shape}"
                raise ValueError(msg)
            values = output.detach().cpu().numpy().copy()
            values.setflags(write=False)
            self._captures.append(CapturedActivation(values=values, metadata=metadata))
            if self._intervention is None:
                return output
            if self._gradient_mode == "disabled":
                with torch.no_grad():
                    replaced = self._intervention(output.detach().clone(), metadata)
            else:
                replaced = self._intervention(output, metadata)
            if replaced.shape != output.shape:
                raise ValueError("intervention must return a Tensor with the original shape")
            return replaced.to(device=output.device, dtype=output.dtype)

        return callback

    def _metadata(self, location: str, tensor: torch.Tensor, call_index: int) -> CaptureMetadata:
        shape = tuple(int(size) for size in tensor.shape)
        return CaptureMetadata(
            location=location,
            call_index=call_index,
            shape=shape,
            batch_axis=0 if tensor.ndim >= 1 else None,
            sequence_axis=None,
            device=str(tensor.device),
            dtype=str(tensor.dtype).removeprefix("torch."),
            source_model_version=self._source_model_version,
        )

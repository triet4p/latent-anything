"""Private helpers for Tensor-valued PyTorch forward-hook outputs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

import torch


def extract_primary_tensor(output: object) -> torch.Tensor:
    """Return the primary Tensor from a supported module output container.

    Hook consumers may target modules whose forward output is a Tensor or a
    plain tuple/list with the activation in the first position.  Other
    containers are intentionally rejected because there is no safe generic
    rule for selecting or reconstructing their primary value.
    """

    if isinstance(output, torch.Tensor):
        return output
    output_type = type(output)
    if output_type is tuple or output_type is list:
        if not output:
            raise TypeError(f"{output_type.__name__} module output is empty; expected a Tensor in position 0")
        primary = cast(tuple[object, ...] | list[object], output)[0]
        if not isinstance(primary, torch.Tensor):
            raise TypeError(
                f"{output_type.__name__} module output position 0 is {type(primary).__name__}; expected Tensor"
            )
        return primary
    if isinstance(output, Mapping):
        raise TypeError("mapping module outputs are ambiguous; expected Tensor, tuple, or list")
    if isinstance(output, (tuple, list)):
        raise TypeError(
            f"custom {output_type.__name__} module output containers are unsupported; expected exact tuple or list"
        )
    raise TypeError(f"module output is {output_type.__name__}; expected Tensor, tuple, or list")


def replace_primary_tensor(output: object, replacement: object) -> object:
    """Replace only the primary Tensor while preserving supported output data."""

    if not isinstance(replacement, torch.Tensor):
        raise TypeError(f"replacement must be a Tensor, got {type(replacement).__name__}")
    extract_primary_tensor(output)
    if isinstance(output, torch.Tensor):
        return replacement
    if type(output) is tuple:
        values = output
        return (replacement, *values[1:])
    if type(output) is list:
        values = output.copy()
        values[0] = replacement
        return values
    raise TypeError(f"cannot reconstruct module output of type {type(output).__name__}")

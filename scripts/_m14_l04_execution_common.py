"""Torch-free common execution facts for M14 L04 handlers."""

from __future__ import annotations

import hashlib
import random
from collections.abc import Mapping
from typing import Any

import numpy as np


class RealExecutionError(RuntimeError):
    """Execution failure carrying sanitized resource facts."""

    def __init__(self, message: str, resources: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.resources = dict(resources)


def seed_everything(seed: int, torch: Any) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parameter_digest(model: Any) -> str:
    digest = hashlib.sha256()
    for parameter in model.parameters():
        digest.update(parameter.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


__all__ = ["RealExecutionError", "parameter_digest", "seed_everything"]

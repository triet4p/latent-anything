"""Private JEPA NPZ checkpoint serialization helpers."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, cast

import numpy as np


@dataclass(frozen=True, slots=True)
class JEPACheckpoint:
    metadata: dict[str, Any]
    scale: np.ndarray
    context_state: dict[str, np.ndarray]
    target_state: dict[str, np.ndarray]
    predictor_state: dict[str, np.ndarray]


def write_jepa_checkpoint(
    path: str | os.PathLike[str],
    *,
    metadata: dict[str, Any],
    scale: np.ndarray,
    context_state: dict[str, np.ndarray],
    target_state: dict[str, np.ndarray],
    predictor_state: dict[str, np.ndarray],
) -> None:
    arrays: dict[str, np.ndarray] = {"scale": scale}
    for prefix, state in (("context", context_state), ("target", target_state), ("predictor", predictor_state)):
        arrays.update({f"{prefix}_{name.replace('.', '_')}": value for name, value in state.items()})
    np.savez(path, **cast(dict[str, Any], {"metadata_json": np.asarray(json.dumps(metadata)), **arrays}))


def read_jepa_checkpoint(path: str | os.PathLike[str]) -> JEPACheckpoint:
    with np.load(path, allow_pickle=False) as data:  # pyright: ignore[reportUnknownMemberType]
        raw = data["metadata_json"].item()
        if not isinstance(raw, str):
            raise ValueError("JEPA checkpoint has no metadata_json string")
        metadata = cast(dict[str, Any], json.loads(raw))

        def read_state(prefix: str) -> dict[str, np.ndarray]:
            return {
                key[len(prefix) + 1 :].replace("_", "."): np.asarray(data[key], dtype=np.float32)
                for key in data.files
                if key.startswith(f"{prefix}_")
            }

        return JEPACheckpoint(
            metadata=metadata,
            scale=np.asarray(data["scale"], dtype=np.float64),
            context_state=read_state("context"),
            target_state=read_state("target"),
            predictor_state=read_state("predictor"),
        )

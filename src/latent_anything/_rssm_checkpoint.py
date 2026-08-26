"""Private RSSM checkpoint serialization and raw load validation."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, cast

import numpy as np


@dataclass(frozen=True, slots=True)
class RSSMCheckpoint:
    """Decoded checkpoint arrays and JSON metadata before model construction."""

    recurrent_weights: np.ndarray
    recurrent_bias: np.ndarray
    emission_weights: np.ndarray
    emission_bias: np.ndarray
    scale: np.ndarray
    metadata: dict[str, Any]


def write_rssm_checkpoint(
    path: str | os.PathLike[str],
    *,
    recurrent_weights: np.ndarray,
    recurrent_bias: np.ndarray,
    emission_weights: np.ndarray,
    emission_bias: np.ndarray,
    scale: np.ndarray,
    config: dict[str, Any],
    source_space_identity: str,
    fit_metadata: dict[str, Any],
) -> None:
    """Write the existing portable RSSM NPZ schema without model knowledge."""

    metadata = {
        "config": config,
        "source_space_identity": source_space_identity,
        "fit_metadata": fit_metadata,
    }
    np.savez(
        path,
        recurrent_weights=recurrent_weights,
        recurrent_bias=recurrent_bias,
        emission_weights=emission_weights,
        emission_bias=emission_bias,
        scale=scale,
        metadata_json=json.dumps(metadata),
    )


def read_rssm_checkpoint(path: str | os.PathLike[str]) -> RSSMCheckpoint:
    """Read the existing NPZ schema and preserve its metadata error contract."""

    with np.load(path, allow_pickle=False) as data:  # pyright: ignore[reportUnknownMemberType]
        metadata_raw = data["metadata_json"].item()
        if not isinstance(metadata_raw, str):
            raise ValueError("RSSM checkpoint has no metadata_json string")
        metadata = cast(dict[str, Any], json.loads(metadata_raw))
        return RSSMCheckpoint(
            recurrent_weights=np.asarray(data["recurrent_weights"], dtype=np.float64),
            recurrent_bias=np.asarray(data["recurrent_bias"], dtype=np.float64),
            emission_weights=np.asarray(data["emission_weights"], dtype=np.float64),
            emission_bias=np.asarray(data["emission_bias"], dtype=np.float64),
            scale=np.asarray(data["scale"], dtype=np.float64),
            metadata=metadata,
        )

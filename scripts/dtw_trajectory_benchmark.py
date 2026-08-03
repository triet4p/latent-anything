"""Benchmark DTW against index-wise distance on controlled policy trajectories."""

from __future__ import annotations

import json
import time
from contextlib import suppress
from pathlib import Path
from typing import Any

import numpy as np

from latent_anything import SE3, DTWConfig, LatentSpace, compute_dtw, indexwise_distance


def _controlled() -> dict[str, Any]:
    space = LatentSpace(dim=1)
    query = np.arange(24, dtype=np.float64).reshape(-1, 1)
    stretched = np.repeat(query, 2, axis=0)
    start = time.perf_counter()
    result = compute_dtw(query, stretched, space)
    elapsed = time.perf_counter() - start
    indexwise = None
    with suppress(ValueError):
        indexwise = indexwise_distance(query, stretched, space)
    return {
        "kind": "controlled_stretch",
        "query_length": len(query),
        "reference_length": len(stretched),
        "dtw_distance": result.distance,
        "indexwise_distance": indexwise,
        "path_length": len(result.path),
        "elapsed_seconds": elapsed,
    }


def _pose_policy_like() -> dict[str, Any]:
    metadata = {"parent_frame": "world", "child_frame": "tool", "position_unit": "m", "angle_unit": "rad"}
    query = np.stack([SE3(translation=np.array([0.05 * i, 0.0, 0.0])).matrix.reshape(-1) for i in range(16)])
    reference = np.stack(
        [
            SE3(translation=np.array([0.05 * min(i, 15), 0.01 * np.sin(i / 3), 0.0])).matrix.reshape(-1)
            for i in range(22)
        ]
    )
    space = LatentSpace(dim=16, geometry="se3", source_model="controlled_policy_pose")
    result = compute_dtw(query, reference, space, config=DTWConfig(normalization="path_length"))
    return {
        "kind": "se3_policy_like",
        "query_length": len(query),
        "reference_length": len(reference),
        "dtw_distance": result.distance,
        "geometry": result.geometry,
        "provenance": result.provenance,
        "path_length": len(result.path),
        "metadata_contract": metadata,
    }


def main() -> None:
    output = {"controlled": _controlled(), "pose_policy_like": _pose_policy_like(), "max_cells": 2_000_000}
    target = Path("artifacts/dtw_trajectory_benchmark.json")
    target.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

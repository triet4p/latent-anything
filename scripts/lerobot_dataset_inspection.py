#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "latent-anything",
#     "lerobot[dataset]>=0.6.0,<0.7.0",
# ]
# ///
"""Inspect LeRobot v3 metadata without making model or task-performance claims."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from latent_anything.integrations.lerobot import describe_lerobot_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo_id", help="Hugging Face dataset repository, e.g. lerobot/aloha_sim_insertion_human")
    parser.add_argument("--revision", default="v3.0", help="Dataset revision to inspect")
    parser.add_argument("--output", type=Path, default=Path("artifacts/lerobot_dataset_inspection.json"))
    args = parser.parse_args()

    from lerobot.datasets import LeRobotDatasetMetadata  # noqa: I001  # pyright: ignore[reportMissingTypeStubs, reportAttributeAccessIssue]

    metadata = LeRobotDatasetMetadata(args.repo_id, revision=args.revision)
    descriptor = describe_lerobot_dataset(metadata)
    episodes = descriptor.episodes
    report = {
        "claim_scope": "dataset schema, episode boundaries, and provenance only; no model claim",
        "dataset": descriptor.to_dict(),
        "inspection": {
            "first_episode": episodes[0].to_dict() if episodes else None,
            "last_episode": episodes[-1].to_dict() if episodes else None,
            "feature_count": len(descriptor.features),
            "camera_count": len(descriptor.cameras),
            "normalization_features": sorted(descriptor.stats),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report["inspection"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

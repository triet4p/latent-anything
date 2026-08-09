"""Small CLI for local run evidence and LeRobot-facing inspection."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from latent_anything.integrations.lerobot_recording import supported_capture_points
from latent_anything.run_record import FileSystemRunRecorder, build_comparison_report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="latent-anything", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    capture = commands.add_parser(
        "capture-points", aliases=["list-capture-points"], help="list supported capture seams"
    )
    capture.add_argument("--policy", choices=["act", "diffusion", "smolvla"])

    dataset = commands.add_parser("inspect-dataset", help="inspect LeRobot dataset metadata")
    dataset.add_argument("repo_id")
    dataset.add_argument("--revision", default="v3.0")
    dataset.add_argument("--output", type=Path)
    dataset.add_argument("--record-root", type=Path)

    policy = commands.add_parser("inspect-policy", help="inspect pinned bridge policy metadata")
    policy.add_argument("--policy", choices=["act", "diffusion", "smolvla"], required=True)
    policy.add_argument("--output", type=Path)

    replay = commands.add_parser("replay-run", aliases=["replay-run-config"], help="materialize a saved run config")
    replay.add_argument("run_id")
    replay.add_argument("--record-root", type=Path, default=Path("artifacts/runs"))
    replay.add_argument("--output", type=Path)

    compare = commands.add_parser("compare-runs", help="write a metric comparison across recorded runs")
    compare.add_argument("run_ids", nargs="*")
    compare.add_argument("--record-root", type=Path, default=Path("artifacts/runs"))
    compare.add_argument("--output", type=Path, default=Path("artifacts/run_comparison.json"))
    return parser


def _write_json(value: object, output: Path | None) -> None:
    encoded = json.dumps(value, indent=2, sort_keys=True, default=str) + "\n"
    if output is None:
        print(encoded, end="")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded, encoding="utf-8")
        print(f"Wrote {output}")


def _policy_snapshot(policy: str) -> dict[str, object]:
    if policy == "act":
        from latent_anything.integrations.lerobot_act import DEFAULT_ACT_CHECKPOINT

        checkpoint = DEFAULT_ACT_CHECKPOINT.to_dict()
    elif policy == "diffusion":
        from latent_anything.integrations.lerobot_diffusion import DEFAULT_DIFFUSION_CHECKPOINT

        checkpoint = DEFAULT_DIFFUSION_CHECKPOINT.to_dict()
    else:
        from latent_anything.integrations.lerobot_smolvla import DEFAULT_SMOLVLA_CHECKPOINT

        checkpoint = DEFAULT_SMOLVLA_CHECKPOINT.to_dict()
    return {
        "policy": policy,
        "checkpoint": checkpoint,
        "capture_points": [point.to_dict() for point in supported_capture_points(policy)],
        "claim_scope": "pinned metadata and supported capture seams; no model-quality claim",
    }


def _inspect_dataset(args: argparse.Namespace) -> int:
    from lerobot.datasets import (  # pyright: ignore[reportMissingTypeStubs, reportAttributeAccessIssue]
        LeRobotDatasetMetadata,
    )

    from latent_anything.integrations.lerobot import describe_lerobot_dataset

    metadata = LeRobotDatasetMetadata(args.repo_id, revision=args.revision)
    descriptor = describe_lerobot_dataset(metadata)
    report: dict[str, object] = {
        "claim_scope": "dataset schema, episode boundaries, and provenance only; no model claim",
        "dataset": descriptor.to_dict(),
        "inspection": {
            "first_episode": descriptor.episodes[0].to_dict() if descriptor.episodes else None,
            "last_episode": descriptor.episodes[-1].to_dict() if descriptor.episodes else None,
            "feature_count": len(descriptor.features),
            "camera_count": len(descriptor.cameras),
            "normalization_features": sorted(descriptor.stats),
        },
    }
    _write_json(report, args.output)
    if args.record_root is not None:
        from latent_anything.integrations.lerobot_recording import record_lerobot_dataset_inspection

        recorder = FileSystemRunRecorder(args.record_root)
        record = record_lerobot_dataset_inspection(
            recorder,
            report,
            config={"repo_id": args.repo_id, "revision": args.revision},
            dataset_revisions={args.repo_id: args.revision},
        )
        print(f"Recorded run {record.run_id}")
    return 0


def _replay(args: argparse.Namespace) -> int:
    record = FileSystemRunRecorder(args.record_root).get(args.run_id)
    payload = {
        "run_id": record.run_id,
        "name": record.name,
        "config": dict(record.config),
        "model_revisions": dict(record.model_revisions),
        "dataset_revisions": dict(record.dataset_revisions),
        "seeds": list(record.seeds),
        "environment": dict(record.environment),
        "metadata": dict(record.metadata),
    }
    _write_json(payload, args.output)
    return 0


def _compare(args: argparse.Namespace) -> int:
    recorder = FileSystemRunRecorder(args.record_root)
    records = tuple(recorder.get(run_id) for run_id in args.run_ids) if args.run_ids else recorder.list()
    report = build_comparison_report(records)
    _write_json(report.to_dict(), args.output)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line interface and return a process status code."""

    args = _parser().parse_args(argv)
    if args.command in {"capture-points", "list-capture-points"}:
        _write_json({"capture_points": [point.to_dict() for point in supported_capture_points(args.policy)]}, None)
        return 0
    if args.command == "inspect-dataset":
        return _inspect_dataset(args)
    if args.command == "inspect-policy":
        _write_json(_policy_snapshot(args.policy), args.output)
        return 0
    if args.command in {"replay-run", "replay-run-config"}:
        return _replay(args)
    if args.command == "compare-runs":
        return _compare(args)
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["latent-anything"]
# ///
"""Compare two or more local Sprint 62 run records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from latent_anything.run_record import FileSystemRunRecorder, build_comparison_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_ids", nargs="+", help="at least two run ids")
    parser.add_argument("--record-root", type=Path, default=Path("artifacts/runs"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/lerobot_run_comparison.json"))
    args = parser.parse_args()
    if len(args.run_ids) < 2:
        parser.error("at least two run ids are required")
    recorder = FileSystemRunRecorder(args.record_root)
    records = tuple(recorder.get(run_id) for run_id in args.run_ids)
    report = build_comparison_report(records, title="LeRobot run comparison")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Comparison written to {args.output}")


if __name__ == "__main__":
    main()

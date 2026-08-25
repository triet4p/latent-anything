#!/usr/bin/env python3
"""Compare two Sprint 77 Phase-A reports without changing workload semantics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def compare(before_path: Path, after_path: Path, output: Path | None = None) -> dict[str, object]:
    before = json.loads(before_path.read_text(encoding="utf-8"))
    after = json.loads(after_path.read_text(encoding="utf-8"))
    before_rows = {str(row["name"]): row for row in before["cases"]}
    after_rows = {str(row["name"]): row for row in after["cases"]}
    if set(before_rows) != set(after_rows):
        raise ValueError("before and after workloads differ")
    rows: list[dict[str, object]] = []
    for name in sorted(before_rows):
        old = before_rows[name]
        new = after_rows[name]
        old_median = float(old["latency_us"]["median"])
        new_median = float(new["latency_us"]["median"])
        rows.append(
            {
                "name": name,
                "before_median_us": old_median,
                "after_median_us": new_median,
                "median_ratio_after_over_before": new_median / old_median if old_median else None,
                "median_delta_percent": (new_median - old_median) / old_median * 100.0 if old_median else None,
                "before_digest": old["correctness_digest"],
                "after_digest": new["correctness_digest"],
                "semantic_digest_preserved": old["correctness_digest"] == new["correctness_digest"],
            }
        )
    report: dict[str, object] = {
        "schema_version": "sprint77-phase-a-comparison-v1",
        "before": str(before_path),
        "after": str(after_path),
        "workload_contract_equal": before["workload_contract"] == after["workload_contract"],
        "environment_equal_except_runtime": {
            "python": before["environment"]["python"] == after["environment"]["python"],
            "numpy": before["environment"]["versions"]["numpy"] == after["environment"]["versions"]["numpy"],
            "torch": before["environment"]["versions"]["torch"] == after["environment"]["versions"]["torch"],
            "seed": before["environment"]["seed"] == after["environment"]["seed"],
        },
        "rows": rows,
    }
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/sprint77_phase_a_comparison.json"))
    args = parser.parse_args()
    report = compare(args.before, args.after, args.output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

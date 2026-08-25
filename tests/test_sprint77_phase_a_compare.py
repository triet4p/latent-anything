"""Comparison contract tests for Phase-A before/after evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from scripts.sprint77_phase_a_compare import compare


def test_comparison_requires_same_workloads_and_preserves_digests(tmp_path: Path) -> None:
    report = {
        "environment": {"python": "3", "seed": 1, "versions": {"numpy": "2", "torch": "2"}},
        "workload_contract": {"seed": 1},
        "cases": [{"name": "x", "latency_us": {"median": 10.0}, "correctness_digest": "a"}],
    }
    before = tmp_path / "before.json"
    after = tmp_path / "after.json"
    before.write_text(json.dumps(report), encoding="utf-8")
    cases = cast(list[dict[str, object]], report["cases"])
    latency = cast(dict[str, object], cases[0]["latency_us"])
    latency["median"] = 5.0
    after.write_text(json.dumps(report), encoding="utf-8")
    result = compare(before, after)
    assert result["workload_contract_equal"] is True
    rows = cast(list[dict[str, object]], result["rows"])
    assert rows[0]["semantic_digest_preserved"] is True
    assert rows[0]["median_ratio_after_over_before"] == 0.5

"""Keep the M14 execution table anchored to real repository paths."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
M14 = ROOT / "docs" / "M14_REAL_SYSTEM_VALIDATION.md"
GAP_MAP = ROOT / "artifacts" / "task_78.38_gap_map.json"
EXECUTION_QUEUE = ROOT / "artifacts" / "task_79.1_execution_queue.json"
LANE_RE = re.compile(r"^\| (L\d{2}) \|")
PATH_RE = re.compile(r"(?:[A-Za-z0-9_.-]+/)*[A-Za-z0-9_.-]+\.(?:py|md|toml|yml|yaml)(?:::[A-Za-z0-9_]+)?")


def _resolve_table_path(token: str) -> Path | None:
    """Resolve a repository path token from the M14 table to this checkout."""
    path_token = token.split("::", 1)[0].replace("\\", "/")
    if not path_token.endswith((".py", ".md", ".toml", ".yml", ".yaml")):
        return None
    if path_token.startswith("artifacts/m14/"):
        return None
    if "*" in path_token:
        raise AssertionError(f"wildcard path is not executable: {token}")
    if path_token == "pyproject.toml" or path_token.startswith(
        ("tests/", "scripts/", "artifacts/", "docs/", ".github/")
    ):
        return ROOT / path_token
    if path_token.startswith("test_"):
        return ROOT / "tests" / path_token
    return ROOT / "src" / "latent_anything" / path_token


def _repo_path_tokens(code_span: str) -> tuple[str, ...]:
    """Extract supported repository-relative paths from a code span."""
    return tuple(PATH_RE.findall(code_span))


def _assert_existing_path(lane: str, token: str) -> None:
    """Validate one path and, when present, its pytest selector."""
    resolved = _resolve_table_path(token)
    if resolved is None:
        return
    assert resolved.is_file(), f"{lane}: missing M14 path {token}"
    if "::" in token:
        function_name = token.split("::", 1)[1]
        source = resolved.read_text(encoding="utf-8")
        assert re.search(rf"^(?:async\s+)?def {re.escape(function_name)}\s*\(", source, re.MULTILINE), (
            f"{lane}: missing pytest selector {token}"
        )


def test_missing_non_python_path_is_rejected() -> None:
    """The same guard used by the table audit must reject a missing Markdown path."""
    missing = "docs/__m14_missing_contract__.md"
    assert _resolve_table_path(missing) is not None
    with pytest.raises(AssertionError, match="missing M14 path"):
        _assert_existing_path("TEST", missing)


def test_m14_has_24_unique_lanes_and_existing_contract_paths() -> None:
    """Fail when a source, test, script, or workflow path silently drifts."""
    rows = {}
    for line in M14.read_text(encoding="utf-8").splitlines():
        match = LANE_RE.match(line)
        if match:
            lane = match.group(1)
            assert lane not in rows, f"duplicate M14 lane: {lane}"
            rows[lane] = line

    assert tuple(rows) == tuple(f"L{i:02d}" for i in range(1, 25))
    for lane, row in rows.items():
        cells = row.split("|")
        assert len(cells) >= 10, f"{lane}: malformed M14 table row"
        assert "uv run" in cells[5], f"{lane}: missing executable command"
        for code_span in re.findall(r"`([^`]+)`", row):
            for token in _repo_path_tokens(code_span):
                _assert_existing_path(lane, token)


def test_sprint79_queue_reconciles_gap_map_and_completed_statuses() -> None:
    """Keep queue records synchronized with gap status and real L01-L03 paths."""
    gap_map = json.loads(GAP_MAP.read_text(encoding="utf-8"))
    queue = json.loads(EXECUTION_QUEUE.read_text(encoding="utf-8"))
    gap_items = {item["id"]: item for item in gap_map["items"]}
    queue_rows = queue["execution_queue"]
    queue_ids = [row["record_id"] for row in queue_rows]

    assert len(gap_items) == len(queue_rows) == 40
    assert len(queue_ids) == len(set(queue_ids))
    assert set(queue_ids) == set(gap_items)
    assert all(row["current_evidence"] == gap_items[row["record_id"]]["status"] for row in queue_rows)

    qualifying = {row["record_id"] for row in queue_rows if row["current_evidence"] in {"D2", "D3"}}
    assert qualifying
    assert all(row["queue_status"] == "satisfied_qualifying" for row in queue_rows if row["record_id"] in qualifying)
    status_counts = Counter(row["queue_status"] for row in queue_rows)
    assert status_counts == Counter(
        {
            "satisfied_qualifying": 8,
            "ready_for_dependency_ordered_execution": 30,
            "co_scheduled_scc_blocked_by_missing_implementation": 2,
        }
    )
    assert queue["reconciliation"]["queue_status_counts"] == dict(status_counts)

    queue_text = EXECUTION_QUEUE.read_text(encoding="utf-8")
    assert "tests/test_latent_anything/test_clustering.py" not in queue_text
    assert "tests/test_latent_anything/test_probes.py" not in queue_text
    l03_rows = [row for row in queue_rows if row["lane_id"] == "L03"]
    assert len(l03_rows) == 3
    for row in l03_rows:
        assert row["command"] == (
            "uv run pytest tests/test_clustering.py tests/test_probes.py tests/test_mlp_probe.py "
            "tests/test_m14_l03_analysis.py -q; uv run python -m scripts.m14_l03_analysis --run-real"
        )
    assert all("memorization control fails as expected" not in row["acceptance"] for row in l03_rows)

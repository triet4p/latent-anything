"""Keep the M14 execution table anchored to real repository paths."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
M14 = ROOT / "docs" / "M14_REAL_SYSTEM_VALIDATION.md"
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

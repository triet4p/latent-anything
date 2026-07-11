"""Regression tests for the evidence-ledger contract."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_evidence_ledger_validator_accepts_repository_inventory() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/validate_evidence_ledger.py", "--json"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout
    payload = json.loads(result.stdout)
    assert len(payload["capabilities"]) >= 100
    assert payload["errors"] == []


def test_evidence_ledger_uses_only_d2_or_d3_for_stable_coverage() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/validate_evidence_ledger.py", "--json"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    assert payload["coverage"]["core"][0] == 0
    assert payload["coverage"]["overall"][0] == 0

"""Regression tests for the evidence-ledger contract."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import validate_evidence_ledger as evidence_ledger

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


def _validate_level(
    monkeypatch: pytest.MonkeyPatch, status: evidence_ledger.Status, roles: tuple[str, ...]
) -> list[str]:
    paths = {
        "source": "src/latent_anything/evaluation.py",
        "test": "tests/test_evaluation.py",
        "benchmark": "scripts/vae_explanation_benchmark.py",
        "config": "pyproject.toml",
        "artifact": "CHANGELOG.md",
    }
    capability_id = "THY-T01-SYNTHETIC"
    records = tuple(evidence_ledger.EvidenceRecord(role=role, path=paths[role]) for role in roles)
    payload: dict[str, object] = {
        "schema_version": 2,
        "contextual_background": {},
        "benchmark_only": [],
        "overrides": {capability_id: {"status": status, "evidence": []}},
    }
    monkeypatch.setattr(evidence_ledger, "_read_ledger", lambda: payload)
    capability = evidence_ledger.Capability(
        capability_id=capability_id,
        tier="T01",
        title="Synthetic",
        source_line=1,
        classification="implementation-applicable",
        status=status,
        evidence=records,
    )
    return evidence_ledger.validate_capabilities((capability,))


@pytest.mark.parametrize(
    ("status", "roles"),
    [
        ("D1", ("source", "test")),
        ("D2", ("source", "test", "benchmark", "config")),
        ("D3", ("source", "test", "benchmark", "config", "artifact")),
    ],
)
def test_evidence_levels_accept_all_required_roles(
    monkeypatch: pytest.MonkeyPatch, status: evidence_ledger.Status, roles: tuple[str, ...]
) -> None:
    assert _validate_level(monkeypatch, status, roles) == []


@pytest.mark.parametrize(
    ("status", "roles", "missing"),
    [
        ("D1", ("source",), "test"),
        ("D2", ("source", "test", "benchmark"), "config"),
        ("D3", ("source", "test", "benchmark", "config"), "artifact"),
    ],
)
def test_evidence_levels_reject_missing_required_roles(
    monkeypatch: pytest.MonkeyPatch,
    status: evidence_ledger.Status,
    roles: tuple[str, ...],
    missing: str,
) -> None:
    errors = _validate_level(monkeypatch, status, roles)
    assert any("lacks roles" in error and missing in error for error in errors)


def test_malformed_typed_evidence_is_reported_without_crashing() -> None:
    records = evidence_ledger.parse_evidence_records(
        "THY-T01-SYNTHETIC",
        [
            {"role": "invalid", "path": "pyproject.toml"},
            {"role": "source"},
            "src/latent_anything/evaluation.py",
        ],
    )
    assert len(records) == 3
    assert all(record.error is not None for record in records)

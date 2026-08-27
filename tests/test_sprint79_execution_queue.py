"""Regression tests for the deterministic Sprint 79 execution queue."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.build_sprint79_execution_queue import (
    effective_ledger_statuses,
    resolve_prerequisite_status,
)

ROOT = Path(__file__).resolve().parents[1]


def test_external_queue_prerequisites_resolve_to_qualifying_ledger_statuses() -> None:
    queue = json.loads((ROOT / "artifacts/task_79.1_execution_queue.json").read_text(encoding="utf-8"))
    external = queue["reconciliation"]["external_prerequisites"]
    assert len(external) == 9
    assert {row["effective_status"] for row in external} == {"D2"}
    assert {row["ledger_section"] for row in external} == {"overrides"}
    assert {row["resolution"] for row in external} == {"satisfied_qualifying"}
    assert (
        effective_ledger_statuses()["THY-T08-JEPA-JOINT-EMBEDDING-PREDICTIVE-ARCHITECTURE-LECUN-2022"][
            "effective_status"
        ]
        == "D2"
    )
    gap_ids = {row["record_id"] for row in queue["execution_queue"]}
    for row in queue["execution_queue"]:
        assert set(row["internal_dependencies"]).issubset(gap_ids)
    assert {row["record_id"] for row in queue["execution_queue"] if row["external_prerequisites"]} == {
        row["record_id"] for row in external
    }


def test_external_prerequisite_statuses_distinguish_unsatisfied_and_missing() -> None:
    assert resolve_prerequisite_status("D3") == "satisfied_qualifying"
    assert resolve_prerequisite_status("D2") == "satisfied_qualifying"
    assert resolve_prerequisite_status("D1") == "unsatisfied"
    assert resolve_prerequisite_status("D0") == "unsatisfied"
    assert resolve_prerequisite_status(None) == "unresolved_missing_ledger_record"
    assert resolve_prerequisite_status("unexpected") == "unresolved_missing_ledger_record"


def test_l05_dependency_cycle_is_one_coscheduled_scc_with_blocker() -> None:
    queue = json.loads((ROOT / "artifacts/task_79.1_execution_queue.json").read_text(encoding="utf-8"))
    groups = queue["reconciliation"]["dependency_cycle_groups"]
    assert groups == [
        {
            "group_id": "SCC-L05-01",
            "records": [
                "THY-T03-NORMALIZING-FLOWS",
                "THY-T04-DENSITY-ESTIMATION-TRONG-LATENT",
            ],
            "co_scheduled": True,
            "lane_id": "L05",
            "shared_artifact": ["artifacts/m14/l05-density.json"],
            "implementation_blocker": (
                "Normalizing Flows has no stable implementation; shared L05 artifact does not promote either claim."
            ),
        }
    ]
    statuses = {
        row["queue_status"] for row in queue["execution_queue"] if row["lane_id"] == "L05" and row["cycle_conflict"]
    }
    assert statuses == {"co_scheduled_scc_blocked_by_missing_implementation"}

"""Build the deterministic Sprint 79 queue from the M14 contract and gap map."""

from __future__ import annotations

import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
GAP_MAP = ROOT / "artifacts/task_78.38_gap_map.json"
M14 = ROOT / "docs/M14_REAL_SYSTEM_VALIDATION.md"
LEDGER = ROOT / "docs/evidence-ledger.json"
OUTPUT = ROOT / "artifacts/task_79.1_execution_queue.json"
REQUIRED_ITEM_FIELDS = (
    "id",
    "status",
    "target",
    "lane",
    "prerequisites",
    "command",
    "tests",
    "acceptance",
    "artifact",
    "blocker",
    "depends_on",
)


def _parse_m14_lanes() -> list[dict[str, Any]]:
    lanes: list[dict[str, Any]] = []
    for line in M14.read_text(encoding="utf-8").splitlines():
        if not re.match(r"^\| L\d{2} \|", line):
            continue
        fields = [field.strip() for field in line.strip().strip("|").split("|")]
        if len(fields) != 8:
            raise ValueError(f"M14 lane row has {len(fields)} fields: {line}")
        lane_id, surface, source_tests, target_backend, environment, acceptance, resources, status = fields
        artifact_match = re.search(r"artifacts/m14/[^` ;|]+", acceptance)
        lanes.append(
            {
                "lane_id": lane_id,
                "surface": surface,
                "source_tests_evidence": source_tests,
                "real_target_backend": target_backend,
                "environment_command": environment,
                "acceptance_artifact": acceptance,
                "artifact_path": artifact_match.group(0) if artifact_match else None,
                "resources_network_cleanup": resources,
                "status_blocker_owner": status,
            }
        )
    return lanes


def effective_ledger_statuses() -> dict[str, dict[str, str]]:
    """Resolve exact prerequisite IDs from the authoritative ledger."""
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    statuses: dict[str, dict[str, str]] = {}
    for section, records in ledger.items():
        if not isinstance(records, dict):
            continue
        for record_id, record in records.items():
            if isinstance(record, dict) and isinstance(record.get("status"), str):
                statuses[record_id] = {"effective_status": record["status"], "section": section}
    return statuses


def resolve_prerequisite_status(effective_status: str | None) -> str:
    """Classify one external prerequisite without collapsing distinct states."""
    if effective_status in {"D2", "D3"}:
        return "satisfied_qualifying"
    if effective_status in {"D0", "D1"}:
        return "unsatisfied"
    return "unresolved_missing_ledger_record"


def _dependency_order(
    items: list[dict[str, Any]],
    lane_index: dict[str, int],
    ledger_statuses: dict[str, dict[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, str]], list[list[str]]]:
    by_id = {item["id"]: item for item in items}
    internal_adjacency: dict[str, list[str]] = {
        item["id"]: [dependency for dependency in item["depends_on"] if dependency in by_id] for item in items
    }
    cycle_groups: list[list[str]] = []

    def visit_cycles(record_id: str, active: list[str], completed: set[str]) -> None:
        if record_id in active:
            cycle = active[active.index(record_id) :]
            normalized = sorted(cycle)
            if normalized not in cycle_groups:
                cycle_groups.append(normalized)
            return
        if record_id in completed:
            return
        for dependency in internal_adjacency[record_id]:
            visit_cycles(dependency, [*active, record_id], completed)
        completed.add(record_id)

    completed: set[str] = set()
    for record_id in by_id:
        visit_cycles(record_id, [], completed)
    cycle_nodes = {record_id for group in cycle_groups for record_id in group}
    external: list[dict[str, str]] = []
    for item in items:
        for dependency in item["depends_on"]:
            if dependency not in by_id:
                ledger_record = ledger_statuses.get(dependency)
                effective_status = ledger_record["effective_status"] if ledger_record else None
                external.append(
                    {
                        "record_id": item["id"],
                        "prerequisite_id": dependency,
                        "effective_status": effective_status or "missing",
                        "ledger_section": ledger_record["section"] if ledger_record else "missing",
                        "resolution": (resolve_prerequisite_status(effective_status)),
                    }
                )

    memo: dict[str, int] = {}

    def depth(record_id: str, trail: tuple[str, ...] = ()) -> int:
        if record_id in memo:
            return memo[record_id]
        internal = [
            dependency
            for dependency in by_id[record_id]["depends_on"]
            if dependency in by_id and not (record_id in cycle_nodes and dependency in cycle_nodes)
        ]
        value = 1 + max((depth(d, (*trail, record_id)) for d in internal), default=0)
        memo[record_id] = value
        return value

    ordered = sorted(
        items,
        key=lambda item: (depth(item["id"]), lane_index[item["lane"]], item["id"]),
    )
    queue: list[dict[str, Any]] = []
    external_by_record: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in external:
        external_by_record[row["record_id"]].append(row)
    for position, item in enumerate(ordered, start=1):
        queue.append(
            {
                "position": position,
                "record_id": item["id"],
                "lane_id": item["lane"],
                "current_evidence": item["status"],
                "target_evidence": item["target"],
                "dependency_depth": depth(item["id"]),
                "internal_dependencies": [dependency for dependency in item["depends_on"] if dependency in by_id],
                "external_prerequisites": [
                    detail["prerequisite_id"] for detail in external_by_record.get(item["id"], [])
                ],
                "external_prerequisite_details": external_by_record.get(item["id"], []),
                "cycle_conflict": next((group for group in cycle_groups if item["id"] in group), None),
                "queue_status": (
                    "co_scheduled_scc_blocked_by_missing_implementation"
                    if item["id"] in cycle_nodes
                    else "requires_prerequisite_resolution"
                    if any(
                        detail["resolution"] != "satisfied_qualifying"
                        for detail in external_by_record.get(item["id"], [])
                    )
                    else "ready_for_dependency_ordered_execution"
                ),
                "command": item["command"],
                "acceptance": item["acceptance"],
                "artifact": item["artifact"],
                "blocker": item["blocker"],
            }
        )
    return queue, external, cycle_groups


def main() -> None:
    gap_map = json.loads(GAP_MAP.read_text(encoding="utf-8"))
    items = gap_map["items"]
    items_by_id = {item["id"]: item for item in items}
    if len(items) != 40 or len({item["id"] for item in items}) != 40:
        raise ValueError("gap map must contain exactly 40 unique records")
    for item in items:
        missing = [field for field in REQUIRED_ITEM_FIELDS if field not in item]
        if missing:
            raise ValueError(f"{item.get('id', '<unknown>')} missing {missing}")

    lanes = _parse_m14_lanes()
    expected_lanes = [f"L{index:02d}" for index in range(1, 25)]
    actual_lanes = [lane["lane_id"] for lane in lanes]
    if actual_lanes != expected_lanes:
        raise ValueError(f"M14 lanes are not exactly L01-L24: {actual_lanes}")
    lane_index = {lane_id: index for index, lane_id in enumerate(actual_lanes)}
    gap_ids_by_lane: dict[str, list[str]] = defaultdict(list)
    for item in items:
        if item["lane"] not in lane_index:
            raise ValueError(f"gap record {item['id']} references unknown lane {item['lane']}")
        gap_ids_by_lane[item["lane"]].append(item["id"])
    for lane in lanes:
        lane["gap_record_ids"] = sorted(gap_ids_by_lane.get(lane["lane_id"], []))
        lane["gap_record_count"] = len(lane["gap_record_ids"])

    ledger_statuses = effective_ledger_statuses()
    execution_queue, external, cycle_groups = _dependency_order(items, lane_index, ledger_statuses)
    source_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    output = {
        "schema_version": "sprint79-execution-queue-v1",
        "purpose": (
            "Deterministic dependency-ordered queue reconciling the 40 Sprint 78.38 gap records with all 24 M14 lanes."
        ),
        "source_commit": source_sha,
        "source_documents": [
            "docs/M14_REAL_SYSTEM_VALIDATION.md",
            "docs/EVIDENCE_GAP_PLAN.md",
            "artifacts/task_78.38_gap_map.json",
        ],
        "reconciliation": {
            "gap_records": len(items),
            "unique_gap_records": len({item["id"] for item in items}),
            "m14_lanes": len(lanes),
            "lane_ids": actual_lanes,
            "mapped_lanes": sum(1 for lane in lanes if lane["gap_record_count"]),
            "unmapped_m14_lanes": [lane["lane_id"] for lane in lanes if not lane["gap_record_count"]],
            "execution_records": len(execution_queue),
            "external_prerequisite_edges": len(external),
            "external_prerequisites": external,
            "dependency_cycle_groups": [
                {
                    "group_id": f"SCC-L05-{index:02d}",
                    "records": group,
                    "co_scheduled": True,
                    "lane_id": items_by_id[group[0]]["lane"],
                    "shared_artifact": sorted({items_by_id[record_id]["artifact"] for record_id in group}),
                    "implementation_blocker": (
                        "Normalizing Flows has no stable implementation; shared L05 artifact "
                        "does not promote either claim."
                    ),
                }
                for index, group in enumerate(cycle_groups, start=1)
            ],
            "external_prerequisite_resolution": {
                resolution: sum(1 for row in external if row["resolution"] == resolution)
                for resolution in (
                    "satisfied_qualifying",
                    "unsatisfied",
                    "unresolved_missing_ledger_record",
                )
            },
        },
        "execution_policy": gap_map["default_execution_policy"],
        "lane_contract": lanes,
        "execution_queue": execution_queue,
        "headline_blockers": gap_map["separate_headline_blockers"],
    }
    OUTPUT.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        f"wrote {OUTPUT.relative_to(ROOT)}: {len(items)} records, "
        f"{len(lanes)} lanes, {len(external)} external prerequisites"
    )


if __name__ == "__main__":
    main()

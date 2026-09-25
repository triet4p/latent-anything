"""Fail-closed preflight for the depth-first stable-release workflow."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "docs" / "sprint-plans" / "sprint-81.md"
DEPTH_REPORT_PATH = ROOT / "docs" / "SPRINT_80_DEPTH_EVIDENCE.md"
RELEASE_GATES_PATH = ROOT / "docs" / "release-gates.json"

_TASK_LINE = re.compile(r"^- \[([ x~])\] (.+)$")
_REQUIRED_PRE_RELEASE_TASKS = (
    (1, "Confirm the Sprint 80 diagnostic-depth gate"),
    (2, "Enforce the depth-first stop-before-release contract"),
    (3, "Finalize version metadata"),
    (4, "Build wheel/sdist from a clean checkout"),
    (8, "State semantic-versioning"),
)
_REQUIRED_EXTERNAL_PREREQUISITES = (
    "release-tag-ruleset",
    "pypi-trusted-publisher",
)
_CORE_DEPTH_GATES = (
    "Capture and provenance",
    "Detection and controls",
    "Localization",
    "Explanation",
    "Causal evidence and comparison",
    "Persistence and reproducibility",
    "Usability",
)
_REPORTING_EVIDENCE = "deterministic rendered report"


def _sprint_tasks(plan: str) -> list[tuple[str, str]]:
    """Return checkbox state and title for tasks in the Sprint 81 task section."""
    section = re.search(r"(?ms)^## Atomic Tasks\s*\n(?P<body>.*?)(?=^## |\Z)", plan)
    if section is None:
        return []
    tasks: list[tuple[str, str]] = []
    for line in section.group("body").splitlines():
        match = _TASK_LINE.fullmatch(line)
        if match is not None:
            tasks.append((match.group(1), match.group(2)))
    return tasks


def _depth_section(report: str) -> str | None:
    """Return the bounded-core gate table and interpretation section."""
    sections = re.search(
        r"(?ms)^## Stable-depth gate interpretation and excluded claims\s*\n(?P<body>.*?)(?=^## |\Z)",
        report,
    )
    return None if sections is None else sections.group("body")


def _external_prerequisite_blockers(manifest_text: str) -> list[str]:
    """Fail closed on missing or unresolved external release prerequisites."""
    try:
        manifest = json.loads(manifest_text)
    except json.JSONDecodeError:
        return ["External release prerequisite manifest is not valid JSON"]
    if not isinstance(manifest, dict) or manifest.get("schema") != "latent-anything-release-prerequisites-v1":
        return ["External release prerequisite manifest has an unknown schema"]

    prerequisites = manifest.get("external_prerequisites")
    if not isinstance(prerequisites, dict):
        return ["External release prerequisite manifest has no prerequisite map"]

    blockers: list[str] = []
    for name in _REQUIRED_EXTERNAL_PREREQUISITES:
        if name not in prerequisites:
            blockers.append(f"Required external prerequisite is missing: {name}")
    for name, prerequisite in prerequisites.items():
        if not isinstance(name, str) or not isinstance(prerequisite, dict):
            blockers.append("External release prerequisite entry is malformed")
            continue
        status = prerequisite.get("status")
        if status != "ready":
            blockers.append(f"External release prerequisite is not ready: {name} (status: {status})")
        evidence = prerequisite.get("evidence")
        if not isinstance(evidence, str) or not evidence.strip():
            blockers.append(f"External release prerequisite has no activation evidence: {name}")
        if name == "pypi-trusted-publisher" and prerequisite.get("environment") != "pypi":
            blockers.append("PyPI Trusted Publisher must use the GitHub Actions pypi environment")

    return blockers


def release_readiness_blockers(plan: str, depth_report: str, release_gates: str) -> list[str]:
    """Return every open pre-release prerequisite or unsupported depth signoff."""
    blockers = _external_prerequisite_blockers(release_gates)
    tasks = _sprint_tasks(plan)
    for task_number, expected_title in _REQUIRED_PRE_RELEASE_TASKS:
        if task_number > len(tasks):
            blockers.append(f"Sprint 81 task {task_number} is missing; release prerequisite mapping needs review")
            continue
        state, title = tasks[task_number - 1]
        if not title.startswith(expected_title):
            blockers.append(f"Sprint 81 task {task_number} changed; release prerequisite mapping needs review")
            continue
        if state != "x":
            blockers.append(f"Sprint 81 task {task_number} is not complete: {title}")

    if "**Sprint 80 diagnostic-depth gate:** **PASS — no unresolved blocker for the bounded supported ordinary-DL core.**" not in depth_report:
        blockers.append("Sprint 80 bounded ordinary-DL depth gate is not explicitly signed off PASS")
    if "This is not `1.0.0` publication approval." not in depth_report:
        blockers.append("Sprint 80 report no longer distinguishes bounded-core signoff from release approval")
    if _REPORTING_EVIDENCE not in depth_report.lower():
        blockers.append("Supported bounded-core reporting evidence is missing")

    gate_section = _depth_section(depth_report)
    if gate_section is None:
        blockers.append("Sprint 80 stable-depth gate disposition table is missing")
    else:
        for gate in _CORE_DEPTH_GATES:
            if re.search(rf"(?m)^\|\s*{re.escape(gate)}\s*\|\s*PASS\b", gate_section) is None:
                blockers.append(f"Supported bounded-core depth gate is not PASS: {gate}")
    if re.search(r"(?im)^.*SmolVLA.*blocked.*non-gating.*$", depth_report) is None:
        blockers.append("Sprint 80 report must preserve SmolVLA as blocked and non-gating")

    return blockers


def main() -> int:
    """Check the current checkout's release evidence and return a process status."""
    try:
        plan = PLAN_PATH.read_text(encoding="utf-8")
        depth_report = DEPTH_REPORT_PATH.read_text(encoding="utf-8")
        release_gates = RELEASE_GATES_PATH.read_text(encoding="utf-8")
    except OSError as error:
        print(f"Release blocked: required evidence cannot be read: {error}", file=sys.stderr)
        return 1

    blockers = release_readiness_blockers(plan, depth_report, release_gates)
    if blockers:
        print("Release blocked; no release tag or publication is authorized:", file=sys.stderr)
        for blocker in blockers:
            print(f"- {blocker}", file=sys.stderr)
        return 1

    print(
        "Release preflight passed for the signed-off bounded ordinary-DL core; "
        "package and workflow checks must still pass in this release job."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

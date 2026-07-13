"""Validate the local, read-only theory capability evidence ledger."""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, cast

Classification = Literal["implementation-applicable", "benchmark-only", "contextual-background"]
Status = Literal["D0", "D1", "D2", "D3"]
EvidenceRole = Literal["source", "test", "benchmark", "config", "artifact"]

ROOT = Path(__file__).resolve().parents[1]
THEORY_PATH = ROOT / "docs" / "THEORY.md"
LEDGER_PATH = ROOT / "docs" / "evidence-ledger.json"
_TIER_PATTERN = re.compile(r"^#{2,} Tầng (?P<number>\d+|bổ sung)(?P<suffix>[A-Z]?)")
_TOPIC_PATTERN = re.compile(r"^- \[[ x~]\] \*\*(?P<title>.+?)\*\*")


@dataclass(frozen=True)
class EvidenceRecord:
    """One typed local evidence link from the schema-v2 ledger."""

    role: str
    path: str
    error: str | None = None


@dataclass(frozen=True)
class Capability:
    """One stable capability inventory record derived from the theory index."""

    capability_id: str
    tier: str
    title: str
    source_line: int
    classification: Classification
    status: Status
    evidence: tuple[EvidenceRecord, ...]


def _slug(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9]+", "-", normalized.lower())).strip("-").upper()


def _read_ledger() -> dict[str, object]:
    return cast(dict[str, object], json.loads(LEDGER_PATH.read_text(encoding="utf-8")))


def _classification(capability_id: str, ledger: dict[str, object]) -> Classification:
    contextual = set(cast(dict[str, str], ledger["contextual_background"]))
    benchmark = set(cast(list[str], ledger["benchmark_only"]))
    if capability_id in contextual:
        return "contextual-background"
    if capability_id in benchmark:
        return "benchmark-only"
    return "implementation-applicable"


def parse_evidence_records(capability_id: str, raw_evidence: object) -> tuple[EvidenceRecord, ...]:
    """Parse evidence without letting malformed ledger data crash validation."""

    if not isinstance(raw_evidence, list):
        return (EvidenceRecord(role="", path="", error=f"{capability_id} evidence must be a list."),)
    evidence_items = cast(list[object], raw_evidence)
    records: list[EvidenceRecord] = []
    valid_roles: set[str] = {"source", "test", "benchmark", "config", "artifact"}
    for index, item in enumerate(evidence_items):
        prefix = f"{capability_id} evidence[{index}]"
        if not isinstance(item, dict):
            records.append(EvidenceRecord(role="", path="", error=f"{prefix} must be an object with role and path."))
            continue
        record = cast(dict[object, object], item)
        role = record.get("role")
        path = record.get("path")
        errors: list[str] = []
        if not isinstance(role, str) or role not in valid_roles:
            errors.append(f"role must be one of {sorted(valid_roles)}")
        if not isinstance(path, str) or not path.strip():
            errors.append("path must be a non-empty string")
        unexpected = sorted(str(key) for key in record if key not in {"role", "path"})
        if unexpected:
            errors.append(f"unexpected fields: {unexpected}")
        records.append(
            EvidenceRecord(
                role=role if isinstance(role, str) else "",
                path=path if isinstance(path, str) else "",
                error=f"{prefix} {'; '.join(errors)}." if errors else None,
            )
        )
    return tuple(records)


def load_capabilities() -> tuple[Capability, ...]:
    """Return the complete capability inventory without modifying repository files."""

    ledger = _read_ledger()
    overrides = cast(dict[str, dict[str, object]], ledger["overrides"])
    current_tier: str | None = None
    supplementary_tier = 0
    capabilities: list[Capability] = []
    for line_number, line in enumerate(THEORY_PATH.read_text(encoding="utf-8").splitlines(), start=1):
        if tier_match := _TIER_PATTERN.match(line):
            if tier_match["number"] == "bổ sung":
                supplementary_tier += 1
                current_tier = f"X{supplementary_tier:02d}"
            else:
                current_tier = f"T{tier_match['number'].zfill(2)}{tier_match['suffix']}"
            continue
        if topic_match := _TOPIC_PATTERN.match(line):
            if current_tier is None:
                raise ValueError(f"Theory topic without a tier at line {line_number}.")
            title = topic_match["title"]
            capability_id = f"THY-{current_tier}-{_slug(title)}"
            override = overrides.get(capability_id, {})
            status = cast(Status, override.get("status", "D0"))
            evidence = parse_evidence_records(capability_id, override.get("evidence", []))
            capabilities.append(
                Capability(
                    capability_id=capability_id,
                    tier=current_tier,
                    title=title,
                    source_line=line_number,
                    classification=_classification(capability_id, ledger),
                    status=status,
                    evidence=evidence,
                )
            )
    return tuple(capabilities)


def validate_capabilities(capabilities: tuple[Capability, ...]) -> list[str]:
    """Return validation errors for the full inventory and its local evidence."""

    ledger = _read_ledger()
    known_ids = {capability.capability_id for capability in capabilities}
    errors: list[str] = []
    if ledger.get("schema_version") != 2:
        errors.append("Evidence ledger schema_version must be 2.")
    if len(known_ids) != len(capabilities):
        errors.append("Duplicate capability IDs were derived from docs/THEORY.md.")
    contextual = cast(dict[str, str], ledger["contextual_background"])
    all_configured_ids = set(contextual)
    all_configured_ids.update(cast(list[str], ledger["benchmark_only"]))
    all_configured_ids.update(cast(dict[str, object], ledger["overrides"]).keys())
    for stale_id in sorted(all_configured_ids - known_ids):
        errors.append(f"Ledger references a stale capability ID: {stale_id}")
    for capability_id, rationale in contextual.items():
        if not rationale.strip():
            errors.append(f"{capability_id} has no contextual-background rationale.")
    for capability in capabilities:
        if capability.status not in {"D0", "D1", "D2", "D3"}:
            errors.append(f"{capability.capability_id} has invalid status {capability.status!r}.")
        if capability.classification == "contextual-background" and capability.status != "D0":
            errors.append(f"{capability.capability_id} is contextual background and must remain D0.")
        if capability.status != "D0" and not capability.evidence:
            errors.append(f"{capability.capability_id} is {capability.status} but has no evidence links.")
        for record in capability.evidence:
            if record.error is not None:
                errors.append(record.error)
            elif not (ROOT / record.path).is_file():
                errors.append(f"{capability.capability_id} links to missing evidence: {record.path}")
        required_roles: set[str] = set()
        if capability.status in {"D1", "D2", "D3"}:
            required_roles.update({"source", "test"})
        if capability.status in {"D2", "D3"}:
            required_roles.update({"benchmark", "config"})
        if capability.status == "D3":
            required_roles.add("artifact")
        roles = {record.role for record in capability.evidence if record.error is None}
        missing_roles = required_roles - roles
        if missing_roles:
            errors.append(f"{capability.capability_id} is {capability.status} but lacks roles: {sorted(missing_roles)}")
    return errors


def coverage_summary(capabilities: tuple[Capability, ...]) -> dict[str, tuple[int, int, float]]:
    """Calculate stable coverage using the exact denominator in the ledger policy."""

    policy = cast(dict[str, object], _read_ledger()["coverage_policy"])
    core_tiers = set(cast(list[str], policy["core_tiers"]))
    qualifying = set(cast(list[str], policy["qualifying_statuses"]))

    def summarize(items: list[Capability]) -> tuple[int, int, float]:
        denominator = len(items)
        numerator = sum(item.status in qualifying for item in items)
        percentage = numerator / denominator if denominator else 0.0
        return numerator, denominator, percentage

    applicable = [item for item in capabilities if item.classification != "contextual-background"]
    core = [item for item in applicable if item.tier in core_tiers]
    return {"core": summarize(core), "overall": summarize(applicable)}


def main() -> int:
    """Print the current coverage and fail non-zero when ledger integrity is broken."""

    # TextIO's platform-specific ``reconfigure`` is not present in typeshed.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Emit the derived inventory and coverage as JSON.")
    args = parser.parse_args()
    capabilities = load_capabilities()
    errors = validate_capabilities(capabilities)
    summary = coverage_summary(capabilities)
    if args.json:
        print(
            json.dumps(
                {
                    "capabilities": [asdict(capability) for capability in capabilities],
                    "coverage": summary,
                    "errors": errors,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print(f"Inventory: {len(capabilities)} capabilities")
        for name, (numerator, denominator, percentage) in summary.items():
            print(f"{name}: {numerator}/{denominator} ({percentage:.1%})")
        for error in errors:
            print(f"ERROR: {error}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())

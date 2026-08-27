"""M14 L03 provenance, independent verdict, and failure-envelope helpers."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import re
import subprocess
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from scripts.m14_l03_plan import EXPECTED_GAP_IDS, EXPECTED_RECORD_IDS, plan_digest

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SHA1_RE = re.compile(r"^[0-9a-f]{40}$")


def _json_safe(value: Any) -> Any:
    """Convert numeric arrays/scalars at the artifact boundary."""
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def artifact_digest(value: Mapping[str, Any]) -> str:
    unsigned = dict(value)
    unsigned.pop("artifact_sha256", None)
    payload = (json.dumps(unsigned, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n").encode()
    return hashlib.sha256(payload).hexdigest()


def record_digest(value: Mapping[str, Any]) -> str:
    """Digest a report component without a self-referential field."""
    unsigned = dict(value)
    unsigned.pop("run_record_sha256", None)
    payload = (
        json.dumps(_json_safe(unsigned), ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def source_digests() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    names = ("m14_l03_analysis.py", "m14_l03_plan.py", "m14_l03_data.py", "m14_l03_metrics.py", "m14_l03_envelope.py")
    return {
        "runner_source_sha256": hashlib.sha256((root / names[0]).read_bytes()).hexdigest(),
        "contract_source_sha256": hashlib.sha256((root / names[1]).read_bytes()).hexdigest(),
        "implementation_source_sha256": hashlib.sha256(
            b"".join((root / name).read_bytes() for name in names)
        ).hexdigest(),
    }


def runtime_versions() -> dict[str, str]:
    """Collect installed runtime versions without guessing missing packages."""
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as package_version

    versions = {"python": platform.python_version(), "platform": platform.platform()}
    for package in ("numpy", "scipy", "scikit-learn", "torch", "transformers"):
        try:
            versions[package] = package_version(package)
        except PackageNotFoundError:
            versions[package] = "not-installed"
    try:
        import torch

        versions["cuda"] = str(torch.version.cuda or "not-available")
    except ImportError:
        versions["cuda"] = "not-available"
    return versions


def validate_artifact(
    artifact: Mapping[str, Any], plan: Mapping[str, Any], current_sources: Mapping[str, str] | None = None
) -> list[str]:
    errors: list[str] = []
    required_fields = {
        "artifact_sha256",
        "plan_sha256",
        "split",
        "records",
        "provenance",
        "accepted_record_ids",
        "accepted_gap_ids",
    }
    if not required_fields.issubset(artifact):
        errors.append("artifact required fields are incomplete")
    if artifact.get("schema_version") != "m14-l03-analysis-artifact-v1" or artifact.get("lane") != "M14-L03":
        errors.append("artifact identity is not M14-L03")
    records = artifact.get("records")
    if (
        not isinstance(records, list)
        or tuple(r.get("record_id") for r in records if isinstance(r, Mapping)) != EXPECTED_RECORD_IDS
    ):
        errors.append("artifact records must be in declared order")
    if isinstance(records, list):
        for index, record in enumerate(records):
            if (
                index >= len(EXPECTED_GAP_IDS)
                or not isinstance(record, Mapping)
                or record.get("gap_id") != EXPECTED_GAP_IDS[index]
            ):
                errors.append(f"record {index} gap mapping is invalid")
            if (
                index
                and isinstance(record, Mapping)
                and record.get("accepted") is True
                and (not isinstance(records[index - 1], Mapping) or records[index - 1].get("accepted") is not True)
            ):
                errors.append(f"record {index} accepted despite blocked dependency")
        accepted_records = [
            str(record["record_id"])
            for record in records
            if isinstance(record, Mapping) and record.get("accepted") is True
        ]
        accepted_gaps = [
            str(record["gap_id"])
            for record in records
            if isinstance(record, Mapping) and record.get("accepted") is True
        ]
        if artifact.get("accepted_record_ids") != accepted_records or artifact.get("accepted_gap_ids") != accepted_gaps:
            errors.append("artifact accepted IDs do not match independent record predicates")
        expected_level = "D2" if accepted_records else "D1"
        if artifact.get("evidence_level") != expected_level or artifact.get("partial_promotion") is not True:
            errors.append("artifact evidence level or partial-promotion marker is invalid")
    if artifact.get("plan_sha256") != plan_digest(plan):
        errors.append("artifact plan digest does not match plan")
    if not isinstance(artifact.get("artifact_sha256"), str) or artifact["artifact_sha256"] != artifact_digest(artifact):
        errors.append("artifact self-digest is invalid")
    provenance = artifact.get("provenance")
    if not isinstance(provenance, Mapping) or not SHA1_RE.fullmatch(str(provenance.get("git_sha", ""))):
        errors.append("artifact provenance must include a committed SHA")
    required_provenance = {
        "runner_source_sha256",
        "contract_source_sha256",
        "implementation_source_sha256",
        "model_id",
        "model_revision",
        "tokenizer",
        "runtime_versions",
        "resources",
        "cleanup",
    }
    if not isinstance(provenance, Mapping) or not required_provenance.issubset(provenance):
        errors.append("artifact provenance contract is incomplete")
    elif any(
        not isinstance(provenance.get(field), str) or not SHA256_RE.fullmatch(provenance[field])
        for field in ("runner_source_sha256", "contract_source_sha256", "implementation_source_sha256")
    ):
        errors.append("artifact source digests must be SHA-256 values")
    if current_sources is not None and isinstance(provenance, Mapping):
        for field in ("runner_source_sha256", "contract_source_sha256", "implementation_source_sha256"):
            if provenance.get(field) != current_sources.get(field):
                errors.append(f"artifact {field} does not match current source")
    if not isinstance(artifact.get("split"), Mapping) or artifact["split"].get("group_overlap") != {
        "train_val": 0,
        "train_test": 0,
        "val_test": 0,
    }:
        errors.append("artifact split must prove zero group overlap")
    if not _all_finite(artifact):
        errors.append("artifact metrics must be finite")
    return errors


def _all_finite(value: Any) -> bool:
    if isinstance(value, Mapping):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_finite(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def failure_envelope(
    plan: Mapping[str, Any],
    error: BaseException,
    *,
    phase: str = "remote-real-run",
    model_attempt: Mapping[str, Any] | None = None,
    resources: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a non-promoting, auditable failure report in memory."""
    return {
        "schema_version": "m14-l03-failure-v1",
        "lane": "M14-L03",
        "status": "failed",
        "phase": phase,
        "command": "uv run python -m scripts.m14_l03_analysis --run-real",
        "git_sha": git_sha(),
        "source_digests": source_digests(),
        "model_attempt": dict(model_attempt or {}),
        "resources": dict(resources or {}),
        "error_type": type(error).__name__,
        "error": str(error),
        "plan_sha256": plan_digest(plan),
        "accepted_record_ids": [],
        "accepted_gap_ids": [],
        "run_record_status": "failed",
        "artifact_written": False,
        "timestamp_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    }


def git_sha() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()


def build_artifact(
    plan: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    split: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    accepted = [dict(r) for r in records if r.get("accepted") is True]
    payload: dict[str, Any] = {
        "schema_version": "m14-l03-analysis-artifact-v1",
        "lane": "M14-L03",
        "evidence_level": "D2" if accepted else "D1",
        "partial_promotion": True,
        "accepted_record_ids": [str(r["record_id"]) for r in accepted],
        "accepted_gap_ids": [str(r["gap_id"]) for r in accepted],
        "records": [_json_safe(dict(r)) for r in records],
        "split": dict(split),
        "provenance": dict(provenance),
        "plan_sha256": plan_digest(plan),
    }
    payload["artifact_sha256"] = artifact_digest(payload)
    return payload


def build_run_record(
    plan: Mapping[str, Any], artifact: Mapping[str, Any], resources: Mapping[str, Any]
) -> dict[str, Any]:
    """Build a provenance-complete run record without persisting it."""
    sources = source_digests()
    result = {
        "schema_version": "m14-l03-analysis-run-v1",
        "lane": "M14-L03",
        "artifact_name": "artifacts/m14/l03-analysis.json",
        "artifact_sha256": artifact.get("artifact_sha256"),
        "plan_sha256": plan_digest(plan),
        "git_sha": artifact.get("provenance", {}).get("git_sha"),
        **sources,
        "command": "uv run python -m scripts.m14_l03_analysis --run-real",
        "timestamp_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "status": "completed",
        "resource_measurement": dict(resources),
        "cleanup": "disposable clone and caches removed after report capture",
        "accepted_record_ids": list(artifact.get("accepted_record_ids", [])),
        "accepted_gap_ids": list(artifact.get("accepted_gap_ids", [])),
    }
    result["run_record_sha256"] = record_digest(result)
    return result


def build_report(plan: Mapping[str, Any], artifact: Mapping[str, Any], run_record: Mapping[str, Any]) -> dict[str, Any]:
    """Build the single JSON object captured before remote cleanup."""
    return {
        "schema_version": "m14-l03-report-v1",
        "status": "success",
        "plan_sha256": plan_digest(plan),
        "artifact": dict(artifact),
        "run_record": dict(run_record),
        "run_record_sha256": record_digest(run_record),
    }


def validate_report(report: Mapping[str, Any], plan: Mapping[str, Any]) -> list[str]:
    """Validate success envelope linkage and both nested records."""
    errors: list[str] = []
    artifact = report.get("artifact")
    run_record = report.get("run_record")
    if report.get("schema_version") != "m14-l03-report-v1" or report.get("status") != "success":
        errors.append("report identity/status is invalid")
    if not isinstance(artifact, Mapping) or not isinstance(run_record, Mapping):
        return errors + ["success report must contain artifact and run_record objects"]
    errors.extend(validate_artifact(artifact, plan, source_digests()))
    if report.get("plan_sha256") != plan_digest(plan):
        errors.append("report plan digest is invalid")
    if report.get("run_record_sha256") != record_digest(run_record):
        errors.append("report run-record digest is invalid")
    if run_record.get("artifact_sha256") != artifact.get("artifact_sha256"):
        errors.append("run record does not link to artifact digest")
    if run_record.get("plan_sha256") != plan_digest(plan):
        errors.append("run record plan digest is invalid")
    if run_record.get("status") != "completed" or not SHA1_RE.fullmatch(str(run_record.get("git_sha", ""))):
        errors.append("run record status or git SHA is invalid")
    if run_record.get("accepted_record_ids") != artifact.get("accepted_record_ids"):
        errors.append("run record accepted IDs do not link to artifact")
    if run_record.get("git_sha") != artifact.get("provenance", {}).get("git_sha"):
        errors.append("run record git SHA does not link to artifact")
    current = source_digests()
    for field in ("runner_source_sha256", "contract_source_sha256", "implementation_source_sha256"):
        if run_record.get(field) != current.get(field):
            errors.append(f"run record {field} does not match current source")
    return errors


def apply_dependency_blocking(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Block each downstream record after the first failed dependency."""
    output: list[dict[str, Any]] = []
    prior_accepted = True
    for record in records:
        item = dict(record)
        accepted = bool(item.get("accepted")) and prior_accepted
        if item.get("accepted") is True and not accepted:
            item["blocked_by_dependency"] = True
        item["accepted"] = accepted
        item["verdict"] = "accepted" if accepted else "failed"
        output.append(item)
        prior_accepted = accepted
    return output

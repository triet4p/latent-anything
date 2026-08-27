"""Declarative plan loading and schema validation for M14 L02."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

PLAN_PATH = Path(__file__).resolve().parents[1] / "artifacts/m14/l02-geometry.plan.json"
SCHEMA_VERSION = "m14-l02-geometry-plan-v1"
EXPECTED_GAP_IDS = (
    "THY-T01-MANIFOLD-HYPOTHESIS",
    "THY-T03-SLERP-SPHERICAL-LINEAR-INTERPOLATION",
    "THY-T04-LERP-LINEAR-INTERPOLATION",
    "THY-T03-RIEMANNIAN-GEOMETRY-CO-BAN",
    "THY-T04-SLERP",
    "THY-T06-TRAJECTORY-SIMILARITY-METRICS",
)
EXPECTED_RECORD_IDS = (
    "manifold_hypothesis",
    "slerp_spherical",
    "lerp_euclidean",
    "riemannian_density_geodesic",
    "slerp_latent_operation",
    "trajectory_similarity_dtw",
)


def section(plan: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = plan.get(name)
    if not isinstance(value, Mapping):
        raise ValueError(f"L02 plan section {name!r} must be an object")
    return value


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def plan_digest(plan: Mapping[str, Any]) -> str:
    unsigned = dict(plan)
    unsigned.pop("plan_sha256", None)
    return hashlib.sha256(_canonical_json(unsigned)).hexdigest()


def _validate_records(plan: Mapping[str, Any], errors: list[str]) -> None:
    records = plan.get("records")
    if not isinstance(records, list) or len(records) != len(EXPECTED_RECORD_IDS):
        errors.append("plan must contain exactly six independent records")
        return
    record_ids: list[str] = []
    gap_ids: list[str] = []
    for record in records:
        if not isinstance(record, Mapping):
            errors.append("each record must be an object")
            continue
        record_id = record.get("record_id")
        record_ids.append(str(record_id))
        if record.get("verdict") != "not-run" or record.get("accepted") is not None:
            errors.append(f"record {record_id!r} must remain not-run")
        mapped = record.get("gap_ids")
        if not isinstance(mapped, list) or len(mapped) != 1:
            errors.append(f"record {record_id!r} must map exactly one gap")
        else:
            gap_ids.extend(str(item) for item in mapped)
    if tuple(record_ids) != EXPECTED_RECORD_IDS:
        errors.append("records must remain in the declared deterministic order")
    if tuple(gap_ids) != EXPECTED_GAP_IDS:
        errors.append("records must map the six gaps exactly once")


def validate_plan(plan: Mapping[str, Any]) -> list[str]:
    """Validate the design-only plan and its predeclared thresholds."""
    required = {
        "schema_version",
        "lane",
        "status",
        "evidence_level",
        "accepted",
        "artifact_contract",
        "execution",
        "artifact_schema",
        "run_record_schema",
        "data",
        "model",
        "density",
        "baseline_contract",
        "trajectory_semantics",
        "provenance_contract",
        "records",
        "plan_sha256",
    }
    errors = [f"missing required field: {field}" for field in sorted(required - set(plan))]
    if errors:
        return errors
    if plan["schema_version"] != SCHEMA_VERSION or plan["lane"] != "M14-L02":
        errors.append("plan identity is not M14-L02")
    if plan["status"] != "design-in-review" or plan["evidence_level"] != "not-run" or plan["accepted"] is not False:
        errors.append("plan must remain design-in-review and not-run")
    artifact = plan["artifact_contract"]
    if (
        not isinstance(artifact, Mapping)
        or artifact.get("accepted_artifact") != "artifacts/m14/l02-geometry.json"
        or artifact.get("design_artifacts_are_not_evidence") is not True
        or artifact.get("shared_across_records") is not True
    ):
        errors.append("artifact contract must identify one shared future artifact")
    artifact_schema = plan["artifact_schema"]
    if (
        not isinstance(artifact_schema, Mapping)
        or not isinstance(artifact_schema.get("required_fields"), list)
        or not {
            "artifact_sha256",
            "dataset",
            "model",
            "density",
            "backend_versions",
            "input_digests",
            "provenance",
            "records",
        }.issubset(artifact_schema["required_fields"])
    ):
        errors.append("artifact schema must declare provenance and independent-record fields")
    run_schema = plan["run_record_schema"]
    if (
        not isinstance(run_schema, Mapping)
        or not isinstance(run_schema.get("required_fields"), list)
        or not {
            "git_sha",
            "artifact_sha256",
            "plan_sha256",
            "runner_source_sha256",
            "contract_source_sha256",
            "timestamp_utc",
        }.issubset(run_schema["required_fields"])
    ):
        errors.append("run-record schema must declare committed provenance fields")
    execution = plan["execution"]
    if not isinstance(execution, Mapping):
        errors.append("execution contract must be an object")
    else:
        if int(execution.get("path_points", 0)) < 3:
            errors.append("path_points must support bounded interpolation")
        if int(execution.get("trajectory_query_points", 0)) <= int(execution.get("path_points", 0)):
            errors.append("trajectory query must be unequal and longer than its reference")
        if not isinstance(execution.get("geodesic"), Mapping) or not isinstance(execution.get("dtw"), Mapping):
            errors.append("execution contract must declare geodesic and DTW options")
    data = plan["data"]
    if (
        not isinstance(data, Mapping)
        or data.get("dataset") != "sklearn.datasets.load_digits"
        or "held-out real images" not in str(data.get("heldout_role", ""))
        or "physical trajectory" not in str(data.get("heldout_role", ""))
    ):
        errors.append("data contract must use held-out real images without a physical-trajectory claim")
    model = plan["model"]
    if not isinstance(model, Mapping) or model.get("adapter") != "latent_anything.adapters.conv_vae.ConvVAE":
        errors.append("model contract must use the existing ConvVAE")
    elif model.get("fit_scope") != "train images only":
        errors.append("ConvVAE fit must be train-only")
    density = plan["density"]
    if not isinstance(density, Mapping) or "train latents only" not in str(density.get("fit_scope", "")):
        errors.append("density fit must be train-only")
    baselines = plan["baseline_contract"]
    if not isinstance(baselines, Mapping) or not {"chance", "shuffled", "strong", "trajectory_alignment"}.issubset(
        baselines
    ):
        errors.append("baseline contract must include chance, shuffled, strong, and trajectory controls")
    semantics = plan["trajectory_semantics"]
    if not isinstance(semantics, Mapping) or "not available" not in str(semantics.get("frechet", "")):
        errors.append("Fréchet semantics must remain unavailable")
    provenance = plan["provenance_contract"]
    if (
        not isinstance(provenance, Mapping)
        or provenance.get("git_sha") != "deferred"
        or provenance.get("runner_source_sha256") != "deferred"
    ):
        errors.append("design provenance must remain deferred")
    _validate_records(plan, errors)
    if not isinstance(plan.get("plan_sha256"), str) or plan["plan_sha256"] != plan_digest(plan):
        errors.append("plan_sha256 does not match canonical unsigned plan")
    return errors


def load_plan(path: Path = PLAN_PATH) -> dict[str, Any]:
    """Load and validate the checked-in plan without writing files."""
    plan = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(plan, dict):
        raise ValueError("L02 plan must be a JSON object")
    errors = validate_plan(plan)
    if errors:
        raise ValueError("invalid L02 plan: " + "; ".join(errors))
    return plan

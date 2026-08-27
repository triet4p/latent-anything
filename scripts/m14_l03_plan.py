"""Immutable, predeclared design contract for the M14 L03 benchmark."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

PLAN_PATH = Path(__file__).resolve().parents[1] / "artifacts/m14/l03-analysis.plan.json"
SCHEMA_VERSION = "m14-l03-analysis-plan-v1"
CANONICAL_COMMAND = "uv run python -m scripts.m14_l03_analysis --run-real"
EXPECTED_RECORD_IDS = ("t03_latent_linear_structure", "t05_linear_probe", "t05_mlp_probe")
EXPECTED_GAP_IDS = (
    "THY-T03-LINEAR-STRUCTURE-TRONG-LATENT",
    "THY-T05-LINEAR-PROBING",
    "THY-T05-NONLINEAR-PROBING",
)


def _canonical(value: Mapping[str, Any]) -> bytes:
    unsigned = dict(value)
    unsigned.pop("plan_sha256", None)
    return (json.dumps(unsigned, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n").encode()


def plan_digest(plan: Mapping[str, Any]) -> str:
    """Hash the unsigned, canonical plan representation."""
    return hashlib.sha256(_canonical(plan)).hexdigest()


def section(plan: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = plan.get(name)
    if not isinstance(value, Mapping):
        raise ValueError(f"L03 plan section {name!r} must be an object")
    return value


def validate_plan(plan: Mapping[str, Any]) -> list[str]:
    """Validate identity, fixed controls, predicates, and dependency order."""
    required = {
        "schema_version",
        "lane",
        "status",
        "evidence_level",
        "accepted",
        "model",
        "data",
        "split",
        "prompt_contract",
        "analysis",
        "controls",
        "records",
        "artifact_contract",
        "artifact_schema",
        "run_record_schema",
        "report_schema",
        "provenance_contract",
        "plan_sha256",
    }
    errors = [f"missing required field: {k}" for k in sorted(required - set(plan))]
    if errors:
        return errors
    if (plan["schema_version"], plan["lane"]) != (SCHEMA_VERSION, "M14-L03"):
        errors.append("plan identity is not M14-L03")
    if (plan["status"], plan["evidence_level"], plan["accepted"]) != ("design-approved", "not-run", False):
        errors.append("plan must remain design-approved and not-run")
    model = section(plan, "model")
    if model.get("integration") != "latent_anything.integrations.transformer_lm.TransformerLMIntegration":
        errors.append("model must name the concrete TransformerLMIntegration")
    if model.get("model_id") != "openai-community/gpt2" or len(str(model.get("revision", ""))) != 40:
        errors.append("model must use the immutable GPT-2 revision")
    if model.get("device") != "cuda" or model.get("inference_batch_size") != 8:
        errors.append("real validation device must be remote CUDA")
    report_schema = section(plan, "report_schema")
    if report_schema.get("schema_version") != "m14-l03-report-v1" or "artifact" not in report_schema.get(
        "required_fields", []
    ):
        errors.append("stdout report schema must link artifact and run record")
    split = section(plan, "split")
    if (split.get("method"), split.get("n_splits"), split.get("seed")) != ("StratifiedGroupKFold", 5, 79):
        errors.append("split must be deterministic five-fold StratifiedGroupKFold seed 79")
    if split.get("partition_folds") != {"train": [0, 1, 2], "val": [3], "test": [4]}:
        errors.append("split must declare 60/20/20 fold assignment")
    data = section(plan, "data")
    if "held-out test labels are never used" not in str(data.get("fit_scope")):
        errors.append("data contract must prohibit test-label fitting and model selection")
    prompts = section(plan, "prompt_contract")
    if prompts.get("max_length") != 64 or prompts.get("pooling") != "attention-mask mean":
        errors.append("prompt max length/pooling contract changed")
    analysis = section(plan, "analysis")
    if analysis.get("pca_components") != 32 or analysis.get("layers") != [0, 4, 8, 12]:
        errors.append("analysis layer/PCA configuration changed")
    if not isinstance(analysis.get("kmeans"), Mapping) or analysis["kmeans"].get("fit_scope") != "train rows only":
        errors.append("KMeans diagnostic must fit train rows only")
    intervals = section(plan, "controls")
    if (intervals.get("bootstrap_resamples"), intervals.get("bootstrap_seed")) != (10000, 7901):
        errors.append("paired held-out bootstrap must remain B=10000 seed 7901")
    records = plan.get("records")
    if (
        not isinstance(records, list)
        or tuple(r.get("record_id") for r in records if isinstance(r, Mapping)) != EXPECTED_RECORD_IDS
    ):
        errors.append("records must be the three exact L03 records in dependency order")
    else:
        for index, record in enumerate(records):
            if not isinstance(record, Mapping) or record.get("gap_id") != EXPECTED_GAP_IDS[index]:
                errors.append(f"record {index} has wrong gap mapping")
            if record.get("depends_on") != ([] if index == 0 else [EXPECTED_RECORD_IDS[index - 1]]):
                errors.append(f"record {index} has wrong dependency")
            if record.get("verdict") != "not-run" or record.get("accepted") is not None:
                errors.append(f"record {index} must remain not-run")
    artifact = section(plan, "artifact_contract")
    if (
        artifact.get("accepted_artifact") != "artifacts/m14/l03-analysis.json"
        or artifact.get("design_plan_is_not_evidence") is not True
    ):
        errors.append("artifact contract must keep the plan separate from evidence")
    provenance = section(plan, "provenance_contract")
    if provenance.get("command") != CANONICAL_COMMAND or provenance.get("git_sha") != "deferred-until-phase-a-commit":
        errors.append("provenance must declare the canonical command and deferred commit SHA")
    if not isinstance(plan.get("plan_sha256"), str) or plan["plan_sha256"] != plan_digest(plan):
        errors.append("plan_sha256 does not match canonical unsigned plan")
    return errors


def load_plan(path: Path = PLAN_PATH) -> dict[str, Any]:
    """Load the immutable design plan without writing anything."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("L03 plan must be a JSON object")
    errors = validate_plan(value)
    if errors:
        raise ValueError("invalid L03 plan: " + "; ".join(errors))
    return value

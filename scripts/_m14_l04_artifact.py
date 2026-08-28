"""Unpromoting artifact builders for the L04 dispatcher."""

from __future__ import annotations

from typing import Any

from scripts._m14_l04_boundary import INTEGRATION_FACTORY
from scripts._m14_l04_digest import canonical_digest, code_sha, source_digests
from scripts.m14_l04_contract import plan_digest

TCAV_RECORD_ID = "t05_tcav"
TCAV_GAP_ID = "THY-T05-CONCEPT-ACTIVATION-VECTORS-TCAV-KIM-ET-AL-2018"


def _record_template(plan: dict[str, Any], item: dict[str, Any], status: str) -> dict[str, Any]:
    token = plan["tokenization_and_sampling"]
    return {
        "record_id": item["record_id"],
        "capability": item["capability"],
        "evidence_level": "D0",
        "status": status,
        "layer": token["hidden_layer"],
        "native_hidden_state_index": token["native_hidden_state_index"],
        "token_ids": {},
        "seed": token["seeds"][0],
        "metrics": {},
        "confidence_intervals": {},
        "controls": {},
        "acceptance": False,
        "failure_ref": None,
    }


def execution_template(plan: dict[str, Any], use_case: str, status: str) -> dict[str, Any]:
    item = next(case for case in plan["real_use_case_checklist"] if case["use_case"] == use_case)
    return {
        "use_case": use_case,
        "record_id": item["record_id"],
        "support_only": item["support_only"],
        "model": item["model"],
        "integration": item["integration"],
        "adapter": item["adapter"],
        "status": status,
        "evidence_eligible": False,
        "acceptance": False,
        "metrics": {},
        "controls": {},
        "failure_ref": None,
    }


def build_artifact(
    plan: dict[str, Any],
    fixture: dict[str, Any],
    use_case: str,
    status: str,
    failure_ref: str | None,
    *,
    injected: bool = False,
    execution_result: dict[str, Any] | None = None,
    resources: dict[str, Any] | None = None,
) -> dict[str, Any]:
    executions = [
        execution_template(
            plan,
            name,
            status if name == use_case else ("blocked_missing_corpus" if name == "TunedLogitLens" else "not_run"),
        )
        for name in (case["use_case"] for case in plan["real_use_case_checklist"])
    ]
    current = next(item for item in executions if item["use_case"] == use_case)
    current["failure_ref"] = failure_ref
    if execution_result is not None and not injected:
        for key in (
            "status",
            "evidence_eligible",
            "acceptance",
            "evidence_level",
            "metrics",
            "controls",
            "control_raw",
            "token_ids",
            "layer",
            "native_hidden_state_index",
            "seeds",
        ):
            if key in execution_result:
                current[key] = execution_result[key]
    accepted_tcav = bool(
        use_case == "TCAV"
        and not injected
        and execution_result is not None
        and execution_result.get("status") == "passed_real_cuda"
        and execution_result.get("evidence_eligible") is True
        and execution_result.get("acceptance") is True
    )
    if accepted_tcav:
        current["evidence_level"] = "D3"
        current["acceptance"] = True
    records = []
    for item in plan["record_order"]:
        record_status = (
            status
            if item["record_id"] == current["record_id"]
            else ("blocked_missing_corpus" if item["record_id"] == "THY-T05-LOGIT-LENS-TUNED-LENS" else "not_run")
        )
        record = _record_template(plan, item, record_status)
        if item["record_id"] == current["record_id"]:
            record["failure_ref"] = failure_ref
        if execution_result is not None and not injected and item["record_id"] == current["record_id"]:
            for key in (
                "status",
                "evidence_level",
                "metrics",
                "confidence_intervals",
                "controls",
                "acceptance",
                "token_ids",
                "layer",
                "native_hidden_state_index",
                "seed",
            ):
                if key in execution_result:
                    record[key] = execution_result[key]
        records.append(record)
    artifact = {
        "schema_version": "m14-l04-explanations-artifact-v1",
        "lane": "L04",
        "use_case": use_case,
        "accepted_gap_ids": [TCAV_GAP_ID] if accepted_tcav else [],
        "accepted_record_ids": [TCAV_RECORD_ID] if accepted_tcav else [],
        "evidence_level": "D3" if accepted_tcav else "D0",
        "partial_promotion": True,
        "model": plan["model"],
        "integration": "TransformerLMIntegration",
        "adapter": "N/A",
        "fixture": fixture,
        "tokenization": plan["tokenization_and_sampling"],
        "split": {
            "train_groups": plan["fixture"]["split"]["train_groups"],
            "holdout_groups": plan["fixture"]["split"]["holdout_groups"],
            "group_overlap": 0,
        },
        "executions": executions,
        "records": records,
        "controls": {
            "thresholds_and_controls": plan["thresholds_and_controls"],
            "evaluation": "not_run_by_dispatcher",
        },
        "provenance": {
            **source_digests(),
            "git_sha": code_sha(),
            "model_id": plan["model"]["id"],
            "model_revision": plan["model"]["revision"],
            "integration": "TransformerLMIntegration",
            "integration_factory": INTEGRATION_FACTORY,
            "adapter": "N/A",
            "evidence_origin": "dependency-injected-offline"
            if injected
            else (
                "real-cuda"
                if execution_result is not None or (resources or {}).get("device") == "cuda"
                else "dispatcher-only-no-model"
            ),
            "network": (resources or {}).get("network", "not attempted"),
            "credentials": "not used",
            "cleanup": (resources or {}).get("cleanup", "not applicable; no model was loaded"),
            "use_case": use_case,
            "plan_sha256": plan_digest(plan),
        },
        "plan_sha256": plan_digest(plan),
    }
    if execution_result is not None and not injected:
        provenance = artifact["provenance"]
        if isinstance(provenance, dict):
            provenance.update(execution_result.get("provenance", {}))
        artifact["raw_summaries"] = execution_result.get("raw_summaries", [])
        artifact["seeds"] = execution_result.get("seeds", [])
        artifact["token_ids"] = execution_result.get("token_ids", {})
    artifact["artifact_sha256"] = canonical_digest(artifact, "artifact_sha256")
    return artifact

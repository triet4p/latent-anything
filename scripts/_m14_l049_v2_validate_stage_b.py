"""Independent Stage B validator for L04.9 v2."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any, cast

from scripts._m14_l049_v2_fixture import validate_fixture
from scripts._m14_l049_v2_schema import (
    PAIRED_SHUFFLED_THRESHOLD,
    RECOVERY_THRESHOLD,
    STAGE_B_SEEDS,
    V2_STAGE_B_SCHEMA,
    VALIDATION_REJECTION_CODES,
    CommitmentPolicy,
    candidate_grid,
    canonical_fixture_bytes,
    directional_recovery,
    fixture_digest,
    is_digest,
)
from scripts._m14_l049_v2_validate_common import (
    addendum_errors,
    canonical_artifact_digest,
    groups,
    metric,
    real_resources,
    runtime_attestation_errors,
    safe_int,
    same_metric,
)
from scripts._m14_l049_v2_validate_stage_a import validate_stage_a_impl


def _mapping_digest(mapping: Mapping[str, str]) -> str:
    return hashlib.sha256(
        canonical_fixture_bytes([{"source": source, "donor": mapping[source]} for source in sorted(mapping)])
    ).hexdigest()


def _label_stratified_mapping(pair_ids: Sequence[str], labels: Mapping[str, int]) -> dict[str, str]:
    """Independently reproduce the producer's deterministic donor mapping."""
    names = sorted(str(pair) for pair in pair_ids)
    if len(names) != 24 or len(set(names)) != 24 or set(labels) != set(names):
        return {}
    if any(label not in {0, 1} for label in labels.values()):
        return {}
    buckets: dict[int, list[str]] = {0: [], 1: []}
    for name in names:
        buckets[labels[name]].append(name)
    if any(len(bucket) < 2 for bucket in buckets.values() if bucket):
        return {}
    return {name: bucket[(index + 1) % len(bucket)] for bucket in buckets.values() for index, name in enumerate(bucket)}


def validate_stage_b_impl(
    artifact: Mapping[str, Any],
    holdout_rows: Sequence[Mapping[str, Any]],
    holdout_seed: bytes,
    candidate_artifact: Mapping[str, Any],
    addendum: Mapping[str, Any],
    train_rows: Sequence[Mapping[str, Any]],
    *,
    policy: CommitmentPolicy,
    validation_context: object = None,
) -> list[str]:
    if validation_context is not None:
        from scripts._m14_l049_v2_validation_context import context_is_valid

        if not context_is_valid(validation_context):
            return ["Stage B validation context is invalid"]
    errors = addendum_errors(addendum, policy)
    errors.extend(validate_fixture(train_rows, holdout_rows))
    holdout_groups, group_errors = groups(holdout_rows, "holdout", 24)
    errors.extend(group_errors)
    if len(holdout_seed) != 32:
        errors.append("Stage B holdout seed is not exactly 256 bits")
    if set(artifact) != {
        "schema_version",
        "stage",
        "status",
        "evidence_level",
        "evidence_eligible",
        "promotion_candidate",
        "acceptance",
        "failure_kind",
        "failure",
        "evaluation_complete",
        "repository_promotion",
        "candidate_artifact_sha256",
        "parent_plan_sha256",
        "addendum_schema",
        "train_fixture_sha256",
        "source_sha256",
        "addendum_sha256",
        "holdout_fixture_sha256",
        "holdout_seed_commitment_sha256",
        "shuffled_mapping",
        "shuffled_mapping_sha256",
        "seed_summaries",
        "controls",
        "resources",
        "runtime_attestation",
        "attestation_sha256",
        "artifact_sha256",
    }:
        errors.append("Stage B artifact fields are invalid")
    if artifact.get("schema_version") != V2_STAGE_B_SCHEMA or artifact.get("stage") != "stage_b_holdout_evaluation":
        errors.append("Stage B schema or stage is invalid")
    candidate_errors = validate_stage_a_impl(
        candidate_artifact,
        train_rows,
        addendum,
        policy=policy,
        validation_context=validation_context,
    )
    if (
        candidate_errors
        or candidate_artifact.get("status") not in {"protocol_fixture", "stage_a_complete"}
        or candidate_artifact.get("selection", {}).get("oof_metric", {}).get("pass") is not True
    ):
        errors.append("Stage B candidate is not a complete passing Stage A artifact")
    chain_fields = {
        "candidate_artifact_sha256": candidate_artifact.get("artifact_sha256"),
        "parent_plan_sha256": policy.parent_plan_sha256,
        "addendum_schema": policy.expected_addendum().get("schema_version"),
        "train_fixture_sha256": candidate_artifact.get("train_fixture_sha256"),
        "source_sha256": candidate_artifact.get("source_sha256"),
        "addendum_sha256": addendum.get("addendum_sha256"),
    }
    for field, expected in chain_fields.items():
        if artifact.get(field) != expected:
            errors.append(f"Stage B {field} commitment chain is invalid")
    if artifact.get("holdout_fixture_sha256") != fixture_digest(holdout_rows):
        errors.append("Stage B holdout fixture digest is invalid")
    commitment = addendum.get("fixture")
    if isinstance(commitment, Mapping):
        if artifact.get("holdout_fixture_sha256") != commitment.get("holdout_content_sha256"):
            errors.append("Stage B holdout fixture commitment mismatch")
        expected_seed_digest = hashlib.sha256(holdout_seed).hexdigest()
        if artifact.get("holdout_seed_commitment_sha256") != expected_seed_digest or artifact.get(
            "holdout_seed_commitment_sha256"
        ) != commitment.get("holdout_seed_commitment_sha256"):
            errors.append("Stage B holdout seed commitment mismatch")
    if artifact.get("failure_kind") in {"runtime_exception", "validation_rejected"}:
        failure = artifact.get("failure")
        expected_failure = (
            {"validation_codes"} if artifact.get("failure_kind") == "validation_rejected" else {"exception_type"}
        )
        if (
            artifact.get("status") != "stage_b_failed"
            or artifact.get("evidence_level") != "D0"
            or artifact.get("evidence_eligible") is not False
            or artifact.get("promotion_candidate") is not False
            or artifact.get("acceptance") is not False
            or not isinstance(failure, Mapping)
            or set(failure) != expected_failure
            or artifact.get("seed_summaries") != []
            or artifact.get("evaluation_complete") is not False
        ):
            errors.append("Stage B runtime-failure envelope is invalid")
        elif artifact.get("failure_kind") == "validation_rejected":
            codes = failure.get("validation_codes")
            if (
                not isinstance(codes, list)
                or not codes
                or any(not isinstance(code, str) or code not in VALIDATION_REJECTION_CODES for code in codes)
            ):
                errors.append("Stage B validation-rejection codes are invalid")
        elif not isinstance(failure.get("exception_type"), str) or not failure.get("exception_type"):
            errors.append("Stage B runtime-failure envelope is invalid")
        resources = artifact.get("resources")
        errors.extend(real_resources(resources, allow_failure=True))
        errors.extend(
            runtime_attestation_errors(
                artifact.get("runtime_attestation"),
                stage="stage_b_holdout_evaluation",
                mode="real",
                group_count=24,
                pair_count=24,
                candidate_count=len(candidate_artifact.get("selection", {}).get("candidate_grid", [])),
                seed_count=len(STAGE_B_SEEDS),
                fixture_sha256=artifact.get("holdout_fixture_sha256"),
                candidate_sha256=artifact.get("candidate_artifact_sha256"),
                source_sha256=artifact.get("source_sha256"),
                addendum_sha256=artifact.get("addendum_sha256"),
                execution_resources=resources,
                partial_failure=True,
                validation_context=validation_context,
            )
        )
        if artifact.get("artifact_sha256") != canonical_artifact_digest(artifact, "artifact_sha256"):
            errors.append("Stage B artifact digest is invalid")
        return errors
    if artifact.get("evaluation_complete") is not True:
        errors.append("Stage B successful/evaluation discriminator is invalid")
    mapping_value = artifact.get("shuffled_mapping")
    mapping: Mapping[str, str] = cast(Mapping[str, str], mapping_value) if isinstance(mapping_value, Mapping) else {}
    pair_ids = sorted({str(row.get("causal_pair_id")) for row in holdout_rows})
    labels = {
        str(row["causal_pair_id"]): int(row["factor_labels"]["clean_label"])
        for row in holdout_rows
        if row.get("condition") == "clean" and isinstance(row.get("factor_labels"), Mapping)
    }
    expected_mapping = _label_stratified_mapping(pair_ids, labels)
    if mapping != expected_mapping or set(mapping) != set(pair_ids) or set(mapping.values()) != set(pair_ids):
        errors.append("Stage B shuffled mapping domain/range is invalid")
    if any(source == donor for source, donor in mapping.items()):
        errors.append("Stage B shuffled mapping contains self donors")
    if artifact.get("shuffled_mapping_sha256") != (_mapping_digest(expected_mapping) if expected_mapping else None):
        errors.append("Stage B shuffled mapping digest is invalid")
    summaries = artifact.get("seed_summaries")
    if not isinstance(summaries, list) or [
        item.get("seed") if isinstance(item, Mapping) else None for item in summaries
    ] != list(STAGE_B_SEEDS):
        errors.append("Stage B seed summaries are not the exact ordered seed list")
        summaries = []
    derived_passes: list[bool] = []
    expected_pairs = set(pair_ids)
    expected_groups = set(holdout_groups)
    for summary in summaries:
        if not isinstance(summary, Mapping):
            errors.append("Stage B seed summary is malformed")
            continue
        seed = safe_int(summary.get("seed"))
        if seed is None:
            errors.append("Stage B seed summary seed is invalid")
            continue
        if summary.get("shuffled_mapping") != mapping:
            errors.append("Stage B seed mapping binding is invalid")
        evidence = summary.get("evidence")
        if not isinstance(evidence, list) or len(evidence) != 24:
            errors.append("Stage B evidence coverage is invalid")
            continue
        true_by_group: dict[str, list[float]] = {}
        shuffled_by_group: dict[str, list[float]] = {}
        seen: set[str] = set()
        for item in evidence:
            if not isinstance(item, Mapping):
                errors.append("Stage B evidence item is malformed")
                continue
            pair, group = item.get("pair_id"), item.get("group_id")
            if (
                not isinstance(pair, str)
                or not isinstance(group, str)
                or pair in seen
                or pair not in expected_pairs
                or group not in expected_groups
            ):
                errors.append("Stage B evidence pair/group domain is invalid")
                continue
            expected_group = next(
                (str(row["group_id"]) for row in holdout_rows if str(row.get("causal_pair_id")) == pair), None
            )
            if group != expected_group:
                errors.append("Stage B evidence pair/group linkage is invalid")
            seen.add(pair)
            true = directional_recovery(
                item.get("clean_margin"), item.get("corrupted_margin"), item.get("patched_margin")
            )
            shuffled = directional_recovery(
                item.get("clean_margin"), item.get("corrupted_margin"), item.get("shuffled_margin")
            )
            if (
                true is None
                or shuffled is None
                or item.get("recovery") != true
                or item.get("shuffled_recovery") != shuffled
            ):
                errors.append("Stage B recovery was not independently recomputed")
                continue
            if item.get("shuffled_donor_pair_id") != mapping.get(pair):
                errors.append("Stage B evidence donor mapping is invalid")
            true_by_group.setdefault(group, []).append(true)
            shuffled_by_group.setdefault(group, []).append(shuffled)
            zero = item.get("zero_strength")
            if (
                not isinstance(zero, Mapping)
                or zero.get("identity") is not True
                or not is_digest(zero.get("selected_logit_digest"))
                or not is_digest(zero.get("relevant_output_digest"))
                or zero.get("selected_logit_digest") != zero.get("corrupted_selected_logit_digest")
                or zero.get("relevant_output_digest") != zero.get("corrupted_relevant_output_digest")
            ):
                errors.append("Stage B zero-strength identity control failed")
            controls = item.get("controls")
            if (
                not isinstance(controls, Mapping)
                or set(controls) != {"wrong_token", "adjacent_layer", "additive", "matched_norm_random"}
                or any(not isinstance(controls[key], Mapping) for key in controls)
            ):
                errors.append("Stage B diagnostic controls are not separately serialized")
        if seen != expected_pairs or set(true_by_group) != expected_groups:
            errors.append("Stage B evidence pair/group membership is incomplete")
            continue
        true_values = [sum(true_by_group[group]) / len(true_by_group[group]) for group in sorted(expected_groups)]
        shuffled_values = [
            sum(shuffled_by_group[group]) / len(shuffled_by_group[group]) for group in sorted(expected_groups)
        ]
        paired = [true - shuffled for true, shuffled in zip(true_values, shuffled_values, strict=True)]
        expected_recovery = metric(true_values, seed, RECOVERY_THRESHOLD)
        expected_paired = metric(paired, seed + 1, PAIRED_SHUFFLED_THRESHOLD)
        if not same_metric(summary.get("recovery"), expected_recovery) or not same_metric(
            summary.get("paired_true_minus_shuffled"), expected_paired
        ):
            errors.append("Stage B seed metrics were not independently recomputed")
        if expected_recovery is not None and expected_paired is not None:
            derived_passes.append(bool(expected_recovery["pass"] and expected_paired["pass"]))
    acceptance = len(derived_passes) == len(STAGE_B_SEEDS) and all(derived_passes)
    if artifact.get("acceptance") is not acceptance:
        errors.append("Stage B acceptance is not bound to recomputed metrics")
    expected_status = "stage_b_complete" if acceptance else "stage_b_failed"
    if artifact.get("status") != expected_status:
        errors.append("Stage B status is not bound to acceptance")
    controls = artifact.get("controls")
    expected_controls = {
        "scope": "diagnostics_only",
        "wrong_token": "separately_serialized",
        "adjacent_layer": "separately_serialized",
        "additive": "separately_serialized",
        "matched_norm_random": "separately_serialized",
        "zero_strength": "exact selected-logit and relevant-output digest identity",
    }
    if controls != expected_controls:
        errors.append("Stage B top-level controls are invalid")
    resources = artifact.get("resources")
    backend = resources.get("execution_backend") if isinstance(resources, Mapping) else None
    if backend in {"synthetic", "cpu"}:
        if resources != {
            "stage": "synthetic_fixture",
            "execution_backend": backend,
            "execution_attempted": False,
            "no_mutation": True,
        }:
            errors.append("Stage B synthetic resources are not the exact D0 contract")
    elif backend == "cuda":
        errors.extend(real_resources(resources, require_measured=acceptance))
    else:
        errors.append("Stage B execution backend is invalid")
    expected_level = (
        "D2" if acceptance and backend == "cuda" and not real_resources(resources, require_measured=True) else "D0"
    )
    if artifact.get("evidence_level") != expected_level or artifact.get("evidence_eligible") is not (False):
        errors.append("Stage B evidence level is invalid")
    if artifact.get("promotion_candidate") is not (expected_level == "D2"):
        errors.append("Stage B promotion-candidate status is invalid")
    if artifact.get("repository_promotion") is not False:
        errors.append("Stage B repository promotion must remain false")
    mode = "real" if backend == "cuda" else "synthetic" if backend in {"cpu", "synthetic"} else "invalid"
    errors.extend(
        runtime_attestation_errors(
            artifact.get("runtime_attestation"),
            stage="stage_b_holdout_evaluation",
            mode=mode,
            group_count=24,
            pair_count=24,
            candidate_count=len(candidate_grid()),
            seed_count=len(STAGE_B_SEEDS),
            fixture_sha256=artifact.get("holdout_fixture_sha256"),
            candidate_sha256=artifact.get("candidate_artifact_sha256"),
            source_sha256=artifact.get("source_sha256"),
            addendum_sha256=artifact.get("addendum_sha256"),
            execution_resources=artifact.get("resources"),
            validation_context=validation_context,
        )
    )
    attestation = artifact.get("runtime_attestation")
    if not isinstance(attestation, Mapping) or artifact.get("attestation_sha256") != attestation.get(
        "attestation_sha256"
    ):
        errors.append("Stage B attestation binding is invalid")
    if artifact.get("artifact_sha256") != canonical_artifact_digest(artifact, "artifact_sha256"):
        errors.append("Stage B artifact digest is invalid")
    return errors


__all__ = ["validate_stage_b_impl"]

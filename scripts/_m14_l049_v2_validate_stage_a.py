"""Independent Stage A validator for L04.9 v2."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

import numpy as np

from scripts._m14_l049_v2_schema import (
    STAGE_A_FAILURE_KINDS,
    TRAIN_GROUP_COUNT,
    V2_STAGE_A_SCHEMA,
    VALIDATION_REJECTION_CODES,
    CommitmentPolicy,
    candidate_grid,
    canonical_json_bytes,
    digest_bytes,
    directional_recovery,
    fixture_digest,
    is_digest,
    top_level_cli_sha256,
)
from scripts._m14_l049_v2_validate_common import (
    addendum_errors,
    candidate_key,
    canonical_artifact_digest,
    groups,
    lower_ci,
    real_resources,
    runtime_attestation_errors,
    safe_float,
)


def _folds(group_ids: Sequence[str]) -> list[list[str]]:
    ordered = sorted(group_ids)
    return [ordered[index : index + 6] for index in range(0, 36, 6)]


def _record_map(
    records: object, group_ids: set[str], errors: list[str]
) -> dict[tuple[str, int, int], Mapping[str, Any]]:
    if not isinstance(records, list):
        errors.append("Stage A score records are malformed")
        return {}
    result: dict[tuple[str, int, int], Mapping[str, Any]] = {}
    candidates = {(item["layer"], item["offset"]) for item in candidate_grid()}
    for record in records:
        if not isinstance(record, Mapping):
            errors.append("Stage A score record is not an object")
            continue
        group = record.get("group_id")
        layer = record.get("layer")
        offset = record.get("offset")
        if (
            not isinstance(group, str)
            or not isinstance(layer, int)
            or isinstance(layer, bool)
            or not isinstance(offset, int)
            or isinstance(offset, bool)
        ):
            errors.append("Stage A score record domain is invalid")
            continue
        key = (group, layer, offset)
        values = record.get("row_scores")
        if key in result or group not in group_ids or (layer, offset) not in candidates:
            errors.append("Stage A score record domain is invalid")
            continue
        if not isinstance(values, list) or len(values) != 2 or any(safe_float(value) is None for value in values):
            errors.append("Stage A score record values are invalid")
            continue
        expected = float(np.mean(np.asarray([safe_float(value) for value in values], dtype=np.float64)))
        if record.get("group_score") != expected:
            errors.append("Stage A group score was not independently recomputed")
        primitive = record.get("primitive_margins")
        if (
            not isinstance(primitive, list)
            or len(primitive) != 2
            or any(
                not isinstance(item, Mapping) or set(item) != {"clean_margin", "corrupted_margin", "patched_margin"}
                for item in primitive
            )
        ):
            errors.append("Stage A primitive margins are missing or malformed")
        else:
            recoveries = [
                directional_recovery(item.get("clean_margin"), item.get("corrupted_margin"), item.get("patched_margin"))
                for item in primitive
            ]
            if any(recovery is None for recovery in recoveries) or any(
                recovery is not None and not np.isclose(recovery, value, rtol=0.0, atol=1e-12)
                for recovery, value in zip(recoveries, values, strict=True)
            ):
                errors.append("Stage A directional recovery was not independently recomputed")
        result[key] = record
    if len(result) != len(group_ids) * len(candidates):
        errors.append("Stage A score record coverage is invalid")
    return result


def validate_stage_a_impl(
    artifact: Mapping[str, Any],
    train_rows: Sequence[Mapping[str, Any]],
    addendum: Mapping[str, Any],
    *,
    policy: CommitmentPolicy,
) -> list[str]:
    errors = addendum_errors(addendum, policy)
    expected_policy = policy.expected_addendum()
    public_train_seed = expected_policy.get("train_seed")
    candidate_policy = expected_policy.get("candidate_selection")
    oof_threshold = candidate_policy.get("oof_lower_ci_strict_gt") if isinstance(candidate_policy, Mapping) else None
    groups_map, group_errors = groups(train_rows, "train", 36)
    errors.extend(group_errors)
    if set(artifact) != {
        "schema_version",
        "stage",
        "status",
        "evidence_level",
        "evidence_eligible",
        "repository_promotion",
        "failure_kind",
        "selection_complete",
        "parent_plan_sha256",
        "addendum_schema",
        "addendum_sha256",
        "source_sha256",
        "public_train_seed",
        "train_fixture_sha256",
        "holdout_commitment",
        "selection",
        "resources",
        "runtime_attestation",
        "attestation_sha256",
        "artifact_sha256",
    }:
        errors.append("Stage A artifact fields are invalid")
    if artifact.get("schema_version") != V2_STAGE_A_SCHEMA or artifact.get("stage") != "stage_a_train_selection":
        errors.append("Stage A schema or stage is invalid")
    if artifact.get("repository_promotion") is not False:
        errors.append("Stage A repository promotion must remain false")
    if artifact.get("parent_plan_sha256") != policy.parent_plan_sha256 or artifact.get(
        "addendum_schema"
    ) != policy.expected_addendum().get("schema_version"):
        errors.append("Stage A parent/addendum binding is invalid")
    expected_addendum = canonical_artifact_digest(addendum, "addendum_sha256")
    if artifact.get("addendum_sha256") != expected_addendum:
        errors.append("Stage A addendum digest is invalid")
    if artifact.get("public_train_seed") != public_train_seed:
        errors.append("Stage A train seed is invalid")
    try:
        expected_train = fixture_digest(train_rows)
    except (TypeError, ValueError, OverflowError):
        expected_train = None
    if artifact.get("train_fixture_sha256") != expected_train:
        errors.append("Stage A train fixture digest is invalid")
    if artifact.get("holdout_commitment") != addendum.get("fixture"):
        errors.append("Stage A holdout commitment is invalid")
    if any(key in artifact for key in ("holdout_rows", "holdout_plaintext", "holdout_seed", "holdout_path")):
        errors.append("Stage A must not contain holdout plaintext, seed, or path")
    selection = artifact.get("selection")
    if not isinstance(selection, Mapping):
        return errors + ["Stage A selection is missing"]

    failure_kind = artifact.get("failure_kind")
    selection_complete = artifact.get("selection_complete")
    if failure_kind is not None and failure_kind not in STAGE_A_FAILURE_KINDS:
        errors.append("Stage A failure discriminator is invalid")
    if artifact.get("status") == "stage_a_failed" and failure_kind in {"runtime_exception", "validation_rejected"}:
        # A runtime exception before selection completes is still a valid,
        # non-promoting Stage A outcome.  It must retain only sanitized
        # exception metadata and execution counters, never fabricate a
        # zero-score protocol fixture.
        if selection_complete is not False:
            errors.append("Stage A runtime failure selection-complete flag is invalid")
        expected_failure_keys = {
            "candidate_grid",
            "score_records",
            "folds",
            "consensus_candidate",
            "consensus_wins",
            "oof_evidence",
            "oof_metric",
            "train_group_ids",
            "failure",
        }
        if set(selection) != expected_failure_keys:
            errors.append("Stage A runtime-failure selection fields are invalid")
        if selection.get("candidate_grid") != candidate_grid():
            errors.append("Stage A runtime-failure candidate grid is invalid")
        if selection.get("score_records") != [] or selection.get("folds") != [] or selection.get("oof_evidence") != []:
            errors.append("Stage A runtime-failure selection contains candidate evidence")
        if selection.get("consensus_candidate") is not None or selection.get("consensus_wins") != 0:
            errors.append("Stage A runtime-failure selection contains a candidate")
        if selection.get("oof_metric") is not None:
            errors.append("Stage A runtime-failure OOF metric must be absent")
        if selection.get("train_group_ids") != sorted(groups_map):
            errors.append("Stage A runtime-failure train group order is invalid")
        failure = selection.get("failure")
        allowed_failure = (
            {"validation_codes"}
            if failure_kind == "validation_rejected"
            else {"exception_type", "shape_field", "expected_shape", "actual_shape"}
        )
        if not isinstance(failure, Mapping) or set(failure) - allowed_failure:
            errors.append("Stage A runtime-failure metadata is invalid")
        elif failure_kind == "validation_rejected":
            codes = failure.get("validation_codes")
            if (
                not isinstance(codes, list)
                or not codes
                or any(not isinstance(code, str) or code not in VALIDATION_REJECTION_CODES for code in codes)
            ):
                errors.append("Stage A validation-rejection codes are invalid")
        elif failure_kind == "runtime_exception":
            if not isinstance(failure.get("exception_type"), str) or not failure.get("exception_type"):
                errors.append("Stage A runtime-failure metadata is invalid")
            elif "shape_field" in failure and (
                not isinstance(failure.get("shape_field"), str)
                or not isinstance(failure.get("expected_shape"), list)
                or not isinstance(failure.get("actual_shape"), list)
                or any(not isinstance(value, int) or isinstance(value, bool) for value in failure["expected_shape"])
                or any(not isinstance(value, int) or isinstance(value, bool) for value in failure["actual_shape"])
            ):
                errors.append("Stage A runtime-failure shape metadata is invalid")
        if artifact.get("evidence_level") != "D0" or artifact.get("evidence_eligible") is not False:
            errors.append("Stage A runtime failure must remain D0 and ineligible")
        resources = artifact.get("resources")
        if not isinstance(resources, Mapping):
            errors.append("Stage A runtime-failure resources are missing")
            mode = "synthetic"
        elif resources.get("execution_backend") == "cuda":
            mode = "real"
            errors.extend(real_resources(resources, allow_failure=True, stage="Stage A"))
        else:
            mode = "synthetic"
            if (
                resources.get("execution_attempted") is not False
                or resources.get("execution_backend") != "none"
                or resources.get("stage") not in {"dispatch", "preflight", "dependency_check"}
                or resources.get("no_mutation") is not True
            ):
                errors.append("Stage A pre-CUDA runtime failure provenance is invalid")
            counters = resources.get("operation_counts")
            if (
                not isinstance(counters, Mapping)
                or set(counters)
                != {
                    "candidate_evaluations",
                    "hooks",
                    "captures",
                    "patches",
                    "controls",
                    "forwards",
                }
                or any(
                    not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in counters.values()
                )
            ):
                errors.append("Stage A pre-CUDA runtime counters are invalid")
        expected_selection_sha = digest_bytes(canonical_json_bytes(selection))
        errors.extend(
            runtime_attestation_errors(
                artifact.get("runtime_attestation"),
                stage="stage_a_train_selection",
                mode=mode,
                group_count=TRAIN_GROUP_COUNT,
                pair_count=TRAIN_GROUP_COUNT,
                candidate_count=len(candidate_grid()),
                seed_count=1,
                fixture_sha256=artifact.get("train_fixture_sha256"),
                candidate_sha256=expected_selection_sha,
                source_sha256=artifact.get("source_sha256"),
                addendum_sha256=artifact.get("addendum_sha256"),
                cli_sha256=top_level_cli_sha256("stage_a_train_selection"),
                execution_resources=resources,
                partial_failure=True,
            )
        )
        attestation = artifact.get("runtime_attestation")
        if not isinstance(attestation, Mapping) or artifact.get("attestation_sha256") != attestation.get(
            "attestation_sha256"
        ):
            errors.append("Stage A attestation binding is invalid")
        if not is_digest(artifact.get("source_sha256")):
            errors.append("Stage A source commitment is invalid")
        if artifact.get("artifact_sha256") != canonical_artifact_digest(artifact, "artifact_sha256"):
            errors.append("Stage A artifact digest is invalid")
        return errors

    if artifact.get("status") == "stage_a_failed" and failure_kind == "semantic_gate":
        if selection_complete is not True:
            errors.append("Stage A semantic failure selection-complete flag is invalid")
        if "failure" in selection:
            errors.append("Stage A semantic failure must not contain runtime failure metadata")
    elif artifact.get("status") == "stage_a_failed":
        errors.append("Stage A failed status discriminator is invalid")
    elif selection_complete is not True or failure_kind is not None:
        errors.append("Stage A successful/protocol discriminator is invalid")

    expected_grid = candidate_grid()
    if selection.get("candidate_grid") != expected_grid or selection.get("train_group_ids") != sorted(groups_map):
        errors.append("Stage A candidate grid or train group order is invalid")
    record_map = _record_map(selection.get("score_records"), set(groups_map), errors)
    folds = selection.get("folds")
    if not isinstance(folds, list) or len(folds) != 6:
        return errors + ["Stage A outer folds are malformed"]
    wins: dict[tuple[int, int], int] = {}
    expected_oof: list[dict[str, Any]] = []
    for index, fold in enumerate(folds):
        if not isinstance(fold, Mapping):
            errors.append("Stage A fold is not an object")
            continue
        validation = fold.get("validation_groups")
        fitting = fold.get("fit_groups")
        expected_validation = _folds(sorted(groups_map))[index]
        expected_fitting = [group for group in sorted(groups_map) if group not in expected_validation]
        if (
            not isinstance(validation, list)
            or not isinstance(fitting, list)
            or validation != expected_validation
            or fitting != expected_fitting
        ):
            errors.append("Stage A fold group partition is invalid")
            continue
        validation_groups = cast(list[str], validation)
        fitting_groups = cast(list[str], fitting)
        ranking: list[dict[str, Any]] = []
        for candidate in expected_grid:
            values = [
                safe_float(record_map[(group, candidate["layer"], candidate["offset"])].get("group_score"))
                for group in fitting_groups
                if (group, candidate["layer"], candidate["offset"]) in record_map
            ]
            if len(values) != len(fitting_groups) or any(value is None for value in values):
                errors.append("Stage A fold score coverage is incomplete")
                continue
            clean_values = [value for value in values if value is not None]
            ci = lower_ci(clean_values, cast(int, public_train_seed) + len(fitting_groups) + candidate["layer"] * 3)
            if ci is None:
                errors.append("Stage A fold bootstrap failed")
                continue
            ranking.append(
                {
                    "layer": candidate["layer"],
                    "offset": candidate["offset"],
                    "mean_recovery": float(np.mean(np.asarray(clean_values, dtype=np.float64))),
                    "lower_ci": ci,
                }
            )
        expected_ranking = sorted(
            ranking, key=lambda item: (-item["mean_recovery"], -item["lower_ci"], candidate_key(item))
        )
        if fold.get("ranking") != expected_ranking:
            errors.append("Stage A ranking was not independently recomputed")
        if not expected_ranking:
            continue
        winner = {"layer": expected_ranking[0]["layer"], "offset": expected_ranking[0]["offset"]}
        if fold.get("winner") != winner:
            errors.append("Stage A winner is invalid")
        wins[(winner["layer"], winner["offset"])] = wins.get((winner["layer"], winner["offset"]), 0) + 1
        for group in validation_groups:
            record = record_map.get((group, winner["layer"], winner["offset"]))
            if record is not None:
                expected_oof.append(
                    {
                        "fold": index,
                        "group_id": group,
                        "layer": winner["layer"],
                        "offset": winner["offset"],
                        "recovery": record.get("group_score"),
                    }
                )
    best = max(wins, key=lambda key: (-wins[key], candidate_key({"layer": key[0], "offset": key[1]})), default=None)
    expected_wins = wins.get(best, 0) if best is not None else 0
    expected_consensus = {"layer": best[0], "offset": best[1]} if best is not None and expected_wins >= 4 else None
    if selection.get("consensus_candidate") != expected_consensus:
        errors.append("Stage A consensus identity is invalid")
    if selection.get("consensus_wins") != expected_wins:
        errors.append("Stage A consensus count is invalid")
    if selection.get("oof_evidence") != expected_oof:
        errors.append("Stage A out-of-fold evidence is invalid")
    oof = selection.get("oof_evidence")
    oof_items: list[Mapping[str, Any]] = []
    if isinstance(oof, list) and all(isinstance(item, Mapping) for item in oof):
        oof_items = cast(list[Mapping[str, Any]], oof)
    values = [safe_float(item.get("recovery")) for item in oof_items]
    if len(values) != 36 or any(value is None for value in values):
        errors.append("Stage A OOF evidence values are invalid")
        expected_metric = None
    else:
        clean_values = [value for value in values if value is not None]
        ci = lower_ci(clean_values, cast(int, public_train_seed))
        fold_positive = all(
            sum(safe_float(item.get("recovery")) or 0.0 for item in oof_items if item.get("fold") == fold) / 6.0 > 0.0
            for fold in range(6)
        )
        positive = sum(value > 0.0 for value in clean_values)
        expected_metric = {
            "point_estimate": float(np.mean(np.asarray(clean_values, dtype=np.float64))),
            "lower_ci_95": ci,
            "threshold": oof_threshold,
            "all_fold_means_positive": fold_positive,
            "positive_groups": positive,
            "required_positive_groups": 24,
            "pass": bool(
                expected_wins >= 4
                and ci is not None
                and ci > cast(float, oof_threshold)
                and fold_positive
                and positive >= 24
            ),
        }
    if selection.get("oof_metric") != expected_metric:
        errors.append("Stage A OOF metric was not independently recomputed")
    pre_resources = artifact.get("resources")
    mode = (
        "real"
        if isinstance(pre_resources, Mapping) and pre_resources.get("execution_backend") == "cuda"
        else "synthetic"
    )
    if expected_metric is not None:
        expected_status = (
            "stage_a_complete"
            if mode == "real" and expected_metric["pass"]
            else "protocol_fixture"
            if mode == "synthetic" and expected_metric["pass"]
            else "stage_a_failed"
        )
        if artifact.get("status") != expected_status:
            errors.append("Stage A status is not bound to recomputed acceptance")
    resources = artifact.get("resources")
    synthetic_resources = {
        "stage": "protocol_fixture",
        "execution_backend": "cpu",
        "execution_attempted": False,
        "no_mutation": True,
    }
    if resources == synthetic_resources:
        if artifact.get("evidence_level") != "D0" or artifact.get("evidence_eligible") is not False:
            errors.append("Stage A offline artifact must remain D0")
    elif isinstance(resources, Mapping) and resources.get("execution_backend") == "cuda":
        mode = "real"
        real_pass = (
            artifact.get("status") == "stage_a_complete" and expected_metric is not None and expected_metric["pass"]
        )
        errors.extend(real_resources(resources, require_measured=bool(real_pass), stage="Stage A"))
        if (
            artifact.get("evidence_level") != ("D1" if real_pass else "D0")
            or artifact.get("evidence_eligible") is not real_pass
        ):
            errors.append("Stage A real evidence level is not bound to gates")
    else:
        errors.append("Stage A resources are not an exact offline or real contract")
    expected_selection_sha = digest_bytes(canonical_json_bytes(selection))
    errors.extend(
        runtime_attestation_errors(
            artifact.get("runtime_attestation"),
            stage="stage_a_train_selection",
            mode=mode,
            group_count=36,
            pair_count=36,
            candidate_count=len(expected_grid),
            seed_count=1,
            fixture_sha256=artifact.get("train_fixture_sha256"),
            candidate_sha256=expected_selection_sha,
            source_sha256=artifact.get("source_sha256"),
            addendum_sha256=artifact.get("addendum_sha256"),
            cli_sha256=top_level_cli_sha256("stage_a_train_selection"),
            execution_resources=artifact.get("resources"),
        )
    )
    attestation = artifact.get("runtime_attestation")
    if not isinstance(attestation, Mapping) or artifact.get("attestation_sha256") != attestation.get(
        "attestation_sha256"
    ):
        errors.append("Stage A attestation binding is invalid")
    if not is_digest(artifact.get("source_sha256")):
        errors.append("Stage A source commitment is invalid")
    if artifact.get("artifact_sha256") != canonical_artifact_digest(artifact, "artifact_sha256"):
        errors.append("Stage A artifact digest is invalid")
    return errors


__all__ = ["validate_stage_a_impl"]

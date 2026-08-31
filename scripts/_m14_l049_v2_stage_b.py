"""Synthetic-capable Stage B holdout evaluator for L04.9 v2.

The real holdout fixture and its 256-bit seed are deliberately supplied by the
caller. No holdout path, plaintext, or seed is embedded in this module.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from scripts._m14_l049_v2_attestation import build_runtime_attestation
from scripts._m14_l049_v2_fixture import TRAIN_FIXTURE_PATH, read_rows, validate_fixture
from scripts._m14_l049_v2_schema import (
    BOOTSTRAP_REPLICATES,
    PAIRED_SHUFFLED_THRESHOLD,
    RECOVERY_THRESHOLD,
    STAGE_B_SEEDS,
    V2_STAGE_A_SCHEMA,
    V2_STAGE_B_SCHEMA,
    canonical_digest,
    canonical_fixture_bytes,
    digest_bytes,
    directional_recovery,
    fixture_digest,
    top_level_cli_sha256,
)
from scripts._m14_l049_v2_stage_a import attempted_real_resources


def label_stratified_shuffled_mapping(
    pair_ids: Sequence[str], pair_clean_labels: Mapping[str, int] | None = None
) -> dict[str, str]:
    """Deterministically rotate pair IDs within clean-label strata."""
    names = sorted(str(pair) for pair in pair_ids)
    if len(names) < 2 or len(set(names)) != len(names):
        raise ValueError("shuffled mapping requires at least two unique pairs")
    labels = {name: int(pair_clean_labels.get(name, 1)) if pair_clean_labels is not None else 1 for name in names}
    if any(label not in {0, 1} for label in labels.values()):
        raise ValueError("shuffled mapping labels are invalid")
    buckets: dict[int, list[str]] = {0: [], 1: []}
    for name in names:
        buckets[labels[name]].append(name)
    if any(len(bucket) < 2 for bucket in buckets.values() if bucket):
        raise ValueError("each non-empty clean-label stratum needs two pairs")
    return {name: bucket[(index + 1) % len(bucket)] for bucket in buckets.values() for index, name in enumerate(bucket)}


def _bootstrap_lower(values: Sequence[float], seed: int) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ValueError("group values must be finite and non-empty")
    rng = np.random.default_rng(int(seed))
    draws = array[rng.integers(0, len(array), size=(BOOTSTRAP_REPLICATES, len(array)))]
    return float(np.quantile(np.mean(draws, axis=1), 0.025))


def _metric(values: Sequence[float], seed: int, threshold: float) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    point = float(np.mean(array))
    lower = _bootstrap_lower(array.tolist(), seed)
    return {
        "point_estimate": point,
        "lower_ci_95": lower,
        "threshold": float(threshold),
        "comparator": ">",
        "aggregation_unit": "independent causal group",
        "pass": bool(lower > threshold),
    }


def _group_mean(values: Mapping[str, list[float]]) -> list[float]:
    return [float(np.mean(values[group])) for group in sorted(values)]


def _pairs(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Mapping[str, Any]]]:
    pairs: dict[str, dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        if row.get("split") != "holdout":
            raise ValueError("Stage B accepts holdout rows only")
        pair = pairs.setdefault(str(row["causal_pair_id"]), {})
        condition = str(row["condition"])
        if condition in pair:
            raise ValueError("Stage B contains duplicate causal-pair conditions")
        pair[condition] = row
    if len(pairs) != 24 or any(set(pair) != {"clean", "corrupted"} for pair in pairs.values()):
        raise ValueError("Stage B requires 24 complete causal pairs")
    return dict(sorted(pairs.items()))


def _candidate_precondition(
    candidate: Mapping[str, Any], train_rows: Sequence[Mapping[str, Any]], addendum: Mapping[str, Any]
) -> None:
    """Apply only producer-local preconditions before running Stage B.

    This deliberately does not import or invoke a validator.  The artifact
    boundary performs the complete independent Stage A recomputation; this
    check only prevents a producer from emitting an obviously unbound or
    malformed candidate before it starts expensive holdout work.
    """
    candidate_sha = candidate.get("artifact_sha256")
    if not isinstance(candidate_sha, str) or candidate_sha != canonical_digest(candidate, "artifact_sha256"):
        raise ValueError("Stage A candidate artifact digest is invalid")
    if candidate.get("schema_version") != V2_STAGE_A_SCHEMA:
        raise ValueError("Stage A candidate schema is invalid")
    if candidate.get("stage") != "stage_a_train_selection":
        raise ValueError("Stage A candidate stage is invalid")
    if candidate.get("parent_plan_sha256") != "f3c315e356af0ee54d4196cc365ee22bd997b069d18a3e72c6b479f94e0b3e1a":
        raise ValueError("Stage A candidate parent commitment is invalid")
    if candidate.get("addendum_schema") != addendum.get("schema_version"):
        raise ValueError("Stage A candidate addendum commitment is invalid")
    if candidate.get("addendum_sha256") != canonical_digest(addendum, "addendum_sha256"):
        raise ValueError("Stage A candidate addendum digest is invalid")
    expected_train = digest_bytes(canonical_fixture_bytes(train_rows))
    if candidate.get("train_fixture_sha256") != expected_train:
        raise ValueError("Stage A candidate train fixture commitment is invalid")
    selection = candidate.get("selection")
    if not isinstance(selection, Mapping) or selection.get("oof_metric", {}).get("pass") is not True:
        raise ValueError("Stage A candidate is not a complete passing artifact")


def evaluate_stage_b(
    holdout_rows: Sequence[Mapping[str, Any]],
    observations: Mapping[str, Mapping[str, Mapping[str, Any]]],
    candidate_artifact: Mapping[str, Any],
    addendum: Mapping[str, Any],
    holdout_seed: bytes,
    *,
    resources: Mapping[str, Any] | None = None,
    cli_sha256: str | None = None,
) -> dict[str, Any]:
    """Evaluate precomputed synthetic/model observations on withheld rows."""
    pairs = _pairs(holdout_rows)
    if len(holdout_seed) != 32:
        raise ValueError("holdout seed must be exactly 256 bits")
    raw = canonical_fixture_bytes(holdout_rows)
    commitment = addendum["fixture"]
    if fixture_digest(holdout_rows) != commitment["holdout_content_sha256"]:
        raise ValueError("holdout fixture digest does not match addendum")
    if hashlib.sha256(holdout_seed).hexdigest() != commitment["holdout_seed_commitment_sha256"]:
        raise ValueError("holdout seed commitment does not match addendum")
    candidate_sha = str(candidate_artifact.get("artifact_sha256", ""))
    train_rows = read_rows(TRAIN_FIXTURE_PATH)[1]
    if validate_fixture(train_rows, holdout_rows):
        raise ValueError("train/holdout fixture contract is invalid")
    _candidate_precondition(candidate_artifact, train_rows, addendum)
    labels = {pair: int(pair_rows["clean"]["factor_labels"]["clean_label"]) for pair, pair_rows in pairs.items()}
    mapping = label_stratified_shuffled_mapping(list(pairs), labels)
    seed_summaries: list[dict[str, Any]] = []
    for seed in STAGE_B_SEEDS:
        true_by_group: dict[str, list[float]] = {}
        shuffled_by_group: dict[str, list[float]] = {}
        evidence: list[dict[str, Any]] = []
        for pair, pair_rows in pairs.items():
            item = observations[pair]
            per_seed = item.get(str(seed))
            if not isinstance(per_seed, Mapping):
                raise ValueError(f"missing observations for pair {pair!r}, seed {seed}")
            true_recovery = directional_recovery(
                per_seed.get("clean_margin"), per_seed.get("corrupted_margin"), per_seed.get("patched_margin")
            )
            shuffled_recovery = directional_recovery(
                per_seed.get("clean_margin"), per_seed.get("corrupted_margin"), per_seed.get("shuffled_margin")
            )
            if true_recovery is None or shuffled_recovery is None:
                raise ValueError(f"invalid directional recovery for pair {pair!r}")
            group = str(pair_rows["clean"]["group_id"])
            true_by_group.setdefault(group, []).append(true_recovery)
            shuffled_by_group.setdefault(group, []).append(shuffled_recovery)
            evidence.append(
                {
                    "pair_id": pair,
                    "group_id": group,
                    "clean_margin": float(per_seed["clean_margin"]),
                    "corrupted_margin": float(per_seed["corrupted_margin"]),
                    "patched_margin": float(per_seed["patched_margin"]),
                    "shuffled_margin": float(per_seed["shuffled_margin"]),
                    "recovery": true_recovery,
                    "shuffled_recovery": shuffled_recovery,
                    "shuffled_donor_pair_id": mapping[pair],
                    "controls": {
                        name: dict(per_seed.get(name, {}))
                        for name in ("wrong_token", "adjacent_layer", "additive", "matched_norm_random")
                    },
                    "zero_strength": {
                        "selected_logit_digest": per_seed.get("zero_strength_selected_logit_digest"),
                        "relevant_output_digest": per_seed.get("zero_strength_relevant_output_digest"),
                        "corrupted_selected_logit_digest": per_seed.get("corrupted_selected_logit_digest"),
                        "corrupted_relevant_output_digest": per_seed.get("corrupted_relevant_output_digest"),
                        "identity": per_seed.get("zero_strength_identity"),
                    },
                }
            )
        true_groups = _group_mean(true_by_group)
        shuffled_groups = _group_mean(shuffled_by_group)
        paired_groups = [true - shuffled for true, shuffled in zip(true_groups, shuffled_groups, strict=True)]
        seed_summaries.append(
            {
                "seed": seed,
                "recovery": _metric(true_groups, seed, RECOVERY_THRESHOLD),
                "paired_true_minus_shuffled": _metric(paired_groups, seed + 1, PAIRED_SHUFFLED_THRESHOLD),
                "evidence": evidence,
                "shuffled_mapping": mapping,
            }
        )
    accepted = all(
        summary["recovery"]["pass"] and summary["paired_true_minus_shuffled"]["pass"] for summary in seed_summaries
    )
    backend = str((resources or {}).get("execution_backend", "synthetic"))
    evidence_level = "D2" if accepted and backend == "cuda" else "D0"
    artifact: dict[str, Any] = {
        "schema_version": V2_STAGE_B_SCHEMA,
        "stage": "stage_b_holdout_evaluation",
        "status": "stage_b_complete" if accepted else "stage_b_failed",
        "evidence_level": evidence_level,
        "evidence_eligible": False,
        "promotion_candidate": bool(accepted and evidence_level == "D2"),
        "acceptance": accepted,
        "failure_kind": None,
        "failure": None,
        "repository_promotion": False,
        "candidate_artifact_sha256": candidate_sha,
        "parent_plan_sha256": candidate_artifact.get("parent_plan_sha256"),
        "addendum_schema": candidate_artifact.get("addendum_schema"),
        "train_fixture_sha256": candidate_artifact.get("train_fixture_sha256"),
        "source_sha256": candidate_artifact.get("source_sha256"),
        "addendum_sha256": canonical_digest(addendum, "addendum_sha256"),
        "holdout_fixture_sha256": digest_bytes(raw),
        "holdout_seed_commitment_sha256": hashlib.sha256(holdout_seed).hexdigest(),
        "shuffled_mapping": mapping,
        "shuffled_mapping_sha256": hashlib.sha256(
            canonical_fixture_bytes([{"source": source, "donor": mapping[source]} for source in sorted(mapping)])
        ).hexdigest(),
        "seed_summaries": seed_summaries,
        "controls": {
            "scope": "diagnostics_only",
            "wrong_token": "separately_serialized",
            "adjacent_layer": "separately_serialized",
            "additive": "separately_serialized",
            "matched_norm_random": "separately_serialized",
            "zero_strength": "exact selected-logit and relevant-output digest identity",
        },
        "resources": dict(
            resources
            or {
                "stage": "synthetic_fixture",
                "execution_backend": "synthetic",
                "execution_attempted": False,
                "no_mutation": True,
            }
        ),
    }
    mode = "real" if backend == "cuda" else "synthetic"
    resolved_cli_sha = cli_sha256 or top_level_cli_sha256("stage_b_holdout_evaluation")
    if resolved_cli_sha is None:
        raise ValueError("Stage B top-level CLI digest is unavailable")
    attestation = build_runtime_attestation(
        stage="stage_b_holdout_evaluation",
        mode=mode,
        group_count=len(pairs),
        pair_count=len(pairs),
        candidate_count=len(candidate_artifact.get("selection", {}).get("candidate_grid", [])),
        seed_count=len(STAGE_B_SEEDS),
        fixture_sha256=artifact["holdout_fixture_sha256"],
        candidate_sha256=candidate_sha,
        source_sha256=str(artifact["source_sha256"]),
        addendum_sha256=str(artifact["addendum_sha256"]),
        cli_sha256=resolved_cli_sha,
        resources=artifact["resources"],
        operation_counts=(
            artifact["resources"].get("operation_counts")
            if mode == "real" and isinstance(artifact["resources"], Mapping)
            else None
        ),
    )
    artifact["runtime_attestation"] = attestation
    artifact["attestation_sha256"] = attestation["attestation_sha256"]
    artifact["artifact_sha256"] = canonical_digest(artifact, "artifact_sha256")
    return artifact


def build_stage_b_failure_artifact(
    rows: Sequence[Mapping[str, Any]],
    candidate: Mapping[str, Any],
    addendum: Mapping[str, Any],
    holdout_seed: bytes,
    *,
    source_sha256: str,
    error: BaseException,
    resources: Mapping[str, Any],
    cli_sha256: str | None = None,
) -> dict[str, Any]:
    """Emit a validator-clean attempted-real D0 with no holdout evidence."""
    raw = canonical_fixture_bytes(rows)
    candidate_sha = str(candidate.get("artifact_sha256", ""))
    resolved_cli = cli_sha256 or top_level_cli_sha256("stage_b_holdout_evaluation")
    if resolved_cli is None:
        raise ValueError("Stage B top-level CLI digest is unavailable")
    resource_payload = dict(resources)
    counters = resource_payload.get("operation_counts")
    if not isinstance(counters, Mapping):
        resource_payload = attempted_real_resources()
        counters = resource_payload["operation_counts"]
    failure: dict[str, Any] = {"exception_type": type(error).__name__}
    attestation = build_runtime_attestation(
        stage="stage_b_holdout_evaluation",
        mode="real",
        group_count=24,
        pair_count=24,
        candidate_count=len(candidate.get("selection", {}).get("candidate_grid", [])),
        seed_count=len(STAGE_B_SEEDS),
        fixture_sha256=digest_bytes(raw),
        candidate_sha256=candidate_sha,
        source_sha256=source_sha256,
        addendum_sha256=canonical_digest(addendum, "addendum_sha256"),
        cli_sha256=resolved_cli,
        resources=resource_payload,
        operation_counts=counters,
    )
    artifact: dict[str, Any] = {
        "schema_version": V2_STAGE_B_SCHEMA,
        "stage": "stage_b_holdout_evaluation",
        "status": "stage_b_failed",
        "evidence_level": "D0",
        "evidence_eligible": False,
        "promotion_candidate": False,
        "acceptance": False,
        "failure_kind": "runtime_exception",
        "failure": failure,
        "repository_promotion": False,
        "candidate_artifact_sha256": candidate_sha,
        "parent_plan_sha256": candidate.get("parent_plan_sha256"),
        "addendum_schema": candidate.get("addendum_schema"),
        "train_fixture_sha256": candidate.get("train_fixture_sha256"),
        "source_sha256": source_sha256,
        "addendum_sha256": canonical_digest(addendum, "addendum_sha256"),
        "holdout_fixture_sha256": digest_bytes(raw),
        "holdout_seed_commitment_sha256": hashlib.sha256(holdout_seed).hexdigest(),
        "shuffled_mapping": {},
        "shuffled_mapping_sha256": hashlib.sha256(canonical_fixture_bytes([])).hexdigest(),
        "seed_summaries": [],
        "controls": {
            "scope": "diagnostics_only",
            "wrong_token": "separately_serialized",
            "adjacent_layer": "separately_serialized",
            "additive": "separately_serialized",
            "matched_norm_random": "separately_serialized",
            "zero_strength": "exact selected-logit and relevant-output digest identity",
        },
        "resources": resource_payload,
        "runtime_attestation": attestation,
        "attestation_sha256": attestation["attestation_sha256"],
    }
    artifact["artifact_sha256"] = canonical_digest(artifact, "artifact_sha256")
    return artifact


__all__ = ["build_stage_b_failure_artifact", "evaluate_stage_b", "label_stratified_shuffled_mapping"]

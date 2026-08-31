"""Offline train-only candidate selection for the L04.9 v2 addendum."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast

import numpy as np

from scripts._m14_l049_v2_attestation import build_runtime_attestation
from scripts._m14_l049_v2_schema import (
    BOOTSTRAP_REPLICATES,
    OOF_RECOVERY_THRESHOLD,
    PARENT_PLAN_SHA256,
    PUBLIC_TRAIN_SEED,
    TRAIN_GROUP_COUNT,
    V2_ADDENDUM_SCHEMA,
    V2_STAGE_A_SCHEMA,
    candidate_grid,
    canonical_digest,
    canonical_fixture_bytes,
    canonical_json_bytes,
    digest_bytes,
    top_level_cli_sha256,
)

ScoreFunction = Callable[[Mapping[str, Any], int, int], float]


def _candidate_key(candidate: Mapping[str, int]) -> tuple[int, int]:
    return int(candidate["layer"]), (0, -1, -2).index(int(candidate["offset"]))


def outer_folds(group_ids: Sequence[str]) -> list[list[str]]:
    groups = sorted(str(group) for group in group_ids)
    if len(groups) != 36 or len(set(groups)) != 36:
        raise ValueError("stage A requires exactly 36 unique train groups")
    return [groups[index : index + 6] for index in range(0, 36, 6)]


def default_train_score(row: Mapping[str, Any], layer: int, offset: int) -> float:
    """Deterministic synthetic sensitivity score; no model or holdout data."""
    token = f"{row['row_id']}|{int(layer)}|{int(offset)}".encode()
    noise = (int.from_bytes(hashlib.sha256(token).digest()[:8], "big") / 2**64 - 0.5) * 0.02
    signal = 0.16 if (int(layer), int(offset)) == (6, 0) else 0.0
    return float(signal + noise)


def _lower_ci(values: Sequence[float], seed: int) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ValueError("bootstrap values must be finite and non-empty")
    rng = np.random.default_rng(int(seed))
    draws = array[rng.integers(0, len(array), size=(BOOTSTRAP_REPLICATES, len(array)))]
    return float(np.quantile(np.mean(draws, axis=1), 0.025))


def _group_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        if row.get("split") != "train":
            raise ValueError("stage A accepts train rows only")
        groups.setdefault(str(row["group_id"]), []).append(row)
    if any(len(value) != 2 for value in groups.values()):
        raise ValueError("stage A requires two rows per train group")
    return dict(sorted(groups.items()))


def _score_records(rows: Sequence[Mapping[str, Any]], scorer: ScoreFunction) -> list[dict[str, Any]]:
    groups = _group_rows(rows)
    records: list[dict[str, Any]] = []
    for group_id, group_rows in groups.items():
        for candidate in candidate_grid():
            values = [float(scorer(row, candidate["layer"], candidate["offset"])) for row in group_rows]
            if not np.isfinite(values).all():
                raise ValueError("train candidate scores must be finite")
            records.append(
                {
                    "group_id": group_id,
                    "layer": candidate["layer"],
                    "offset": candidate["offset"],
                    "row_scores": values,
                    "group_score": float(np.mean(values)),
                }
            )
    return records


def _rank(records: Sequence[Mapping[str, Any]], groups: Sequence[str]) -> list[dict[str, Any]]:
    selected = [record for record in records if str(record["group_id"]) in set(groups)]
    ranked: list[dict[str, Any]] = []
    for candidate in candidate_grid():
        values = [
            float(record["group_score"])
            for record in selected
            if record["layer"] == candidate["layer"] and record["offset"] == candidate["offset"]
        ]
        if len(values) != len(groups):
            raise ValueError("candidate fold scores are incomplete")
        ranked.append(
            {
                "layer": candidate["layer"],
                "offset": candidate["offset"],
                "mean_recovery": float(np.mean(values)),
                "lower_ci": _lower_ci(values, seed=PUBLIC_TRAIN_SEED + len(groups) + candidate["layer"] * 3),
            }
        )
    return sorted(ranked, key=lambda item: (-item["mean_recovery"], -item["lower_ci"], _candidate_key(item)))


def select_stage_a(rows: Sequence[Mapping[str, Any]], scorer: ScoreFunction = default_train_score) -> dict[str, Any]:
    """Select a candidate using six outer folds and train groups only."""
    groups = _group_rows(rows)
    folds = outer_folds(list(groups))
    records = _score_records(rows, scorer)
    fold_records: list[dict[str, Any]] = []
    wins: dict[tuple[int, int], int] = {}
    oof: list[dict[str, Any]] = []
    for fold_index, validation_groups in enumerate(folds):
        fit_groups = [group for group in groups if group not in validation_groups]
        ranking = _rank(records, fit_groups)
        winner = {"layer": int(ranking[0]["layer"]), "offset": int(ranking[0]["offset"])}
        key = (winner["layer"], winner["offset"])
        wins[key] = wins.get(key, 0) + 1
        for group in validation_groups:
            match = next(
                record
                for record in records
                if record["group_id"] == group
                and record["layer"] == winner["layer"]
                and record["offset"] == winner["offset"]
            )
            oof.append(
                {
                    "fold": fold_index,
                    "group_id": group,
                    "layer": winner["layer"],
                    "offset": winner["offset"],
                    "recovery": float(match["group_score"]),
                }
            )
        fold_records.append(
            {
                "fold": fold_index,
                "fit_groups": fit_groups,
                "validation_groups": validation_groups,
                "ranking": ranking,
                "winner": winner,
            }
        )
    consensus = max(
        wins.items(), key=lambda item: (-item[1], _candidate_key({"layer": item[0][0], "offset": item[0][1]}))
    )
    oof_values = [float(item["recovery"]) for item in oof]
    lower = _lower_ci(oof_values, seed=PUBLIC_TRAIN_SEED)
    positive = sum(value > 0.0 for value in oof_values)
    passed = bool(
        consensus[1] >= 4
        and lower > OOF_RECOVERY_THRESHOLD
        and all(float(np.mean([item["recovery"] for item in oof if item["fold"] == fold])) > 0.0 for fold in range(6))
        and positive >= 24
    )
    return {
        "candidate_grid": candidate_grid(),
        "score_records": records,
        "folds": fold_records,
        "consensus_candidate": {"layer": consensus[0][0], "offset": consensus[0][1]},
        "consensus_wins": int(consensus[1]),
        "oof_evidence": oof,
        "oof_metric": {
            "point_estimate": float(np.mean(oof_values)),
            "lower_ci_95": lower,
            "threshold": OOF_RECOVERY_THRESHOLD,
            "all_fold_means_positive": all(
                float(np.mean([item["recovery"] for item in oof if item["fold"] == fold])) > 0.0 for fold in range(6)
            ),
            "positive_groups": positive,
            "required_positive_groups": 24,
            "pass": passed,
        },
        "train_group_ids": list(groups),
    }


def build_stage_a_artifact(
    rows: Sequence[Mapping[str, Any]],
    addendum: Mapping[str, Any],
    *,
    source_sha256: str,
    scorer: ScoreFunction = default_train_score,
    resources: Mapping[str, Any] | None = None,
    execution_mode: str = "synthetic",
    cli_sha256: str | None = None,
) -> dict[str, Any]:
    """Build a train-only artifact for offline or injected real execution."""
    if execution_mode not in {"synthetic", "real"}:
        raise ValueError("Stage A execution mode is invalid")
    selection = select_stage_a(rows, scorer)
    resource_payload = (
        dict(resources)
        if resources is not None
        else {
            "stage": "protocol_fixture",
            "execution_backend": "cpu",
            "execution_attempted": False,
            "no_mutation": True,
        }
    )
    finalizer = resource_payload.pop("finalize", None)
    if callable(finalizer):
        finalized = finalizer()
        if not isinstance(finalized, Mapping):
            raise ValueError("runtime resource finalizer returned an invalid mapping")
        resource_payload = dict(finalized)
    raw = canonical_fixture_bytes(rows)
    artifact: dict[str, Any] = {
        "schema_version": V2_STAGE_A_SCHEMA,
        "stage": "stage_a_train_selection",
        "status": (
            "stage_a_complete"
            if execution_mode == "real" and selection["oof_metric"]["pass"]
            else "protocol_fixture"
            if execution_mode == "synthetic" and selection["oof_metric"]["pass"]
            else "stage_a_failed"
        ),
        "evidence_level": "D1" if execution_mode == "real" and selection["oof_metric"]["pass"] else "D0",
        "evidence_eligible": bool(execution_mode == "real" and selection["oof_metric"]["pass"]),
        "repository_promotion": False,
        "parent_plan_sha256": PARENT_PLAN_SHA256,
        "addendum_schema": V2_ADDENDUM_SCHEMA,
        "addendum_sha256": canonical_digest(addendum, "addendum_sha256"),
        "source_sha256": str(source_sha256),
        "public_train_seed": PUBLIC_TRAIN_SEED,
        "train_fixture_sha256": digest_bytes(raw),
        "holdout_commitment": dict(addendum["fixture"]),
        "selection": selection,
        "resources": resource_payload,
    }
    selection_sha = digest_bytes(canonical_json_bytes(selection))
    resolved_cli_sha = cli_sha256 or top_level_cli_sha256("stage_a_train_selection")
    if resolved_cli_sha is None:
        raise ValueError("Stage A top-level CLI digest is unavailable")
    attestation = build_runtime_attestation(
        stage="stage_a_train_selection",
        mode=execution_mode,
        group_count=TRAIN_GROUP_COUNT,
        pair_count=TRAIN_GROUP_COUNT,
        candidate_count=len(candidate_grid()),
        seed_count=1,
        fixture_sha256=artifact["train_fixture_sha256"],
        candidate_sha256=selection_sha,
        source_sha256=str(source_sha256),
        addendum_sha256=artifact["addendum_sha256"],
        cli_sha256=resolved_cli_sha,
        resources=artifact["resources"],
        operation_counts=(
            artifact["resources"].get("operation_counts")
            if execution_mode == "real" and isinstance(artifact["resources"], Mapping)
            else None
        ),
    )
    artifact["runtime_attestation"] = attestation
    artifact["attestation_sha256"] = attestation["attestation_sha256"]
    artifact["artifact_sha256"] = canonical_digest(artifact, "artifact_sha256")
    return artifact


def run_real_stage_a(
    rows: Sequence[Mapping[str, Any]],
    addendum: Mapping[str, Any],
    *,
    source_sha256: str,
    runtime: Mapping[str, Any],
    cli_sha256: str | None = None,
) -> dict[str, Any]:
    """Run the train-only real boundary through an injected runtime.

    The runtime owns model loading, hooks, captures, interventions, and
    cleanup.  Keeping it injected makes behavioral tests deterministic while
    the CLI can refuse to fabricate a result when CUDA is unavailable.
    """
    scorer = runtime.get("score")
    resources = runtime.get("resources")
    if not callable(scorer) or not isinstance(resources, Mapping):
        return build_stage_a_artifact(
            rows,
            addendum,
            source_sha256=source_sha256,
            scorer=lambda _row, _layer, _offset: 0.0,
            execution_mode="synthetic",
            cli_sha256=cli_sha256,
        )
    try:
        return build_stage_a_artifact(
            rows,
            addendum,
            source_sha256=source_sha256,
            scorer=cast(ScoreFunction, scorer),
            resources=resources,
            execution_mode="real",
            cli_sha256=cli_sha256,
        )
    except (TypeError, ValueError, OverflowError):
        return build_stage_a_artifact(
            rows,
            addendum,
            source_sha256=source_sha256,
            scorer=lambda _row, _layer, _offset: 0.0,
            execution_mode="synthetic",
            cli_sha256=cli_sha256,
        )


__all__ = [
    "ScoreFunction",
    "build_stage_a_artifact",
    "default_train_score",
    "outer_folds",
    "run_real_stage_a",
    "select_stage_a",
]

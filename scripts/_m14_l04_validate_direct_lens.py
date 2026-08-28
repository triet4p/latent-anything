"""Strict validation for the support-only real direct logit-lens payload."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from scripts._m14_l04_fixture_contract import read_fixture
from scripts._m14_l04_ig_metrics import metric

SEEDS = [17, 29, 41, 53, 67]
LAYERS = list(range(13))
BOOTSTRAP_REPLICATES = 2000
METRICS = {
    "terminal_logit_parity": ("direct_parity_atol", "<="),
    "terminal_logit_relative_parity": ("direct_parity_rtol", "<="),
}
CONTROL_NAMES = {
    "target_non_target_selectivity",
    "shuffled_target_labels",
    "randomized_target_tokens",
    "terminal_post_ln_f_parity",
}


def _finite(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _metric_errors(value: object, label: str, threshold: float, comparator: str, *, statistic: str) -> list[str]:
    if not isinstance(value, Mapping):
        return [f"real direct lens metric {label} is not an object"]
    errors: list[str] = []
    point = value.get("point_estimate")
    interval = value.get("confidence_interval_95")
    if not _finite(point):
        errors.append(f"real direct lens metric {label} point estimate is not finite")
    if not isinstance(interval, list) or len(interval) != 2 or not all(_finite(item) for item in interval):
        errors.append(f"real direct lens metric {label} CI is invalid")
    elif float(interval[0]) > float(interval[1]):
        errors.append(f"real direct lens metric {label} CI is not ordered")
    if value.get("units") != "dimensionless" or value.get("aggregation_unit") != "independent causal group":
        errors.append(f"real direct lens metric {label} units/aggregation are invalid")
    if value.get("statistic") != statistic or value.get("comparator") != comparator:
        errors.append(f"real direct lens metric {label} statistic/comparator is not frozen")
    if not _finite(value.get("threshold")) or float(cast(float, value["threshold"])) != threshold:
        errors.append(f"real direct lens metric {label} threshold is not frozen")
    if not isinstance(value.get("pass"), bool):
        errors.append(f"real direct lens metric {label} pass is not boolean")
    if _finite(point) and _finite(value.get("threshold")):
        expected = float(cast(float, point)) <= float(cast(float, value["threshold"]))
        if value.get("pass") is not expected:
            errors.append(f"real direct lens metric {label} pass is inconsistent")
    return errors


def _rows_and_parity(
    raw: object, artifact: Mapping[str, Any]
) -> tuple[list[str], list[float], list[float], list[float], list[float], list[float]]:
    errors: list[str] = []
    abs_values: list[float] = []
    rel_values: list[float] = []
    heldout_margins: list[float] = []
    shuffled_margins: list[float] = []
    randomized_margins: list[float] = []
    try:
        _raw_fixture, fixture_rows = read_fixture(Path("artifacts/m14/l04-prompt-factor-fixture.jsonl"))
    except (OSError, ValueError):
        return (
            ["real direct lens frozen fixture cannot be loaded"],
            abs_values,
            rel_values,
            heldout_margins,
            shuffled_margins,
            randomized_margins,
        )
    expected = {
        str(row["row_id"]): (str(row["group_id"]), str(row["split"]), str(row["causal_pair_id"]))
        for row in fixture_rows
    }
    if not isinstance(raw, list) or [item.get("seed") for item in raw if isinstance(item, Mapping)] != SEEDS:
        return (
            ["real direct lens per-seed summaries are incomplete"],
            abs_values,
            rel_values,
            heldout_margins,
            shuffled_margins,
            randomized_margins,
        )
    row_keys = {
        "row_id",
        "group_id",
        "split",
        "causal_pair_id",
        "target_position",
        "target_token_id",
        "other_token_id",
        "target_probabilities",
        "other_probabilities",
        "target_margin",
        "shuffled_target_margin",
        "randomized_target_margin",
        "finite",
    }
    for summary in raw:
        if not isinstance(summary, Mapping) or set(summary) != {
            "seed",
            "layer_indices",
            "native_hidden_state_indices",
            "rows",
            "terminal_logit_max_abs_error",
            "terminal_logit_max_relative_error",
        }:
            errors.append("real direct lens per-seed summary schema is invalid")
            continue
        if summary.get("layer_indices") != LAYERS or summary.get("native_hidden_state_indices") != LAYERS:
            errors.append("real direct lens layer/native index provenance is invalid")
        if not _finite(summary.get("terminal_logit_max_abs_error")) or not _finite(
            summary.get("terminal_logit_max_relative_error")
        ):
            errors.append("real direct lens parity values are not finite")
        else:
            abs_values.append(float(summary["terminal_logit_max_abs_error"]))
            rel_values.append(float(summary["terminal_logit_max_relative_error"]))
        values = summary.get("rows")
        if not isinstance(values, list) or len(values) != len(expected):
            errors.append("real direct lens row coverage is invalid")
            continue
        observed: dict[str, tuple[str, str, str]] = {}
        summary_holdout: dict[str, list[float]] = {}
        summary_shuffled: dict[str, list[float]] = {}
        summary_randomized: dict[str, list[float]] = {}
        for row in values:
            if not isinstance(row, Mapping) or set(row) != row_keys:
                errors.append("real direct lens row schema is invalid")
                continue
            row_id = row.get("row_id")
            if not isinstance(row_id, str) or row_id in observed:
                errors.append("real direct lens row IDs are not unique")
                continue
            observed[row_id] = (str(row.get("group_id")), str(row.get("split")), str(row.get("causal_pair_id")))
            if observed[row_id] != expected.get(row_id):
                errors.append("real direct lens row provenance is invalid")
            if any("prompt" in str(key).lower() for key in row):
                errors.append("real direct lens summaries must not contain prompt text")
            target = row.get("target_probabilities")
            other = row.get("other_probabilities")
            if (
                not isinstance(target, list)
                or not isinstance(other, list)
                or len(target) != 13
                or len(other) != 13
                or not all(_finite(item) and 0.0 <= float(item) <= 1.0 for item in [*target, *other])
            ):
                errors.append("real direct lens layerwise probabilities are invalid")
                continue
            if not _finite(row.get("target_margin")) or float(row["target_margin"]) != float(target[-1]) - float(
                other[-1]
            ):
                errors.append("real direct lens target margin is not recomputed from probabilities")
            if not _finite(row.get("shuffled_target_margin")) or not _finite(row.get("randomized_target_margin")):
                errors.append("real direct lens null margins are invalid")
            if row.get("finite") is not True:
                errors.append("real direct lens finite flag is invalid")
            if row.get("target_token_id") == row.get("other_token_id") or not isinstance(
                row.get("target_position"), int
            ):
                errors.append("real direct lens token/position provenance is invalid")
            if row.get("split") == "holdout":
                group_id = str(row["group_id"])
                summary_holdout.setdefault(group_id, []).append(float(row["target_margin"]))
                summary_shuffled.setdefault(group_id, []).append(float(row["shuffled_target_margin"]))
                summary_randomized.setdefault(group_id, []).append(float(row["randomized_target_margin"]))
        for group_id in sorted(summary_holdout):
            heldout_margins.append(float(sum(summary_holdout[group_id]) / len(summary_holdout[group_id])))
            shuffled_margins.append(float(sum(summary_shuffled[group_id]) / len(summary_shuffled[group_id])))
            randomized_margins.append(float(sum(summary_randomized[group_id]) / len(summary_randomized[group_id])))
        if observed != expected:
            errors.append("real direct lens row group coverage is invalid")
    if len(heldout_margins) != 20:
        # The frozen fixture has four held-out groups per seed.
        errors.append("real direct lens held-out row coverage is invalid")
    if not isinstance(artifact.get("fixture"), Mapping) or artifact["fixture"].get("rows") != 24:
        errors.append("real direct lens fixture linkage is invalid")
    return errors, abs_values, rel_values, heldout_margins, shuffled_margins, randomized_margins


def validate_real_direct_lens_execution(
    entry: Mapping[str, Any], artifact: Mapping[str, Any], plan: Mapping[str, Any]
) -> list[str]:
    """Validate all direct-lens parity, provenance, controls, and raw links."""
    errors: list[str] = []
    if entry.get("status") != "passed_real_cuda":
        return ["real direct lens validation requires passed_real_cuda status"]
    if entry.get("support_only") is not True or entry.get("evidence_eligible") is not True:
        errors.append("direct lens must remain support-only while retaining runtime eligibility")
    if entry.get("acceptance") is not True or entry.get("evidence_level") != "D0":
        errors.append("direct lens must not be promoted")
    if entry.get("layer") != 6 or entry.get("native_hidden_state_index") != 7 or entry.get("seeds") != SEEDS:
        errors.append("direct lens layer/seed linkage is invalid")
    token_ids = entry.get("token_ids")
    if (
        not isinstance(token_ids, Mapping)
        or set(token_ids) != {" true", " false"}
        or not all(isinstance(value, int) and not isinstance(value, bool) for value in token_ids.values())
        or token_ids[" true"] == token_ids[" false"]
    ):
        errors.append("direct lens target token IDs are invalid")
    if entry.get("target_token_strings") != {" true": " true", " false": " false"}:
        errors.append("direct lens target token strings are invalid")
    metrics = entry.get("metrics")
    thresholds = plan["thresholds_and_controls"]["lens"]
    if not isinstance(metrics, Mapping) or set(metrics) != set(METRICS):
        errors.append("real direct lens metrics have the wrong schema")
    else:
        for name, (threshold_key, comparator) in METRICS.items():
            errors.extend(
                _metric_errors(metrics[name], name, float(thresholds[threshold_key]), comparator, statistic="median")
            )
    raw_errors, abs_values, rel_values, heldout_margins, shuffled_margins, randomized_margins = _rows_and_parity(
        artifact.get("raw_summaries"), artifact
    )
    errors.extend(raw_errors)
    if len(abs_values) == len(SEEDS) and isinstance(metrics, Mapping):
        expected_abs = metric(
            abs_values,
            seed=SEEDS[0],
            threshold=float(thresholds["direct_parity_atol"]),
            comparator="<=",
            statistic="median",
        )
        expected_rel = metric(
            rel_values,
            seed=SEEDS[1],
            threshold=float(thresholds["direct_parity_rtol"]),
            comparator="<=",
            statistic="median",
        )
        for key, expected_value in (
            ("terminal_logit_parity", expected_abs),
            ("terminal_logit_relative_parity", expected_rel),
        ):
            actual = metrics.get(key)
            if not isinstance(actual, Mapping) or any(
                actual.get(field) != expected_value.get(field)
                for field in ("point_estimate", "confidence_interval_95", "threshold", "comparator", "pass")
            ):
                errors.append(f"real direct lens {key} is not recomputed from raw summaries")
    diagnostics = entry.get("diagnostics")
    if not isinstance(diagnostics, Mapping) or set(diagnostics) != {"heldout_target_non_target_selectivity"}:
        errors.append("real direct lens diagnostics have the wrong schema")
    elif not isinstance(diagnostics["heldout_target_non_target_selectivity"], Mapping):
        errors.append("real direct lens selectivity diagnostic is invalid")
    elif len(heldout_margins) == 20:
        grouped = [
            float(sum(heldout_margins[offset + index] for offset in range(0, len(heldout_margins), 4)) / len(SEEDS))
            for index in range(4)
        ]
        expected_selectivity = metric(grouped, seed=SEEDS[2], threshold=0.0, comparator=">")
        observed = diagnostics["heldout_target_non_target_selectivity"]
        if any(
            observed.get(field) != expected_selectivity.get(field)
            for field in ("point_estimate", "confidence_interval_95", "threshold", "comparator", "pass")
        ):
            errors.append("real direct lens selectivity is not recomputed from held-out groups")
    controls = entry.get("controls")
    if not isinstance(controls, Mapping) or set(controls) != CONTROL_NAMES:
        errors.append("real direct lens controls have the wrong schema")
    else:
        metrics_map = cast(Mapping[str, Any], metrics)
        for name, control in controls.items():
            if (
                not isinstance(control, Mapping)
                or not isinstance(control.get("metrics"), Mapping)
                or not isinstance(control.get("pass"), bool)
            ):
                errors.append(f"real direct lens control {name} schema is invalid")
                continue
            expected_metrics = (
                {"max_abs_error", "max_relative_error"} if name == "terminal_post_ln_f_parity" else {"finite_fraction"}
            )
            if set(control["metrics"]) != expected_metrics:
                errors.append(f"real direct lens control {name} metric schema is invalid")
                continue
            if name == "terminal_post_ln_f_parity":
                expected = {
                    "max_abs_error": metrics_map.get("terminal_logit_parity"),
                    "max_relative_error": metrics_map.get("terminal_logit_relative_parity"),
                }
                for key, value in expected.items():
                    if control["metrics"].get(key) != value:
                        errors.append(f"real direct lens control {name} is not linked to parity metrics")
                expected_pass = all(
                    isinstance(value, Mapping) and value.get("pass") is True for value in expected.values()
                )
            else:
                value = control["metrics"]["finite_fraction"]
                if not isinstance(value, Mapping) or value.get("threshold") != 1.0 or value.get("comparator") != ">=":
                    errors.append(f"real direct lens control {name} finite gate is invalid")
                expected_pass = isinstance(value, Mapping) and value.get("pass") is True
            if control.get("pass") is not expected_pass:
                errors.append(f"real direct lens control {name} pass is inconsistent")
    raw_control = entry.get("control_raw")
    if not isinstance(raw_control, Mapping) or set(raw_control) != {
        "holdout_group_margins",
        "shuffled_group_margins",
        "randomized_group_margins",
    }:
        errors.append("real direct lens control raw schema is invalid")
    else:
        for key in raw_control:
            values = raw_control[key]
            if not isinstance(values, list) or len(values) != 20 or not all(_finite(value) for value in values):
                errors.append(f"real direct lens raw control {key} coverage is invalid")
        expected_raw = {
            "holdout_group_margins": heldout_margins,
            "shuffled_group_margins": shuffled_margins,
            "randomized_group_margins": randomized_margins,
        }
        for key, expected in expected_raw.items():
            if raw_control.get(key) != expected:
                errors.append(f"real direct lens raw control {key} is not linked to raw summaries")
    provenance = entry.get("provenance")
    required_provenance = {
        "runtime": "real TransformerLMIntegration",
        "model_id": plan["model"]["id"],
        "model_revision": plan["model"]["revision"],
        "target_position": "last non-padding token",
        "native_layer_indices": LAYERS,
        "network": "enabled",
        "deterministic_algorithms": True,
        "aggregation_unit": "independent causal group",
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
    }
    if not isinstance(provenance, Mapping) or any(
        provenance.get(key) != value for key, value in required_provenance.items()
    ):
        errors.append("real direct lens runtime provenance is incomplete")
    if (
        artifact.get("accepted_record_ids") != []
        or artifact.get("accepted_gap_ids") != []
        or artifact.get("evidence_level") != "D0"
    ):
        errors.append("direct lens support-only execution cannot promote a ledger record")
    if artifact.get("use_case") != "DirectLogitLens":
        errors.append("direct lens artifact use-case linkage is invalid")
    if isinstance(token_ids, Mapping):
        for summary in artifact.get("raw_summaries", []):
            if isinstance(summary, Mapping) and isinstance(summary.get("rows"), list):
                for row in summary["rows"]:
                    if isinstance(row, Mapping) and row.get("target_token_id") not in token_ids.values():
                        errors.append("direct lens row token ID is not linked to target provenance")
    return errors


__all__ = ["validate_real_direct_lens_execution"]

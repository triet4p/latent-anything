"""Unpromoting artifact builders for the L04 dispatcher."""

from __future__ import annotations

import hashlib
import math
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

from scripts._m14_l04_boundary import INTEGRATION_FACTORY
from scripts._m14_l04_contract_common import canonical_json_bytes
from scripts._m14_l04_digest import canonical_digest, code_sha, source_digests
from scripts._m14_l04_fixture_contract import FIXTURE_PATH, fixture_digests, read_fixture
from scripts.m14_l04_contract import plan_digest

TCAV_RECORD_ID = "t05_tcav"
TCAV_GAP_ID = "THY-T05-CONCEPT-ACTIVATION-VECTORS-TCAV-KIM-ET-AL-2018"
TUNED_RECORD_ID = "THY-T05-LOGIT-LENS-TUNED-LENS"
TUNED_GAP_ID = "THY-T05-LOGIT-LENS-TUNED-LENS"
DISENTANGLEMENT_RECORD_ID = "THY-T03-DISENTANGLEMENT"
DISENTANGLEMENT_GAP_ID = "THY-T03-DISENTANGLEMENT"
ACTIVATION_PATCHING_RECORD_ID = "THY-T05-ACTIVATION-PATCHING"
ACTIVATION_PATCHING_GAP_ID = "THY-T05-ACTIVATION-PATCHING"
PHASE_A_NON_PROMOTING_USE_CASES = frozenset({"AdditiveSteering"})
ADDITIVE_COMPLETED_STATUS = "completed_real_cuda_d0"
ADDITIVE_SEEDS = (17, 29, 41, 53, 67)
ADDITIVE_STRENGTHS = ("0.0", "0.25", "0.5", "1.0")
ADDITIVE_METRIC_FIELDS: set[str] = {
    "point_estimate",
    "confidence_interval_95",
    "units",
    "aggregation_unit",
    "statistic",
    "threshold",
    "comparator",
    "pass",
}
ADDITIVE_RESOURCE_FIELDS: set[str] = {
    "device",
    "network",
    "resource_peak",
    "cleanup",
    "cleanup_complete",
    "cleanup_error",
    "execution_attempted",
    "execution_backend",
    "stage",
    "failure_stage",
}
ADDITIVE_PEAK_FIELDS: set[str] = {
    "cuda_device",
    "elapsed_seconds",
    "max_memory_allocated_bytes",
    "max_memory_reserved_bytes",
    "max_rss_bytes",
    "rss_source",
    "rss_unit",
}
ADDITIVE_PROVENANCE_FIELDS = frozenset(
    {
        "runtime",
        "model_revision",
        "target_token_ids",
        "target_token_strings",
        "target_position",
        "direction_fit",
        "network",
        "device",
        "execution_attempted",
        "execution_backend",
        "stage",
        "deterministic_algorithms",
        "runtime_versions",
        "resource_peak",
        "budget_pass",
        "cleanup",
        "cleanup_complete",
        "model_parameter_digest_before",
        "model_parameter_digest_after",
        "model_parameter_digest_algorithm",
        "bootstrap_replicates",
        "aggregation_unit",
        "off_target_aggregation",
        "use_case",
        "shuffled_label_policy",
        "shuffled_label_cardinality",
        "shuffled_label_identity_assignment",
        "execution_result_digest",
    }
)
ADDITIVE_LINKED_FIELDS = frozenset(
    {
        "metrics",
        "controls",
        "direction_norm",
        "layer",
        "native_hidden_state_index",
        "token_ids",
        "target_token_strings",
        "seeds",
        "strength_grid",
        "train_groups",
        "holdout_groups",
        "resources",
        "provenance",
        "criteria",
        "raw_summaries",
        "holdout_evidence",
        "semantic_candidate",
        "no_mutation",
        "budget_pass",
        "model_parameter_digest_before",
        "model_parameter_digest_after",
        "failure_reason",
    }
)
ADDITIVE_EXECUTION_BASE_FIELDS = frozenset(
    {
        "use_case",
        "record_id",
        "support_only",
        "model",
        "integration",
        "adapter",
        "status",
        "evidence_level",
        "evidence_eligible",
        "acceptance",
        "failure_ref",
        "metrics",
        "controls",
    }
)
ADDITIVE_RECORD_BASE_FIELDS = frozenset(
    {
        "record_id",
        "capability",
        "evidence_level",
        "status",
        "layer",
        "native_hidden_state_index",
        "token_ids",
        "metrics",
        "controls",
        "acceptance",
        "failure_ref",
        "confidence_intervals",
        "seed",
    }
)
ADDITIVE_ARTIFACT_BASE_FIELDS = frozenset(
    {
        "schema_version",
        "lane",
        "use_case",
        "accepted_gap_ids",
        "accepted_record_ids",
        "evidence_level",
        "partial_promotion",
        "model",
        "integration",
        "adapter",
        "fixture",
        "tokenization",
        "split",
        "executions",
        "records",
        "controls",
        "provenance",
        "artifact_sha256",
        "plan_sha256",
    }
)
_ADDITIVE_SENSITIVE_KEYS = frozenset({"prompt", "path", "cache", "raw_output", "credentials", "weights"})
_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ROW_RE = re.compile(r"^l04-g(?:0[1-9]|1[0-2])-(?:clean|corrupted)$")
_SAFE_GROUP_RE = re.compile(r"^g(?:0[1-9]|1[0-2])$")
_SAFE_PAIR_RE = re.compile(r"^p(?:0[1-9]|1[0-2])$")
_SAFE_CUDA_DEVICE_RE = re.compile(r"^cuda(?::(?:0|[1-9][0-9]*))?$")
_ADDITIVE_HOLDOUT_GROUPS = tuple(f"g{i:02d}" for i in range(9, 13))
_ADDITIVE_TRAIN_GROUPS = frozenset(f"g{i:02d}" for i in range(1, 9))


def _keys(value: dict[str, Any]) -> set[str]:
    keys: set[str] = set()
    for key in value:
        if type(key) is not str:
            raise ValueError("additive mapping keys must be strings")
        keys.add(key)
    return keys


def _reject_sensitive_keys(value: Any, *, path: tuple[str, ...] = ()) -> None:
    """Reject untrusted sensitive fields in retained additive payloads."""
    if isinstance(value, dict):
        for key, nested in value.items():
            key_text = str(key)
            # The dispatcher has two fixed declarations with these names:
            # fixture.path identifies the authored fixture and provenance.credentials
            # records that credentials were not used.  They are not handler output.
            if key_text.lower() in _ADDITIVE_SENSITIVE_KEYS and path + (key_text,) not in {
                ("fixture", "path"),
                ("provenance", "credentials"),
            }:
                raise ValueError(f"additive sensitive field {key_text!r} is not retained")
            _reject_sensitive_keys(nested, path=path + (key_text,))
    elif isinstance(value, list):
        for nested in value:
            _reject_sensitive_keys(nested, path=path)


def _finite_number(value: Any, label: str, *, non_negative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"additive {label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (non_negative and result < 0):
        raise ValueError(f"additive {label} is non-finite or negative")
    return result


def _execution_result_digest(value: dict[str, Any]) -> str:
    """Hash the canonical handler result without its derived provenance field."""
    unsigned = deepcopy(value)
    unsigned.pop("execution_result_digest", None)
    provenance = unsigned.get("provenance")
    if isinstance(provenance, dict):
        provenance.pop("execution_result_digest", None)
    try:
        payload = canonical_json_bytes(unsigned)
    except (TypeError, ValueError) as exc:
        raise ValueError("additive execution result digest cannot be computed") from exc
    return hashlib.sha256(payload).hexdigest()


def _metric(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or _keys(value) != ADDITIVE_METRIC_FIELDS:
        raise ValueError(f"additive {label} metric schema is invalid")
    _finite_number(value["point_estimate"], f"{label}.point_estimate")
    interval = value["confidence_interval_95"]
    if not isinstance(interval, list) or len(interval) != 2:
        raise ValueError(f"additive {label} confidence interval is invalid")
    for index, item in enumerate(interval):
        _finite_number(item, f"{label}.confidence_interval_95[{index}]")
    if value["units"] not in {"logits", "absolute logit margin difference"}:
        raise ValueError(f"additive {label} units are invalid")
    if value["aggregation_unit"] != "independent causal group" or value["statistic"] != "mean":
        raise ValueError(f"additive {label} aggregation is invalid")
    _finite_number(value["threshold"], f"{label}.threshold")
    if value["comparator"] not in {"<=", "<", ">=", ">"} or not isinstance(value["pass"], bool):
        raise ValueError(f"additive {label} verdict is invalid")
    return deepcopy(value)


def _seed_metrics(value: Any, label: str) -> dict[str, dict[str, Any]]:
    expected = {str(seed) for seed in ADDITIVE_SEEDS}
    if not isinstance(value, dict) or _keys(value) != expected:
        raise ValueError(f"additive {label} seed mapping is invalid")
    return {seed: _metric(value[seed], f"{label}[{seed}]") for seed in sorted(value)}


def _sanitize_controls(value: Any) -> dict[str, Any]:
    names = ("zero_strength", "randomized_direction", "shuffled_labels", "off_target_token", "matched_norm_direction")
    if not isinstance(value, dict) or _keys(value) != set(names):
        raise ValueError("additive controls schema is invalid")
    result: dict[str, Any] = {}
    for name in names:
        control = value[name]
        if (
            not isinstance(control, dict)
            or _keys(control) != {"pass", "by_seed"}
            or not isinstance(control["pass"], bool)
        ):
            raise ValueError(f"additive control {name} schema is invalid")
        result[name] = {"pass": control["pass"], "by_seed": _seed_metrics(control["by_seed"], f"control {name}")}
    return result


def _sanitize_raw_summaries(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) != len(ADDITIVE_SEEDS):
        raise ValueError("additive raw summaries must contain five seeds")
    result: list[dict[str, Any]] = []
    for summary in value:
        fields = {
            "seed",
            "holdout_groups",
            "target_effect",
            "selectivity",
            "off_target_token",
            "off_target_pair_changes",
            "zero_strength_identity",
            "strength_curve",
            "control_effects",
            "control_direction_norms",
            "finite",
            "shuffled_label_provenance",
        }
        if not isinstance(summary, dict) or _keys(summary) != fields:
            raise ValueError("additive raw summary schema is invalid")
        seed = summary["seed"]
        if type(seed) is not int or seed not in ADDITIVE_SEEDS:
            raise ValueError("additive raw summary seed is invalid")
        groups = summary["holdout_groups"]
        if not isinstance(groups, list) or groups != [f"g{i:02d}" for i in range(9, 13)]:
            raise ValueError("additive holdout groups are invalid")
        for name in ("target_effect", "selectivity", "off_target_token", "zero_strength_identity"):
            _metric(summary[name], f"raw summary {name}")
        changes = summary["off_target_pair_changes"]
        if not isinstance(changes, list) or len(changes) != 4:
            raise ValueError("additive off-target pair changes are invalid")
        for item in changes:
            _finite_number(item, "off-target pair change", non_negative=True)
        curve = summary["strength_curve"]
        if not isinstance(curve, dict) or _keys(curve) != set(ADDITIVE_STRENGTHS):
            raise ValueError("additive strength curve schema is invalid")
        for strength in ADDITIVE_STRENGTHS:
            _metric(curve[strength], f"strength curve {strength}")
        effects = summary["control_effects"]
        if not isinstance(effects, dict) or _keys(effects) != {"randomized", "shuffled", "matched_norm"}:
            raise ValueError("additive control effects schema is invalid")
        for name, metric_value in effects.items():
            _metric(metric_value, f"control effect {name}")
        norms = summary["control_direction_norms"]
        if not isinstance(norms, dict) or _keys(norms) != {"randomized", "shuffled", "matched_norm"}:
            raise ValueError("additive control direction norms schema is invalid")
        for name, norm in norms.items():
            _finite_number(norm, f"control direction norm {name}", non_negative=True)
        if not isinstance(summary["finite"], bool):
            raise ValueError("additive finite marker is invalid")
        shuffled = summary["shuffled_label_provenance"]
        if (
            not isinstance(shuffled, dict)
            or _keys(shuffled)
            != {"policy", "permutation_digest", "row_count", "positive_count", "negative_count", "identity_permutation"}
            or shuffled["policy"]
            != "uniform permutation of balanced train-example labels; independent of fitted labels"
            or not isinstance(shuffled["permutation_digest"], str)
            or not _DIGEST_RE.fullmatch(shuffled["permutation_digest"])
            or shuffled["row_count"] != 16
            or shuffled["positive_count"] != 8
            or shuffled["negative_count"] != 8
            or shuffled["identity_permutation"] is not False
        ):
            raise ValueError("additive shuffled-label permutation provenance is invalid")
        result.append(
            {
                "seed": seed,
                "holdout_groups": list(groups),
                "target_effect": _metric(summary["target_effect"], "target effect"),
                "selectivity": _metric(summary["selectivity"], "selectivity"),
                "off_target_token": _metric(summary["off_target_token"], "off-target token"),
                "off_target_pair_changes": [float(item) for item in changes],
                "zero_strength_identity": _metric(summary["zero_strength_identity"], "zero strength"),
                "strength_curve": {
                    strength: _metric(curve[strength], f"strength curve {strength}") for strength in ADDITIVE_STRENGTHS
                },
                "control_effects": {
                    name: _metric(effects[name], f"control effect {name}")
                    for name in ("randomized", "shuffled", "matched_norm")
                },
                "control_direction_norms": {
                    name: float(norms[name]) for name in ("randomized", "shuffled", "matched_norm")
                },
                "finite": summary["finite"],
                "shuffled_label_provenance": deepcopy(shuffled),
            }
        )
    if {item["seed"] for item in result} != set(ADDITIVE_SEEDS):
        raise ValueError("additive raw summary seeds are incomplete")
    return result


def _sanitize_holdout(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) != 8:
        raise ValueError("additive holdout summary must contain eight rows")
    result: list[dict[str, Any]] = []
    seen_rows: set[str] = set()
    group_conditions: dict[str, set[str]] = {}
    group_pairs: dict[str, str] = {}
    pair_rows: dict[str, dict[str, str]] = {}
    for row in value:
        fields = {"row_id", "group_id", "causal_pair_id", "condition", "split", "baseline_margin", "strength_effects"}
        if not isinstance(row, dict) or _keys(row) != fields:
            raise ValueError("additive holdout summary schema is invalid")
        if not isinstance(row["row_id"], str) or not _SAFE_ROW_RE.fullmatch(row["row_id"]):
            raise ValueError("additive holdout row identity is invalid")
        if not isinstance(row["group_id"], str) or not _SAFE_GROUP_RE.fullmatch(row["group_id"]):
            raise ValueError("additive holdout group identity is invalid")
        if not isinstance(row["causal_pair_id"], str) or not _SAFE_PAIR_RE.fullmatch(row["causal_pair_id"]):
            raise ValueError("additive holdout pair identity is invalid")
        if row["condition"] not in {"clean", "corrupted"} or row["split"] != "holdout":
            raise ValueError("additive holdout condition/split is invalid")
        _finite_number(row["baseline_margin"], "holdout baseline margin")
        effects = row["strength_effects"]
        if not isinstance(effects, dict) or _keys(effects) != set(ADDITIVE_STRENGTHS):
            raise ValueError("additive holdout strength effects are invalid")
        for strength in ADDITIVE_STRENGTHS:
            _finite_number(effects[strength], f"holdout strength {strength}")
        row_id = row["row_id"]
        group_id = row["group_id"]
        pair_id = row["causal_pair_id"]
        if row_id in seen_rows:
            raise ValueError("additive holdout row IDs must be unique")
        seen_rows.add(row_id)
        if group_id not in _ADDITIVE_HOLDOUT_GROUPS or group_id in _ADDITIVE_TRAIN_GROUPS:
            raise ValueError("additive holdout group overlaps train or is not canonical")
        conditions = group_conditions.setdefault(group_id, set())
        conditions.add(row["condition"])
        existing_pair = group_pairs.setdefault(group_id, pair_id)
        if existing_pair != pair_id:
            raise ValueError("additive holdout group/pair linkage is invalid")
        pair = pair_rows.setdefault(pair_id, {})
        if row["condition"] in pair:
            raise ValueError("additive holdout pair contains duplicate condition")
        pair[row["condition"]] = row_id
        result.append(
            {
                "row_id": row["row_id"],
                "group_id": row["group_id"],
                "causal_pair_id": row["causal_pair_id"],
                "condition": row["condition"],
                "split": row["split"],
                "baseline_margin": float(row["baseline_margin"]),
                "strength_effects": {strength: float(effects[strength]) for strength in ADDITIVE_STRENGTHS},
            }
        )
    expected_rows = {
        f"l04-{group}-{condition}" for group in _ADDITIVE_HOLDOUT_GROUPS for condition in ("clean", "corrupted")
    }
    if seen_rows != expected_rows:
        raise ValueError("additive holdout rows are not the canonical g09-g12 pairs")
    if set(group_conditions) != set(_ADDITIVE_HOLDOUT_GROUPS) or any(
        conditions != {"clean", "corrupted"} for conditions in group_conditions.values()
    ):
        raise ValueError("additive holdout groups must each contain clean and corrupted rows")
    expected_pairs = {f"p{i:02d}" for i in range(9, 13)}
    if set(pair_rows) != expected_pairs or any(set(pair) != {"clean", "corrupted"} for pair in pair_rows.values()):
        raise ValueError("additive holdout causal pairs are not canonical")
    if any(group_pairs[group] != f"p{group[1:]}" for group in _ADDITIVE_HOLDOUT_GROUPS):
        raise ValueError("additive holdout group/pair linkage is not canonical")
    return result


def _validate_fixture_metadata(value: Any) -> None:
    """Require fixture metadata to be authored, canonical, and independently rehashed."""
    expected_keys = {"path", "rows", "content_sha256", "split_sha256", "pair_sha256"}
    if not isinstance(value, dict) or _keys(value) != expected_keys:
        raise ValueError("additive fixture metadata schema is invalid")
    if value["path"] != "artifacts/m14/l04-prompt-factor-fixture.jsonl" or value["rows"] != 24:
        raise ValueError("additive fixture metadata identity is invalid")
    if not all(
        isinstance(value[name], str) and _DIGEST_RE.fullmatch(value[name]) for name in expected_keys - {"path", "rows"}
    ):
        raise ValueError("additive fixture metadata digest format is invalid")
    try:
        raw, rows = read_fixture(Path(FIXTURE_PATH))
        expected = fixture_digests(raw, rows)
    except (OSError, KeyError, StopIteration, TypeError, ValueError) as exc:
        raise ValueError("additive authored fixture digest cannot be recomputed") from exc
    for name, digest in expected.items():
        if value[name] != digest:
            raise ValueError(f"additive fixture {name} does not match authored canonical bytes")


def _validate_additive_internal_links(
    result: dict[str, Any], summaries: list[dict[str, Any]], holdout: list[dict[str, Any]], direction_norm: float
) -> None:
    """Bind derived per-seed views before any artifact fields are assembled."""
    by_seed = {summary["seed"]: summary for summary in summaries}
    if set(by_seed) != set(ADDITIVE_SEEDS):
        raise ValueError("additive seed summaries are not complete")
    metrics = result["metrics"]
    controls = result["controls"]
    for seed in ADDITIVE_SEEDS:
        summary = by_seed[seed]
        seed_key = str(seed)
        for name in ("target_effect", "selectivity", "off_target_token", "zero_strength_identity"):
            if metrics[name].get(seed_key) != summary[name]:
                raise ValueError(f"additive metric {name} does not link to raw summary")
        for name, summary_name in (
            ("zero_strength", "zero_strength_identity"),
            ("off_target_token", "off_target_token"),
        ):
            if controls[name]["by_seed"].get(seed_key) != summary[summary_name]:
                raise ValueError(f"additive control {name} does not link to raw summary")
        for name in ("randomized_direction", "shuffled_labels", "matched_norm_direction"):
            summary_name = {
                "randomized_direction": "randomized",
                "shuffled_labels": "shuffled",
                "matched_norm_direction": "matched_norm",
            }[name]
            if controls[name]["by_seed"].get(seed_key) != summary["control_effects"][summary_name]:
                raise ValueError(f"additive control {name} does not link to raw summary")
        matched = summary["control_direction_norms"]["matched_norm"]
        if not math.isclose(matched, direction_norm, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("additive matched-norm diagnostic does not equal fitted direction norm")

    # The holdout rows are the sole source for the metric point estimates.
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in holdout:
        grouped.setdefault(row["group_id"], []).append(row)
    if set(grouped) != set(_ADDITIVE_HOLDOUT_GROUPS) or any(len(rows) != 2 for rows in grouped.values()):
        raise ValueError("additive holdout derivation groups are incomplete")
    for seed in ADDITIVE_SEEDS:
        summary = by_seed[seed]
        for strength in ADDITIVE_STRENGTHS:
            group_values = [
                sum(row["strength_effects"][strength] for row in grouped[group]) / 2.0
                for group in _ADDITIVE_HOLDOUT_GROUPS
            ]
            observed = summary["strength_curve"][strength]["point_estimate"]
            if not math.isclose(observed, sum(group_values) / len(group_values), rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError("additive strength-curve metric does not derive from holdout evidence")
        target_values = [
            sum(row["strength_effects"]["1.0"] for row in grouped[group]) / 2.0 for group in _ADDITIVE_HOLDOUT_GROUPS
        ]
        zero_values = [
            sum(abs(row["strength_effects"]["0.0"]) for row in grouped[group]) / 2.0
            for group in _ADDITIVE_HOLDOUT_GROUPS
        ]
        off_values = list(summary["off_target_pair_changes"])
        selectivity_values = [target - off for target, off in zip(target_values, off_values, strict=True)]
        derived = {
            "target_effect": sum(target_values) / 4.0,
            "selectivity": sum(selectivity_values) / 4.0,
            "off_target_token": sum(off_values) / 4.0,
            "zero_strength_identity": sum(zero_values) / 4.0,
        }
        for name, point in derived.items():
            if not math.isclose(summary[name]["point_estimate"], point, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError(f"additive raw summary {name} does not derive from holdout evidence")


def _sanitize_resources(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or not set(value).issubset(ADDITIVE_RESOURCE_FIELDS):
        raise ValueError("additive resource schema contains unexpected fields")
    required = {"device", "network", "resource_peak", "cleanup", "execution_attempted", "execution_backend", "stage"}
    if not required.issubset(value) or not isinstance(value["execution_attempted"], bool):
        raise ValueError("additive resource schema is incomplete")
    if value["execution_backend"] not in {"cuda", "none"} or value["stage"] not in {
        "dispatch",
        "preflight",
        "dependency_check",
        "cuda_check",
        "model_load",
        "scoring",
        "cleanup",
        "complete",
        "execution",
    }:
        raise ValueError("additive resource execution tuple is invalid")
    if value["execution_attempted"] != (value["execution_backend"] == "cuda"):
        raise ValueError("additive resource execution tuple is incoherent")
    if not isinstance(value["network"], str) or value["network"] not in {"enabled", "not attempted"}:
        raise ValueError("additive resource network is invalid")
    if (
        not isinstance(value["cleanup"], str)
        or value["cleanup"] not in {"pending", "not applicable; no model was loaded"}
        and not value["cleanup"].startswith(("CUDA synchronized;", "failure cleanup incomplete:"))
    ):
        raise ValueError("additive resource cleanup is invalid")
    if "cleanup_complete" in value and not isinstance(value["cleanup_complete"], bool):
        raise ValueError("additive cleanup marker is invalid")
    if value.get("cleanup_complete") is False and (
        "cleanup_error" not in value
        or not isinstance(value["cleanup_error"], str)
        or not re.fullmatch(r"[A-Za-z][A-Za-z0-9]*(?:Error|Exception)", value["cleanup_error"])
    ):
        raise ValueError("additive cleanup error marker is invalid")
    if "cleanup_error" in value and value.get("cleanup_complete") is not False:
        raise ValueError("additive cleanup error marker is incoherent")
    if value.get("cleanup_complete") is True and not value["cleanup"].startswith("CUDA synchronized;"):
        raise ValueError("additive completed cleanup marker is incoherent")
    failure_stage = value.get("failure_stage")
    if failure_stage is not None and failure_stage not in {
        "dispatch",
        "preflight",
        "dependency_check",
        "cuda_check",
        "model_load",
        "scoring",
        "cleanup",
        "complete",
        "execution",
    }:
        raise ValueError("additive failure stage marker is invalid")
    if failure_stage == "preflight" and value["execution_attempted"]:
        raise ValueError("attempted additive execution cannot claim preflight failure stage")
    peak = value["resource_peak"]
    if not value["execution_attempted"]:
        if (
            peak != "not measured"
            or value["device"] not in {"not used", "not attempted"}
            or value["network"] != "not attempted"
        ):
            raise ValueError("additive pre-CUDA resource provenance is invalid")
        return deepcopy(value)
    if (
        not isinstance(value["device"], str)
        or _SAFE_CUDA_DEVICE_RE.fullmatch(value["device"]) is None
        or value["network"] != "enabled"
    ):
        raise ValueError("additive CUDA resource provenance is invalid")
    # A post-CUDA failure may occur before the peak sampler runs. Preserve the
    # attempted backend/stage and admit only this explicit partial envelope.
    if peak == "not measured":
        if value["stage"] == "complete":
            raise ValueError("additive completed resource peak is missing")
        if value.get("cleanup_complete") is True:
            raise ValueError("additive partial resource peak cannot claim completed cleanup")
        return deepcopy(value)
    if (
        not isinstance(peak, dict)
        or _keys(peak) != ADDITIVE_PEAK_FIELDS
        or peak["cuda_device"] != value["device"]
        or _SAFE_CUDA_DEVICE_RE.fullmatch(peak["cuda_device"]) is None
        or peak["rss_unit"] != "bytes"
        or peak["rss_source"] not in {"resource.getrusage(RUSAGE_SELF).ru_maxrss", "psutil.Process.memory_info().rss"}
    ):
        raise ValueError("additive resource peak schema is invalid")
    elapsed = _finite_number(peak["elapsed_seconds"], "elapsed_seconds", non_negative=True)
    allocated = _finite_number(peak["max_memory_allocated_bytes"], "allocated bytes", non_negative=True)
    reserved = _finite_number(peak["max_memory_reserved_bytes"], "reserved bytes", non_negative=True)
    rss = _finite_number(peak["max_rss_bytes"], "RSS bytes", non_negative=True)
    if any(
        type(peak[name]) is not int
        for name in ("max_memory_allocated_bytes", "max_memory_reserved_bytes", "max_rss_bytes")
    ):
        raise ValueError("additive resource byte counters must be integers")
    if elapsed > 1800.0 or allocated > 6 * 1024**3 or reserved > 6 * 1024**3 or rss > 4 * 1024**3:
        raise ValueError("additive resource budget exceeded")
    return deepcopy(value)


def _sanitize_provenance(value: Any, resources: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(value, dict) or not set(value).issubset(ADDITIVE_PROVENANCE_FIELDS):
        raise ValueError("additive provenance contains unexpected fields")
    required = {
        "runtime",
        "model_revision",
        "target_token_ids",
        "target_token_strings",
        "target_position",
        "direction_fit",
        "network",
        "device",
        "execution_attempted",
        "execution_backend",
        "stage",
        "deterministic_algorithms",
        "runtime_versions",
        "resource_peak",
        "budget_pass",
        "cleanup",
        "cleanup_complete",
        "model_parameter_digest_before",
        "model_parameter_digest_after",
        "model_parameter_digest_algorithm",
        "bootstrap_replicates",
        "aggregation_unit",
        "off_target_aggregation",
        "use_case",
        "shuffled_label_policy",
        "shuffled_label_cardinality",
        "shuffled_label_identity_assignment",
        "execution_result_digest",
    }
    if _keys(value) != required:
        raise ValueError("additive provenance schema is incomplete")
    if (
        value["runtime"] != "real TransformerLMIntegration"
        or value["model_revision"] != "e7da7f221d5bf496a48136c0cd264e630fe9fcc8"
        or value["target_position"] != "last non-padding token"
        or value["direction_fit"] != "clean-minus-corrupted train pairs only; normalized before holdout scoring"
        or value["model_parameter_digest_algorithm"] != "sha256/canonical-ordered-named-parameters-v1"
        or value["aggregation_unit"] != "independent causal group"
        or value["off_target_aggregation"] != "mean(abs(clean-corrupted effect per causal pair))"
        or value["use_case"] != "AdditiveSteering"
    ):
        raise ValueError("additive provenance identity is invalid")
    if value["target_token_ids"] != {" true": 2081, " false": 3991} or value["target_token_strings"] != {
        " true": " true",
        " false": " false",
    }:
        raise ValueError("additive token provenance is invalid")
    if (
        not isinstance(value["deterministic_algorithms"], bool)
        or not isinstance(value["runtime_versions"], dict)
        or _keys(value["runtime_versions"]) != {"python", "platform", "numpy", "torch", "transformers", "tokenizers"}
        or not value["runtime_versions"]
        or any(not isinstance(k, str) or not isinstance(v, str) or not v for k, v in value["runtime_versions"].items())
    ):
        raise ValueError("additive runtime provenance is invalid")
    if (
        value["network"] != resources["network"]
        or value["device"] != resources["device"]
        or value["execution_attempted"] != resources["execution_attempted"]
        or value["execution_backend"] != resources["execution_backend"]
        or value["stage"] != resources["stage"]
        or value["resource_peak"] != resources["resource_peak"]
        or value["cleanup"] != resources["cleanup"]
        or value["cleanup_complete"] != resources.get("cleanup_complete")
    ):
        raise ValueError("additive provenance/resource linkage is invalid")
    if not isinstance(value["budget_pass"], bool) or value["budget_pass"] != (
        isinstance(value["resource_peak"], dict)
        and _finite_number(value["resource_peak"]["elapsed_seconds"], "elapsed_seconds", non_negative=True) <= 1800.0
        and value["resource_peak"]["max_memory_allocated_bytes"] <= 6 * 1024**3
        and value["resource_peak"]["max_memory_reserved_bytes"] <= 6 * 1024**3
        and value["resource_peak"]["max_rss_bytes"] <= 4 * 1024**3
    ):
        raise ValueError("additive budget provenance is invalid")
    for name in ("model_parameter_digest_before", "model_parameter_digest_after"):
        if not isinstance(value[name], str) or not _DIGEST_RE.fullmatch(value[name]):
            raise ValueError(f"additive {name} format is invalid")
    if not isinstance(value["execution_result_digest"], str) or not _DIGEST_RE.fullmatch(
        value["execution_result_digest"]
    ):
        raise ValueError("additive execution result digest format is invalid")
    if value["execution_result_digest"] != _execution_result_digest(result):
        raise ValueError("additive execution result digest linkage is invalid")
    if value["model_parameter_digest_before"] != result.get("model_parameter_digest_before") or value[
        "model_parameter_digest_after"
    ] != result.get("model_parameter_digest_after"):
        raise ValueError("additive parameter digest linkage is invalid")
    if (
        value["model_parameter_digest_before"] != value["model_parameter_digest_after"]
        or result.get("no_mutation") is not True
    ):
        raise ValueError("additive parameter mutation is not proven absent")
    if type(value["bootstrap_replicates"]) is not int or value["bootstrap_replicates"] != 2000:
        raise ValueError("additive bootstrap provenance is invalid")
    if (
        value["shuffled_label_policy"]
        != "uniform permutation of balanced train-example labels; independent of fitted labels"
        or value["shuffled_label_cardinality"] != {"rows": 16, "positive": 8, "negative": 8}
        or value["shuffled_label_identity_assignment"] is not False
    ):
        raise ValueError("additive shuffled-label policy provenance is invalid")
    return deepcopy(value)


def _sanitize_additive_result(
    result: dict[str, Any], resources: dict[str, Any], plan: dict[str, Any] | None = None
) -> dict[str, Any]:
    allowed = {
        "status",
        "evidence_eligible",
        "acceptance",
        "evidence_level",
        "semantic_candidate",
        "criteria",
        "failure_reason",
        "metrics",
        "controls",
        "raw_summaries",
        "holdout_evidence",
        "token_ids",
        "target_token_strings",
        "layer",
        "native_hidden_state_index",
        "seeds",
        "strength_grid",
        "train_groups",
        "holdout_groups",
        "direction_norm",
        "no_mutation",
        "budget_pass",
        "model_parameter_digest_before",
        "model_parameter_digest_after",
        "provenance",
        "resources",
    }
    if _keys(result) != allowed:
        raise ValueError("additive execution result contains unexpected fields")
    _reject_sensitive_keys(result)
    if (
        result.get("status") not in {ADDITIVE_COMPLETED_STATUS, "failed"}
        or result.get("evidence_eligible") is not False
        or result.get("acceptance") is not False
        or result.get("evidence_level") != "D0"
    ):
        raise ValueError("additive execution result must be a non-promoting D0 result")
    if (
        not isinstance(result.get("semantic_candidate"), bool)
        or not isinstance(result.get("no_mutation"), bool)
        or result.get("no_mutation") is not True
    ):
        raise ValueError("additive execution criteria are invalid")
    if result.get("failure_reason") not in {
        None,
        "cleanup incomplete",
        "one or more additive steering criteria failed",
    }:
        raise ValueError("additive failure reason is not sanitized")
    if (
        result.get("metrics") is None
        or not isinstance(result["metrics"], dict)
        or _keys(result["metrics"]) != {"target_effect", "selectivity", "off_target_token", "zero_strength_identity"}
    ):
        raise ValueError("additive metrics schema is invalid")
    if result.get("token_ids") != {" true": 2081, " false": 3991}:
        raise ValueError("additive token IDs are not the frozen constants")
    if result.get("target_token_strings") != {" true": " true", " false": " false"}:
        raise ValueError("additive token strings are not the frozen constants")
    if type(result.get("layer")) is not int or result["layer"] != 6:
        raise ValueError("additive layer is not the frozen constant")
    if type(result.get("native_hidden_state_index")) is not int or result["native_hidden_state_index"] != 7:
        raise ValueError("additive native hidden-state index is not the frozen constant")
    if result.get("seeds") != list(ADDITIVE_SEEDS):
        raise ValueError("additive seeds are not the frozen constants")
    if result.get("strength_grid") != [float(value) for value in ADDITIVE_STRENGTHS]:
        raise ValueError("additive strength grid is not the frozen constants")
    if result.get("train_groups") != [f"g{i:02d}" for i in range(1, 9)]:
        raise ValueError("additive train groups are invalid")
    if result.get("holdout_groups") != [f"g{i:02d}" for i in range(9, 13)]:
        raise ValueError("additive holdout groups are invalid")
    metrics = {
        name: _seed_metrics(result["metrics"][name], f"metrics {name}")
        for name in ("target_effect", "selectivity", "off_target_token", "zero_strength_identity")
    }
    controls = _sanitize_controls(result.get("controls"))
    summaries = _sanitize_raw_summaries(result.get("raw_summaries"))
    holdout = _sanitize_holdout(result.get("holdout_evidence"))
    safe_resources = _sanitize_resources(result.get("resources"))
    if safe_resources.get("execution_attempted") is not True:
        raise ValueError("additive completed-result handler must record an attempted CUDA execution")
    if safe_resources != resources:
        raise ValueError("additive resource argument does not match handler result")
    safe_provenance = _sanitize_provenance(result.get("provenance"), safe_resources, result)
    if plan is not None:
        from scripts._m14_l04_plan_contract import validate_plan

        plan_errors = validate_plan(plan)
        if plan_errors:
            raise ValueError("additive plan is not the frozen plan: " + "; ".join(plan_errors))
        steering = plan["thresholds_and_controls"]["steering"]
        expected_metrics = {
            "target_effect": (float(steering["target_effect_ci_lower_strict_gt_logits"]), ">", "logits"),
            "selectivity": (float(steering["selectivity_ci_lower_strict_gt_logits"]), ">", "logits"),
            "off_target_token": (float(steering["off_target_absolute_effect_max_logits"]), "<=", "logits"),
            "zero_strength_identity": (
                float(steering["zero_strength_identity_atol"]),
                "<=",
                "absolute logit margin difference",
            ),
        }
        control_declarations = {
            "zero_strength": expected_metrics["zero_strength_identity"],
            "off_target_token": expected_metrics["off_target_token"],
            "randomized_direction": expected_metrics["off_target_token"],
            "shuffled_labels": expected_metrics["off_target_token"],
            "matched_norm_direction": expected_metrics["off_target_token"],
        }
        for metric_name, (threshold, comparator, units) in expected_metrics.items():
            for metric_value in metrics[metric_name].values():
                if (
                    metric_value["threshold"] != threshold
                    or metric_value["comparator"] != comparator
                    or metric_value["units"] != units
                ):
                    raise ValueError(f"additive {metric_name} metric is not pinned to frozen steering declarations")
        for summary in summaries:
            for metric_name, (threshold, comparator, units) in expected_metrics.items():
                metric_value = summary[metric_name]
                if (
                    metric_value["threshold"] != threshold
                    or metric_value["comparator"] != comparator
                    or metric_value["units"] != units
                ):
                    raise ValueError(
                        f"additive raw summary {metric_name} is not pinned to frozen steering declarations"
                    )
                for metric_value in summary["strength_curve"].values():
                    if (
                        metric_value["threshold"] != expected_metrics["target_effect"][0]
                        or metric_value["comparator"] != expected_metrics["target_effect"][1]
                        or metric_value["units"] != expected_metrics["target_effect"][2]
                    ):
                        raise ValueError("additive strength curve metric declaration is invalid")
                for name, metric_value in summary["control_effects"].items():
                    declaration = control_declarations[
                        {
                            "randomized": "randomized_direction",
                            "shuffled": "shuffled_labels",
                            "matched_norm": "matched_norm_direction",
                        }[name]
                    ]
                    if (
                        metric_value["threshold"] != declaration[0]
                        or metric_value["comparator"] != declaration[1]
                        or metric_value["units"] != declaration[2]
                    ):
                        raise ValueError("additive control metric declaration is invalid")
    criteria = result.get("criteria")
    if (
        not isinstance(criteria, dict)
        or set(criteria)
        != {
            "target_effect",
            "selectivity",
            "off_target",
            "zero_strength_identity",
            "controls",
            "finite",
            "budget",
            "no_mutation",
            "parameter_digest_equal",
            "cleanup_complete",
        }
        or any(not isinstance(v, bool) for v in criteria.values())
    ):
        raise ValueError("additive criteria schema is invalid")
    if (
        criteria["no_mutation"] is not True
        or criteria["parameter_digest_equal"] is not True
        or criteria["cleanup_complete"] != safe_resources.get("cleanup_complete")
    ):
        raise ValueError("additive criteria/digest linkage is invalid")
    expected_semantic_candidate = all(criteria.values())
    if result["semantic_candidate"] != expected_semantic_candidate:
        raise ValueError("additive semantic candidate is not bound to criteria")
    if result["budget_pass"] != criteria["budget"]:
        raise ValueError("additive budget result is not bound to criteria")
    direction_norm = _finite_number(result.get("direction_norm"), "direction norm")
    if direction_norm <= 0:
        raise ValueError("additive direction norm must be finite and positive")
    _validate_additive_internal_links(result, summaries, holdout, direction_norm)
    if result["status"] == ADDITIVE_COMPLETED_STATUS and safe_resources.get("cleanup_complete") is not True:
        raise ValueError("additive completed status requires completed cleanup")
    if result["status"] == "failed" and safe_resources.get("cleanup_complete") is True:
        raise ValueError("additive failed status is inconsistent with cleanup")
    return {
        "status": result["status"],
        "evidence_eligible": False,
        "acceptance": False,
        "evidence_level": "D0",
        "semantic_candidate": result["semantic_candidate"],
        "criteria": deepcopy(criteria),
        "failure_reason": result.get("failure_reason"),
        "metrics": metrics,
        "controls": controls,
        "raw_summaries": summaries,
        "holdout_evidence": holdout,
        "token_ids": {" true": 2081, " false": 3991},
        "target_token_strings": {" true": " true", " false": " false"},
        "layer": 6,
        "native_hidden_state_index": 7,
        "seeds": list(ADDITIVE_SEEDS),
        "strength_grid": [float(s) for s in ADDITIVE_STRENGTHS],
        "train_groups": [f"g{i:02d}" for i in range(1, 9)],
        "holdout_groups": [f"g{i:02d}" for i in range(9, 13)],
        "direction_norm": direction_norm,
        "no_mutation": True,
        "budget_pass": bool(result.get("budget_pass")),
        "model_parameter_digest_before": result["model_parameter_digest_before"],
        "model_parameter_digest_after": result["model_parameter_digest_after"],
        "provenance": safe_provenance,
        "resources": safe_resources,
        "execution_result_digest": safe_provenance["execution_result_digest"],
    }


# Publicly named aliases let the independent validator reuse the frozen schema
# without suppressing strict type-checker private-use diagnostics.
sanitize_additive_result = _sanitize_additive_result
sanitize_additive_resources = _sanitize_resources
reject_additive_sensitive_keys = _reject_sensitive_keys
validate_fixture_metadata = _validate_fixture_metadata


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
        "evidence_level": "D0",
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
    if use_case == "AdditiveSteering":
        _validate_fixture_metadata(fixture)
        if resources is not None:
            _sanitize_resources(resources)
    execution_attempted = bool(
        (resources or {}).get("execution_attempted", execution_result is not None and not injected)
    )
    execution_backend = (resources or {}).get("execution_backend")
    if execution_backend is None:
        execution_backend = "cuda" if execution_attempted and not injected else "none"
    execution_stage = (resources or {}).get("stage")
    if execution_stage is None:
        execution_stage = (
            "complete"
            if status in {"passed_real_cuda", ADDITIVE_COMPLETED_STATUS}
            else ("cleanup" if execution_attempted else "dispatch")
        )
    elif execution_stage == "dispatch" and execution_attempted and execution_result is not None:
        execution_stage = "complete" if status == "passed_real_cuda" else "cleanup"
    executions = [
        execution_template(
            plan,
            name,
            status if name == use_case else ("blocked_missing_corpus" if name == "TunedLogitLens" else "not_run"),
        )
        for name in (case["use_case"] for case in plan["real_use_case_checklist"])
    ]
    current: dict[str, Any] = next(item for item in executions if item["use_case"] == use_case)
    current["failure_ref"] = failure_ref
    phase_a_non_promoting = use_case in PHASE_A_NON_PROMOTING_USE_CASES
    sanitized_additive: dict[str, Any] | None = None
    if use_case == "AdditiveSteering" and execution_result is not None and not injected:
        sanitized_additive = _sanitize_additive_result(execution_result, resources or {}, plan)
        current.update(
            {
                "status": sanitized_additive["status"],
                "evidence_level": "D0",
                "evidence_eligible": False,
                "acceptance": False,
                "metrics": sanitized_additive["metrics"],
                "controls": sanitized_additive["controls"],
                "provenance": sanitized_additive["provenance"],
                "resources": sanitized_additive["resources"],
                "criteria": sanitized_additive["criteria"],
                "semantic_candidate": sanitized_additive["semantic_candidate"],
                "failure_reason": sanitized_additive["failure_reason"],
                "raw_summaries": sanitized_additive["raw_summaries"],
                "holdout_evidence": sanitized_additive["holdout_evidence"],
                "token_ids": sanitized_additive["token_ids"],
                "target_token_strings": sanitized_additive["target_token_strings"],
                "layer": sanitized_additive["layer"],
                "native_hidden_state_index": sanitized_additive["native_hidden_state_index"],
                "seeds": sanitized_additive["seeds"],
                "strength_grid": sanitized_additive["strength_grid"],
                "train_groups": sanitized_additive["train_groups"],
                "holdout_groups": sanitized_additive["holdout_groups"],
                "model_parameter_digest_before": sanitized_additive["model_parameter_digest_before"],
                "model_parameter_digest_after": sanitized_additive["model_parameter_digest_after"],
                "direction_norm": sanitized_additive["direction_norm"],
                "no_mutation": sanitized_additive["no_mutation"],
                "budget_pass": sanitized_additive["budget_pass"],
            }
        )
    elif execution_result is not None and not injected:
        for key in (
            "status",
            "evidence_eligible",
            "acceptance",
            "evidence_level",
            "metrics",
            "confidence_intervals",
            "controls",
            "control_raw",
            "diagnostics",
            "raw_summaries",
            "fixture_linkage",
            "budget_pass",
            "provenance",
            "token_ids",
            "target_token_strings",
            "raw_token_linkage",
            "layer",
            "native_hidden_state_index",
            "seed",
            "seeds",
            "no_mutation",
            "model_parameter_digest_before",
            "model_parameter_digest_after",
            "resources",
        ):
            if key in execution_result and not (
                phase_a_non_promoting and key in {"evidence_eligible", "acceptance", "evidence_level"}
            ):
                current[key] = execution_result[key]
    if phase_a_non_promoting:
        # Phase A runtime diagnostics are deliberately not promotion-capable.
        # Keep the runtime status/metrics for auditability, but never let a
        # handler result manufacture an accepted D3 execution or record.
        current["evidence_level"] = "D0"
        current["evidence_eligible"] = False
        current["acceptance"] = False
        if execution_result is None:
            current["resources"] = deepcopy(resources or {})
    accepted_tcav = bool(
        use_case == "TCAV"
        and not injected
        and execution_result is not None
        and execution_result.get("status") == "passed_real_cuda"
        and execution_result.get("evidence_eligible") is True
        and execution_result.get("acceptance") is True
    )
    accepted_tuned = bool(
        use_case == "TunedLogitLens"
        and not injected
        and execution_result is not None
        and execution_result.get("status") == "passed_real_cuda"
        and execution_result.get("evidence_eligible") is True
        and execution_result.get("acceptance") is True
    )
    accepted_disentanglement = bool(
        use_case == "Disentanglement"
        and not injected
        and execution_result is not None
        and execution_result.get("status") == "passed_real_cuda"
        and execution_result.get("evidence_eligible") is True
        and execution_result.get("acceptance") is True
    )
    accepted_activation = bool(
        use_case == "TrueActivationPatching"
        and not injected
        and execution_result is not None
        and execution_result.get("status") == "passed_real_cuda"
        and execution_result.get("evidence_eligible") is True
        and execution_result.get("acceptance") is True
    )
    if accepted_tcav or accepted_tuned or accepted_disentanglement or accepted_activation:
        current["evidence_level"] = "D2" if accepted_disentanglement else "D3"
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
                if key in execution_result and not (phase_a_non_promoting and key in {"evidence_level", "acceptance"}):
                    record[key] = execution_result[key]
            if phase_a_non_promoting:
                record["evidence_level"] = "D0"
                record["acceptance"] = False
            if sanitized_additive is not None:
                for field in ADDITIVE_LINKED_FIELDS:
                    if field in sanitized_additive:
                        record[field] = deepcopy(sanitized_additive[field])
        elif use_case == "AdditiveSteering" and item["record_id"] == current["record_id"]:
            record["resources"] = deepcopy(resources or {})
        records.append(record)
    artifact = {
        "schema_version": "m14-l04-explanations-artifact-v1",
        "lane": "L04",
        "use_case": use_case,
        "accepted_gap_ids": [TCAV_GAP_ID]
        if accepted_tcav
        else (
            [TUNED_GAP_ID]
            if accepted_tuned
            else (
                [DISENTANGLEMENT_GAP_ID]
                if accepted_disentanglement
                else ([ACTIVATION_PATCHING_GAP_ID] if accepted_activation else [])
            )
        ),
        "accepted_record_ids": [TCAV_RECORD_ID]
        if accepted_tcav
        else (
            [TUNED_RECORD_ID]
            if accepted_tuned
            else (
                [DISENTANGLEMENT_RECORD_ID]
                if accepted_disentanglement
                else ([ACTIVATION_PATCHING_RECORD_ID] if accepted_activation else [])
            )
        ),
        "evidence_level": "D2"
        if accepted_disentanglement
        else ("D3" if (accepted_tcav or accepted_tuned or accepted_activation) else "D0"),
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
            else ("real-cuda" if execution_attempted and execution_backend == "cuda" else "dispatcher-only-no-model"),
            "network": (resources or {}).get("network", "not attempted"),
            "device": (resources or {}).get("device", "not used"),
            "credentials": "not used",
            "cleanup": (resources or {}).get("cleanup", "not applicable; no model was loaded"),
            "execution_attempted": execution_attempted,
            "execution_backend": execution_backend,
            "stage": execution_stage,
            "resource_peak": (resources or {}).get("resource_peak", "not measured"),
            "use_case": use_case,
            "plan_sha256": plan_digest(plan),
        },
        "plan_sha256": plan_digest(plan),
    }
    if use_case == "AdditiveSteering":
        artifact["resources"] = deepcopy(resources or {})
    if execution_result is not None and not injected and sanitized_additive is None:
        provenance = artifact["provenance"]
        if isinstance(provenance, dict):
            provenance.update(execution_result.get("provenance", {}))
        artifact["raw_summaries"] = execution_result.get("raw_summaries", [])
        artifact["fixture_linkage"] = execution_result.get("fixture_linkage", [])
        artifact["diagnostics"] = execution_result.get("diagnostics", {})
        artifact["seeds"] = execution_result.get("seeds", [])
        artifact["token_ids"] = execution_result.get("token_ids", {})
        artifact["target_token_strings"] = execution_result.get("target_token_strings", {})
        artifact["raw_token_linkage"] = execution_result.get("raw_token_linkage", {})
    elif sanitized_additive is not None:
        artifact_provenance = artifact["provenance"]
        if isinstance(artifact_provenance, dict):
            artifact_provenance.update(sanitized_additive["provenance"])
        artifact["raw_summaries"] = sanitized_additive["raw_summaries"]
        artifact["holdout_evidence"] = sanitized_additive["holdout_evidence"]
        artifact["metrics"] = sanitized_additive["metrics"]
        artifact["criteria"] = sanitized_additive["criteria"]
        artifact["semantic_candidate"] = sanitized_additive["semantic_candidate"]
        artifact["resources"] = sanitized_additive["resources"]
        artifact["model_parameter_digest_before"] = sanitized_additive["model_parameter_digest_before"]
        artifact["model_parameter_digest_after"] = sanitized_additive["model_parameter_digest_after"]
        artifact["no_mutation"] = sanitized_additive["no_mutation"]
        artifact["budget_pass"] = sanitized_additive["budget_pass"]
        artifact_controls = artifact.get("controls")
        if not isinstance(artifact_controls, dict):
            raise ValueError("additive dispatcher controls are malformed")
        artifact_controls["additive"] = deepcopy(sanitized_additive["controls"])
        for field in ADDITIVE_LINKED_FIELDS:
            if field not in {"provenance", "resources", "controls"}:
                artifact[field] = deepcopy(sanitized_additive[field])
    if use_case == "AdditiveSteering":
        # The authored fixture path and the credentials declaration are fixed
        # dispatcher metadata, not handler output; every other retained value
        # must be free of sensitive/unknown keys.
        _reject_sensitive_keys(artifact)
    artifact["artifact_sha256"] = canonical_digest(artifact, "artifact_sha256")
    return artifact

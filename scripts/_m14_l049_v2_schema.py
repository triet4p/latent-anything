"""Private schemas and immutable constants for the L04.9 v2 addendum."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

from scripts._m14_l04_contract_common import canonical_json_bytes

V2_ADDENDUM_PATH = Path(__file__).resolve().parents[1] / "artifacts/m14/l04-explanations.v2.addendum.json"
V2_ADDENDUM_SCHEMA = "m14-l04.9-v2-addendum-v1"
V2_FIXTURE_SCHEMA = "m14-l04.9-v2-causal-pair-v1"
V2_STAGE_A_SCHEMA = "m14-l04.9-v2-stage-a-v1"
V2_STAGE_B_SCHEMA = "m14-l04.9-v2-stage-b-v1"
RUNTIME_ATTESTATION_SCHEMA = "m14-l04.9-v2-runtime-attestation-v1"
STAGE_A_FAILURE_KINDS = ("runtime_exception", "semantic_gate")
RUNTIME_EVENT_CODES = (
    "fixture_loaded",
    "candidate_scored",
    "hooks_registered",
    "captures_complete",
    "interventions_complete",
    "controls_complete",
    "cleanup_complete",
    "resources_recorded",
)
EXPECTED_RUNTIME_MODEL = "openai-community/gpt2@e7da7f221d5bf496a48136c0cd264e630fe9fcc8"
EXPECTED_RUNTIME_INTEGRATION = "TransformerLMIntegration"
EXPECTED_RUNTIME_DTYPE = "float32"
TOP_LEVEL_CLI_FILES = {
    "stage_a_train_selection": "m14_l049_v2_stage_a.py",
    "stage_b_holdout_evaluation": "m14_l049_v2_stage_b.py",
}
AUTHORING_MANIFEST_DIGEST_FIELD = "manifest_sha256"
AUTHORING_MANIFEST_DIGEST_RULE = (
    "sha256(canonical UTF-8 JSON object with manifest_sha256 omitted, sorted keys, compact separators, LF terminated)"
)
AUTHORING_MANIFEST_FILE_SHA256 = "2849b07fd719a0a761f433892fcc031c2ab17012a538daba322dd6fa50674974"
EXPECTED_ADDENDUM_SHA256 = "dea04fa5ce916327832324f882a951a69121dccd6f5cfa3e8cb0194d17a67243"
EXPECTED_AUTHORING_MANIFEST_SHA256 = "c63059b05fb45c984bdff2ebd7ecaeee0ff0ca98dab3cc81b845bedb2e1c83c7"
EXPECTED_HOLDOUT_CONTENT_SHA256 = "295ef5f558315c629d68e2d0216567a67163e5ef4adaaf3bbc9fe8a4da96dd5f"
EXPECTED_HOLDOUT_SEED_COMMITMENT_SHA256 = "b8e5e28908c2d2925a5bf5dcc69d852b4e31584f23f0ced2903a70f10d36b5e1"
PARENT_PLAN_SHA256 = "f3c315e356af0ee54d4196cc365ee22bd997b069d18a3e72c6b479f94e0b3e1a"
PUBLIC_TRAIN_SEED = 79049
TRAIN_GROUP_COUNT = 36
HOLDOUT_GROUP_COUNT = 24
ROWS_PER_GROUP = 2
CANDIDATE_LAYERS = tuple(range(12))
CANDIDATE_OFFSETS = (0, -1, -2)
BOOTSTRAP_REPLICATES = 2000
STAGE_B_SEEDS = (1701, 2901, 4101, 5301, 6701)
POWER_SIMULATION_SEED = 7904901
RECOVERY_THRESHOLD = 0.10
PAIRED_SHUFFLED_THRESHOLD = 0.05
OOF_RECOVERY_THRESHOLD = 0.05
DENOMINATOR_EPSILON = 1e-12
DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
V2_ROW_KEYS = (
    "row_id",
    "group_id",
    "causal_pair_id",
    "condition",
    "split",
    "prompt_family",
    "prompt",
    "target_text",
    "factor_labels",
)


@dataclass(frozen=True)
class CommitmentPolicy:
    """Immutable complete policy document used by the independent core."""

    expected_addendum_json: str
    addendum_sha256: str
    parent_plan_sha256: str
    holdout_content_sha256: str
    holdout_seed_commitment_sha256: str

    @classmethod
    def from_addendum(cls, addendum: Mapping[str, Any]) -> CommitmentPolicy:
        raw = canonical_json_bytes(dict(addendum)).decode("utf-8")
        return cls(
            expected_addendum_json=raw,
            addendum_sha256=str(addendum.get("addendum_sha256", "")),
            parent_plan_sha256=str(addendum.get("parent_plan_sha256", "")),
            holdout_content_sha256=str(addendum.get("fixture", {}).get("holdout_content_sha256", "")),
            holdout_seed_commitment_sha256=str(addendum.get("fixture", {}).get("holdout_seed_commitment_sha256", "")),
        )

    def expected_addendum(self) -> dict[str, Any]:
        return cast(dict[str, Any], json.loads(self.expected_addendum_json))


@lru_cache(maxsize=1)
def pinned_commitment_policy() -> CommitmentPolicy:
    return CommitmentPolicy.from_addendum(cast(dict[str, Any], json.loads(V2_ADDENDUM_PATH.read_bytes())))


def digest_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def top_level_cli_sha256(stage: str) -> str | None:
    """Hash the stage entry point, never its private producer module."""
    filename = TOP_LEVEL_CLI_FILES.get(stage)
    if filename is None:
        return None
    try:
        return digest_bytes(Path(__file__).with_name(filename).read_bytes())
    except OSError:
        return None


def canonical_digest(value: Mapping[str, Any], field: str) -> str:
    unsigned = dict(value)
    unsigned.pop(field, None)
    return digest_bytes(canonical_json_bytes(unsigned))


def is_digest(value: object) -> bool:
    return isinstance(value, str) and DIGEST_RE.fullmatch(value) is not None


def canonical_fixture_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(
        json.dumps(dict(row), ensure_ascii=True, separators=(",", ":"), allow_nan=False).encode("utf-8") + b"\n"
        for row in rows
    )


def fixture_digest(rows: Sequence[Mapping[str, Any]]) -> str:
    return digest_bytes(canonical_fixture_bytes(rows))


def directional_recovery(clean: object, corrupted: object, patched: object) -> float | None:
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in (clean, corrupted, patched)):
        return None
    try:
        clean_value = float(cast(float, clean))
        corrupted_value = float(cast(float, corrupted))
        patched_value = float(cast(float, patched))
    except (TypeError, ValueError, OverflowError):
        return None
    denominator = clean_value - corrupted_value
    if not all(math.isfinite(value) for value in (clean_value, corrupted_value, patched_value)):
        return None
    if not math.isfinite(denominator) or abs(denominator) <= DENOMINATOR_EPSILON:
        return None
    result = (patched_value - corrupted_value) / denominator
    return result if math.isfinite(result) else None


def candidate_grid() -> list[dict[str, int]]:
    return [{"layer": layer, "offset": offset} for layer in CANDIDATE_LAYERS for offset in CANDIDATE_OFFSETS]


def validate_addendum(addendum: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    expected_top_level = {
        "authoring",
        "addendum_sha256",
        "candidate_selection",
        "controls",
        "fixture",
        "parent_plan_sha256",
        "power",
        "protocol",
        "runtime_boundary",
        "schema_version",
        "stage_b_bootstrap_seeds",
        "train_seed",
        "v1_denylist",
        "v1_exposed_holdout_groups",
    }
    if set(addendum) != expected_top_level:
        errors.append("v2 addendum top-level policy fields are invalid")
    if addendum.get("schema_version") != V2_ADDENDUM_SCHEMA:
        errors.append("v2 addendum schema version is invalid")
    try:
        recomputed_addendum = canonical_digest(addendum, "addendum_sha256")
    except (TypeError, ValueError, OverflowError):
        recomputed_addendum = None
    if addendum.get("addendum_sha256") != EXPECTED_ADDENDUM_SHA256 or recomputed_addendum != EXPECTED_ADDENDUM_SHA256:
        errors.append("v2 addendum digest is invalid")
    if addendum.get("parent_plan_sha256") != PARENT_PLAN_SHA256:
        errors.append("v2 addendum parent plan digest is invalid")
    if addendum.get("v1_exposed_holdout_groups") != [f"g{i:02d}" for i in range(9, 13)]:
        errors.append("v2 addendum v1 exposed holdout denylist is invalid")
    if addendum.get("v1_denylist") != {
        "artifact_sha256": "historical-v1-failure-nonpromoting",
        "record_id": "THY-T05-ACTIVATION-PATCHING",
        "groups": [f"g{i:02d}" for i in range(9, 13)],
    }:
        errors.append("v2 addendum v1 denylist is invalid")
    fixture = addendum.get("fixture")
    if not isinstance(fixture, Mapping):
        errors.append("v2 addendum fixture commitment is missing")
        return errors
    expected_counts = {
        "train_groups": TRAIN_GROUP_COUNT,
        "holdout_groups": HOLDOUT_GROUP_COUNT,
        "rows_per_group": ROWS_PER_GROUP,
        "train_rows": TRAIN_GROUP_COUNT * ROWS_PER_GROUP,
        "holdout_rows": HOLDOUT_GROUP_COUNT * ROWS_PER_GROUP,
    }
    for key, expected in expected_counts.items():
        if fixture.get(key) != expected:
            errors.append(f"v2 addendum fixture count {key} is invalid")
    for key, expected in {
        "holdout_content_sha256": EXPECTED_HOLDOUT_CONTENT_SHA256,
        "holdout_seed_commitment_sha256": EXPECTED_HOLDOUT_SEED_COMMITMENT_SHA256,
    }.items():
        if fixture.get(key) != expected or not is_digest(fixture.get(key)):
            errors.append(f"v2 addendum fixture digest {key} is invalid")
    authoring = addendum.get("authoring")
    if (
        not isinstance(authoring, Mapping)
        or not is_digest(authoring.get(AUTHORING_MANIFEST_DIGEST_FIELD))
        or not isinstance(authoring.get("manifest"), Mapping)
        or authoring.get("manifest_digest_rule") != AUTHORING_MANIFEST_DIGEST_RULE
        or authoring.get("manifest_file_sha256") != AUTHORING_MANIFEST_FILE_SHA256
    ):
        errors.append("v2 addendum authoring manifest digest is invalid")
    manifest = authoring.get("manifest") if isinstance(authoring, Mapping) else None
    expected_manifest = {
        "generator": "scripts._m14_l049_v2_fixture.generate_rows",
        "generator_revision": "v1",
        "holdout_content_sha256": EXPECTED_HOLDOUT_CONTENT_SHA256,
        "holdout_groups": HOLDOUT_GROUP_COUNT,
        "holdout_seed_commitment_sha256": EXPECTED_HOLDOUT_SEED_COMMITMENT_SHA256,
        "manifest_sha256": EXPECTED_AUTHORING_MANIFEST_SHA256,
        "near_duplicate_policy": "unique family and vocabulary tokens; no cross-split shared lexical identifiers",
        "prompt_families_disjoint": True,
        "rows_per_group": ROWS_PER_GROUP,
        "schema_version": "m14-l04.9-v2-authoring-manifest-v1",
        "train_groups": TRAIN_GROUP_COUNT,
        "train_seed": PUBLIC_TRAIN_SEED,
        "vocabulary_disjoint": True,
    }
    if manifest != expected_manifest:
        errors.append("v2 authoring manifest policy is invalid")
    if isinstance(authoring, Mapping) and (
        authoring.get("manifest_sha256") != EXPECTED_AUTHORING_MANIFEST_SHA256
        or authoring.get("manifest_file_sha256") != AUTHORING_MANIFEST_FILE_SHA256
        or authoring.get("generator") != "scripts._m14_l049_v2_fixture.generate_rows"
        or authoring.get("generator_revision") != "v1"
        or authoring.get("manifest_digest_rule") != AUTHORING_MANIFEST_DIGEST_RULE
        or authoring.get("not_human_blind") is not True
        or authoring.get("withheld_computationally") is not True
    ):
        errors.append("v2 authoring manifest commitments are invalid")
    expected_candidate_selection = {
        "consensus_min_wins": 4,
        "fold_count": 6,
        "fold_validation_groups": 6,
        "layers": list(CANDIDATE_LAYERS),
        "offsets": list(CANDIDATE_OFFSETS),
        "oof_groups": 36,
        "oof_min_positive_groups": 24,
        "oof_lower_ci_strict_gt": 0.05,
        "oof_fold_mean_strict_gt": 0.0,
        "ranking": ["mean_recovery_desc", "lower_ci_desc", "layer_asc", "offset_order_0_-1_-2"],
    }
    if addendum.get("candidate_selection") != expected_candidate_selection:
        errors.append("v2 candidate-selection policy is invalid")
    expected_controls = [
        "true_clean_donor_replacement",
        "zero_strength_identity",
        "wrong_token_diagnostic",
        "adjacent_layer_diagnostic",
        "additive_diagnostic",
        "matched_norm_random_diagnostic",
        "label_stratified_shuffled_donor",
    ]
    if addendum.get("controls") != expected_controls:
        errors.append("v2 controls policy is invalid")
    expected_power = {
        "assumptions": {
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "decision_lower_ci": 0.05,
            "effect_mean": 0.16,
            "effect_sd": 0.3,
            "groups": 24,
            "null_mean": 0.0,
            "simulations": 2000,
        },
        "false_negative_risk": 0.43799999999999994,
        "false_positive_rate": 0.0115,
        "power": 0.562,
        "result_sha256": "2102e8bb02e092f4cef5ac5b42290019e2fae3393d4c089fa8e7aa2c494ba431",
        "schema": "m14-l04.9-v2-power-v1",
        "seed": POWER_SIMULATION_SEED,
    }
    if addendum.get("power") != expected_power:
        errors.append("v2 power policy is invalid")
    expected_protocol = {
        "directional_recovery": "(patched-corrupted)/(clean-corrupted)",
        "denominator_epsilon": 1e-12,
        "d3_scope": "held-out causal recovery plus donor specificity; not localization",
        "stage_b_bootstrap_seeds": list(STAGE_B_SEEDS),
        "stage_b_recovery_lower_ci_strict_gt": 0.1,
        "stage_b_shuffled_margin_lower_ci_strict_gt": 0.05,
        "stage_b_seed_semantics": "reproducibility summaries, not independent samples",
    }
    if addendum.get("protocol") != expected_protocol:
        errors.append("v2 protocol policy is invalid")
    if fixture.get("schema") != V2_FIXTURE_SCHEMA or fixture.get("row_keys") != list(V2_ROW_KEYS):
        errors.append("v2 addendum fixture schema is invalid")
    if addendum.get("train_seed") != PUBLIC_TRAIN_SEED or addendum.get("stage_b_bootstrap_seeds") != list(
        STAGE_B_SEEDS
    ):
        errors.append("v2 addendum public/bootstrap seeds are invalid")
    boundary = addendum.get("runtime_boundary")
    if (
        not isinstance(boundary, Mapping)
        or boundary.get("integration") != "TransformerLMIntegration"
        or boundary.get("model_adapter") != "N/A"
        or boundary.get("device") != "cuda"
        or boundary.get("model") != "openai-community/gpt2@e7da7f221d5bf496a48136c0cd264e630fe9fcc8"
        or not isinstance(boundary.get("source"), str)
        or not isinstance(boundary.get("recipient"), str)
    ):
        errors.append("v2 addendum runtime boundary is invalid")
    if "holdout_seed" in addendum or "holdout_path" in addendum or "holdout_plaintext" in addendum:
        errors.append("v2 addendum must not contain holdout seed, path, or plaintext")
    return errors


__all__ = [
    "BOOTSTRAP_REPLICATES",
    "AUTHORING_MANIFEST_DIGEST_FIELD",
    "AUTHORING_MANIFEST_DIGEST_RULE",
    "AUTHORING_MANIFEST_FILE_SHA256",
    "EXPECTED_ADDENDUM_SHA256",
    "EXPECTED_AUTHORING_MANIFEST_SHA256",
    "EXPECTED_HOLDOUT_CONTENT_SHA256",
    "EXPECTED_HOLDOUT_SEED_COMMITMENT_SHA256",
    "CANDIDATE_LAYERS",
    "CANDIDATE_OFFSETS",
    "DENOMINATOR_EPSILON",
    "HOLDOUT_GROUP_COUNT",
    "OOF_RECOVERY_THRESHOLD",
    "PARENT_PLAN_SHA256",
    "PAIRED_SHUFFLED_THRESHOLD",
    "POWER_SIMULATION_SEED",
    "PUBLIC_TRAIN_SEED",
    "RECOVERY_THRESHOLD",
    "ROWS_PER_GROUP",
    "STAGE_B_SEEDS",
    "TRAIN_GROUP_COUNT",
    "V2_ADDENDUM_PATH",
    "V2_ADDENDUM_SCHEMA",
    "V2_FIXTURE_SCHEMA",
    "V2_ROW_KEYS",
    "V2_STAGE_A_SCHEMA",
    "V2_STAGE_B_SCHEMA",
    "RUNTIME_ATTESTATION_SCHEMA",
    "STAGE_A_FAILURE_KINDS",
    "RUNTIME_EVENT_CODES",
    "EXPECTED_RUNTIME_MODEL",
    "EXPECTED_RUNTIME_INTEGRATION",
    "EXPECTED_RUNTIME_DTYPE",
    "candidate_grid",
    "canonical_digest",
    "canonical_fixture_bytes",
    "digest_bytes",
    "directional_recovery",
    "fixture_digest",
    "is_digest",
    "validate_addendum",
]

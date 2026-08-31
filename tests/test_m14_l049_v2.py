"""Offline contract tests for the preregistered L04.9 v2 stages."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from scripts._m14_l049_v2_fixture import authoring_manifest_digest, generate_rows, read_rows
from scripts._m14_l049_v2_power import POWER_ASSUMPTIONS, frozen_power_result, power_digest, validate_power_result
from scripts._m14_l049_v2_promotion import build_promotion_record, validate_promotion_record
from scripts._m14_l049_v2_real_runtime import _hidden, _patch_positions
from scripts._m14_l049_v2_retention import build_retention_record, validate_retention_record
from scripts._m14_l049_v2_schema import (
    STAGE_B_SEEDS,
    V2_ADDENDUM_PATH,
    CommitmentPolicy,
    canonical_digest,
    canonical_fixture_bytes,
    fixture_digest,
)
from scripts._m14_l049_v2_stage_a import build_stage_a_artifact, run_real_stage_a
from scripts._m14_l049_v2_stage_b import evaluate_stage_b, label_stratified_shuffled_mapping
from scripts._m14_l049_v2_transport import build_transport_metadata, validate_transport_metadata
from scripts._m14_l049_v2_validate import validate_stage_a, validate_stage_b
from scripts._m14_l049_v2_validate_stage_a import validate_stage_a_impl
from scripts._m14_l049_v2_validate_stage_b import validate_stage_b_impl

ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "artifacts/m14/l04-l049-v2-train.jsonl"
STAGE_A_FAILURE_SIDECAR = ROOT / (
    "artifacts/m14/l04-explanations.ssh.L049V2StageA.41828c2e12e1efacb80e8cb5a0c62e4e69a688b2.sidecar.json"
)
STAGE_A_FAILURE_RAW = ROOT / (
    "artifacts/m14/l04-explanations.ssh.L049V2StageA.41828c2e12e1efacb80e8cb5a0c62e4e69a688b2.raw.txt"
)
SOURCE_COMMIT = "1" * 40
SOURCE_TREE = "2" * 64


def _base() -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    rows = read_rows(TRAIN_PATH)[1]
    addendum = json.loads(V2_ADDENDUM_PATH.read_bytes())
    artifact = build_stage_a_artifact(rows, addendum, source_sha256="a" * 64)
    return rows, addendum, artifact


def test_failed_stage_a_sidecar_is_canonical_and_sanitized() -> None:
    sidecar = json.loads(STAGE_A_FAILURE_SIDECAR.read_bytes())
    assert (
        canonical_digest(sidecar, "sidecar_sha256")
        == sidecar["sidecar_sha256"]
        == "b08b979f982beac73130da38c78201048dcf37d7fd426dbf15cac5935a9e20ad"
    )
    assert sidecar["source"] == {
        "commit_sha": "41828c2e12e1efacb80e8cb5a0c62e4e69a688b2",
        "tree_sha256": "1178bd4a339d773fb74c86b4046b087311a0b70d",
        "use_case": "L049V2StageA",
    }
    assert sidecar["raw_capture"] == {
        "bytes": 7314,
        "sha256": "9d3682dbe0f5faa0a65881f4f79d5d946e323b5e959b29650df96355a66e2f6f",
    }
    assert sidecar["reason_code"] == "no_triad_bundle_status_66"
    assert sidecar["artifact"]["failure"] == "no_triad"
    assert sidecar["artifact"]["audit"] == "not_created"
    assert sidecar["repository_promotion"] is False
    assert sidecar["raw_retention_status"] == "deleted_by_owner_exception"
    assert sidecar["standard_finalize"] is False
    assert not STAGE_A_FAILURE_RAW.exists()
    assert sidecar["owner_exception"] == {
        "deletion_verification": {
            "absent_after_delete": True,
            "pre_delete_bytes": 7314,
            "pre_delete_sha256": "9d3682dbe0f5faa0a65881f4f79d5d946e323b5e959b29650df96355a66e2f6f",
        },
        "previous_sidecar_sha256": "92ec7b7dfd194b3edb440867fabb9a6befd89d2399b66bada512a73b23b1fd04",
        "reason": "no_triad_bundle_status_66",
        "standard_finalize": False,
    }
    serialized = json.dumps(sidecar, sort_keys=True)
    assert all(secret not in serialized for secret in ("traceback", "PROMPT", "holdout_plaintext", "BEGIN PRIVATE"))


def test_real_runtime_uses_independent_clean_source_and_corrupt_recipient_positions() -> None:
    clean_hidden = np.arange(1 * 22 * 4, dtype=np.float32).reshape(1, 22, 4)
    corrupt_hidden = np.arange(1 * 21 * 4, dtype=np.float32).reshape(1, 21, 4) + 100.0
    clean = SimpleNamespace(attention_mask=np.ones((1, 22), dtype=np.int64), hidden_states=[])
    corrupt = SimpleNamespace(attention_mask=np.ones((1, 21), dtype=np.int64), hidden_states=[])
    clean_position, corrupt_position = _patch_positions(clean, corrupt, clean_hidden, corrupt_hidden, -1)
    assert (clean_position, corrupt_position) == (20, 19)
    direction = np.zeros_like(corrupt_hidden)
    direction[0, corrupt_position] = clean_hidden[0, clean_position] - corrupt_hidden[0, corrupt_position]
    np.testing.assert_array_equal(direction[0, 19], clean_hidden[0, 20] - corrupt_hidden[0, 19])
    assert np.count_nonzero(direction) == 4


def test_real_runtime_hidden_layer_helper_rejects_missing_native_index() -> None:
    result = SimpleNamespace(hidden_states=[SimpleNamespace(layer=0, values=np.zeros((1, 3, 4)))])
    with pytest.raises(ValueError, match="native hidden state 7 is missing"):
        _hidden(result, 6, role="clean source")


def test_real_stage_a_index_error_emits_sanitized_d0_artifact() -> None:
    rows, addendum, _ = _base()
    finalized = False
    resources: dict[str, Any] = {
        "stage": "real_runtime",
        "execution_attempted": True,
        "execution_backend": "cuda",
        "model": "openai-community/gpt2@e7da7f221d5bf496a48136c0cd264e630fe9fcc8",
        "model_revision": "openai-community/gpt2@e7da7f221d5bf496a48136c0cd264e630fe9fcc8",
        "integration": "TransformerLMIntegration",
        "model_adapter": "N/A",
        "device": "cuda",
        "backend": "cuda",
        "dtype": "float32",
        "hook": {"registered": 0, "capture_calls": 0, "removed": 0},
        "intervention": {"patch_calls": 0, "control_calls": 0, "forward_calls": 0},
        "operation_counts": {
            "candidate_evaluations": 1,
            "hooks": 0,
            "captures": 1,
            "patches": 0,
            "controls": 0,
            "forwards": 1,
        },
        "cleanup": {"hook_count": 0, "completed": True},
        "resource_peak": {
            "peak_cpu_bytes": 1,
            "peak_gpu_bytes": 1,
            "unit": "bytes",
            "budget_cpu_bytes": 6_000_000_000,
            "budget_gpu_bytes": 6_000_000_000,
        },
        "no_mutation": True,
    }

    def finalize() -> dict[str, Any]:
        nonlocal finalized
        finalized = True
        return {key: value for key, value in resources.items() if key != "finalize"}

    resources["finalize"] = finalize

    def failing_score(*_args: Any) -> float:
        raise IndexError("prompt payload must never be serialized")

    artifact = run_real_stage_a(
        rows,
        addendum,
        source_sha256="a" * 64,
        runtime={"score": failing_score, "resources": resources},
    )
    assert finalized is True
    assert artifact["status"] == "stage_a_failed"
    assert artifact["evidence_level"] == "D0"
    assert artifact["evidence_eligible"] is False
    assert artifact["selection"]["failure"] == {"exception_type": "IndexError"}
    assert "prompt payload" not in json.dumps(artifact)
    assert validate_stage_a(artifact, rows, addendum) == []


def _real_resources_for_failure() -> dict[str, Any]:
    return {
        "stage": "real_runtime",
        "execution_attempted": True,
        "execution_backend": "cuda",
        "model": "openai-community/gpt2@e7da7f221d5bf496a48136c0cd264e630fe9fcc8",
        "model_revision": "openai-community/gpt2@e7da7f221d5bf496a48136c0cd264e630fe9fcc8",
        "integration": "TransformerLMIntegration",
        "model_adapter": "N/A",
        "device": "cuda",
        "backend": "cuda",
        "dtype": "float32",
        "hook": {"registered": 1, "capture_calls": 1, "removed": 0},
        "intervention": {"patch_calls": 0, "control_calls": 0, "forward_calls": 1},
        "operation_counts": {
            "candidate_evaluations": 1,
            "hooks": 1,
            "captures": 1,
            "patches": 0,
            "controls": 0,
            "forwards": 1,
        },
        "cleanup": {"hook_count": 1, "completed": True},
        "resource_peak": {
            "peak_cpu_bytes": 1,
            "peak_gpu_bytes": 1,
            "unit": "bytes",
            "budget_cpu_bytes": 6_000_000_000,
            "budget_gpu_bytes": 6_000_000_000,
        },
        "no_mutation": True,
    }


def test_stage_a_cleanup_failure_alone_is_sanitized_and_validator_clean() -> None:
    rows, addendum, _ = _base()
    resources = _real_resources_for_failure()
    calls = 0

    class SecretCleanupError(RuntimeError):
        pass

    def finalize() -> dict[str, Any]:
        nonlocal calls
        calls += 1
        raise SecretCleanupError("cleanup prompt secret must never escape")

    resources["finalize"] = finalize
    artifact = run_real_stage_a(
        rows,
        addendum,
        source_sha256="a" * 64,
        runtime={"score": lambda *_args: 0.0, "resources": resources},
    )
    assert calls == 1
    assert artifact["status"] == "stage_a_failed"
    assert artifact["evidence_level"] == "D0"
    assert artifact["selection"]["consensus_candidate"] is None
    assert artifact["selection"]["score_records"] == []
    assert artifact["resources"]["cleanup"] == {
        "attempted": True,
        "completed": False,
        "hooks_remaining": 1,
        "error_type": "CleanupError",
        "reason": "finalizer_exception",
        "stage": "cleanup",
    }
    assert "cleanup prompt secret" not in json.dumps(artifact)
    assert validate_stage_a(artifact, rows, addendum) == []


def test_stage_a_runtime_and_cleanup_failures_preserve_primary_error() -> None:
    rows, addendum, _ = _base()
    resources = _real_resources_for_failure()
    calls = 0

    def finalize() -> dict[str, Any]:
        nonlocal calls
        calls += 1
        raise RuntimeError("cleanup secret must never escape")

    resources["finalize"] = finalize

    def score(*_args: Any) -> float:
        raise IndexError("runtime prompt secret must never escape")

    artifact = run_real_stage_a(
        rows,
        addendum,
        source_sha256="a" * 64,
        runtime={"score": score, "resources": resources},
    )
    assert calls == 1
    assert artifact["selection"]["failure"] == {"exception_type": "IndexError"}
    assert artifact["resources"]["cleanup"]["error_type"] == "RuntimeError"
    assert artifact["resources"]["cleanup"]["completed"] is False
    assert "prompt secret" not in json.dumps(artifact)
    assert validate_stage_a(artifact, rows, addendum) == []


@pytest.mark.parametrize("finalizer_result", [None, {"stage": "cleanup"}])
def test_stage_a_invalid_finalizer_result_is_sanitized_and_validator_clean(finalizer_result: object) -> None:
    rows, addendum, _ = _base()
    resources = _real_resources_for_failure()
    calls = 0

    def finalize() -> object:
        nonlocal calls
        calls += 1
        return finalizer_result

    resources["finalize"] = finalize
    artifact = run_real_stage_a(
        rows,
        addendum,
        source_sha256="a" * 64,
        runtime={"score": lambda *_args: 0.0, "resources": resources},
    )
    assert calls == 1
    assert artifact["selection"]["failure"] == {"exception_type": "TypeError"}
    assert artifact["resources"]["cleanup"] == {
        "attempted": True,
        "completed": False,
        "hooks_remaining": 1,
        "error_type": "TypeError",
        "reason": "finalizer_invalid_result",
        "stage": "cleanup",
    }
    assert artifact["runtime_attestation"]["cleanup_hook_count"] == 1
    assert validate_stage_a(artifact, rows, addendum) == []


@pytest.mark.parametrize("forged_remaining", [0, 999_999])
def test_stage_a_rehashed_forged_cleanup_remaining_hooks_fails_validation(forged_remaining: int) -> None:
    rows, addendum, _ = _base()
    resources = _real_resources_for_failure()

    def finalize() -> object:
        raise RuntimeError("cleanup failure")

    resources["finalize"] = finalize
    artifact = run_real_stage_a(
        rows,
        addendum,
        source_sha256="a" * 64,
        runtime={"score": lambda *_args: 0.0, "resources": resources},
    )
    artifact["resources"]["cleanup"]["hooks_remaining"] = forged_remaining
    artifact["runtime_attestation"]["cleanup_hook_count"] = forged_remaining
    artifact["runtime_attestation"]["attestation_sha256"] = canonical_digest(
        artifact["runtime_attestation"], "attestation_sha256"
    )
    artifact["attestation_sha256"] = artifact["runtime_attestation"]["attestation_sha256"]
    artifact["artifact_sha256"] = canonical_digest(artifact, "artifact_sha256")
    errors = validate_stage_a(artifact, rows, addendum)
    assert any("counter-derived" in error or "cleanup count" in error for error in errors)


def test_stage_a_cli_runtime_index_error_writes_complete_d0_triad(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import torch

    import scripts._m14_l049_v2_real_runtime as real_runtime
    from scripts.m14_l049_v2_stage_a import main

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def failing_runtime(_rows: Any) -> Any:
        raise IndexError("prompt payload must never reach the envelope")

    monkeypatch.setattr(real_runtime, "build_stage_a_runtime", failing_runtime)
    output = tmp_path / "artifact.json"
    main(
        [
            "--train-fixture",
            str(TRAIN_PATH),
            "--output",
            str(output),
            "--source-commit-sha",
            SOURCE_COMMIT,
            "--run-real",
        ]
    )
    triad = sorted(tmp_path.glob("l04-explanations.L049V2StageA.attempt1.*.json"))
    assert [path.suffixes[-2] for path in triad] == [".failure", ".partial", ".run"]
    partial = json.loads(next(path for path in triad if path.name.endswith(".partial.json")).read_bytes())
    assert partial["artifact"]["status"] == "stage_a_failed"
    assert partial["artifact"]["evidence_level"] == "D0"
    assert "prompt payload" not in json.dumps(partial)


def _synthetic_stage_b() -> tuple[dict[str, Any], list[dict[str, Any]], bytes, dict[str, Any], dict[str, Any]]:
    train, addendum, _ = _base()
    holdout_seed = b"synthetic-v2-holdout-seed-32-bytes"[:32]
    holdout = generate_rows("holdout", 24, int.from_bytes(holdout_seed, "big"))
    synthetic_addendum = copy.deepcopy(addendum)
    synthetic_addendum["fixture"]["holdout_content_sha256"] = fixture_digest(holdout)
    synthetic_addendum["fixture"]["holdout_seed_commitment_sha256"] = (
        __import__("hashlib").sha256(holdout_seed).hexdigest()
    )
    synthetic_manifest = synthetic_addendum["authoring"]["manifest"]
    synthetic_manifest["holdout_content_sha256"] = synthetic_addendum["fixture"]["holdout_content_sha256"]
    synthetic_manifest["holdout_seed_commitment_sha256"] = synthetic_addendum["fixture"][
        "holdout_seed_commitment_sha256"
    ]
    synthetic_manifest["manifest_sha256"] = authoring_manifest_digest(synthetic_manifest)
    synthetic_addendum["authoring"]["manifest_sha256"] = synthetic_manifest["manifest_sha256"]
    synthetic_addendum["addendum_sha256"] = canonical_digest(synthetic_addendum, "addendum_sha256")
    candidate = build_stage_a_artifact(train, synthetic_addendum, source_sha256="b" * 64)
    observations: dict[str, dict[str, dict[str, Any]]] = {}
    pair_ids = sorted({str(row["causal_pair_id"]) for row in holdout})
    for pair in pair_ids:
        observations[pair] = {}
        for seed in STAGE_B_SEEDS:
            observations[pair][str(seed)] = {
                "clean_margin": 0.0,
                "corrupted_margin": -1.0,
                "patched_margin": 0.9,
                "shuffled_margin": -0.5,
                "zero_strength_selected_logit_digest": "c" * 64,
                "zero_strength_relevant_output_digest": "d" * 64,
                "corrupted_selected_logit_digest": "c" * 64,
                "corrupted_relevant_output_digest": "d" * 64,
                "zero_strength_identity": True,
                "wrong_token": {"effect": 0.0},
                "adjacent_layer": {"effect": 0.0},
                "additive": {"effect": 0.0},
                "matched_norm_random": {"effect": 0.0},
            }
    return candidate, holdout, holdout_seed, synthetic_addendum, observations


def test_v2_addendum_and_stage_a_bind_public_boundary() -> None:
    train, addendum, artifact = _base()
    assert validate_stage_a(artifact, train, addendum) == []
    assert not (ROOT / "artifacts/m14/l04-l049-v2-stage-a.json").exists()
    assert addendum["parent_plan_sha256"] == "f3c315e356af0ee54d4196cc365ee22bd997b069d18a3e72c6b479f94e0b3e1a"
    assert addendum["v1_exposed_holdout_groups"] == ["g09", "g10", "g11", "g12"]
    assert "holdout_seed" not in addendum and "holdout_path" not in addendum and "holdout_plaintext" not in addendum
    assert artifact["status"] == "protocol_fixture"
    assert artifact["evidence_level"] == "D0" and artifact["evidence_eligible"] is False
    assert artifact["resources"]["execution_backend"] == "cpu"
    assert artifact["resources"]["execution_attempted"] is False
    assert artifact["selection"]["consensus_wins"] >= 4
    assert artifact["selection"]["oof_metric"]["positive_groups"] >= 24


def test_stage_a_never_accepts_holdout_rows() -> None:
    train, addendum, artifact = _base()
    mutated = train + [dict(generate_rows("holdout", 1, 1)[0])]
    assert validate_stage_a(artifact, mutated, addendum)


@pytest.mark.parametrize(
    "mutator",
    [
        lambda value: value["selection"]["score_records"][0].update({"group_score": 999.0}),
        lambda value: value["selection"]["folds"][0]["winner"].update({"layer": 0}),
        lambda value: value["selection"].update({"consensus_wins": -1}),
    ],
)
def test_stage_a_tampering_is_rejected(mutator: Any) -> None:
    train, addendum, artifact = _base()
    mutated = copy.deepcopy(artifact)
    mutator(mutated)
    assert validate_stage_a(mutated, train, addendum)


@pytest.mark.parametrize(
    "field",
    ["lower_ci_95", "all_fold_means_positive", "positive_groups", "pass", "status", "evidence_level"],
)
def test_stage_a_gate_and_status_tampering_is_rejected_after_rehash(field: str) -> None:
    train, addendum, artifact = _base()
    mutated = copy.deepcopy(artifact)
    if field == "status":
        mutated["status"] = "stage_a_complete"
    elif field == "evidence_level":
        mutated["evidence_level"] = "D1"
    elif field == "pass":
        mutated["selection"]["oof_metric"]["pass"] = False
    elif field == "lower_ci_95":
        mutated["selection"]["oof_metric"]["lower_ci_95"] = -1.0
    elif field == "all_fold_means_positive":
        mutated["selection"]["oof_metric"]["all_fold_means_positive"] = False
    else:
        mutated["selection"]["oof_metric"]["positive_groups"] = 0
    mutated["artifact_sha256"] = canonical_digest(mutated, "artifact_sha256")
    assert validate_stage_a(mutated, train, addendum)


def test_v2_authoring_manifest_self_digest_tampering_is_rejected() -> None:
    train, addendum, artifact = _base()
    mutated = copy.deepcopy(addendum)
    mutated["authoring"]["manifest_sha256"] = "0" * 64
    mutated["addendum_sha256"] = canonical_digest(mutated, "addendum_sha256")
    assert validate_stage_a(artifact, train, mutated)


def test_v2_authoring_manifest_file_commitment_tampering_is_rejected() -> None:
    train, addendum, artifact = _base()
    mutated = copy.deepcopy(addendum)
    mutated["authoring"]["manifest_file_sha256"] = "0" * 64
    mutated["addendum_sha256"] = canonical_digest(mutated, "addendum_sha256")
    assert validate_stage_a(artifact, train, mutated)


def test_stage_b_synthetic_path_is_d0_but_fully_validated() -> None:
    candidate, holdout, _seed, addendum, observations = _synthetic_stage_b()
    artifact = evaluate_stage_b(holdout, observations, candidate, addendum, _seed)
    assert validate_stage_b(artifact, holdout, _seed, candidate, addendum, read_rows(TRAIN_PATH)[1])
    assert artifact["evidence_level"] == "D0"
    assert artifact["evidence_eligible"] is False
    assert artifact["repository_promotion"] is False
    assert artifact["runtime_attestation"]["mode"] == "synthetic"
    assert label_stratified_shuffled_mapping(["a", "b", "c", "d"], {"a": 0, "b": 0, "c": 1, "d": 1}) == {
        "a": "b",
        "b": "a",
        "c": "d",
        "d": "c",
    }


@pytest.mark.parametrize("malformed", [10**1000, float("nan"), {"nested": [1, 2]}])
def test_stage_a_malformed_numeric_artifact_fails_closed(malformed: Any) -> None:
    train, addendum, artifact = _base()
    mutated = copy.deepcopy(artifact)
    mutated["selection"]["oof_evidence"][0]["recovery"] = malformed
    if malformed == malformed:
        mutated["artifact_sha256"] = canonical_digest(mutated, "artifact_sha256")
    assert validate_stage_a(mutated, train, addendum)


def test_stage_b_malformed_input_fails_closed_without_exception() -> None:
    train, addendum, candidate = _base()
    errors = validate_stage_b({"artifact_sha256": "0" * 64}, [], b"x" * 32, candidate, addendum, train)
    assert errors


def test_stage_b_commitment_and_denominator_mismatches_fail_closed() -> None:
    candidate, holdout, seed, addendum, observations = _synthetic_stage_b()
    bad_addendum = copy.deepcopy(addendum)
    bad_addendum["fixture"]["holdout_content_sha256"] = "e" * 64
    with pytest.raises(ValueError, match="fixture digest"):
        evaluate_stage_b(holdout, observations, candidate, bad_addendum, seed)
    bad_observations = copy.deepcopy(observations)
    pair = sorted(bad_observations)[0]
    for seed_key in bad_observations[pair]:
        bad_observations[pair][seed_key]["clean_margin"] = -1.0
    with pytest.raises(ValueError, match="directional recovery"):
        evaluate_stage_b(holdout, bad_observations, candidate, addendum, seed)


def test_stage_b_holdout_seed_commitment_mismatch_fails_closed() -> None:
    candidate, holdout, _seed, addendum, observations = _synthetic_stage_b()
    with pytest.raises(ValueError, match="seed commitment"):
        evaluate_stage_b(holdout, observations, candidate, addendum, b"x" * 32)


def test_v2_source_does_not_embed_external_holdout_path_or_seed() -> None:
    for path in ROOT.glob("scripts/*m14_l049_v2*.py"):
        text = path.read_text(encoding="utf-8")
        assert "latent-anything-holdout" not in text
        assert "holdout_seed.bin" not in text
    assert canonical_fixture_bytes(read_rows(TRAIN_PATH)[1]).startswith(b'{"row_id":')


def test_v2_transport_and_retention_do_not_mutate_v1_protocol() -> None:
    _, addendum, artifact = _base()
    retention = build_retention_record(artifact, "stage_a_train_selection")
    assert validate_retention_record(retention, artifact) == []
    transport = build_transport_metadata(
        "stage_a_train_selection", artifact, addendum, source_commit_sha=SOURCE_COMMIT, source_tree_sha256=SOURCE_TREE
    )
    assert (
        validate_transport_metadata(
            transport,
            "stage_a_train_selection",
            artifact,
            addendum,
            expected_source_commit_sha=SOURCE_COMMIT,
            expected_source_tree_sha256=SOURCE_TREE,
            policy=CommitmentPolicy.from_addendum(addendum),
        )
        == []
    )
    assert transport["v1_exact_three_member_protocol"] == "unchanged and not reused"
    assert transport["network"] == "owner-gated; not invoked in Phase A"
    assert transport["cli_sha256"] != artifact["source_sha256"]


def test_transport_rehashed_producer_module_cli_forgery_fails() -> None:
    _, addendum, artifact = _base()
    transport = build_transport_metadata(
        "stage_a_train_selection", artifact, addendum, source_commit_sha=SOURCE_COMMIT, source_tree_sha256=SOURCE_TREE
    )
    transport["cli_sha256"] = artifact["source_sha256"]
    transport["bundle_sha256"] = canonical_digest(transport, "bundle_sha256")
    transport["transport_sha256"] = canonical_digest(transport, "transport_sha256")
    assert validate_transport_metadata(
        transport,
        "stage_a_train_selection",
        artifact,
        addendum,
        expected_source_commit_sha=SOURCE_COMMIT,
        expected_source_tree_sha256=SOURCE_TREE,
        policy=CommitmentPolicy.from_addendum(addendum),
    )


def test_stage_b_validator_has_no_producer_oracle_import() -> None:
    validator_source = (ROOT / "scripts/_m14_l049_v2_validate_stage_b.py").read_text(encoding="utf-8")
    assert "_m14_l049_v2_stage_b" not in validator_source
    assert "label_stratified_shuffled_mapping" not in validator_source


@pytest.mark.parametrize("field", ["events", "counts", "commitments", "resources", "transcript_sha256"])
def test_runtime_attestation_semantic_tampering_fails_after_rehash(field: str) -> None:
    train, addendum, artifact = _base()
    mutated = copy.deepcopy(artifact)
    if field == "events":
        mutated["runtime_attestation"]["events"][0]["code"] = "forged"
    elif field == "counts":
        mutated["runtime_attestation"]["counts"]["captures"] += 1
    elif field == "commitments":
        mutated["runtime_attestation"]["commitments"]["fixture_sha256"] = "f" * 64
    elif field == "resources":
        mutated["runtime_attestation"]["resources"]["peak_cpu_bytes"] = 1
    else:
        mutated["runtime_attestation"][field] = "e" * 64
    mutated["runtime_attestation"]["attestation_sha256"] = canonical_digest(
        mutated["runtime_attestation"], "attestation_sha256"
    )
    mutated["attestation_sha256"] = mutated["runtime_attestation"]["attestation_sha256"]
    mutated["artifact_sha256"] = canonical_digest(mutated, "artifact_sha256")
    assert validate_stage_a(mutated, train, addendum)


def test_transport_rehashed_marker_tampering_fails() -> None:
    _, addendum, artifact = _base()
    transport = build_transport_metadata(
        "stage_a_train_selection", artifact, addendum, source_commit_sha=SOURCE_COMMIT, source_tree_sha256=SOURCE_TREE
    )
    transport["cli_invocation_count"] = 2
    transport["bundle_sha256"] = canonical_digest(transport, "bundle_sha256")
    transport["transport_sha256"] = canonical_digest(transport, "transport_sha256")
    assert validate_transport_metadata(
        transport,
        "stage_a_train_selection",
        artifact,
        addendum,
        expected_source_commit_sha=SOURCE_COMMIT,
        expected_source_tree_sha256=SOURCE_TREE,
    )


def test_stage_b_d3_self_declaration_is_rejected_after_rehash() -> None:
    candidate, holdout, seed, addendum, observations = _synthetic_stage_b()
    artifact = evaluate_stage_b(holdout, observations, candidate, addendum, seed)
    artifact["evidence_level"] = "D3"
    artifact["evidence_eligible"] = True
    artifact["promotion_candidate"] = True
    artifact["artifact_sha256"] = canonical_digest(artifact, "artifact_sha256")
    assert validate_stage_b(artifact, holdout, seed, candidate, addendum, read_rows(TRAIN_PATH)[1])


def test_d3_promotion_requires_independent_real_and_retention_prerequisites() -> None:
    candidate, holdout, seed, addendum, observations = _synthetic_stage_b()
    stage_b = evaluate_stage_b(holdout, observations, candidate, addendum, seed)
    transport = build_transport_metadata(
        "stage_b_holdout_evaluation",
        stage_b,
        addendum,
        source_commit_sha=SOURCE_COMMIT,
        source_tree_sha256=SOURCE_TREE,
        policy=CommitmentPolicy.from_addendum(addendum),
    )
    audit = {"status": "forged", "members": {}}
    with pytest.raises(ValueError, match="promotion prerequisites"):
        build_promotion_record(
            stage_b,
            candidate,
            addendum,
            read_rows(TRAIN_PATH)[1],
            holdout,
            seed,
            transport,
            audit,
            expected_source_commit_sha=SOURCE_COMMIT,
            expected_source_tree_sha256=SOURCE_TREE,
            policy=CommitmentPolicy.from_addendum(addendum),
        )
    assert validate_promotion_record(
        {},
        stage_b,
        candidate,
        addendum,
        read_rows(TRAIN_PATH)[1],
        holdout,
        seed,
        transport,
        audit,
        expected_source_commit_sha=SOURCE_COMMIT,
        expected_source_tree_sha256=SOURCE_TREE,
        policy=CommitmentPolicy.from_addendum(addendum),
    )


def test_d3_promotion_builds_from_valid_d2_and_reopened_triplet(tmp_path: Path) -> None:
    candidate, holdout, seed, addendum, observations = _synthetic_stage_b()
    policy = CommitmentPolicy.from_addendum(addendum)
    train = read_rows(TRAIN_PATH)[1]
    candidate = build_stage_a_artifact(
        train,
        addendum,
        source_sha256="b" * 64,
        execution_mode="real",
        resources={
            "stage": "real_runtime",
            "execution_attempted": True,
            "execution_backend": "cuda",
            "model": "openai-community/gpt2@e7da7f221d5bf496a48136c0cd264e630fe9fcc8",
            "model_revision": "openai-community/gpt2@e7da7f221d5bf496a48136c0cd264e630fe9fcc8",
            "integration": "TransformerLMIntegration",
            "model_adapter": "N/A",
            "device": "cuda",
            "backend": "cuda",
            "dtype": "float32",
            "hook": {"registered": 1296, "capture_calls": 1368, "removed": 1296},
            "intervention": {"patch_calls": 1296, "control_calls": 0, "forward_calls": 1368},
            "operation_counts": {
                "candidate_evaluations": 2592,
                "hooks": 1296,
                "captures": 1368,
                "patches": 1296,
                "controls": 0,
                "forwards": 1368,
            },
            "cleanup": {"hook_count": 0, "completed": True},
            "resource_peak": {
                "peak_cpu_bytes": 1,
                "peak_gpu_bytes": 1,
                "unit": "bytes",
                "budget_cpu_bytes": 2,
                "budget_gpu_bytes": 2,
            },
            "no_mutation": True,
        },
    )
    assert validate_stage_a_impl(candidate, train, addendum, policy=policy) == []
    resources = {
        "stage": "real_runtime",
        "execution_attempted": True,
        "execution_backend": "cuda",
        "model": "openai-community/gpt2@e7da7f221d5bf496a48136c0cd264e630fe9fcc8",
        "model_revision": "openai-community/gpt2@e7da7f221d5bf496a48136c0cd264e630fe9fcc8",
        "integration": "TransformerLMIntegration",
        "model_adapter": "N/A",
        "device": "cuda",
        "backend": "cuda",
        "dtype": "float32",
        "hook": {"registered": 24, "capture_calls": 552, "removed": 24},
        "intervention": {"patch_calls": 24, "control_calls": 480, "forward_calls": 552},
        "operation_counts": {
            "candidate_evaluations": 120,
            "hooks": 24,
            "captures": 552,
            "patches": 24,
            "controls": 480,
            "forwards": 552,
        },
        "cleanup": {"hook_count": 0, "completed": True},
        "resource_peak": {
            "peak_cpu_bytes": 1,
            "peak_gpu_bytes": 1,
            "unit": "bytes",
            "budget_cpu_bytes": 2,
            "budget_gpu_bytes": 2,
        },
        "no_mutation": True,
    }
    stage_b = evaluate_stage_b(holdout, observations, candidate, addendum, seed, resources=resources)
    assert validate_stage_b_impl(stage_b, holdout, seed, candidate, addendum, train, policy=policy) == []
    transport = build_transport_metadata(
        "stage_b_holdout_evaluation",
        stage_b,
        addendum,
        source_commit_sha=SOURCE_COMMIT,
        source_tree_sha256=SOURCE_TREE,
        policy=policy,
    )
    members: dict[str, dict[str, Any]] = {}
    for kind in ("partial", "run", "failure"):
        path = tmp_path / f"{kind}.json"
        raw = json.dumps({"kind": kind}, separators=(",", ":")).encode()
        path.write_bytes(raw)
        members[kind] = {
            "path": str(path),
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "reopen_validation": "PASS",
        }
    pending = {
        "schema_version": "m14-l04.9-v2-pending-retention-audit-v1",
        "stage": stage_b["stage"],
        "status": "quarantined_pending_delete",
        "raw_capture_bytes": 7,
        "raw_capture_sha256": "3" * 64,
        "reopen_validation": "PASS",
    }
    pending["pending_audit_sha256"] = canonical_digest(pending, "pending_audit_sha256")
    audit: dict[str, Any] = {
        "schema_version": "m14-l04.9-v2-retained-triplet-audit-v1",
        "stage": stage_b["stage"],
        "status": stage_b["status"],
        "source_commit_sha": SOURCE_COMMIT,
        "source_tree_sha256": SOURCE_TREE,
        "cli_sha256": transport["cli_sha256"],
        "cli_invocation_count": 1,
        "cleanup_marker_count": 1,
        "raw_before_parse": True,
        "raw_status": "deleted_verified",
        "raw_absent": True,
        "reopen_validation": "PASS",
        "transport_sha256": transport["transport_sha256"],
        "members": members,
        "pending_audit": pending,
        "pending_audit_sha256": pending["pending_audit_sha256"],
        "raw_capture_bytes": 7,
        "raw_capture_sha256": "3" * 64,
        "raw_predelete_reopen_validation": "PASS",
        "deleted_verified": True,
        "source_sha256": stage_b["source_sha256"],
        "one_cli_invocation": True,
        "cleanup_status": "PASS",
    }
    audit["audit_sha256"] = canonical_digest(audit, "audit_sha256")
    record = build_promotion_record(
        stage_b,
        candidate,
        addendum,
        train,
        holdout,
        seed,
        transport,
        audit,
        expected_source_commit_sha=SOURCE_COMMIT,
        expected_source_tree_sha256=SOURCE_TREE,
        policy=policy,
    )
    assert record["evidence_level"] == "D3" and record["evidence_eligible"] is True
    members["run"]["bytes"] += 1
    audit["audit_sha256"] = canonical_digest(audit, "audit_sha256")
    assert validate_promotion_record(
        record,
        stage_b,
        candidate,
        addendum,
        train,
        holdout,
        seed,
        transport,
        audit,
        expected_source_commit_sha=SOURCE_COMMIT,
        expected_source_tree_sha256=SOURCE_TREE,
        policy=policy,
    )


def test_injected_real_stage_a_records_d1_attestation_and_executes_scorer() -> None:
    train, addendum, _ = _base()
    calls = 0

    def score(_row: dict[str, Any], layer: int, offset: int) -> float:
        nonlocal calls
        calls += 1
        return 0.16 if (layer, offset) == (6, 0) else 0.0

    resources: dict[str, Any] = {
        "stage": "real_runtime",
        "execution_attempted": True,
        "execution_backend": "cuda",
        "model": "openai-community/gpt2@e7da7f221d5bf496a48136c0cd264e630fe9fcc8",
        "model_revision": "openai-community/gpt2@e7da7f221d5bf496a48136c0cd264e630fe9fcc8",
        "integration": "TransformerLMIntegration",
        "model_adapter": "N/A",
        "device": "cuda",
        "backend": "cuda",
        "dtype": "float32",
        "hook": {"registered": 1296, "capture_calls": 1368, "removed": 1296},
        "intervention": {"patch_calls": 1296, "control_calls": 0, "forward_calls": 1368},
        "operation_counts": {
            "candidate_evaluations": 2592,
            "hooks": 1296,
            "captures": 1368,
            "patches": 1296,
            "controls": 0,
            "forwards": 1368,
        },
        "cleanup": {"hook_count": 0, "completed": True},
        "resource_peak": {
            "peak_cpu_bytes": 1,
            "peak_gpu_bytes": 1,
            "unit": "bytes",
            "budget_cpu_bytes": 2,
            "budget_gpu_bytes": 2,
        },
        "no_mutation": True,
    }
    artifact = run_real_stage_a(
        train,
        addendum,
        source_sha256="a" * 64,
        runtime={"score": score, "resources": resources},
    )
    assert calls > 0
    assert artifact["status"] == "stage_a_complete"
    assert artifact["evidence_level"] == "D1"
    assert validate_stage_a(artifact, train, addendum) == []


def test_power_simulation_uses_declared_draws_and_frozen_result() -> None:
    result = frozen_power_result()
    assert POWER_ASSUMPTIONS["simulations"] == 2000
    assert POWER_ASSUMPTIONS["bootstrap_replicates"] == 2000
    assert result["digest_sha256"] == power_digest(result)
    assert validate_power_result(result) == []


@pytest.mark.parametrize("mutation", ["count", "result", "digest"])
def test_power_tampering_fails_after_rehash(mutation: str) -> None:
    result = frozen_power_result()
    if mutation == "count":
        result["assumptions"]["simulations"] = 1999
    elif mutation == "result":
        result["result"]["power"] = 0.999
    result["digest_sha256"] = "0" * 64 if mutation == "digest" else power_digest(result)
    assert validate_power_result(result)

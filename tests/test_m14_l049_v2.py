"""Offline contract tests for the preregistered L04.9 v2 stages."""

from __future__ import annotations

import builtins
import copy
import hashlib
import json
import os
import shutil
import subprocess
import weakref
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

import scripts._m14_l049_v2_inputs as stage_b_inputs
import scripts._m14_l049_v2_promotion as promotion
import scripts._m14_l049_v2_real_runtime as real_runtime
import scripts._m14_l049_v2_stage_a as stage_a_module
import scripts._m14_l049_v2_validate_common as validate_common
import scripts.m14_l049_v2_load_stress as load_stress
import scripts.m14_l049_v2_preflight as stage_b_preflight
import scripts.m14_l049_v2_resource_probe as resource_probe
from scripts._m14_l049_v2_fixture import authoring_manifest_digest, generate_rows, read_rows, validate_fixture
from scripts._m14_l049_v2_inputs import (
    CANONICAL_STAGE_B_HOLDOUT,
    CANONICAL_STAGE_B_MANIFEST,
    CANONICAL_STAGE_B_SEED,
    SOURCE_KEYED_STAGE_B_CANDIDATE,
    canonical_stage_b_paths,
    validate_canonical_stage_b_inputs,
)
from scripts._m14_l049_v2_power import POWER_ASSUMPTIONS, frozen_power_result, power_digest, validate_power_result
from scripts._m14_l049_v2_promotion import (
    RealEvidenceCommitment,
    RealPromotionPolicy,
    _canonical_mapping_matches,
    _canonical_rows_matches,
    _repository_tree_errors,
    _safe_bound_file,
    _validate_official_audit,
    _validate_real_sidecars,
    build_legacy_promotion_record,
    load_real_promotion_policy,
    validate_legacy_promotion_record,
    validate_promotion_record,
)
from scripts._m14_l049_v2_real_runtime import ResourceTracker, _hidden, _patch_positions, attempted_runtime_resources
from scripts._m14_l049_v2_retention import build_retention_record, validate_retention_record
from scripts._m14_l049_v2_schema import (
    STAGE_B_SEEDS,
    V2_ADDENDUM_PATH,
    CommitmentPolicy,
    canonical_digest,
    canonical_fixture_bytes,
    canonical_json_bytes,
    fixture_digest,
    top_level_cli_sha256,
)
from scripts._m14_l049_v2_stage_a import (
    build_stage_a_artifact,
    normalize_attempted_real_resources,
    run_real_stage_a,
    run_stage_a_candidate_workload,
)
from scripts._m14_l049_v2_stage_b import (
    build_stage_b_failure_artifact,
    build_stage_b_validation_rejected_artifact,
    evaluate_stage_b,
    label_stratified_shuffled_mapping,
)
from scripts._m14_l049_v2_transport import build_transport_metadata, validate_transport_metadata
from scripts._m14_l049_v2_validate import validate_stage_a, validate_stage_b
from scripts._m14_l049_v2_validate_stage_a import validate_stage_a_impl
from scripts._m14_l049_v2_validate_stage_b import validate_stage_b_impl

ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "artifacts/m14/l04-l049-v2-train.jsonl"
STAGE_A_FAILURE_SIDECAR = ROOT / (
    "artifacts/m14/l04-explanations.ssh.L049V2StageA.41828c2e12e1efacb80e8cb5a0c62e4e69a688b2.sidecar.json"
)
CURRENT_STAGE_A_FAILURE_SIDECAR = ROOT / (
    "artifacts/m14/l04-explanations.ssh.L049V2StageA.3b15627585a0fc07e28c0f8b5d0118630f3ded5d.sidecar.json"
)
CURRENT_STAGE_A_RESOURCE_ASSESSMENT = ROOT / (
    "artifacts/m14/l04-explanations.ssh.L049V2StageA.66455a526f6974b31974f058dda341817dea2998.assessment.sidecar.json"
)
CURRENT_STAGE_A_VALIDATION_REJECTION_ASSESSMENT = ROOT / (
    "artifacts/m14/l04-explanations.ssh.L049V2StageA.8cfe9b0a47c001f7f228f33313d5c99be8ee9cb5.assessment.sidecar.json"
)
CURRENT_INCIDENT_ASSESSMENT = ROOT / (
    "artifacts/m14/l04-explanations.ssh.L049V2StageA.13bf46e7b748f6fa64bf5f44cd80c194d1ca889d.incident-assessment.sidecar.json"
)
CURRENT_B295_ASSESSMENT = ROOT / (
    "artifacts/m14/l04-explanations.ssh.L049V2StageA.b295a506933e18f6d9139b0439f0e80d6ed441e8.assessment.sidecar.json"
)
CURRENT_A205_RAW_ONLY_ASSESSMENT = ROOT / (
    "artifacts/m14/l04-explanations.ssh.L049V2StageA.a205ca7f0f4714c045027094208804c479a85445.assessment.sidecar.json"
)
CURRENT_5D6_ASSESSMENT = ROOT / (
    "artifacts/m14/l04-explanations.ssh.L049V2StageA.5d6d8fb5e06890cf9615936f049681a6d1e52228.assessment.sidecar.json"
)
CURRENT_RESOURCE_PROBE_ASSESSMENT = ROOT / (
    "artifacts/m14/l049-v2-resource-probe.67d2fb7649543ffc679e521f4f2a2ee970c55c63.assessment.sidecar.json"
)
CURRENT_855F_ASSESSMENT = ROOT / (
    "artifacts/m14/l04-explanations.ssh.L049V2StageA.855f440b87e62c875ba32ae584a77e3cd2394025.assessment.sidecar.json"
)
CURRENT_LOAD_STRESS_ASSESSMENT = ROOT / (
    "artifacts/m14/l049-v2-load-stress.32211433134facb901098c1a6313d010f22495a0.assessment.sidecar.json"
)
CURRENT_D1_ASSESSMENT = ROOT / (
    "artifacts/m14/l04-explanations.ssh.L049V2StageA.76a45ea74fbb2843b7d109855c2c387ab98b3e47.d1-assessment.sidecar.json"
)
CURRENT_D2_ASSESSMENT = ROOT / (
    "artifacts/m14/l04-explanations.ssh.L049V2StageB.6af20749b305f591d2c90d868cb09e71f623bdd0.d2-assessment.sidecar.json"
)
CURRENT_D1_AUDIT = ROOT / (
    "artifacts/m14/l04-explanations.ssh.L049V2StageA.76a45ea74fbb2843b7d109855c2c387ab98b3e47.audit.json"
)
CURRENT_D2_AUDIT = ROOT / (
    "artifacts/m14/l04-explanations.ssh.L049V2StageB.6af20749b305f591d2c90d868cb09e71f623bdd0.audit.json"
)
CURRENT_D1_CANDIDATE = ROOT / (
    "artifacts/m14/l04-explanations.L049V2StageA.76a45ea74fbb2843b7d109855c2c387ab98b3e47.candidate.json"
)
STAGE_B_PROVISIONING_ASSESSMENT = ROOT / (
    "artifacts/m14/l04-explanations.L049V2StageB.provisioning-assessment.sidecar.json"
)
STAGE_A_FAILURE_RAW = ROOT / (
    "artifacts/m14/l04-explanations.ssh.L049V2StageA.41828c2e12e1efacb80e8cb5a0c62e4e69a688b2.raw.txt"
)
SOURCE_COMMIT = "1" * 40
SOURCE_TREE = "2" * 64


def test_canonical_stage_b_inputs_are_provisioned_and_independently_validated() -> None:
    assert validate_canonical_stage_b_inputs(ROOT) == []
    paths = canonical_stage_b_paths(ROOT)
    assert set(paths) == {"manifest", "holdout", "seed", "candidate"}
    assert paths["manifest"].relative_to(ROOT) == CANONICAL_STAGE_B_MANIFEST
    assert paths["holdout"].relative_to(ROOT) == CANONICAL_STAGE_B_HOLDOUT
    assert paths["seed"].relative_to(ROOT) == CANONICAL_STAGE_B_SEED
    assert paths["candidate"].relative_to(ROOT) == SOURCE_KEYED_STAGE_B_CANDIDATE
    holdout_rows = read_rows(paths["holdout"])[1]
    train_rows = read_rows(TRAIN_PATH)[1]
    assert len(holdout_rows) == 48
    assert len({row["group_id"] for row in holdout_rows}) == 24
    assert validate_fixture(train_rows, holdout_rows) == []


def _build_temp_stage_b_git_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "stage-b-input-repo"
    destination = repo / "artifacts" / "m14"
    destination.mkdir(parents=True)
    for source in canonical_stage_b_paths(ROOT).values():
        shutil.copyfile(source, destination / source.name)
    subprocess.run(["git", "init", "--quiet", str(repo)], check=True, capture_output=True)
    relative_paths = [source.relative_to(ROOT).as_posix() for source in canonical_stage_b_paths(ROOT).values()]
    subprocess.run(["git", "-C", str(repo), "add", "--", *relative_paths], check=True, capture_output=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=Stage B Test",
            "-c",
            "user.email=stage-b-test@example.invalid",
            "commit",
            "--quiet",
            "-m",
            "provisioned inputs",
        ],
        check=True,
        capture_output=True,
    )
    return repo


def test_stage_b_preflight_requires_all_four_inputs_tracked(tmp_path: Path) -> None:
    repo = _build_temp_stage_b_git_repo(tmp_path)
    assert validate_canonical_stage_b_inputs(repo, require_tracked=True) == []
    assert stage_b_preflight.main(["--repo-root", str(repo), "--require-tracked"]) == 0


def test_stage_b_preflight_rejects_nested_repo_root_before_input_checks(tmp_path: Path) -> None:
    parent = _build_temp_stage_b_git_repo(tmp_path)
    nested = parent / "nested-checkout"
    destination = nested / "artifacts" / "m14"
    destination.mkdir(parents=True)
    for source in canonical_stage_b_paths(ROOT).values():
        shutil.copyfile(source, destination / source.name)
    relative_paths = [path.relative_to(parent).as_posix() for path in canonical_stage_b_paths(nested).values()]
    subprocess.run(["git", "-C", str(parent), "add", "--", *relative_paths], check=True, capture_output=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(parent),
            "-c",
            "user.name=Stage B Test",
            "-c",
            "user.email=stage-b-test@example.invalid",
            "commit",
            "--quiet",
            "-m",
            "nested inputs",
        ],
        check=True,
        capture_output=True,
    )
    assert validate_canonical_stage_b_inputs(nested, require_tracked=True) == ["canonical_repo_root_mismatch"]


def test_stage_b_preflight_rejects_non_git_root_before_input_checks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "non-git-input-root"
    destination = repo / "artifacts" / "m14"
    destination.mkdir(parents=True)
    for source in canonical_stage_b_paths(ROOT).values():
        shutil.copyfile(source, destination / source.name)
    monkeypatch.setattr(
        stage_b_inputs,
        "_safe_regular_under",
        lambda *_args: pytest.fail("input shape was checked"),
    )
    assert validate_canonical_stage_b_inputs(repo, require_tracked=True) == ["canonical_repo_root_mismatch"]


@pytest.mark.parametrize("failure", ["timeout", "oserror"])
def test_stage_b_preflight_fails_closed_on_repo_root_probe_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    repo = _build_temp_stage_b_git_repo(tmp_path)

    def fail_probe(*_args: object, **_kwargs: object) -> object:
        if failure == "timeout":
            raise subprocess.TimeoutExpired("git", 10)
        raise OSError("probe failed")

    monkeypatch.setattr(stage_b_inputs.subprocess, "run", fail_probe)
    assert validate_canonical_stage_b_inputs(repo, require_tracked=True) == ["canonical_repo_root_mismatch"]


@pytest.mark.skipif(os.name != "nt", reason="case normalization is platform-specific")
def test_stage_b_preflight_normalizes_repo_root_case(tmp_path: Path) -> None:
    repo = _build_temp_stage_b_git_repo(tmp_path)
    altered = Path(str(repo).swapcase())
    assert validate_canonical_stage_b_inputs(altered, require_tracked=True) == []


@pytest.mark.parametrize("missing_key", ["manifest", "holdout", "seed", "candidate"])
def test_stage_b_preflight_rejects_valid_unindexed_input(tmp_path: Path, missing_key: str) -> None:
    repo = _build_temp_stage_b_git_repo(tmp_path)
    missing = canonical_stage_b_paths(repo)[missing_key].relative_to(repo).as_posix()
    remove_command = ["git", "-C", str(repo), "rm", "--cached", "--quiet", "--", missing]
    subprocess.run(remove_command, check=True, capture_output=True)
    assert validate_canonical_stage_b_inputs(repo, require_tracked=True) == ["canonical_input_untracked"]


def test_stage_b_preflight_is_validate_only_and_sanitized(capsys: pytest.CaptureFixture[str]) -> None:
    assert stage_b_preflight.main(["--repo-root", str(ROOT)]) == 0
    output = capsys.readouterr().out
    assert json.loads(output) == {"stage": "stage_b_input_preflight", "status": "PASS", "evaluation": "not_run"}
    assert "holdout" not in output
    assert "seed" not in output


def test_stage_b_provisioning_assessment_is_canonical_and_pending() -> None:
    sidecar = json.loads(STAGE_B_PROVISIONING_ASSESSMENT.read_bytes())
    assert sidecar["status"] == "ready_for_stage_b"
    assert sidecar["sidecar_sha256"] == canonical_digest(sidecar, "sidecar_sha256")
    assert sidecar["assessment"] == {
        "evaluation": False,
        "promotion": False,
        "standard_finalize": False,
        "d2": False,
        "d3": False,
        "status": "inputs_validated_stage_b_ready",
    }
    assert sidecar["inputs"]["holdout"]["commitment_valid"] is True
    assert sidecar["inputs"]["holdout"]["train_overlap_valid"] is True
    assert sidecar["inputs"]["seed"]["value_redacted"] is True
    serialized = json.dumps(sidecar)
    assert "C:/" not in serialized and "\\\\" not in serialized


def test_current_stage_b_d2_assessment_binds_finalized_evidence_without_leaks() -> None:
    sidecar = json.loads(CURRENT_D2_ASSESSMENT.read_bytes())
    assert canonical_digest(sidecar, "sidecar_sha256") == sidecar["sidecar_sha256"]
    assert sidecar["sidecar_sha256"] == "eae9c36ce5d57f94ce41b7ab5cb0277fa089e58c67332fe4bf36dd248f8de280"
    assert sidecar["source"] == {
        "commit_sha": "6af20749b305f591d2c90d868cb09e71f623bdd0",
        "tree_sha256": "a0f1fb55c8d112128d81f3942132657100eac00f",
        "exact_source_verified": True,
        "use_case": "L049V2StageB",
        "provisioning_commit_sha": "7d1e23fdbc385909f964df05360f01027d3b6c35",
        "provisioning_tree_sha256": "5f43b035a043faf97237cd87aa621bec61c805b1",
        "predecessor_d1_commit_sha": "76a45ea74fbb2843b7d109855c2c387ab98b3e47",
        "predecessor_d1_tree_sha256": "392d241719b10fe6a946f20d203b9e0ff0f5f46c",
    }
    assert sidecar["status"] == "deleted_verified"
    assert sidecar["assessment"] == {
        "evidence_level": "D2",
        "evaluation_complete": True,
        "evidence_eligible": True,
        "promotion_candidate": True,
        "repository_promotion": False,
        "semantic_finalization": False,
        "standard_finalize": True,
        "retention_finalized": True,
        "d3": False,
        "status": "d2_evaluation_complete_retention_finalized",
    }
    assert sidecar["semantic"]["groups"] == 24
    assert sidecar["semantic"]["rows"] == 48
    assert sidecar["semantic"]["seed_summaries"] == 5
    assert sidecar["semantic"]["bootstrap_replicates"] == 2000
    assert sidecar["semantic"]["recovery"]["all_seed_gates_pass"] is True
    assert sidecar["semantic"]["paired_true_minus_shuffled"]["all_seed_gates_pass"] is True
    assert sidecar["resources"]["cleanup"] == {"completed": True, "hook_count": 0, "attestation": "PASS"}
    assert sidecar["validation"]["stage_b_validator_errors"] == []
    assert sidecar["execution"]["invocation_invariant"] == "satisfied"
    assert sidecar["execution"]["ssh_processes"] == 1
    assert sidecar["execution"]["stage_b_cli_invocations"] == 1

    raw = sidecar["evidence"]["raw_capture"]
    assert raw["present"] is False
    assert raw["present_before_finalize"] is True
    assert raw["local_absence_proof"] is True
    assert raw["absence_verified_by"] == "official_finalize_delete"
    assert not (ROOT / raw["path"]).exists()

    evidence = [sidecar["evidence"]["audit"]]
    evidence.extend(sidecar["evidence"]["triad"].values())
    for item in evidence:
        path = ROOT / item["path"]
        assert path.is_file()
        assert path.stat().st_size == item["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
        assert item["present"] is True
    assert sidecar["evidence"]["audit"]["prior_pending_bytes"] == 3241
    assert sidecar["evidence"]["audit"]["prior_pending_sha256"] == (
        "bfbf41e2e97d8818d4f42dc44d732ecd87e303129a1eae1bafe73857f2b8d2be"
    )
    assert sidecar["retention"]["raw_retention_status"] == "deleted_verified"
    assert sidecar["retention"]["raw_local_absence_verified"] is True
    assert sidecar["retention"]["standard_finalize"] is True
    assert sidecar["retention"]["finalize_delete"]["raw_target_only"] is True
    assert sidecar["retention"]["finalize_delete"]["triad_survives"] == "yes"
    assert sidecar["retention"]["previous_pending_sidecar_sha256"] == (
        "76ba6a0079c61df0f0de4040a279b41f00ca039ab1b75d44614b41a8817410f6"
    )
    assert sidecar["retention"]["repository_promotion"] is False
    serialized = json.dumps(sidecar, sort_keys=True).lower()
    assert all(secret not in serialized for secret in ("traceback", "prompt", "seed value", "begin private"))
    assert "C:/" not in serialized and "\\\\" not in serialized


def test_stage_b_docs_reconcile_historical_provisioning_with_current_d2() -> None:
    m14 = (ROOT / "docs/M14_REAL_SYSTEM_VALIDATION.md").read_text(encoding="utf-8")
    sprint = (ROOT / "docs/sprint-plans/sprint-79.md").read_text(encoding="utf-8")
    summary = (ROOT / "artifacts/task_79_l04_9_summary.md").read_text(encoding="utf-8")
    d2_link = "l04-explanations.ssh.L049V2StageB.6af20749b305f591d2c90d868cb09e71f623bdd0.d2-assessment.sidecar.json"
    assert "historical provisioning" in m14
    assert d2_link in m14
    assert "evaluation remains not run" not in m14
    assert "At this preregistration checkpoint" in sprint
    assert d2_link in summary
    assert "v2 preregistration — frozen historical contract" in summary


def test_canonical_stage_b_inputs_reject_candidate_tampering(tmp_path: Path) -> None:
    destination = tmp_path / "artifacts" / "m14"
    destination.mkdir(parents=True)
    for source in canonical_stage_b_paths(ROOT).values():
        target = destination / source.name
        target.write_bytes(source.read_bytes())
    (destination / SOURCE_KEYED_STAGE_B_CANDIDATE.name).write_bytes(b"{}")
    assert "candidate_file_commitment" in validate_canonical_stage_b_inputs(tmp_path)


def test_canonical_stage_b_inputs_reject_wrong_source_keyed_candidate_path(tmp_path: Path) -> None:
    destination = tmp_path / "artifacts" / "m14"
    destination.mkdir(parents=True)
    for source in canonical_stage_b_paths(ROOT).values():
        (destination / source.name).write_bytes(source.read_bytes())
    expected = destination / SOURCE_KEYED_STAGE_B_CANDIDATE.name
    expected.rename(destination / "artifacts.L049V2StageA.wrong-source.candidate.json")
    assert "canonical_input_shape" in validate_canonical_stage_b_inputs(tmp_path)


def test_canonical_stage_b_inputs_reject_symlinked_holdout(tmp_path: Path) -> None:
    destination = tmp_path / "artifacts" / "m14"
    destination.mkdir(parents=True)
    paths = canonical_stage_b_paths(ROOT)
    for key, source in paths.items():
        target = destination / source.name
        if key == "holdout":
            try:
                target.symlink_to(source)
            except OSError:
                pytest.skip("symlink creation is unavailable")
        else:
            target.write_bytes(source.read_bytes())
    assert "canonical_input_shape" in validate_canonical_stage_b_inputs(tmp_path)


def test_current_d1_candidate_and_assessment_are_canonical_and_stage_b_ready() -> None:
    candidate = json.loads(CURRENT_D1_CANDIDATE.read_bytes())
    rows = read_rows(TRAIN_PATH)[1]
    addendum = json.loads(V2_ADDENDUM_PATH.read_bytes())
    assert validate_stage_a(candidate, rows, addendum) == []
    assert candidate["selection"]["consensus_candidate"] == {"layer": 10, "offset": 0}
    assert candidate["artifact_sha256"] == canonical_digest(candidate, "artifact_sha256")
    assert not {"raw", "bundle", "transport", "holdout", "seed", "path"}.intersection(candidate)

    sidecar = json.loads(CURRENT_D1_ASSESSMENT.read_bytes())
    assert canonical_digest(sidecar, "sidecar_sha256") == sidecar["sidecar_sha256"]
    assert sidecar["status"] == "deleted_verified"
    assert sidecar["assessment"]["evidence_level"] == "D1"
    assert sidecar["assessment"]["selected_candidate"] == {"layer": 10, "offset": 0}
    assert sidecar["assessment"]["stage_b_access"] is False
    assert sidecar["assessment"]["holdout_access"] is False
    assert sidecar["assessment"]["promotion"] is False
    assert sidecar["assessment"]["finalization"] is False
    assert sidecar["assessment"]["retention_finalized"] is True
    assert sidecar["standard_finalize"] is True
    assert sidecar["retention"]["standard_finalize"] is True
    assert sidecar["retention"]["raw_local_absence_verified"] is True
    assert sidecar["retention"]["previous_pending_sidecar_sha256"] == (
        "237ba264988af961bcba793aec05cc9d8331afddaca55ec2f52204bbbc06d83e"
    )
    assert sidecar["retention"]["finalize_delete"]["raw_target_only"] is True
    assert sidecar["retention"]["finalize_delete"]["triad_survives"] == "yes"
    assert sidecar["retention"]["finalize_delete"]["candidate_survives"] == "yes"
    assert sidecar["evidence"]["raw_capture"]["present"] is False
    assert sidecar["evidence"]["raw_capture"]["local_absence_proof"] is True
    assert not (ROOT / sidecar["evidence"]["raw_capture"]["path"]).exists()
    assert sidecar["evidence"]["audit"] == {
        "bytes": 3397,
        "final_reopen_validation": "PASS",
        "path": "artifacts/m14/l04-explanations.ssh.L049V2StageA.76a45ea74fbb2843b7d109855c2c387ab98b3e47.audit.json",
        "present": True,
        "raw_status": "deleted_verified",
        "sha256": "a1b60ec6804e0468716398c75c9e3508a1c982c0b312fcd8fb1c5aab737e166d",
        "validation": {"archive": "PASS", "envelopes": "PASS"},
    }


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


def test_validation_rejection_sidecar_records_owner_exception_delete() -> None:
    sidecar = json.loads(CURRENT_STAGE_A_VALIDATION_REJECTION_ASSESSMENT.read_bytes())
    assert (
        canonical_digest(sidecar, "sidecar_sha256")
        == sidecar["sidecar_sha256"]
        == "56b2a93b9ca9da42aa66e09a32fb9a99af7aee569f34e7bfdd58f86f97613abb"
    )
    assert sidecar["raw_capture"] == {
        "bytes": 5791,
        "sha256": "85255e5eb1924e3e30946e93a530ab7746dc0ed560bfff20c3124df59bf1dd06",
    }
    assert sidecar["artifact"]["failure_kind"] == "validation_rejected"
    assert sidecar["artifact"]["validation_codes"] == ["validation_rejected_unavailable_resource"]
    assert sidecar["artifact"]["selected_candidate"] is None
    assert sidecar["raw_retention_status"] == "deleted_by_owner_exception"
    assert sidecar["standard_finalize"] is False
    assert sidecar["retention"]["reason"] == "no_triad_bundle_status_66/validation_rejected_unavailable_resource"
    assert sidecar["retention"]["previous_sidecar_sha256"] == (
        "b64cd9c8f0b20652904bae5694dca9c3fa0cf3e6dfaf22f6fcdc6e02aec8ef6a"
    )
    assert sidecar["owner_exception"]["deletion_verification"]["absent_after_delete"] is True
    assert sidecar["repository_promotion"] is False


def test_current_failed_stage_a_sidecar_records_validator_misclassification() -> None:
    sidecar = json.loads(CURRENT_STAGE_A_FAILURE_SIDECAR.read_bytes())
    assert (
        canonical_digest(sidecar, "sidecar_sha256")
        == sidecar["sidecar_sha256"]
        == "02a355cd6dffe6d85e07a0cce2175c126a4f512056bccb88408cadf744ecde93"
    )
    assert sidecar["source"] == {
        "commit_sha": "3b15627585a0fc07e28c0f8b5d0118630f3ded5d",
        "tree_sha256": "2052472177cb7284027571241fbdeb41ff7dd8c2",
        "use_case": "L049V2StageA",
    }
    assert sidecar["raw_capture"] == {
        "bytes": 6008,
        "sha256": "757af5cce5b4e8aa4c5b476ecc52d69ae192423179c23b3fc148510a8eafc212",
    }
    assert sidecar["markers"] == {
        "bundle_status": 66,
        "cli_status": 1,
        "final_status": 1,
        "remote_cleanup": "PASS",
        "transport_cleanup": "PASS",
        "transport_decode_match": "PASS",
        "transport_decode_status": 0,
    }
    assert sidecar["semantic"] == {
        "evidence_status": "unavailable_validator_rejection",
        "selection_status": "reached",
        "status": "selection_reached",
    }
    assert sidecar["artifact"] == {
        "audit": "not_created",
        "bundle": "not_created",
        "failure": "no_triad",
        "partial": "not_created",
        "run": "not_created",
        "selected_candidate": None,
    }
    assert sidecar["reason_code"] == "no_triad_bundle_status_66/semantic_gate_d0_validator_misclassification"
    assert sidecar["raw_retention_status"] == "deleted_by_owner_exception"
    assert sidecar["standard_finalize"] is False
    assert sidecar["owner_exception"] == {
        "deletion_verification": {
            "absent_after_delete": True,
            "pre_delete_bytes": 6008,
            "pre_delete_sha256": "757af5cce5b4e8aa4c5b476ecc52d69ae192423179c23b3fc148510a8eafc212",
        },
        "previous_sidecar_sha256": "a6a2afe995abdaa8996e202f51851813f6a7f7ca580715ff90109885afe39fe9",
        "reason": "no_triad_bundle_status_66/semantic_gate_d0_validator_misclassification",
        "standard_finalize": False,
    }
    assert not (
        ROOT / "artifacts/m14/l04-explanations.ssh.L049V2StageA.3b15627585a0fc07e28c0f8b5d0118630f3ded5d.raw.txt"
    ).exists()
    assert sidecar["repository_promotion"] is False
    serialized = json.dumps(sidecar, sort_keys=True)
    assert all(secret not in serialized for secret in ("traceback", "PROMPT", "holdout_plaintext", "BEGIN PRIVATE"))


def test_current_stage_a_resource_assessment_is_canonical_and_sanitized() -> None:
    sidecar = json.loads(CURRENT_STAGE_A_RESOURCE_ASSESSMENT.read_bytes())
    assert canonical_digest(sidecar, "sidecar_sha256") == sidecar["sidecar_sha256"]
    assert sidecar["sidecar_sha256"] == "d45dcaca9b4cfbe1bf6c8b3f4507b9b80afb483f5b5609d965883fdc9c4dc8b1"
    assert sidecar["source"]["commit_sha"] == "66455a526f6974b31974f058dda341817dea2998"
    assert sidecar["raw_capture"] == {
        "bytes": 54754,
        "sha256": "c366462b3f5dee243832e539db7c4495a018e8988305d2c54c0e83fdcd36767f",
    }
    assert sidecar["resource_assessment"]["reason_code"] == "measured_source_with_zero_peaks"
    assert sidecar["resource_assessment"]["resource_provenance_valid"] is False
    assert sidecar["artifact"]["evidence_level"] == "D0"
    assert sidecar["transport"] == {
        "payload_sha256": "a78e29527f6f4810729c63a13d61b3e83fdf11de4421c69b4bca0966cdc09950",
        "decode_status": 0,
        "decode_sha256": "a78e29527f6f4810729c63a13d61b3e83fdf11de4421c69b4bca0966cdc09950",
        "decode_match": "PASS",
        "cli_status": 0,
        "bundle_status": 0,
        "final_status": 0,
        "bundle_bytes": 36149,
        "bundle_sha256": "28e3df35ecb6817ae86804df511e232a3ba377d0993752826fc4fb93d2a5883e",
        "cleanup": "PASS",
        "transport_cleanup": "PASS",
    }
    assert set(sidecar["deleted_evidence"]) == {
        "artifacts/m14/l04-explanations.ssh.L049V2StageA.66455a526f6974b31974f058dda341817dea2998.raw.txt",
        "artifacts/m14/l04-explanations.L049V2StageA.attempt1.failure.json",
        "artifacts/m14/l04-explanations.L049V2StageA.attempt1.partial.json",
        "artifacts/m14/l04-explanations.L049V2StageA.attempt1.run.json",
        "artifacts/m14/l04-explanations.ssh.L049V2StageA.66455a526f6974b31974f058dda341817dea2998.audit.json",
    }
    assert sidecar["retention"]["raw_retention_status"] == "deleted_by_owner_exception"
    assert sidecar["retention"]["reason"] == "historical_resource_provenance_invalid_measured_zero_peaks"
    assert (
        sidecar["retention"]["previous_sidecar_sha256"]
        == "84dfffd53c28fb12cbb3ee8640616c3f95f1b938d076e075437ea5c290447f75"
    )
    assert sidecar["retention"]["standard_finalize"] is False
    assert sidecar["owner_exception"]["deletion_verification"]["absent_after_delete"] is True
    assert all(
        item["absent_after_delete"] is True
        for item in sidecar["owner_exception"]["deletion_verification"]["files"].values()
    )
    for path, record in sidecar["owner_exception"]["deletion_verification"]["files"].items():
        assert record["absent_after_delete"] is True
        assert not (ROOT / path).exists()
    serialized = json.dumps(sidecar, sort_keys=True).lower()
    assert all(secret not in serialized for secret in ("prompt", "holdout", "traceback", "private key"))


def test_current_incident_assessment_binds_owner_exception_deletion_without_secrets() -> None:
    sidecar = json.loads(CURRENT_INCIDENT_ASSESSMENT.read_bytes())
    assert canonical_digest(sidecar, "sidecar_sha256") == sidecar["sidecar_sha256"]
    assert sidecar["sidecar_sha256"] == "77aedee5a182c843dbc20aa1e989e0e54eb5dc79656ce5f0498722c121d5225b"
    assert sidecar["source"] == {
        "commit_sha": "13bf46e7b748f6fa64bf5f44cd80c194d1ca889d",
        "use_case": "L049V2StageA",
        "exact_source_verified": True,
    }
    assert sidecar["status"] == "deleted_by_owner_exception"
    assert sidecar["assessment"]["promotion"] is False
    assert sidecar["assessment"]["finalization"] is False
    assert sidecar["assessment"]["standard_finalize"] is False
    assert sidecar["assessment"]["repository_promotion"] is False
    assert sidecar["standard_finalize"] is False
    assert sidecar["repository_promotion"] is False
    assert sidecar["execution"] == {
        "completed_payloads": 1,
        "ssh_launches_reported": 2,
        "second_launch_aborted": True,
        "second_remote_reach": "uncertain",
        "invocation_invariant": "not_satisfied",
        "evidence_limitation": sidecar["execution"]["evidence_limitation"],
    }
    evidence_items = [sidecar["evidence"]["raw_capture"], sidecar["evidence"]["audit"]]
    evidence_items.extend(sidecar["evidence"]["triad"].values())
    assert all(item["path"].startswith("artifacts/") for item in evidence_items)
    assert all(item["present_after_delete"] is False for item in evidence_items)
    assert sidecar["retention"]["raw_and_triad"] == "deleted_by_owner_exception"
    assert sidecar["retention"]["standard_finalize"] is False
    assert sidecar["owner_exception"]["previous_status"] == "pending"
    assert sidecar["owner_exception"]["standard_finalize"] is False
    assert sidecar["owner_exception"]["repository_promotion"] is False
    assert sidecar["owner_exception"]["pre_delete_verification"]["all_files_verified"] is True
    assert sidecar["owner_exception"]["post_delete_absence"]["all_files_absent"] is True
    for item in evidence_items:
        assert not (ROOT / item["path"]).exists()
    assert all(
        record["absent"] is True for record in sidecar["owner_exception"]["post_delete_absence"]["files"].values()
    )
    serialized = json.dumps(sidecar, sort_keys=True)
    assert all(secret not in serialized for secret in ("traceback", "holdout_plaintext", "BEGIN PRIVATE"))


def test_current_b295_assessment_binds_owner_exception_evidence_deletion() -> None:
    sidecar = json.loads(CURRENT_B295_ASSESSMENT.read_bytes())
    assert canonical_digest(sidecar, "sidecar_sha256") == sidecar["sidecar_sha256"]
    assert sidecar["source"]["commit_sha"] == "b295a506933e18f6d9139b0439f0e80d6ed441e8"
    assert sidecar["status"] == "deleted_by_owner_exception"
    assert sidecar["assessment"]["finalizer_rejection_code"] == "finalizer_resource_peak_fields"
    assert sidecar["assessment"]["selection_evaluations"] == 2592
    assert sidecar["assessment"]["selection_reuse"] == "discarded_not_reusable"
    assert sidecar["assessment"]["resource_measurement"]["gpu_subfield"] == "unknown"
    assert sidecar["execution"]["invocation_invariant"] == "satisfied"
    assert sidecar["transport"]["mutex_digest"] == "821f69e4c0f206a1128ba3aa61fa6004ef9206ad430f00419d480d5aacbba1ea"
    assert all(
        not item["present_after_delete"] for item in [sidecar["evidence"]["raw_capture"], sidecar["evidence"]["audit"]]
    )
    assert all(not item["present_after_delete"] for item in sidecar["evidence"]["triad"].values())
    assert sidecar["evidence"]["audit"]["rebuild"] == {
        "previous_bytes": 2638,
        "previous_sha256": "c9fdfcfa5f60a010c403e143062ee6f7a9820f588308b7f13912240093456fd4",
        "current_bytes": 3276,
        "current_sha256": "914d374748e803513d3aeba04c556fa17584ebdabd3efdad4dcde6bf26e43a91",
        "reason": "distinguish_generic_archive_member_names_from_source_unique_local_paths",
    }
    assert all(
        sidecar["evidence"]["triad"][kind]["path"].split(".")[-3] == "b295a506933e18f6d9139b0439f0e80d6ed441e8"
        for kind in ("partial", "run", "failure")
    )
    assert sidecar["owner_exception"]["pre_delete_verification"]["all_files_verified"] is True
    assert sidecar["owner_exception"]["post_delete_absence"]["all_files_absent"] is True
    serialized = json.dumps(sidecar, sort_keys=True)
    assert all(secret not in serialized for secret in ("traceback", "holdout_plaintext", "BEGIN PRIVATE"))


def test_current_a205_raw_only_assessment_is_canonical_and_pending() -> None:
    sidecar = json.loads(CURRENT_A205_RAW_ONLY_ASSESSMENT.read_bytes())
    assert canonical_digest(sidecar, "sidecar_sha256") == sidecar["sidecar_sha256"]
    assert sidecar["source"] == {
        "commit_sha": "a205ca7f0f4714c045027094208804c479a85445",
        "tree_sha256": "a96ca4612c5b04a5bf7a370baec20b17d4b28b9f",
        "use_case": "L049V2StageA",
        "exact_source_verified": True,
    }
    raw = sidecar["evidence"]["raw_capture"]
    assert raw["bytes"] == 6598
    assert raw["sha256"] == "6080a35c40369c225e8611891f5403b0b53c194b065473c885ea73d58464b674"
    assert raw["written_before_parse"] is True
    assert raw["present_before_delete"] is True
    assert raw["present_after_delete"] is False
    assert sidecar["status"] == "deleted_by_owner_exception"
    assert sidecar["execution"]["selection_status"] == "not_reached"
    assert sidecar["execution"]["model_adapter_integration"] == "not_reached"
    assert sidecar["execution"]["resource_finalizer_status"] == "not_reached"
    assert sidecar["assessment"]["failure_kind"] == "remote_stage_input_missing"
    assert sidecar["assessment"]["promotion"] is False
    assert sidecar["assessment"]["finalization"] is False
    assert sidecar["evidence"]["audit"] is None
    assert sidecar["evidence"]["bundle"] is None
    assert sidecar["evidence"]["triad"] is None
    assert sidecar["markers"]["remote_cleanup"] == "PASS"
    assert sidecar["markers"]["transport_cleanup"] == "PASS"
    assert sidecar["markers"]["cli_status"] == 1
    assert sidecar["markers"]["bundle_status"] == 66
    assert sidecar["retention"]["status"] == "deleted_by_owner_exception"
    assert sidecar["owner_exception"]["pre_delete_verification"]["all_files_verified"] is True
    assert sidecar["owner_exception"]["post_delete_absence"]["all_files_absent"] is True
    assert sidecar["owner_exception"]["post_delete_absence"]["files"][raw["path"]]["absent"] is True
    serialized = json.dumps(sidecar, sort_keys=True).lower()
    assert all(secret not in serialized for secret in ("holdout plaintext", "traceback", "begin private"))
    assert "f:\\ai-ml\\latent-anything" not in serialized


def test_current_5d6_assessment_binds_owner_exception_deletion() -> None:
    sidecar = json.loads(CURRENT_5D6_ASSESSMENT.read_bytes())
    assert canonical_digest(sidecar, "sidecar_sha256") == sidecar["sidecar_sha256"]
    assert sidecar["sidecar_sha256"] == "949cc9d93e72b5fa1f01744aeef200469bc8e8d17f35b3742249922d69d2bc29"
    assert sidecar["source"] == {
        "commit_sha": "5d6d8fb5e06890cf9615936f049681a6d1e52228",
        "tree_sha256": "07bea97c9c4b55919fa707d473e2cecf7e1392a6",
        "use_case": "L049V2StageA",
        "exact_source_verified": True,
    }
    assert sidecar["status"] == "deleted_by_owner_exception"
    assert sidecar["execution"] == {
        "completed_payloads": 1,
        "ssh_launches_reported": 1,
        "remote_cli_invocations": 1,
        "retry_performed": False,
        "second_launch_aborted": False,
        "remote_reach": "reached",
        "cuda_proof": "PASS",
        "remote_cleanup": "PASS",
        "remote_cleanup_interpretation": "Recorded marker only; no independent remote absence proof is claimed.",
        "transport_cleanup": "PASS",
        "mutex_key_sha256": "80741d63253909a313149b756cf1668f0a339fd191ee7bf4a114bc74b7a276fb",
        "argv_sha256": "d8066963bd854746f2d1d2a3be716e64f1cca58602854bd75572bf7053a5159a",
    }
    raw = sidecar["evidence"]["raw_capture"]
    assert raw["bytes"] == 9779
    assert raw["sha256"] == "6dd1741c94b5af2fc084667129197304b5de2b51d023920a22164a33d342c4d2"
    triad = sidecar["evidence"]["triad"]
    assert sidecar["evidence"]["audit"]["sha256"] == "a4599c0d4154314576b82bdd9eb1132d4fea29b44cef5b69dde019dfc20c6827"
    assert sidecar["evidence"]["bundle"]["sha256"] == "87e063d18d0f654792af9afc9de3b3b4519a82348367c672bcc191f290543efe"
    assert sidecar["assessment"]["remote_cleanup_claim"] == "not_claimed"
    assert not sidecar["evidence"]["raw_capture"]["present"]
    assert not sidecar["evidence"]["audit"]["present"]
    assert all(not sidecar["evidence"]["triad"][kind]["present"] for kind in ("partial", "run", "failure"))
    evidence_items = [sidecar["evidence"]["raw_capture"], sidecar["evidence"]["audit"]]
    evidence_items.extend(sidecar["evidence"]["triad"][kind] for kind in ("partial", "run", "failure"))
    assert all(item["local_absence_proof"] is True for item in evidence_items)
    assert all(not (ROOT / item["path"]).exists() for item in evidence_items)
    assert all(sidecar["source"]["commit_sha"] in triad[kind]["path"] for kind in ("partial", "run", "failure"))
    assert sidecar["evidence"]["audit"]["rebuild"] == {
        "status": "canonical_retain_reopened",
        "previous_bytes": 3234,
        "previous_sha256": "a4599c0d4154314576b82bdd9eb1132d4fea29b44cef5b69dde019dfc20c6827",
        "current_bytes": 3234,
        "current_sha256": "a4599c0d4154314576b82bdd9eb1132d4fea29b44cef5b69dde019dfc20c6827",
        "archive_member_names_preserved": True,
        "source_unique_local_paths": True,
    }
    assert sidecar["assessment"]["evidence_level"] == "D0"
    assert sidecar["assessment"]["selection_reuse"] == "discarded_not_reusable"
    assert sidecar["assessment"]["resource_measurement"]["exact_peak_subcondition"] == "unknown"
    assert sidecar["assessment"]["root_cause"]["cuda_device_contract_inference"]["status"] == (
        "not_supported_by_artifact"
    )
    assert sidecar["assessment"]["promotion"] is False
    assert sidecar["assessment"]["finalization"] is False
    assert sidecar["retention"]["status"] == "deleted_by_owner_exception"
    assert sidecar["retention"]["deletion"]["postdelete_local_absence"] is True
    assert sidecar["retention"]["deletion"]["irreversible_not_recoverable_from_git"] is True
    serialized = json.dumps(sidecar, sort_keys=True).lower()
    assert all(secret not in serialized for secret in ("traceback", "begin private", "f:\\ai-ml"))


def test_current_resource_probe_assessment_is_canonical_and_scope_bounded() -> None:
    sidecar = json.loads(CURRENT_RESOURCE_PROBE_ASSESSMENT.read_bytes())
    assert canonical_digest(sidecar, "sidecar_sha256") == sidecar["sidecar_sha256"]
    assert sidecar["sidecar_sha256"] == "d0ee03f491930877c0695e19ca7b72d8ea2d85d2d0c1e600f3e98b0048b2c12e"
    assert sidecar["source"] == {
        "commit_sha": "67d2fb7649543ffc679e521f4f2a2ee970c55c63",
        "tree_sha256": "72d18bfe8f899243ad1aad20000d86c3b69a85a8",
        "use_case": "L049V2ResourceProbe",
        "exact_source_verified": True,
    }
    assert sidecar["probe_cli"]["path"] == "scripts/m14_l049_v2_resource_probe.py"
    assert sidecar["probe_cli"]["version"] == "m14-l049-v2-resource-probe-v1"
    raw = sidecar["evidence"]["raw_capture"]
    assert raw == {
        "path": "artifacts/m14/l049-v2-resource-probe.67d2fb7649543ffc679e521f4f2a2ee970c55c63.raw.txt",
        "bytes": 344,
        "sha256": "e812a99ccd7388a5183879d54a40e5d5f5f4913b1ce3944772cc17e24ffb49aa",
        "written_before_parse": True,
        "present_at_assessment": False,
        "local_absence_proof": True,
    }
    assert sidecar["markers"]["count"] == 8
    assert sidecar["markers"]["order"] == list(resource_probe._PROBE_MARKER_NAMES)
    assert sidecar["markers"]["parser"] == "PASS"
    assert sidecar["markers"]["fixed_sanitized"] is True
    assert sidecar["execution"] == {
        "ssh_processes": 1,
        "probe_invocations": 1,
        "ssh_exit": 0,
        "provenance": "executor_recorded_only",
        "remote_exact_sha_check": "PASS",
        "remote_cuda_preflight": "PASS",
    }
    assert all(value == "not_attempted" for value in sidecar["scope"].values())
    assert sidecar["assessment"]["diagnostic_status"] == "bounded_resource_only"
    assert sidecar["assessment"]["probe_cleanup"] == "self_attested"
    assert sidecar["assessment"]["remote_checkout_cleanup"] == "unverified"
    assert sidecar["assessment"]["remote_checkout_absence_proof"] is False
    assert sidecar["evidence"]["raw_capture"]["local_absence_proof"] is True
    assert not (ROOT / sidecar["evidence"]["raw_capture"]["path"]).exists()
    assert sidecar["status"] == "deleted_by_owner_exception"
    assert sidecar["assessment"]["standard_finalize"] is False
    assert sidecar["assessment"]["repository_promotion"] is False
    assert sidecar["retention"]["status"] == "deleted_by_owner_exception"
    assert sidecar["retention"]["deletion"]["postdelete_local_absence"] is True
    assert sidecar["retention"]["deletion"]["irreversible_not_recoverable_from_git"] is True
    serialized = json.dumps(sidecar, sort_keys=True).lower()
    assert all(
        secret not in serialized for secret in ("traceback", "holdout_plaintext", "private key", "f:\\ai-ml", "cuda:0")
    )


def test_current_load_stress_assessment_is_canonical_and_scope_bounded() -> None:
    sidecar = json.loads(CURRENT_LOAD_STRESS_ASSESSMENT.read_bytes())
    assert canonical_digest(sidecar, "sidecar_sha256") == sidecar["sidecar_sha256"]
    assert sidecar["sidecar_sha256"] == "fbf4d3534ff0db52010829166166a1f0ce04bd4f8dd75638c1c29122ca0452d4"
    assert sidecar["source"] == {
        "commit_sha": "32211433134facb901098c1a6313d010f22495a0",
        "tree_sha256": "b592505dd9920f73d1d419e2302fb7a8ab6351ae",
        "use_case": "L049V2LoadStress",
        "exact_source_verified": True,
    }
    assert sidecar["stress_cli"]["path"] == "scripts/m14_l049_v2_load_stress.py"
    assert sidecar["stress_cli"]["version"] == "m14-l049-v2-load-stress-v1"
    assert sidecar["stress_cli"]["script_bytes"] == 10245
    assert sidecar["stress_cli"]["script_sha256"] == "d546836764e0792c7377ced8f56ca06cba58de524bec5357ebb2f61e1c5dcc47"
    assert sidecar["transport_payload"]["decode_match"] == "PASS"
    assert sidecar["transport_payload"]["payload_sha256"] == sidecar["transport_payload"]["decoded_payload_sha256"]
    raw = sidecar["evidence"]["raw_capture"]
    assert raw == {
        "path": "artifacts/m14/l049-v2-load-stress.32211433134facb901098c1a6313d010f22495a0.raw.txt",
        "bytes": 1175,
        "sha256": "c320b802c905fa325fdf9e20f42e4db130ad99a26a108dc0396e81e3e68d13f9",
        "written_before_parse": True,
        "present_at_assessment": False,
        "local_absence_proof": True,
    }
    assert not (ROOT / raw["path"]).exists()
    assert sidecar["markers"]["inner"]["count"] == 12
    assert sidecar["markers"]["inner"]["order"] == list(load_stress._MARKERS)
    assert sidecar["markers"]["inner"]["parser"] == "PASS"
    assert sidecar["markers"]["inner"]["tamper_validation"] == "PASS"
    assert sidecar["markers"]["inner"]["fixed_sanitized"] is True
    assert sidecar["markers"]["outer"] == {
        "source_sha": "32211433134facb901098c1a6313d010f22495a0",
        "git_sha_match": "PASS",
        "cuda_present": True,
        "precheck": "PASS",
        "exec_status": 0,
        "markers_present": True,
        "outer_cleanup": "PASS",
        "transport_cleanup": "PASS",
    }
    assert sidecar["markers"]["inner"]["raw_values"] == {
        "status": "PASS",
        "finalizer_code": "NONE",
        "measurement_status": "available",
        "measurement_reason": "none",
        "cpu_provenance": True,
        "gpu_provenance": True,
        "device_canonical": True,
        "counters_complete": True,
        "cpu_budget_ok": True,
        "gpu_allocated_budget_ok": True,
        "gpu_reserved_budget_ok": True,
        "cleanup": "PASS",
    }
    assert sidecar["workload"] == {
        "scope": "train_only",
        "expected_records": 1296,
        "expected_scorer_calls": 2592,
        "counter_match": True,
        "records_published": False,
        "selection_published": False,
        "folds_published": False,
        "oof_published": False,
        "artifact_published": False,
        "holdout_access": False,
        "stage_b_access": False,
    }
    assert sidecar["execution"]["provenance"] == "executor_recorded_only"
    assert sidecar["execution"]["ssh_processes"] == 1
    assert sidecar["execution"]["stress_invocations"] == 1
    assert sidecar["assessment"]["semantic_status"] == "not_evaluated"
    assert sidecar["assessment"]["internal_stress_cleanup"] == "self_attested"
    assert sidecar["assessment"]["remote_checkout_cleanup"] == "unverified"
    assert sidecar["assessment"]["remote_checkout_absence_proof"] is False
    assert sidecar["assessment"]["promotion"] is False
    assert sidecar["assessment"]["finalization"] is False
    assert sidecar["retention"]["status"] == "deleted_by_owner_exception"
    assert sidecar["retention"]["deletion"]["postdelete_local_absence"] is True
    assert sidecar["retention"]["deletion"]["irreversible_not_recoverable_from_git"] is True
    serialized = json.dumps(sidecar, sort_keys=True).lower()
    assert all(
        secret not in serialized
        for secret in ("traceback", "holdout plaintext", "begin private", "f:\\ai-ml", "cuda:0")
    )


def test_current_855f_assessment_binds_owner_exception_and_stress_link() -> None:
    sidecar = json.loads(CURRENT_855F_ASSESSMENT.read_bytes())
    assert canonical_digest(sidecar, "sidecar_sha256") == sidecar["sidecar_sha256"]
    assert sidecar["sidecar_sha256"] == "00e3b60269c14108d9e244b55db9f64d27e45d99386a0f3462ec5002489dbfdd"
    assert sidecar["status"] == "deleted_by_owner_exception"
    assert sidecar["retention"]["status"] == "deleted_by_owner_exception"
    assert sidecar["retention"]["deletion"] == {
        "authorization": "owner_explicit",
        "predelete_verification": (
            "All five sidecar-bound paths were regular, non-reparse, workspace-contained, "
            "untracked, and matched recorded size and SHA-256."
        ),
        "postdelete_local_absence": True,
        "standard_finalize": False,
        "repository_promotion": False,
        "irreversible_not_recoverable_from_git": True,
    }
    assert sidecar["owner_exception"]["pre_delete_verification"]["all_files_verified"] is True
    assert sidecar["owner_exception"]["post_delete_absence"]["all_files_absent"] is True
    assert sidecar["load_stress_diagnostic"] == {
        "entrypoint": "scripts/m14_l049_v2_load_stress.py",
        "scope": "train_only",
        "status": "executed_once_pass",
        "assessment_sidecar": (
            "artifacts/m14/l049-v2-load-stress.32211433134facb901098c1a6313d010f22495a0.assessment.sidecar.json"
        ),
        "execution_provenance": "executor_recorded_only; see source-bound assessment sidecar for fixed marker facts",
        "emits": "fixed_sanitized_markers_only",
        "selection_or_oof_emitted": False,
        "holdout_or_stage_b_access": False,
    }
    for item in [sidecar["evidence"]["raw_capture"], sidecar["evidence"]["audit"]]:
        assert item["present"] is False
        assert item["local_absence_proof"] is True
    for kind in ("partial", "run", "failure"):
        assert sidecar["evidence"]["triad"][kind]["present"] is False
        assert sidecar["evidence"]["triad"][kind]["local_absence_proof"] is True
    serialized = json.dumps(sidecar, sort_keys=True).lower()
    assert all(secret not in serialized for secret in ("traceback", "holdout plaintext", "begin private", "f:\\ai-ml"))


class _FakeCuda:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def is_available(self) -> bool:
        return True

    def current_device(self) -> int:
        self.calls.append("current")
        return 0

    def reset_peak_memory_stats(self, device: int) -> None:
        assert device == 0
        self.calls.append("reset")

    def synchronize(self, device: int) -> None:
        assert device == 0
        self.calls.append("sync")

    def max_memory_allocated(self, device: int) -> int:
        assert device == 0
        return 200

    def max_memory_reserved(self, device: int) -> int:
        assert device == 0
        return 400


def test_resource_tracker_records_nonzero_cuda_and_linux_rss_provenance(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(real_runtime.sys, "platform", "linux")
    cuda = _FakeCuda()
    clock_values = iter((10.0, 12.5))
    fake_resource = SimpleNamespace(RUSAGE_SELF=0, getrusage=lambda _kind: SimpleNamespace(ru_maxrss=3))
    tracker = ResourceTracker(
        torch_module=SimpleNamespace(cuda=cuda), resource_module=fake_resource, clock=lambda: next(clock_values)
    )
    tracker.start()
    tracker.finish()
    assert cuda.calls == ["current", "reset", "sync"]
    assert tracker.resource_peak() == {
        "peak_cpu_bytes": 3072,
        "peak_gpu_bytes": 200,
        "peak_gpu_reserved_bytes": 400,
        "unit": "bytes",
        "budget_cpu_bytes": 6_000_000_000,
        "budget_gpu_bytes": 6_000_000_000,
        "measurement_status": "available",
        "measurement_reason": None,
        "elapsed_seconds": 2.5,
        "elapsed_source": "time.perf_counter",
        "cpu_source": "resource.ru_maxrss_linux_kib",
        "gpu_source": "torch.cuda.max_memory_allocated",
        "gpu_reserved_source": "torch.cuda.max_memory_reserved",
        "gpu_device": "cuda:0",
    }


def test_runtime_forward_snapshot_retains_only_scorer_owned_values(monkeypatch: pytest.MonkeyPatch) -> None:
    result = SimpleNamespace(attention_mask=np.ones((1, 2), dtype=np.int64), logits=np.ones((1, 2, 4)))
    raw = {3: np.ones((1, 2, 2), dtype=np.float32)}
    monkeypatch.setattr(real_runtime, "_forward", lambda *_args, **_kwargs: (result, raw))
    monkeypatch.setattr(real_runtime, "_margin", lambda *_args, **_kwargs: 0.5)

    snapshot = real_runtime._forward_snapshot(  # pyright: ignore[reportPrivateUsage]
        object(),
        "prompt",
        margin_targets=(" true",),
        raw_capture_layers=(3,),
    )
    assert set(snapshot.__dataclass_fields__) == {"attention_mask", "margins", "raw_states"}
    assert not hasattr(snapshot, "logits")
    assert snapshot.margins == {" true": 0.5}
    raw[3][0, 0, 0] = 99.0
    result.attention_mask[0, 0] = 0
    assert snapshot.raw_states is not raw
    assert snapshot.raw_states[3][0, 0, 0] == 1.0
    assert snapshot.attention_mask[0, 0] == 1
    with pytest.raises(TypeError):
        snapshot.margins["false"] = 0.0  # type: ignore[index]
    with pytest.raises(TypeError):
        snapshot.raw_states[4] = np.zeros((1, 2, 2))  # type: ignore[index]
    with pytest.raises(ValueError):
        snapshot.raw_states[3][0, 0, 0] = 0.0
    with pytest.raises(ValueError):
        snapshot.raw_states[3].setflags(write=True)


def test_runtime_forward_snapshot_releases_full_result_after_extraction(monkeypatch: pytest.MonkeyPatch) -> None:
    holder: dict[str, weakref.ReferenceType[Any]] = {}

    class _Result:
        pass

    def make_forward(*_args: Any, **_kwargs: Any) -> tuple[Any, dict[int, np.ndarray]]:
        result = _Result()
        result.attention_mask = np.ones((1, 2), dtype=np.int64)
        result.logits = np.ones((1, 2, 4))
        raw = {3: np.ones((1, 2, 2), dtype=np.float32)}
        holder["result"] = weakref.ref(result)
        holder["logits"] = weakref.ref(result.logits)
        holder["raw"] = weakref.ref(raw[3])
        return result, raw

    monkeypatch.setattr(real_runtime, "_forward", make_forward)
    monkeypatch.setattr(real_runtime, "_margin", lambda *_args, **_kwargs: 0.5)
    snapshot = real_runtime._forward_snapshot(object(), "prompt", margin_targets=(" true",), raw_capture_layers=(3,))  # type: ignore[reportPrivateUsage]
    assert snapshot.raw_states[3] is not holder["raw"]()
    del snapshot
    import gc

    gc.collect()
    assert holder["result"]() is None
    assert holder["logits"]() is None
    assert holder["raw"]() is None


def test_canonical_candidate_workload_matches_stage_a_call_count() -> None:
    rows = read_rows(TRAIN_PATH)[1]
    calls = 0

    def scorer(_row: Mapping[str, Any], _layer: int, _offset: int) -> float:
        nonlocal calls
        calls += 1
        return 0.1

    records: list[dict[str, Any]] = []
    assert run_stage_a_candidate_workload(rows, scorer, on_record=records.append) == 1296
    assert calls == 2592
    assert len(records) == 1296


def test_load_stress_marker_validator_is_fixed_and_fail_closed() -> None:
    records = (
        ("L049_V2_LOAD_STRESS_STATUS", "PASS"),
        ("L049_V2_LOAD_STRESS_FINALIZER_CODE", "NONE"),
        ("L049_V2_LOAD_STRESS_MEASUREMENT_STATUS", "available"),
        ("L049_V2_LOAD_STRESS_MEASUREMENT_REASON", "none"),
        ("L049_V2_LOAD_STRESS_CPU_PROVENANCE", "true"),
        ("L049_V2_LOAD_STRESS_GPU_PROVENANCE", "true"),
        ("L049_V2_LOAD_STRESS_DEVICE_CANONICAL", "true"),
        ("L049_V2_LOAD_STRESS_COUNTERS_COMPLETE", "true"),
        ("L049_V2_LOAD_STRESS_CPU_BUDGET_OK", "true"),
        ("L049_V2_LOAD_STRESS_GPU_ALLOCATED_BUDGET_OK", "true"),
        ("L049_V2_LOAD_STRESS_GPU_RESERVED_BUDGET_OK", "true"),
        ("L049_V2_LOAD_STRESS_CLEANUP", "PASS"),
    )
    output = "\n".join(f"{name}={value}" for name, value in records)
    assert load_stress.validate_load_stress_output(output) == []
    assert load_stress.validate_load_stress_output(
        output.replace("L049_V2_LOAD_STRESS_STATUS=PASS", "L049_V2_LOAD_STRESS_STATUS=FAIL")
    )
    assert load_stress.validate_load_stress_output(
        output.replace("L049_V2_LOAD_STRESS_MEASUREMENT_REASON=none", "L049_V2_LOAD_STRESS_MEASUREMENT_REASON=secret")
    )


def test_load_stress_safe_markers_guard_unhashable_metadata() -> None:
    resources = {"resource_peak": {"measurement_status": [], "measurement_reason": {"secret": True}}}
    markers = load_stress._safe_markers(  # pyright: ignore[reportPrivateUsage]
        resources,
        workload_ok=False,
        cleanup_ok=False,
        rejection_code=["secret"],  # type: ignore[arg-type]
    )
    output = "\n".join(f"{name}={value}" for name, value in markers)
    assert load_stress.validate_load_stress_output(output) == []
    assert "secret" not in output


def test_load_stress_requires_valid_finalizer_before_cleanup_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    resources = _real_resources_for_complete_selection()
    calls = 0

    def invalid_finalizer() -> object:
        nonlocal calls
        calls += 1
        return {"secret": "invalid"}

    resources["finalize"] = invalid_finalizer
    monkeypatch.setattr(load_stress, "read_rows", lambda _path: (b"", []))
    monkeypatch.setattr(load_stress, "build_stage_a_runtime", lambda _rows: (lambda *_args: 0.0, resources))
    monkeypatch.setattr(load_stress, "run_stage_a_candidate_workload", lambda *_args, **_kwargs: 1296)
    markers, code = load_stress.run_load_stress()
    output = "\n".join(f"{name}={value}" for name, value in markers)
    assert calls == 1
    assert code == 1
    assert "L049_V2_LOAD_STRESS_STATUS=FAIL" in output
    assert "L049_V2_LOAD_STRESS_FINALIZER_CODE=finalizer_top_level_fields" in output
    assert "L049_V2_LOAD_STRESS_CLEANUP=FAIL" in output
    assert load_stress.validate_load_stress_output(output) == []


@pytest.mark.parametrize("finalizer_mode", ["missing", "raising"])
def test_load_stress_missing_or_throwing_finalizer_is_sanitized(
    monkeypatch: pytest.MonkeyPatch, finalizer_mode: str
) -> None:
    resources = _real_resources_for_complete_selection()
    if finalizer_mode == "raising":

        def raising_finalizer() -> object:
            raise RuntimeError("secret cleanup detail")

        resources["finalize"] = raising_finalizer
    monkeypatch.setattr(load_stress, "read_rows", lambda _path: (b"", []))
    monkeypatch.setattr(load_stress, "build_stage_a_runtime", lambda _rows: (lambda *_args: 0.0, resources))
    monkeypatch.setattr(load_stress, "run_stage_a_candidate_workload", lambda *_args, **_kwargs: 1296)
    markers, code = load_stress.run_load_stress()
    output = "\n".join(f"{name}={value}" for name, value in markers)
    assert code == 1
    assert "L049_V2_LOAD_STRESS_STATUS=FAIL" in output
    assert "L049_V2_LOAD_STRESS_CLEANUP=FAIL" in output
    assert "secret" not in output
    assert load_stress.validate_load_stress_output(output) == []


def test_resource_tracker_reset_failure_preserves_attempted_canonical_device() -> None:
    class _ResetFailCuda(_FakeCuda):
        def reset_peak_memory_stats(self, device: int) -> None:
            assert device == 0
            raise RuntimeError("secret reset detail")

    tracker = ResourceTracker(
        torch_module=SimpleNamespace(cuda=_ResetFailCuda()),
        resource_module=SimpleNamespace(RUSAGE_SELF=0, getrusage=lambda _kind: SimpleNamespace(ru_maxrss=1)),
        clock=iter((1.0, 2.0)).__next__,
    )
    tracker.start()
    tracker.finish()
    resources = real_runtime._runtime_resources(real_runtime._new_counters(), tracker)
    peak = resources["resource_peak"]
    assert peak["measurement_reason"] == "cuda_reset_failed"
    assert peak["gpu_device"] == "cuda:0"
    assert peak["gpu_source"] == peak["gpu_reserved_source"] == "unavailable"
    assert peak["peak_gpu_bytes"] == peak["peak_gpu_reserved_bytes"] == 0
    assert stage_a_module._finalizer_rejection_code(resources) is None
    assert validate_common.real_resources(resources) == []


def test_resource_tracker_marks_zero_peak_unavailable_without_exception_text() -> None:
    cuda = _FakeCuda()
    cuda.max_memory_allocated = lambda _device: 0  # type: ignore[method-assign]
    cuda.max_memory_reserved = lambda _device: 0  # type: ignore[method-assign]
    fake_resource = SimpleNamespace(RUSAGE_SELF=0, getrusage=lambda _kind: SimpleNamespace(ru_maxrss=1))
    tracker = ResourceTracker(
        torch_module=SimpleNamespace(cuda=cuda), resource_module=fake_resource, clock=iter((1.0, 2.0)).__next__
    )
    tracker.start()
    tracker.finish()
    peak = tracker.resource_peak()
    assert peak["measurement_status"] == "unavailable"
    assert peak["measurement_reason"] == "cuda_zero_peak"
    assert peak["gpu_device"] == "cuda:0"
    resources = real_runtime._runtime_resources(real_runtime._new_counters(), tracker)
    assert stage_a_module._finalizer_rejection_code(resources) is None
    assert validate_common.real_resources(resources) == []
    assert "exception" not in json.dumps(peak).lower()


@pytest.mark.parametrize(
    ("allocated", "reserved", "reason"),
    [
        (200, 0, "cuda_zero_peak"),
        (0, 400, "cuda_zero_peak"),
        (400, 200, "cuda_peak_query_failed"),
        (-1, 400, "cuda_peak_query_failed"),
        (True, 400, "cuda_peak_query_failed"),
        (float("nan"), 400, "cuda_peak_query_failed"),
    ],
)
def test_resource_tracker_publishes_no_asymmetric_or_invalid_cuda_pair(
    allocated: object, reserved: object, reason: str
) -> None:
    cuda = _FakeCuda()
    cuda.max_memory_allocated = lambda _device: allocated  # type: ignore[method-assign]
    cuda.max_memory_reserved = lambda _device: reserved  # type: ignore[method-assign]
    tracker = ResourceTracker(
        torch_module=SimpleNamespace(cuda=cuda),
        resource_module=SimpleNamespace(RUSAGE_SELF=0, getrusage=lambda _kind: SimpleNamespace(ru_maxrss=1)),
        clock=iter((1.0, 2.0)).__next__,
    )
    tracker.start()
    tracker.finish()
    peak = tracker.resource_peak()
    assert peak["measurement_status"] == "unavailable"
    assert peak["measurement_reason"] == reason
    assert peak["peak_gpu_bytes"] == 0
    assert peak["peak_gpu_reserved_bytes"] == 0
    assert peak["gpu_source"] == "unavailable"
    assert peak["gpu_reserved_source"] == "unavailable"
    assert peak["gpu_device"] == "cuda:0"
    resources = real_runtime._runtime_resources(real_runtime._new_counters(), tracker)
    assert stage_a_module._finalizer_rejection_code(resources) is None
    assert validate_common.real_resources(resources) == []


@pytest.mark.parametrize("failing_query", ["allocated", "reserved"])
def test_resource_tracker_clears_cuda_pair_when_either_query_raises(failing_query: str) -> None:
    cuda = _FakeCuda()
    if failing_query == "allocated":
        cuda.max_memory_allocated = lambda _device: (_ for _ in ()).throw(RuntimeError("secret"))  # type: ignore[method-assign]
    else:
        cuda.max_memory_reserved = lambda _device: (_ for _ in ()).throw(RuntimeError("secret"))  # type: ignore[method-assign]
    tracker = ResourceTracker(
        torch_module=SimpleNamespace(cuda=cuda),
        resource_module=SimpleNamespace(RUSAGE_SELF=0, getrusage=lambda _kind: SimpleNamespace(ru_maxrss=1)),
        clock=iter((1.0, 2.0)).__next__,
    )
    tracker.start()
    tracker.finish()
    peak = tracker.resource_peak()
    assert peak["measurement_status"] == "unavailable"
    assert peak["measurement_reason"] == "cuda_peak_query_failed"
    assert peak["peak_gpu_bytes"] == peak["peak_gpu_reserved_bytes"] == 0
    assert peak["gpu_device"] == "cuda:0"
    resources = real_runtime._runtime_resources(real_runtime._new_counters(), tracker)
    assert stage_a_module._finalizer_rejection_code(resources) is None
    assert validate_common.real_resources(resources) == []
    assert "secret" not in json.dumps(peak)


def test_failed_finalizer_normalization_uses_live_operation_snapshot_for_projections() -> None:
    resources = _real_resources_for_complete_selection()
    resources["hook"] = {"registered": 0, "capture_calls": 0, "removed": 0}
    resources["intervention"] = {"patch_calls": 0, "control_calls": 0, "forward_calls": 0}
    projected = normalize_attempted_real_resources(resources)
    assert projected["operation_counts"] == {
        "candidate_evaluations": 2592,
        "hooks": 1296,
        "captures": 1368,
        "patches": 1296,
        "controls": 0,
        "forwards": 1368,
    }
    assert projected["hook"] == {"registered": 1296, "capture_calls": 1368, "removed": 1296}
    assert projected["intervention"] == {"patch_calls": 1296, "control_calls": 0, "forward_calls": 1368}


def test_resource_tracker_uses_psutil_rss_fallback_when_resource_is_zero() -> None:
    cuda = _FakeCuda()
    fake_resource = SimpleNamespace(RUSAGE_SELF=0, getrusage=lambda _kind: SimpleNamespace(ru_maxrss=0))
    fake_psutil = SimpleNamespace(Process=lambda: SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=4096)))
    tracker = ResourceTracker(
        torch_module=SimpleNamespace(cuda=cuda),
        resource_module=fake_resource,
        psutil_module=fake_psutil,
        clock=iter((1.0, 2.0)).__next__,
    )
    tracker.start()
    tracker.finish()
    peak = tracker.resource_peak()
    assert peak["peak_cpu_bytes"] == 4096
    assert peak["cpu_source"] == "psutil.Process.memory_info.rss"
    assert peak["measurement_status"] == "available"


def test_resource_tracker_uses_macos_ru_maxrss_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(real_runtime.sys, "platform", "darwin")
    fake_resource = SimpleNamespace(RUSAGE_SELF=0, getrusage=lambda _kind: SimpleNamespace(ru_maxrss=4096))
    tracker = ResourceTracker(
        torch_module=SimpleNamespace(cuda=_FakeCuda()),
        resource_module=fake_resource,
        clock=iter((1.0, 2.0)).__next__,
    )
    tracker.start()
    tracker.finish()
    peak = tracker.resource_peak()
    assert peak["peak_cpu_bytes"] == 4096
    assert peak["cpu_source"] == "resource.ru_maxrss_macos_bytes"


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
            "peak_gpu_reserved_bytes": 1,
            "unit": "bytes",
            "budget_cpu_bytes": 6_000_000_000,
            "budget_gpu_bytes": 6_000_000_000,
            "measurement_status": "available",
            "measurement_reason": None,
            "elapsed_seconds": 1.0,
            "elapsed_source": "time.perf_counter",
            "cpu_source": "resource.ru_maxrss_linux_kib",
            "gpu_source": "torch.cuda.max_memory_allocated",
            "gpu_reserved_source": "torch.cuda.max_memory_reserved",
            "gpu_device": "cuda:0",
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
    assert artifact["failure_kind"] == "runtime_exception"
    assert artifact["selection_complete"] is False
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
            "peak_gpu_reserved_bytes": 1,
            "unit": "bytes",
            "budget_cpu_bytes": 6_000_000_000,
            "budget_gpu_bytes": 6_000_000_000,
            "measurement_status": "available",
            "measurement_reason": None,
            "elapsed_seconds": 1.0,
            "elapsed_source": "time.perf_counter",
            "cpu_source": "resource.ru_maxrss_linux_kib",
            "gpu_source": "torch.cuda.max_memory_allocated",
            "gpu_reserved_source": "torch.cuda.max_memory_reserved",
            "gpu_device": "cuda:0",
        },
        "no_mutation": True,
    }


def _real_resources_for_complete_selection() -> dict[str, Any]:
    resources = _real_resources_for_failure()
    resources.update(
        {
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
        }
    )
    return resources


def test_real_stage_a_semantic_gate_failure_is_validator_clean() -> None:
    rows, addendum, _ = _base()
    artifact = run_real_stage_a(
        rows,
        addendum,
        source_sha256="a" * 64,
        runtime={"score": lambda *_args: 0.0, "resources": _real_resources_for_complete_selection()},
    )
    assert artifact["status"] == "stage_a_failed"
    assert artifact["evidence_level"] == "D0"
    assert artifact["evidence_eligible"] is False
    assert artifact["failure_kind"] == "semantic_gate"
    assert artifact["selection_complete"] is True
    assert "failure" not in artifact["selection"]
    assert artifact["selection"]["score_records"]
    assert artifact["selection"]["folds"]
    assert artifact["selection"]["consensus_candidate"] is not None
    assert artifact["selection"]["oof_metric"]["pass"] is False
    assert validate_stage_a(artifact, rows, addendum) == []


def test_real_stage_a_unavailable_resource_measurement_remains_d0() -> None:
    rows, addendum, _ = _base()
    resources = _real_resources_for_complete_selection()
    resources["resource_peak"].update(
        {
            "peak_cpu_bytes": 0,
            "peak_gpu_bytes": 0,
            "peak_gpu_reserved_bytes": 0,
            "measurement_status": "unavailable",
            "measurement_reason": "cuda_zero_peak",
            "cpu_source": "unavailable",
            "gpu_source": "unavailable",
            "gpu_reserved_source": "unavailable",
            "gpu_device": "unavailable",
        }
    )
    artifact = run_real_stage_a(
        rows,
        addendum,
        source_sha256="a" * 64,
        runtime={"score": lambda *_args: 0.0, "resources": resources},
    )
    assert artifact["evidence_level"] == "D0"
    assert validate_stage_a(artifact, rows, addendum) == []


def test_real_stage_a_measured_zero_peak_is_rejected_after_rehash() -> None:
    rows, addendum, _ = _base()
    artifact = run_real_stage_a(
        rows,
        addendum,
        source_sha256="a" * 64,
        runtime={"score": lambda *_args: 0.0, "resources": _real_resources_for_complete_selection()},
    )
    artifact["resources"]["resource_peak"]["peak_gpu_bytes"] = 0
    artifact["runtime_attestation"]["resources"]["peak_gpu_bytes"] = 0
    artifact["runtime_attestation"]["attestation_sha256"] = canonical_digest(
        artifact["runtime_attestation"], "attestation_sha256"
    )
    artifact["artifact_sha256"] = canonical_digest(artifact, "artifact_sha256")
    errors = validate_stage_a(artifact, rows, addendum)
    assert "Stage A measured resource provenance is invalid" in errors


def test_stage_a_recomputes_directional_recovery_from_primitive_margins() -> None:
    rows, addendum, _ = _base()
    artifact = run_real_stage_a(
        rows,
        addendum,
        source_sha256="a" * 64,
        runtime={
            "score": lambda *_args: {
                "clean_margin": 4.0,
                "corrupted_margin": 1.0,
                "patched_margin": 2.5,
            },
            "resources": _real_resources_for_complete_selection(),
        },
    )
    record = artifact["selection"]["score_records"][0]
    assert record["group_score"] == pytest.approx(0.5)
    assert record["primitive_margins"][0] == {
        "clean_margin": 4.0,
        "corrupted_margin": 1.0,
        "patched_margin": 2.5,
    }
    assert validate_stage_a(artifact, rows, addendum) == []


def test_stage_a_rehashed_primitive_margin_tampering_is_rejected() -> None:
    rows, addendum, _ = _base()
    artifact = run_real_stage_a(
        rows,
        addendum,
        source_sha256="a" * 64,
        runtime={"score": lambda *_args: 0.5, "resources": _real_resources_for_complete_selection()},
    )
    artifact["selection"]["score_records"][0]["primitive_margins"][0]["patched_margin"] = 9.0
    artifact["artifact_sha256"] = canonical_digest(artifact, "artifact_sha256")
    errors = validate_stage_a(artifact, rows, addendum)
    assert "Stage A directional recovery was not independently recomputed" in errors


def test_real_stage_b_runtime_exception_emits_validator_clean_attempted_d0() -> None:
    candidate, holdout, seed, addendum, _observations = _synthetic_stage_b()
    artifact = build_stage_b_failure_artifact(
        holdout,
        candidate,
        addendum,
        seed,
        source_sha256=str(candidate["source_sha256"]),
        error=IndexError("secret prompt must not escape"),
        resources=attempted_runtime_resources(),
        cli_sha256=top_level_cli_sha256("stage_b_holdout_evaluation"),
    )
    assert artifact["status"] == "stage_b_failed"
    assert artifact["evidence_level"] == "D0"
    assert artifact["failure_kind"] == "runtime_exception"
    assert artifact["seed_summaries"] == []
    assert "secret prompt" not in json.dumps(artifact)
    assert (
        validate_stage_b_impl(
            artifact,
            holdout,
            seed,
            candidate,
            addendum,
            read_rows(TRAIN_PATH)[1],
            policy=CommitmentPolicy.from_addendum(addendum),
        )
        == []
    )


def test_real_stage_a_semantic_no_consensus_is_validator_clean() -> None:
    rows, addendum, _ = _base()
    targets = [
        0,
        0,
        1,
        1,
        1,
        2,
        0,
        0,
        0,
        2,
        3,
        5,
        2,
        2,
        3,
        3,
        4,
        5,
        0,
        3,
        5,
        5,
        5,
        5,
        0,
        1,
        2,
        3,
        3,
        4,
        1,
        2,
        2,
        2,
        3,
        4,
    ]

    def score(row: Mapping[str, Any], layer: int, offset: int) -> float:
        group_index = int(str(row["group_id"])[-3:]) - 1
        return 1.0 if (layer, offset) == (targets[group_index], 0) else -1.0

    artifact = run_real_stage_a(
        rows,
        addendum,
        source_sha256="a" * 64,
        runtime={"score": score, "resources": _real_resources_for_complete_selection()},
    )
    assert artifact["status"] == "stage_a_failed"
    assert artifact["failure_kind"] == "semantic_gate"
    assert artifact["selection_complete"] is True
    assert artifact["selection"]["consensus_candidate"] is None
    assert 0 < artifact["selection"]["consensus_wins"] < 4
    assert artifact["selection"]["oof_metric"]["pass"] is False
    assert validate_stage_a(artifact, rows, addendum) == []


def test_real_stage_a_semantic_gate_failure_cli_writes_complete_d0_triad(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import torch

    import scripts._m14_l049_v2_real_runtime as real_runtime
    from scripts.m14_l049_v2_stage_a import main

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        real_runtime,
        "build_stage_a_runtime",
        lambda _rows: (lambda *_args: 0.0, _real_resources_for_complete_selection()),
    )
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
    artifact = partial["artifact"]
    assert artifact["failure_kind"] == "semantic_gate"
    assert artifact["selection_complete"] is True
    assert artifact["evidence_level"] == "D0"
    assert validate_stage_a(artifact, read_rows(TRAIN_PATH)[1], json.loads(V2_ADDENDUM_PATH.read_bytes())) == []


def test_real_stage_a_resource_projection_drops_untrusted_values(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import torch

    import scripts._m14_l049_v2_real_runtime as real_runtime
    import scripts.m14_l049_v2_stage_a as cli

    resources = _real_resources_for_complete_selection()
    resources["resource_peak"]["measurement_reason"] = "resource secret must never escape"
    resources["untrusted_extra"] = {"nested_secret": "must never escape"}
    resources["hook"]["nested_secret"] = "must never escape"
    resources["resource_peak"]["nested_secret"] = {"deep": "must never escape"}
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(real_runtime, "build_stage_a_runtime", lambda _rows: (lambda *_args: 0.0, resources))
    output = tmp_path / "artifact.json"
    cli.main(
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
    artifact = json.loads(output.read_bytes())
    assert artifact["failure_kind"] == "semantic_gate"
    assert artifact["selection_complete"] is True
    assert "failure" not in artifact["selection"]
    assert artifact["evidence_level"] == "D0"
    assert validate_stage_a(artifact, read_rows(TRAIN_PATH)[1], json.loads(V2_ADDENDUM_PATH.read_bytes())) == []
    assert "resource secret" not in json.dumps(artifact)
    assert "must never escape" not in json.dumps(artifact)
    assert set(artifact["resources"]) == {
        "stage",
        "execution_attempted",
        "execution_backend",
        "model",
        "model_revision",
        "integration",
        "model_adapter",
        "device",
        "backend",
        "dtype",
        "hook",
        "intervention",
        "operation_counts",
        "cleanup",
        "resource_peak",
        "no_mutation",
    }
    assert len(list(tmp_path.glob("l04-explanations.L049V2StageA.attempt1.*.json"))) == 3


@pytest.mark.parametrize("bad_scalar", [float("nan"), float("inf"), float("-inf"), 10**1000, "not-a-number"])
def test_real_resource_projection_is_fail_closed_for_adversarial_scalars(bad_scalar: object) -> None:
    resources = _real_resources_for_complete_selection()
    resources["network"] = "legacy network secret"
    resources["nested_sentinel"] = {"secret": "must not escape"}
    resources["operation_counts"]["forwards"] = bad_scalar
    resources["resource_peak"]["elapsed_seconds"] = bad_scalar
    projected = normalize_attempted_real_resources(resources)
    assert set(projected) == {
        "stage",
        "execution_attempted",
        "execution_backend",
        "model",
        "model_revision",
        "integration",
        "model_adapter",
        "device",
        "backend",
        "dtype",
        "hook",
        "intervention",
        "operation_counts",
        "cleanup",
        "resource_peak",
        "no_mutation",
    }
    assert "network" not in json.dumps(projected)
    assert "must not escape" not in json.dumps(projected)
    assert projected["operation_counts"]["forwards"] == 0
    assert projected["resource_peak"]["measurement_status"] == "unavailable"
    assert projected["resource_peak"]["measurement_reason"] == "tracker_unstarted"


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
        "hooks_remaining": 0,
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
        "hooks_remaining": 0,
        "error_type": "TypeError",
        "reason": "finalizer_invalid_result",
        "stage": "cleanup",
        "finalizer_rejection_code": "finalizer_not_mapping"
        if finalizer_result is None
        else "finalizer_top_level_missing_fields",
    }
    assert artifact["runtime_attestation"]["cleanup_hook_count"] == 0
    assert validate_stage_a(artifact, rows, addendum) == []


def test_stage_a_finalizer_rejection_preserves_live_partial_counters() -> None:
    rows, addendum, _ = _base()
    resources = _real_resources_for_failure()
    resources["operation_counts"] = {
        "candidate_evaluations": 0,
        "hooks": 0,
        "captures": 0,
        "patches": 0,
        "controls": 0,
        "forwards": 0,
    }

    def score(*_args: Any) -> float:
        resources["operation_counts"]["candidate_evaluations"] += 1
        return 0.0

    resources["finalize"] = lambda: {"stage": "cleanup"}
    artifact = run_real_stage_a(
        rows,
        addendum,
        source_sha256="a" * 64,
        runtime={"score": score, "resources": resources},
    )
    counts = artifact["resources"]["operation_counts"]
    assert counts["candidate_evaluations"] > 0
    assert artifact["resources"]["cleanup"]["finalizer_rejection_code"] == "finalizer_top_level_missing_fields"
    assert validate_stage_a(artifact, rows, addendum) == []


def test_stage_a_production_runtime_closure_publishes_live_resources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows, addendum, _ = _base()

    class FakeIntegration:
        def __init__(self, **_kwargs: Any) -> None:
            pass

    margin_calls = 0

    def fake_forward(*_args: Any, **kwargs: Any) -> tuple[Any, dict[int, np.ndarray]]:
        counters = kwargs.get("counters")
        if isinstance(counters, dict):
            counters["forwards"] += 1
            counters["captures"] += 1
            if kwargs.get("operation") == "patch":
                counters["patches"] += 1
                counters["hooks"] += 1
            elif kwargs.get("operation") == "control":
                counters["controls"] += 1
        return object(), {0: np.ones((1, 2, 2), dtype=np.float32)}

    def fake_margin(*_args: Any, **_kwargs: Any) -> float:
        nonlocal margin_calls
        value = (1.0, 0.0, 1.0)[margin_calls % 3]
        margin_calls += 1
        return value

    monkeypatch.setattr(real_runtime, "TransformerLMIntegration", FakeIntegration)
    monkeypatch.setattr(real_runtime, "_forward", fake_forward)
    monkeypatch.setattr(real_runtime, "_raw_hidden", lambda *_args, **_kwargs: np.ones((1, 2, 2), dtype=np.float32))
    monkeypatch.setattr(real_runtime, "_patch_positions", lambda *_args, **_kwargs: (0, 0))
    monkeypatch.setattr(real_runtime, "_margin", fake_margin)
    scorer, resources = real_runtime.build_stage_a_runtime(rows)
    artifact = run_real_stage_a(
        rows, addendum, source_sha256="a" * 64, runtime={"score": scorer, "resources": resources}
    )
    assert artifact["resources"]["operation_counts"]["candidate_evaluations"] > 0
    assert artifact["resources"]["cleanup"] == {"hook_count": 0, "completed": True}
    assert artifact["evidence_level"] in {"D0", "D1"}
    assert validate_stage_a(artifact, rows, addendum) == []


@pytest.mark.parametrize(
    ("label", "mutate", "expected"),
    [
        ("missing", lambda value: value.pop("stage"), "finalizer_top_level_missing_fields"),
        ("extra", lambda value: value.update({"unexpected": "secret"}), "finalizer_top_level_extra_fields"),
        ("identity", lambda value: value.update({"stage": "cleanup"}), "finalizer_identity_fields"),
        (
            "bool_counter",
            lambda value: value["operation_counts"].update({"captures": True}),
            "finalizer_operation_counts",
        ),
        (
            "numpy_counter",
            lambda value: value["operation_counts"].update({"captures": np.int64(1)}),
            "finalizer_operation_counts",
        ),
        (
            "float_peak",
            lambda value: value["resource_peak"].update({"peak_cpu_bytes": 1.0}),
            "finalizer_resource_peak_primitive_types",
        ),
        (
            "hook_counter_invariant",
            lambda value: value["hook"].update({"removed": 0}),
            "finalizer_cross_field_invariants",
        ),
        (
            "intervention_counter_invariant",
            lambda value: value["intervention"].update({"patch_calls": 0}),
            "finalizer_cross_field_invariants",
        ),
        (
            "cleanup_shape",
            lambda value: value.update({"cleanup": {"hook_count": 1, "completed": True}}),
            "finalizer_cleanup_fields",
        ),
        (
            "reserved_below_allocated",
            lambda value: value["resource_peak"].update({"peak_gpu_bytes": 3, "peak_gpu_reserved_bytes": 2}),
            "finalizer_resource_peak_cross_invariants",
        ),
        (
            "unknown_query_failure",
            lambda value: value["resource_peak"].update(
                {"measurement_status": "unavailable", "measurement_reason": "query secret"}
            ),
            "finalizer_resource_peak_status_reason",
        ),
    ],
)
def test_finalizer_checker_is_single_source_of_truth(label: str, mutate: Any, expected: str) -> None:
    del label
    value = _real_resources_for_complete_selection()
    mutate(value)
    code = stage_a_module._finalizer_rejection_code(value)
    assert code == expected
    assert stage_a_module._valid_finalizer_resources(value) is False
    assert code in stage_a_module.FINALIZER_REJECTION_CODES
    assert "secret" not in json.dumps({"finalizer_rejection_code": code})


def test_finalizer_checker_accepts_sanitized_query_failure() -> None:
    value = _real_resources_for_complete_selection()
    peak = value["resource_peak"]
    peak.update(
        {
            "peak_gpu_bytes": 0,
            "peak_gpu_reserved_bytes": 0,
            "measurement_status": "unavailable",
            "measurement_reason": "cuda_peak_query_failed",
            "elapsed_seconds": 1.0,
            "gpu_source": "unavailable",
            "gpu_reserved_source": "unavailable",
            "gpu_device": "unavailable",
        }
    )
    assert stage_a_module._finalizer_rejection_code(value) is None
    assert stage_a_module._valid_finalizer_resources(value) is True


@pytest.mark.parametrize(
    ("label", "mutate", "expected"),
    [
        ("peak_shape", lambda value: value["resource_peak"].pop("unit"), "finalizer_resource_peak_shape"),
        (
            "peak_bool",
            lambda value: value["resource_peak"].update({"peak_cpu_bytes": True}),
            "finalizer_resource_peak_primitive_types",
        ),
        (
            "peak_numpy_integer",
            lambda value: value["resource_peak"].update({"peak_cpu_bytes": np.int64(1)}),
            "finalizer_resource_peak_primitive_types",
        ),
        (
            "peak_float_integer",
            lambda value: value["resource_peak"].update({"peak_cpu_bytes": 1.0}),
            "finalizer_resource_peak_primitive_types",
        ),
        (
            "peak_nan_elapsed",
            lambda value: value["resource_peak"].update({"elapsed_seconds": float("nan")}),
            "finalizer_resource_peak_elapsed",
        ),
        (
            "peak_negative_elapsed",
            lambda value: value["resource_peak"].update({"elapsed_seconds": -1.0}),
            "finalizer_resource_peak_elapsed",
        ),
        (
            "peak_source",
            lambda value: value["resource_peak"].update({"gpu_source": "secret-source"}),
            "finalizer_resource_peak_source_device",
        ),
        (
            "peak_device",
            lambda value: value["resource_peak"].update({"gpu_device": "device-secret"}),
            "finalizer_resource_peak_source_device",
        ),
        (
            "peak_status",
            lambda value: value["resource_peak"].update({"measurement_status": "unknown"}),
            "finalizer_resource_peak_status_reason",
        ),
        (
            "peak_reason",
            lambda value: value["resource_peak"].update({"measurement_reason": "query-secret"}),
            "finalizer_resource_peak_status_reason",
        ),
        (
            "peak_availability",
            lambda value: value["resource_peak"].update({"gpu_source": "unavailable"}),
            "finalizer_resource_peak_availability_provenance",
        ),
        (
            "peak_budget_type",
            lambda value: value["resource_peak"].update({"budget_gpu_bytes": 6_000_000_001}),
            "finalizer_resource_peak_budget_fields",
        ),
        (
            "peak_budget_exceeded",
            lambda value: value["resource_peak"].update({"peak_gpu_bytes": 5_999_999_999, "budget_gpu_bytes": 1}),
            "finalizer_resource_peak_gpu_allocated_peak",
        ),
        (
            "peak_cross_invariant",
            lambda value: value["resource_peak"].update({"peak_gpu_bytes": 2, "peak_gpu_reserved_bytes": 1}),
            "finalizer_resource_peak_cross_invariants",
        ),
    ],
)
def test_finalizer_resource_peak_subcodes_are_ordered_and_non_sensitive(label: str, mutate: Any, expected: str) -> None:
    del label
    value = _real_resources_for_complete_selection()
    mutate(value)
    code = stage_a_module._finalizer_rejection_code(value)
    assert code == expected
    assert code in stage_a_module.FINALIZER_REJECTION_CODES
    assert "secret" not in json.dumps({"finalizer_rejection_code": code})


def test_resource_peak_device_contract_rejects_noncanonical_text_independently() -> None:
    resources = _real_resources_for_complete_selection()
    resources["resource_peak"]["gpu_device"] = "cuda:GPU"
    assert stage_a_module._finalizer_rejection_code(resources) == "finalizer_resource_peak_source_device"
    errors = validate_common.real_resources(resources)
    assert errors
    assert all("cuda:GPU" not in error for error in errors)


@pytest.mark.parametrize(
    ("label", "reason", "mutate", "valid"),
    [
        (
            "rss_fully_unavailable_with_attempted_device",
            "rss_unavailable",
            lambda peak: peak.update(
                {
                    "peak_cpu_bytes": 0,
                    "peak_gpu_bytes": 0,
                    "peak_gpu_reserved_bytes": 0,
                    "cpu_source": "unavailable",
                    "gpu_source": "unavailable",
                    "gpu_reserved_source": "unavailable",
                    "gpu_device": "cuda:0",
                    "measurement_status": "unavailable",
                    "measurement_reason": "rss_unavailable",
                    "elapsed_seconds": None,
                }
            ),
            False,
        ),
        (
            "rss_measured_gpu",
            "rss_unavailable",
            lambda peak: peak.update(
                {
                    "peak_cpu_bytes": 0,
                    "cpu_source": "unavailable",
                    "measurement_status": "unavailable",
                    "measurement_reason": "rss_unavailable",
                    "elapsed_seconds": None,
                }
            ),
            True,
        ),
        (
            "clock_measured_gpu",
            "clock_invalid",
            lambda peak: peak.update(
                {"measurement_status": "unavailable", "measurement_reason": "clock_invalid", "elapsed_seconds": None}
            ),
            True,
        ),
        (
            "clock_fully_unavailable",
            "clock_invalid",
            lambda peak: peak.update(
                {
                    "peak_cpu_bytes": 0,
                    "peak_gpu_bytes": 0,
                    "peak_gpu_reserved_bytes": 0,
                    "cpu_source": "unavailable",
                    "gpu_source": "unavailable",
                    "gpu_reserved_source": "unavailable",
                    "gpu_device": "unavailable",
                    "measurement_status": "unavailable",
                    "measurement_reason": "clock_invalid",
                    "elapsed_seconds": None,
                }
            ),
            True,
        ),
    ],
)
def test_producer_and_independent_validator_agree_on_unavailable_gpu_contract(
    label: str, reason: str, mutate: Any, valid: bool
) -> None:
    del label, reason
    resources = _real_resources_for_complete_selection()
    mutate(resources["resource_peak"])
    producer_valid = stage_a_module._finalizer_rejection_code(resources) is None
    public_valid = validate_common.real_resources(resources) == []
    assert producer_valid is valid
    assert public_valid is valid
    if not valid:
        assert stage_a_module._finalizer_rejection_code(resources) == "finalizer_resource_peak_availability_provenance"


@pytest.mark.parametrize("reason", sorted(stage_a_module._FINALIZER_MEASUREMENT_REASONS))
def test_finalizer_checker_accepts_each_allowlisted_unavailable_reason(reason: str) -> None:
    value = _real_resources_for_complete_selection()
    peak = value["resource_peak"]
    peak.update({"measurement_status": "unavailable", "measurement_reason": reason})
    if reason in {"cuda_unavailable", "cuda_reset_failed", "cuda_peak_query_failed", "cuda_zero_peak"}:
        peak.update(
            {
                "peak_gpu_bytes": 0,
                "peak_gpu_reserved_bytes": 0,
                "gpu_source": "unavailable",
                "gpu_reserved_source": "unavailable",
                "gpu_device": "unavailable",
            }
        )
    elif reason in {"tracker_unstarted", "resource_measurement_invalid"}:
        peak.update(
            {
                "peak_cpu_bytes": 0,
                "peak_gpu_bytes": 0,
                "peak_gpu_reserved_bytes": 0,
                "cpu_source": "unavailable",
                "gpu_source": "unavailable",
                "gpu_reserved_source": "unavailable",
                "gpu_device": "unavailable",
            }
        )
    elif reason == "rss_unavailable":
        peak.update({"peak_cpu_bytes": 0, "cpu_source": "unavailable"})
    assert stage_a_module._finalizer_rejection_code(value) is None
    assert stage_a_module._valid_finalizer_resources(value) is True


def test_resource_probe_uses_production_tracker_without_model_or_fixture(capsys: pytest.CaptureFixture[str]) -> None:
    class _ProbeTorch:
        cuda = _FakeCuda()

        @staticmethod
        def empty(shape: tuple[int, ...], *, device: str) -> object:
            assert shape == (1,)
            assert device == "cuda"
            return object()

    resources, rejection = resource_probe.run_resource_probe(_ProbeTorch)
    assert rejection is None
    assert resources["operation_counts"] == dict.fromkeys(real_runtime._COUNT_KEYS, 0)
    assert resources["resource_peak"]["measurement_status"] == "available"
    resource_probe.emit_resource_probe(resources, rejection)
    lines = capsys.readouterr().out.strip().splitlines()
    assert {line.split("=", 1)[0] for line in lines} == {
        "L049_V2_RESOURCE_PROBE_STATUS",
        "L049_V2_RESOURCE_PROBE_FINALIZER_CODE",
        "L049_V2_RESOURCE_PROBE_MEASUREMENT_STATUS",
        "L049_V2_RESOURCE_PROBE_MEASUREMENT_REASON",
        "L049_V2_RESOURCE_PROBE_CPU_PROVENANCE",
        "L049_V2_RESOURCE_PROBE_GPU_PROVENANCE",
        "L049_V2_RESOURCE_PROBE_DEVICE_CANONICAL",
        "L049_V2_RESOURCE_PROBE_CLEANUP",
    }
    output = "\n".join(lines)
    assert resource_probe.validate_resource_probe_output(output) == []
    assert "PASS" in output
    assert "openai-community" not in output
    assert "train" not in output.lower()
    assert "F:\\" not in output


def test_resource_probe_unavailable_path_is_sanitized(capsys: pytest.CaptureFixture[str]) -> None:
    class _UnavailableCuda:
        @staticmethod
        def is_available() -> bool:
            return False

    class _UnavailableTorch:
        cuda = _UnavailableCuda()

    resources, rejection = resource_probe.run_resource_probe(_UnavailableTorch)
    assert rejection is None
    assert resources["resource_peak"]["measurement_reason"] == "cuda_unavailable"
    resource_probe.emit_resource_probe(resources, rejection)
    output = capsys.readouterr().out
    assert resource_probe.validate_resource_probe_output(output) == []
    assert "L049_V2_RESOURCE_PROBE_STATUS=PASS" in output
    assert "L049_V2_RESOURCE_PROBE_MEASUREMENT_REASON=cuda_unavailable" in output
    assert "traceback" not in output.lower()


def test_resource_probe_emitter_allowlists_untrusted_marker_values(capsys: pytest.CaptureFixture[str]) -> None:
    class _UnavailableCuda:
        @staticmethod
        def is_available() -> bool:
            return False

    class _UnavailableTorch:
        cuda = _UnavailableCuda()

    resources, _ = resource_probe.run_resource_probe(_UnavailableTorch)
    resources["resource_peak"].update(
        {"measurement_status": {"secret": "status"}, "measurement_reason": "secret-reason"}
    )
    resource_probe.emit_resource_probe(resources, "secret-finalizer-code")
    output = capsys.readouterr().out
    assert "L049_V2_RESOURCE_PROBE_MEASUREMENT_STATUS=unknown" in output
    assert "L049_V2_RESOURCE_PROBE_MEASUREMENT_REASON=unknown" in output
    assert "L049_V2_RESOURCE_PROBE_FINALIZER_CODE=finalizer_not_mapping" in output
    assert resource_probe.validate_resource_probe_output(output) == []
    assert "secret" not in output


def test_resource_probe_validator_rejects_inconsistent_forged_markers() -> None:
    valid = [
        "L049_V2_RESOURCE_PROBE_STATUS=PASS",
        "L049_V2_RESOURCE_PROBE_FINALIZER_CODE=NONE",
        "L049_V2_RESOURCE_PROBE_MEASUREMENT_STATUS=unavailable",
        "L049_V2_RESOURCE_PROBE_MEASUREMENT_REASON=cuda_zero_peak",
        "L049_V2_RESOURCE_PROBE_CPU_PROVENANCE=true",
        "L049_V2_RESOURCE_PROBE_GPU_PROVENANCE=false",
        "L049_V2_RESOURCE_PROBE_DEVICE_CANONICAL=true",
        "L049_V2_RESOURCE_PROBE_CLEANUP=PASS",
    ]
    for index, replacement in (
        (0, "L049_V2_RESOURCE_PROBE_STATUS=FAIL"),
        (1, "L049_V2_RESOURCE_PROBE_FINALIZER_CODE=finalizer_resource_peak_shape"),
        (2, "L049_V2_RESOURCE_PROBE_MEASUREMENT_STATUS=unknown"),
        (5, "L049_V2_RESOURCE_PROBE_GPU_PROVENANCE=true"),
    ):
        forged = list(valid)
        forged[index] = replacement
        assert resource_probe.validate_resource_probe_output("\n".join(forged))

    available_inconsistent = list(valid)
    available_inconsistent[2] = "L049_V2_RESOURCE_PROBE_MEASUREMENT_STATUS=available"
    available_inconsistent[3] = "L049_V2_RESOURCE_PROBE_MEASUREMENT_REASON=none"
    available_inconsistent[4] = "L049_V2_RESOURCE_PROBE_CPU_PROVENANCE=false"
    assert resource_probe.validate_resource_probe_output("\n".join(available_inconsistent))

    unknown_failure = list(valid)
    unknown_failure[0] = "L049_V2_RESOURCE_PROBE_STATUS=FAIL"
    unknown_failure[1] = "L049_V2_RESOURCE_PROBE_FINALIZER_CODE=finalizer_resource_peak_status_reason"
    unknown_failure[2] = "L049_V2_RESOURCE_PROBE_MEASUREMENT_STATUS=unknown"
    unknown_failure[3] = "L049_V2_RESOURCE_PROBE_MEASUREMENT_REASON=unknown"
    unknown_failure[4] = "L049_V2_RESOURCE_PROBE_CPU_PROVENANCE=false"
    unknown_failure[6] = "L049_V2_RESOURCE_PROBE_DEVICE_CANONICAL=false"
    assert resource_probe.validate_resource_probe_output("\n".join(unknown_failure)) == []


def test_stage_a_late_failure_does_not_double_invoke_finalizer(monkeypatch: pytest.MonkeyPatch) -> None:
    rows, addendum, _ = _base()
    resources = _real_resources_for_complete_selection()
    calls = 0

    def finalize() -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return _real_resources_for_complete_selection()

    resources["finalize"] = finalize

    def late_failure(*_args: Any, resources: Mapping[str, Any], **_kwargs: Any) -> dict[str, Any]:
        callback = resources.get("finalize")
        assert callable(callback)
        callback()
        raise ValueError("late artifact failure")

    monkeypatch.setattr(stage_a_module, "build_stage_a_artifact", late_failure)
    artifact = run_real_stage_a(
        rows,
        addendum,
        source_sha256="a" * 64,
        runtime={"score": lambda *_args: 0.0, "resources": resources},
    )
    assert calls == 1
    assert artifact["failure_kind"] == "runtime_exception"
    assert validate_stage_a(artifact, rows, addendum) == []


@pytest.mark.parametrize("forged_remaining", [0, 999_999])
def test_stage_a_rehashed_cleanup_remaining_hooks_obeys_live_counter_projection(forged_remaining: int) -> None:
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
    if forged_remaining == 0:
        assert errors == []
    else:
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


def _stage_b_cli_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path, dict[str, Any]]:
    candidate, holdout, seed, addendum, _observations = _synthetic_stage_b()
    holdout_path = tmp_path / "holdout.jsonl"
    seed_path = tmp_path / "holdout.seed"
    candidate_path = tmp_path / "candidate.json"
    addendum_path = tmp_path / "addendum.json"
    holdout_path.write_bytes(canonical_fixture_bytes(holdout))
    seed_path.write_bytes(seed)
    candidate_path.write_bytes(canonical_json_bytes(candidate))
    addendum_path.write_bytes(canonical_json_bytes(addendum))
    return holdout_path, seed_path, candidate_path, addendum_path, candidate


def test_stage_a_cli_helper_import_failure_is_attempted_real_d0(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import torch

    import scripts.m14_l049_v2_stage_a as cli

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    original_import = builtins.__import__

    def blocked_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "scripts._m14_l049_v2_real_runtime":
            raise ImportError("runtime helper secret must never escape")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    output = tmp_path / "stage-a.json"
    cli.main(
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
    artifact = json.loads(output.read_bytes())
    assert artifact["status"] == "stage_a_failed"
    assert artifact["evidence_level"] == "D0"
    assert artifact["resources"]["execution_attempted"] is True
    assert artifact["resources"]["execution_backend"] == "cuda"
    assert "runtime helper secret" not in json.dumps(artifact)


def test_stage_b_cli_helper_import_failure_is_attempted_real_d0(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import torch

    import scripts.m14_l049_v2_stage_b as cli

    holdout_path, seed_path, candidate_path, addendum_path, _candidate = _stage_b_cli_inputs(tmp_path)
    monkeypatch.setattr(cli, "V2_ADDENDUM_PATH", addendum_path)
    monkeypatch.setattr(cli, "validate_stage_b", lambda *_args: [])
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    original_import = builtins.__import__

    def blocked_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "scripts._m14_l049_v2_real_runtime":
            raise ImportError("Stage B helper secret must never escape")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    output = tmp_path / "stage-b.json"
    cli.main(
        [
            "--holdout-fixture",
            str(holdout_path),
            "--holdout-seed",
            str(seed_path),
            "--candidate-manifest",
            str(candidate_path),
            "--output",
            str(output),
            "--source-commit-sha",
            SOURCE_COMMIT,
            "--run-real",
        ]
    )
    artifact = json.loads(output.read_bytes())
    assert artifact["status"] == "stage_b_failed"
    assert artifact["evidence_level"] == "D0"
    assert artifact["resources"]["execution_attempted"] is True
    assert artifact["resources"]["execution_backend"] == "cuda"
    assert "Stage B helper secret" not in json.dumps(artifact)


def test_stage_b_cli_evaluation_failure_is_attempted_real_d0(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import torch

    import scripts._m14_l049_v2_real_runtime as runtime_module
    import scripts._m14_l049_v2_stage_b as stage_b_module
    import scripts.m14_l049_v2_stage_b as cli

    holdout_path, seed_path, candidate_path, addendum_path, _candidate = _stage_b_cli_inputs(tmp_path)
    monkeypatch.setattr(cli, "V2_ADDENDUM_PATH", addendum_path)
    monkeypatch.setattr(cli, "validate_stage_b", lambda *_args: [])
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    resources = attempted_runtime_resources()
    monkeypatch.setattr(runtime_module, "build_stage_b_runtime", lambda *_args: ({}, resources))

    def fail_evaluation(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise IndexError("Stage B evaluation prompt secret must never escape")

    monkeypatch.setattr(stage_b_module, "evaluate_stage_b", fail_evaluation)
    output = tmp_path / "stage-b-eval-failure.json"
    cli.main(
        [
            "--holdout-fixture",
            str(holdout_path),
            "--holdout-seed",
            str(seed_path),
            "--candidate-manifest",
            str(candidate_path),
            "--output",
            str(output),
            "--source-commit-sha",
            SOURCE_COMMIT,
            "--run-real",
        ]
    )
    artifact = json.loads(output.read_bytes())
    assert artifact["status"] == "stage_b_failed"
    assert artifact["failure"] == {"exception_type": "IndexError"}

    assert artifact["seed_summaries"] == []
    assert "evaluation prompt secret" not in json.dumps(artifact)


def test_stage_b_validation_rejection_fallback_is_checked_by_real_validator() -> None:
    candidate, holdout, seed, addendum, _observations = _synthetic_stage_b()
    resources = attempted_runtime_resources()
    resources["resource_peak"]["measurement_reason"] = "Stage B resource secret must never escape"
    resources["untrusted_extra"] = {"nested_secret": "must never escape"}
    resources["cleanup"]["nested_secret"] = "must never escape"
    artifact = build_stage_b_validation_rejected_artifact(
        holdout,
        candidate,
        addendum,
        seed,
        source_sha256=str(candidate["source_sha256"]),
        resources=resources,
        validation_codes=["validation_rejected_unavailable_resource"],
        cli_sha256=top_level_cli_sha256("stage_b_holdout_evaluation"),
    )
    assert artifact["failure_kind"] == "validation_rejected"
    assert artifact["evaluation_complete"] is False
    assert artifact["failure"] == {"validation_codes": ["validation_rejected_unavailable_resource"]}
    assert (
        validate_stage_b_impl(
            artifact,
            holdout,
            seed,
            candidate,
            addendum,
            read_rows(TRAIN_PATH)[1],
            policy=CommitmentPolicy.from_addendum(addendum),
        )
        == []
    )
    assert "resource secret" not in json.dumps(artifact)
    assert "must never escape" not in json.dumps(artifact)
    assert "untrusted_extra" not in json.dumps(artifact)


def test_stage_b_success_resource_projection_drops_untrusted_values() -> None:
    candidate, holdout, seed, addendum, observations = _synthetic_stage_b()
    resources = _real_resources_for_complete_selection()
    resources["network"] = "legacy network secret"
    resources["resource_peak"]["nested_sentinel"] = {"secret": "must not escape"}
    artifact = evaluate_stage_b(holdout, observations, candidate, addendum, seed, resources=resources)
    assert set(artifact["resources"]) == {
        "stage",
        "execution_attempted",
        "execution_backend",
        "model",
        "model_revision",
        "integration",
        "model_adapter",
        "device",
        "backend",
        "dtype",
        "hook",
        "intervention",
        "operation_counts",
        "cleanup",
        "resource_peak",
        "no_mutation",
    }
    serialized = json.dumps(artifact)
    assert "legacy network secret" not in serialized
    assert "must not escape" not in serialized


def test_stage_b_runtime_uses_shared_tracker_envelope_and_idempotent_finish(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate, holdout, seed, addendum, _observations = _synthetic_stage_b()
    finish_calls: list[int] = []

    class FakeTracker:
        def __init__(self) -> None:
            self._finished = False

        def start(self) -> None:
            pass

        def finish(self) -> None:
            if not self._finished:
                self._finished = True
                finish_calls.append(1)

        def resource_peak(self) -> dict[str, Any]:
            return copy.deepcopy(_real_resources_for_complete_selection()["resource_peak"])

    class FakeIntegration:
        def __init__(self, **_kwargs: Any) -> None:
            pass

    def fake_forward(*_args: Any, **kwargs: Any) -> tuple[Any, dict[int, np.ndarray]]:
        counters = kwargs.get("counters")
        if isinstance(counters, dict):
            counters["forwards"] += 1
            counters["captures"] += 1
            if kwargs.get("operation") == "patch":
                counters["patches"] += 1
                counters["hooks"] += 1
            elif kwargs.get("operation") == "control":
                counters["controls"] += 1
        return object(), {0: np.ones((1, 2, 2), dtype=np.float32)}

    monkeypatch.setattr(real_runtime, "ResourceTracker", FakeTracker)
    monkeypatch.setattr(real_runtime, "TransformerLMIntegration", FakeIntegration)
    monkeypatch.setattr(real_runtime, "_forward", fake_forward)
    monkeypatch.setattr(real_runtime, "_raw_hidden", lambda *_args, **_kwargs: np.ones((1, 2, 2), dtype=np.float32))
    monkeypatch.setattr(real_runtime, "_patch_positions", lambda *_args, **_kwargs: (0, 0))
    margin_calls = 0

    def fake_margin(*_args: Any, **_kwargs: Any) -> float:
        nonlocal margin_calls
        value = (1.0, 0.0, 0.5)[margin_calls % 3]
        margin_calls += 1
        return value

    monkeypatch.setattr(real_runtime, "_margin", fake_margin)

    observations, resources = real_runtime.build_stage_b_runtime(
        holdout, {"selection": {"consensus_candidate": {"layer": 0, "offset": 0}}}
    )
    assert len(finish_calls) == 1
    assert resources["operation_counts"]["candidate_evaluations"] == 120
    assert resources["operation_counts"]["controls"] == 480
    artifact = evaluate_stage_b(holdout, observations, candidate, addendum, seed, resources=resources)
    assert (
        validate_stage_b_impl(
            artifact,
            holdout,
            seed,
            candidate,
            addendum,
            read_rows(TRAIN_PATH)[1],
            policy=CommitmentPolicy.from_addendum(addendum),
        )
        == []
    )
    assert artifact["resources"]["operation_counts"] == resources["operation_counts"]


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
        build_legacy_promotion_record(
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
    assert validate_legacy_promotion_record(
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
    for malformed in (None, []):
        assert validate_legacy_promotion_record(
            cast(Any, malformed),
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
        ) == (
            ["v2 promotion malformed input"] if not isinstance(malformed, Mapping) else ["v2 promotion malformed input"]
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
                "peak_gpu_reserved_bytes": 1,
                "unit": "bytes",
                "budget_cpu_bytes": 2,
                "budget_gpu_bytes": 2,
                "measurement_status": "available",
                "measurement_reason": None,
                "elapsed_seconds": 1.0,
                "elapsed_source": "time.perf_counter",
                "cpu_source": "resource.ru_maxrss_linux_kib",
                "gpu_source": "torch.cuda.max_memory_allocated",
                "gpu_reserved_source": "torch.cuda.max_memory_reserved",
                "gpu_device": "cuda:0",
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
            "peak_gpu_reserved_bytes": 1,
            "unit": "bytes",
            "budget_cpu_bytes": 2,
            "budget_gpu_bytes": 2,
            "measurement_status": "available",
            "measurement_reason": None,
            "elapsed_seconds": 1.0,
            "elapsed_source": "time.perf_counter",
            "cpu_source": "resource.ru_maxrss_linux_kib",
            "gpu_source": "torch.cuda.max_memory_allocated",
            "gpu_reserved_source": "torch.cuda.max_memory_reserved",
            "gpu_device": "cuda:0",
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
    record = build_legacy_promotion_record(
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
    assert validate_legacy_promotion_record(
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


def _real_promotion_fixture() -> dict[str, Any]:
    """Load retained D1/D2 evidence without creating a promotion artifact."""
    d1 = json.loads(CURRENT_D1_ASSESSMENT.read_bytes())
    d2 = json.loads(CURRENT_D2_ASSESSMENT.read_bytes())
    provisioning = json.loads(STAGE_B_PROVISIONING_ASSESSMENT.read_bytes())
    d1_audit = json.loads(CURRENT_D1_AUDIT.read_bytes())
    d2_audit = json.loads(CURRENT_D2_AUDIT.read_bytes())
    partial_path = ROOT / d2["evidence"]["triad"]["partial"]["path"]
    stage_b = json.loads(partial_path.read_bytes())["artifact"]
    candidate = json.loads(CURRENT_D1_CANDIDATE.read_bytes())
    addendum = json.loads(V2_ADDENDUM_PATH.read_bytes())
    paths = canonical_stage_b_paths(ROOT)
    return {
        "d1": d1,
        "d2": d2,
        "provisioning": provisioning,
        "d1_audit": d1_audit,
        "d2_audit": d2_audit,
        "stage_b": stage_b,
        "candidate": candidate,
        "addendum": addendum,
        "train": read_rows(TRAIN_PATH)[1],
        "holdout": read_rows(paths["holdout"])[1],
        "seed": paths["seed"].read_bytes(),
        "policy": CommitmentPolicy.from_addendum(addendum),
    }


def _real_promotion_policy(fixture: dict[str, Any]) -> RealPromotionPolicy:
    d1, d2, provisioning = fixture["d1"], fixture["d2"], fixture["provisioning"]
    d1_evidence = d1["evidence"]
    d2_evidence, d2_inputs = d2["evidence"], d2["inputs"]
    d2_audit = fixture["d2_audit"]

    def commitment(item: Mapping[str, Any]) -> RealEvidenceCommitment:
        return RealEvidenceCommitment(str(item["path"]), int(item["bytes"]), str(item["sha256"]))

    return RealPromotionPolicy(
        source_commit_sha=d2["source"]["commit_sha"],
        source_tree_algorithm="sha1",
        source_tree_oid=d2["source"]["tree_sha256"],
        d1_assessment=RealEvidenceCommitment(
            CURRENT_D1_ASSESSMENT.relative_to(ROOT).as_posix(),
            CURRENT_D1_ASSESSMENT.stat().st_size,
            hashlib.sha256(CURRENT_D1_ASSESSMENT.read_bytes()).hexdigest(),
        ),
        d1_assessment_canonical_sha256=d1["sidecar_sha256"],
        d1_audit=commitment(d1_evidence["audit"]),
        d1_candidate=commitment(d1_evidence["candidate"]),
        d1_source_commit_sha=d1["source"]["commit_sha"],
        d1_source_tree_algorithm="sha1",
        d1_source_tree_oid=d1["source"]["tree_sha256"],
        d1_pending_sidecar_sha256=d1["retention"]["previous_pending_sidecar_sha256"],
        d1_pending_audit_sha256="0c81ddedac08d2747d20982f4f2e221183ed9e380504917550b6cdfd680f9d7c",
        d1_pending_audit_bytes=3243,
        provisioning_source_commit_sha=provisioning["source"]["commit_sha"],
        provisioning_source_tree_algorithm="sha1",
        provisioning_source_tree_oid=provisioning["source"]["tree_sha256"],
        d2_assessment=RealEvidenceCommitment(
            CURRENT_D2_ASSESSMENT.relative_to(ROOT).as_posix(),
            CURRENT_D2_ASSESSMENT.stat().st_size,
            hashlib.sha256(CURRENT_D2_ASSESSMENT.read_bytes()).hexdigest(),
        ),
        d2_assessment_canonical_sha256=d2["sidecar_sha256"],
        d2_audit=commitment(d2_evidence["audit"]),
        d2_pending_sidecar_sha256=d2["retention"]["previous_pending_sidecar_sha256"],
        d2_pending_audit_sha256=d2_evidence["audit"]["prior_pending_sha256"],
        d2_pending_audit_bytes=d2_evidence["audit"]["prior_pending_bytes"],
        provisioning_assessment=RealEvidenceCommitment(
            STAGE_B_PROVISIONING_ASSESSMENT.relative_to(ROOT).as_posix(),
            STAGE_B_PROVISIONING_ASSESSMENT.stat().st_size,
            hashlib.sha256(STAGE_B_PROVISIONING_ASSESSMENT.read_bytes()).hexdigest(),
        ),
        provisioning_assessment_canonical_sha256=provisioning["sidecar_sha256"],
        manifest=commitment(d2_inputs["manifest"]),
        holdout=RealEvidenceCommitment(
            d2_inputs["holdout"]["path"],
            d2_inputs["holdout"]["bytes"],
            hashlib.sha256(canonical_stage_b_paths(ROOT)["holdout"].read_bytes()).hexdigest(),
        ),
        seed=RealEvidenceCommitment(
            d2_inputs["seed"]["path"], d2_inputs["seed"]["bytes"], d2_inputs["seed"]["commitment_sha256"]
        ),
        candidate=commitment(d2_inputs["candidate"]),
        parent_plan_sha256=fixture["policy"].parent_plan_sha256,
        addendum_schema=fixture["addendum"]["schema_version"],
        candidate_artifact_sha256=fixture["candidate"]["artifact_sha256"],
        stage_b_artifact_sha256=fixture["stage_b"]["artifact_sha256"],
        stage_b_attestation_sha256=fixture["stage_b"]["attestation_sha256"],
        cli_sha256=fixture["stage_b"]["runtime_attestation"]["cli_sha256"],
        transport_payload_sha256=d2_audit["transport"]["payload_sha256"],
        transport_decode_sha256=d2_audit["transport"]["decode_sha256"],
        raw_capture=RealEvidenceCommitment(
            d2["evidence"]["raw_capture"]["path"],
            d2_audit["raw_capture"]["bytes"],
            d2_audit["raw_capture"]["sha256"],
        ),
        bundle=RealEvidenceCommitment("<bundle>", d2_audit["bundle"]["bytes"], d2_audit["bundle"]["sha256"]),
        bundle_members=tuple(
            RealEvidenceCommitment(name, item["bytes"], item["sha256"])
            for name, item in d2_audit["bundle"]["members"].items()
        ),
        triad=tuple(commitment(item) for item in d2_evidence["triad"].values()),
    )


def test_real_promotion_contract_accepts_official_finalized_evidence_chain() -> None:
    fixture = _real_promotion_fixture()
    real_policy = _real_promotion_policy(fixture)
    source = fixture["d2"]["source"]
    assert (
        _validate_real_sidecars(
            fixture["d1"],
            fixture["d2"],
            fixture["provisioning"],
            fixture["candidate"],
            repo_root=ROOT,
            expected_source_commit_sha=source["commit_sha"],
            expected_source_tree_algorithm="sha1",
            expected_source_tree_oid=source["tree_sha256"],
            policy=fixture["policy"],
            real_policy=real_policy,
        )[0]
        == []
    )
    assert (
        _validate_official_audit(
            fixture["d2_audit"],
            fixture["d2"],
            repo_root=ROOT,
            expected_source_commit_sha=source["commit_sha"],
            expected_use_case="L049V2StageB",
            real_policy=real_policy,
        )
        == []
    )
    assert (
        _validate_official_audit(
            fixture["d1_audit"],
            fixture["d1"],
            repo_root=ROOT,
            expected_source_commit_sha=fixture["d1"]["source"]["commit_sha"],
            expected_use_case="L049V2StageA",
            real_policy=real_policy,
        )
        == []
    )
    assert fixture["d2_audit"]["transport"]["payload_sha256"] != fixture["stage_b"]["artifact_sha256"]


@pytest.mark.parametrize(
    ("field", "value"),
    [("raw_status", "retained_pending_finalize"), ("source_sha", "0" * 40), ("mode", "legacy_guess")],
)
def test_real_promotion_official_audit_tamper_matrix_rejects(field: str, value: object) -> None:
    fixture = _real_promotion_fixture()
    real_policy = _real_promotion_policy(fixture)
    audit = copy.deepcopy(fixture["d2_audit"])
    audit[field] = value
    source = fixture["d2"]["source"]
    assert _validate_official_audit(
        audit,
        fixture["d2"],
        repo_root=ROOT,
        expected_source_commit_sha=source["commit_sha"],
        expected_use_case="L049V2StageB",
        real_policy=real_policy,
    )


def test_real_promotion_bundle_member_tamper_and_no_record_side_effect() -> None:
    fixture = _real_promotion_fixture()
    real_policy = _real_promotion_policy(fixture)
    audit = copy.deepcopy(fixture["d2_audit"])
    member_name = next(iter(audit["bundle"]["members"]))
    audit["bundle"]["members"][member_name]["sha256"] = "0" * 64
    source = fixture["d2"]["source"]
    assert _validate_official_audit(
        audit,
        fixture["d2"],
        repo_root=ROOT,
        expected_source_commit_sha=source["commit_sha"],
        expected_use_case="L049V2StageB",
        real_policy=real_policy,
    )
    assert not list(ROOT.glob("artifacts/m14/*d3-promotion-real-v2*"))


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("source", "tree_sha256", "f" * 40),
        ("assessment", "evidence_eligible", False),
        ("retention", "previous_pending_sidecar_sha256", "0" * 64),
    ],
)
def test_real_promotion_sidecar_tamper_matrix_rejects(section: str, field: str, value: object) -> None:
    fixture = _real_promotion_fixture()
    real_policy = _real_promotion_policy(fixture)
    d2 = copy.deepcopy(fixture["d2"])
    d2[section][field] = value
    source = fixture["d2"]["source"]
    errors, _ = _validate_real_sidecars(
        fixture["d1"],
        d2,
        fixture["provisioning"],
        fixture["candidate"],
        repo_root=ROOT,
        expected_source_commit_sha=source["commit_sha"],
        expected_source_tree_algorithm="sha1",
        expected_source_tree_oid=source["tree_sha256"],
        policy=fixture["policy"],
        real_policy=real_policy,
    )
    assert errors


def test_real_promotion_public_validator_is_independent_and_fail_closed() -> None:
    fixture = _real_promotion_fixture()
    real_policy = _real_promotion_policy(fixture)
    source = fixture["d2"]["source"]
    errors = validate_promotion_record(
        {},
        fixture["stage_b"],
        fixture["candidate"],
        fixture["addendum"],
        fixture["train"],
        fixture["holdout"],
        fixture["seed"],
        fixture["d2_audit"]["transport"],
        fixture["d2_audit"],
        d1_assessment=fixture["d1"],
        d2_assessment=fixture["d2"],
        provisioning_assessment=fixture["provisioning"],
        repo_root=ROOT,
        expected_source_commit_sha=source["commit_sha"],
        expected_source_tree_algorithm="sha1",
        expected_source_tree_oid=source["tree_sha256"],
        policy=fixture["policy"],
        real_policy=real_policy,
    )
    assert "real promotion record fields are invalid" in errors


def test_real_promotion_tree_commitment_is_checked_against_git_metadata() -> None:
    fixture = _real_promotion_fixture()
    source = fixture["d2"]["source"]
    assert _repository_tree_errors(ROOT, source["commit_sha"], "sha1", source["tree_sha256"]) == []
    assert _repository_tree_errors(ROOT, source["commit_sha"], "sha256", source["tree_sha256"])
    assert _repository_tree_errors(ROOT, source["commit_sha"], "sha1", "f" * 40)


@pytest.mark.parametrize("malformed", [None, [], {"schema_version": []}, {"schema_version": {}}])
def test_real_promotion_public_validator_rejects_malformed_top_level_without_leak(
    malformed: object,
) -> None:
    fixture = _real_promotion_fixture()
    real_policy = _real_promotion_policy(fixture)
    source = fixture["d2"]["source"]
    errors = validate_promotion_record(
        cast(Any, malformed),
        fixture["stage_b"],
        fixture["candidate"],
        fixture["addendum"],
        fixture["train"],
        fixture["holdout"],
        fixture["seed"],
        fixture["d2_audit"]["transport"],
        fixture["d2_audit"],
        d1_assessment=fixture["d1"],
        d2_assessment=fixture["d2"],
        provisioning_assessment=fixture["provisioning"],
        repo_root=ROOT,
        expected_source_commit_sha=source["commit_sha"],
        expected_source_tree_algorithm="sha1",
        expected_source_tree_oid=source["tree_sha256"],
        policy=fixture["policy"],
        real_policy=real_policy,
    )
    assert errors


def test_real_promotion_public_validator_rejects_malformed_nested_inputs_without_leak() -> None:
    fixture = _real_promotion_fixture()
    real_policy = _real_promotion_policy(fixture)
    source = fixture["d2"]["source"]

    def invoke(**overrides: Any) -> list[str]:
        values: dict[str, Any] = {
            "record": {},
            "stage_b": fixture["stage_b"],
            "candidate": fixture["candidate"],
            "addendum": fixture["addendum"],
            "train_rows": fixture["train"],
            "holdout_rows": fixture["holdout"],
            "holdout_seed": fixture["seed"],
            "transport": fixture["d2_audit"]["transport"],
            "retention_audit": fixture["d2_audit"],
            "d1_assessment": fixture["d1"],
            "d2_assessment": fixture["d2"],
            "provisioning_assessment": fixture["provisioning"],
            "repo_root": ROOT,
            "expected_source_commit_sha": source["commit_sha"],
            "expected_source_tree_algorithm": "sha1",
            "expected_source_tree_oid": source["tree_sha256"],
            "policy": fixture["policy"],
            "real_policy": real_policy,
        }
        values.update(overrides)
        return validate_promotion_record(**values)

    for key, bad in (
        ("stage_b", []),
        ("retention_audit", None),
        ("d1_assessment", {"evidence": []}),
        ("real_policy", []),
        ("train_rows", [[]]),
        ("holdout_seed", bytearray(b"seed")),
    ):
        errors = invoke(**{key: bad})
        assert errors == ["real promotion malformed input"]
        assert all("secret" not in error and "traceback" not in error for error in errors)


@pytest.mark.parametrize(
    ("stage", "section", "field", "value"),
    [
        ("d1", "retention", "standard_finalize", False),
        ("d1", "finalize_delete", "executed", "manual"),
        ("d1", "raw_capture", "present", True),
        ("d2", "retention", "raw_present", True),
        ("d2", "finalize_delete", "mode", "manual"),
        ("d2", "raw_capture", "written_before_parse", False),
    ],
)
def test_real_promotion_requires_exact_finalized_lifecycle(stage: str, section: str, field: str, value: object) -> None:
    fixture = _real_promotion_fixture()
    real_policy = _real_promotion_policy(fixture)
    d1 = copy.deepcopy(fixture["d1"])
    d2 = copy.deepcopy(fixture["d2"])
    target = d1 if stage == "d1" else d2
    if section == "raw_capture":
        target["evidence"][section][field] = value
    elif section == "finalize_delete":
        target["retention"][section][field] = value
    else:
        target[section][field] = value
    errors, _ = _validate_real_sidecars(
        d1,
        d2,
        fixture["provisioning"],
        fixture["candidate"],
        repo_root=ROOT,
        expected_source_commit_sha=fixture["d2"]["source"]["commit_sha"],
        expected_source_tree_algorithm="sha1",
        expected_source_tree_oid=fixture["d2"]["source"]["tree_sha256"],
        policy=fixture["policy"],
        real_policy=real_policy,
    )
    assert errors
    assert any("lifecycle" in error or "finalize-delete" in error or "raw-capture" in error for error in errors)


def test_real_promotion_policy_pin_cannot_be_replaced_by_self_consistency() -> None:
    fixture = _real_promotion_fixture()
    real_policy = _real_promotion_policy(fixture)
    forged = replace(real_policy, d2_pending_sidecar_sha256="0" * 64)
    errors, _ = _validate_real_sidecars(
        fixture["d1"],
        fixture["d2"],
        fixture["provisioning"],
        fixture["candidate"],
        repo_root=ROOT,
        expected_source_commit_sha=fixture["d2"]["source"]["commit_sha"],
        expected_source_tree_algorithm="sha1",
        expected_source_tree_oid=fixture["d2"]["source"]["tree_sha256"],
        policy=fixture["policy"],
        real_policy=forged,
    )
    assert errors


def test_real_promotion_policy_is_loaded_from_pinned_canonical_files() -> None:
    fixture = _real_promotion_fixture()
    assert load_real_promotion_policy(ROOT) == _real_promotion_policy(fixture)


def test_real_promotion_bound_file_read_rejects_untracked_file(tmp_path: Path) -> None:
    repo = tmp_path / "untracked-evidence-repo"
    repo.mkdir()
    subprocess.run(["git", "init", "--quiet", str(repo)], check=True, capture_output=True)
    evidence = repo / "evidence.json"
    evidence.write_text("{}\n", encoding="utf-8")
    assert _safe_bound_file(repo, "evidence.json") is None


@pytest.mark.parametrize(
    "path",
    [
        "artifacts/m14/./file",
        "artifacts/m14/../file",
        "artifacts/m14//file",
        "artifacts/m14///file",
        "artifacts/m14/file/",
        "artifacts\\m14\\file",
        "artifacts/m14\\file",
        "C:/repo/artifacts/m14/file",
        "C:repo/artifacts/m14/file",
        "C:\\repo\\artifacts\\m14\\file",
        "\\\\server\\share\\artifacts\\m14\\file",
        "//server/share/artifacts/m14/file",
        "/artifacts/m14/file",
    ],
)
def test_real_promotion_bound_path_rejects_noncanonical_raw_spellings(
    path: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject path tricks before tracked lookup or file reads can normalize them."""

    def fail_if_tracked_lookup_reached(*_args: object, **_kwargs: object) -> bool:
        raise AssertionError("tracked lookup must happen after raw path validation")

    monkeypatch.setattr(promotion, "_tracked_exact", fail_if_tracked_lookup_reached)
    assert _safe_bound_file(ROOT, path) is None


def test_real_promotion_bound_path_accepts_only_canonical_posix_syntax() -> None:
    resolved = _safe_bound_file(ROOT, "artifacts/m14/l04-l049-v2-train.jsonl")
    assert resolved is not None
    assert resolved[0] == TRAIN_PATH


def test_real_promotion_canonical_mapping_preserves_nested_json_types(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected: dict[str, Any] = {"nested": {"value": 656, "flag": True}}
    monkeypatch.setattr(promotion, "_canonical_json", lambda *_args: expected)
    assert _canonical_mapping_matches(ROOT, "candidate", expected)
    assert not _canonical_mapping_matches(ROOT, "candidate", {"nested": {"value": 656.0, "flag": True}})
    assert not _canonical_mapping_matches(ROOT, "candidate", {"nested": {"value": 656, "flag": 1}})


def test_real_promotion_canonical_rows_preserve_nested_json_types(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected: list[dict[str, Any]] = [{"nested": {"value": 656, "flag": True}}]
    monkeypatch.setattr(promotion, "_canonical_raw", lambda *_args: (ROOT / "rows.jsonl", b""))
    monkeypatch.setattr(promotion, "read_rows", lambda *_args: (b"", expected))
    assert _canonical_rows_matches(ROOT, "train", expected)
    assert not _canonical_rows_matches(ROOT, "train", [{"nested": {"value": 656.0, "flag": True}}])
    assert not _canonical_rows_matches(ROOT, "train", [{"nested": {"value": 656, "flag": 1}}])


def test_real_promotion_rejects_self_rehashed_canonical_mapping_mutations() -> None:
    fixture = _real_promotion_fixture()
    real_policy = _real_promotion_policy(fixture)
    d2 = copy.deepcopy(fixture["d2"])
    d2["assessment"]["attacker_extra"] = True
    d2["sidecar_sha256"] = canonical_digest(d2, "sidecar_sha256")
    errors, _ = _validate_real_sidecars(
        fixture["d1"],
        d2,
        fixture["provisioning"],
        fixture["candidate"],
        repo_root=ROOT,
        expected_source_commit_sha=fixture["d2"]["source"]["commit_sha"],
        expected_source_tree_algorithm="sha1",
        expected_source_tree_oid=fixture["d2"]["source"]["tree_sha256"],
        policy=fixture["policy"],
        real_policy=real_policy,
    )
    assert errors

    candidate = copy.deepcopy(fixture["candidate"])
    candidate["attacker_extra"] = True
    candidate["artifact_sha256"] = canonical_digest(candidate, "artifact_sha256")
    errors, _ = _validate_real_sidecars(
        fixture["d1"],
        fixture["d2"],
        fixture["provisioning"],
        candidate,
        repo_root=ROOT,
        expected_source_commit_sha=fixture["d2"]["source"]["commit_sha"],
        expected_source_tree_algorithm="sha1",
        expected_source_tree_oid=fixture["d2"]["source"]["tree_sha256"],
        policy=fixture["policy"],
        real_policy=real_policy,
    )
    assert errors


def test_real_promotion_rejects_stage_b_artifact_that_differs_from_partial() -> None:
    fixture = _real_promotion_fixture()
    real_policy = _real_promotion_policy(fixture)
    stage_b = copy.deepcopy(fixture["stage_b"])
    stage_b["attacker_extra"] = True
    source = fixture["d2"]["source"]
    errors = validate_promotion_record(
        {},
        stage_b,
        fixture["candidate"],
        fixture["addendum"],
        fixture["train"],
        fixture["holdout"],
        fixture["seed"],
        fixture["d2_audit"]["transport"],
        fixture["d2_audit"],
        d1_assessment=fixture["d1"],
        d2_assessment=fixture["d2"],
        provisioning_assessment=fixture["provisioning"],
        repo_root=ROOT,
        expected_source_commit_sha=source["commit_sha"],
        expected_source_tree_algorithm="sha1",
        expected_source_tree_oid=source["tree_sha256"],
        policy=fixture["policy"],
        real_policy=real_policy,
    )
    assert "real promotion Stage B artifact mapping is not canonical" in errors


def test_real_promotion_manual_valid_record_validates_without_builder_or_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    d3_outputs_before = tuple(sorted(ROOT.glob("artifacts/m14/*d3-promotion-real-v2*")))
    assert d3_outputs_before == ()

    def fail_if_builder_called(*_args: object, **_kwargs: object) -> object:
        pytest.fail("manual validator must not call the D3 builder")

    monkeypatch.setattr(promotion, "build_promotion_record", fail_if_builder_called)
    fixture = _real_promotion_fixture()
    real_policy = _real_promotion_policy(fixture)
    d1, d2, provisioning = fixture["d1"], fixture["d2"], fixture["provisioning"]
    d1e, d2e = d1["evidence"], d2["evidence"]
    audit = fixture["d2_audit"]
    paths = audit["final_payload"]["paths"]
    members = audit["bundle"]["members"]
    record: dict[str, Any] = {
        "schema_version": "m14-l04.9-v2-d3-promotion-real-v2",
        "stage": fixture["stage_b"]["stage"],
        "status": "accepted",
        "evidence_level": "D3",
        "evidence_eligible": True,
        "repository_promotion": True,
        "promotion_candidate": True,
        "stage_b_artifact_sha256": fixture["stage_b"]["artifact_sha256"],
        "stage_b_attestation_sha256": fixture["stage_b"]["attestation_sha256"],
        "candidate_artifact_sha256": fixture["candidate"]["artifact_sha256"],
        "candidate_file_sha256": d2["inputs"]["candidate"]["sha256"],
        "parent_plan_sha256": fixture["policy"].parent_plan_sha256,
        "addendum_schema": fixture["addendum"]["schema_version"],
        "source_commit_sha": d2["source"]["commit_sha"],
        "source_tree": {"algorithm": "sha1", "oid": d2["source"]["tree_sha256"]},
        "d1_source_commit_sha": d1["source"]["commit_sha"],
        "d1_source_tree": {"algorithm": "sha1", "oid": d1["source"]["tree_sha256"]},
        "provisioning_source_commit_sha": provisioning["source"]["commit_sha"],
        "provisioning_source_tree": {"algorithm": "sha1", "oid": provisioning["source"]["tree_sha256"]},
        "d1_pending_sidecar_sha256": d1["retention"]["previous_pending_sidecar_sha256"],
        "d1_pending_audit_sha256": real_policy.d1_pending_audit_sha256,
        "d1_pending_audit_bytes": real_policy.d1_pending_audit_bytes,
        "d2_pending_audit_sha256": d2e["audit"]["prior_pending_sha256"],
        "d2_pending_audit_bytes": d2e["audit"]["prior_pending_bytes"],
        "cli_sha256": fixture["stage_b"]["runtime_attestation"]["cli_sha256"],
        "transport_payload_sha256": audit["transport"]["payload_sha256"],
        "transport_decode_sha256": audit["transport"]["decode_sha256"],
        "transport_decode_match": fixture["d2_audit"]["transport"].get("decode_match", "PASS"),
        "bundle_bytes": audit["bundle"]["bytes"],
        "bundle_sha256": audit["bundle"]["sha256"],
        "bundle_member_sha256": {name: item["sha256"] for name, item in members.items()},
        "retention_audit_sha256": d2e["audit"]["sha256"],
        "retention_audit_schema": audit["schema_version"],
        "d1_assessment_sha256": d1["sidecar_sha256"],
        "d1_audit_sha256": d1e["audit"]["sha256"],
        "d1_candidate_file_sha256": d1e["candidate"]["sha256"],
        "d2_assessment_sha256": d2["sidecar_sha256"],
        "d2_audit_sha256": d2e["audit"]["sha256"],
        "provisioning_assessment_sha256": provisioning["sidecar_sha256"],
        "provisioning_manifest_sha256": provisioning["inputs"]["manifest"]["sha256"],
        "provisioning_holdout_sha256": provisioning["inputs"]["holdout"]["sha256"],
        "provisioning_seed_commitment_sha256": provisioning["inputs"]["seed"]["commitment_sha256"],
        "retained_member_sha256": {Path(item["path"]).name: item["sha256"] for item in paths.values()},
        "pending_retention_sidecar_sha256": d2["retention"]["previous_pending_sidecar_sha256"],
    }
    record["promotion_sha256"] = canonical_digest(record, "promotion_sha256")
    assert (
        validate_promotion_record(
            record,
            fixture["stage_b"],
            fixture["candidate"],
            fixture["addendum"],
            fixture["train"],
            fixture["holdout"],
            fixture["seed"],
            audit["transport"],
            audit,
            d1_assessment=d1,
            d2_assessment=d2,
            provisioning_assessment=provisioning,
            repo_root=ROOT,
            expected_source_commit_sha=d2["source"]["commit_sha"],
            expected_source_tree_algorithm="sha1",
            expected_source_tree_oid=d2["source"]["tree_sha256"],
            policy=fixture["policy"],
            real_policy=real_policy,
        )
        == []
    )
    assert tuple(sorted(ROOT.glob("artifacts/m14/*d3-promotion-real-v2*"))) == d3_outputs_before


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
            "peak_gpu_reserved_bytes": 1,
            "unit": "bytes",
            "budget_cpu_bytes": 2,
            "budget_gpu_bytes": 2,
            "measurement_status": "available",
            "measurement_reason": None,
            "elapsed_seconds": 1.0,
            "elapsed_source": "time.perf_counter",
            "cpu_source": "resource.ru_maxrss_linux_kib",
            "gpu_source": "torch.cuda.max_memory_allocated",
            "gpu_reserved_source": "torch.cuda.max_memory_reserved",
            "gpu_device": "cuda:0",
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

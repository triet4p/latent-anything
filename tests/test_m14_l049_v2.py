"""Offline contract tests for the preregistered L04.9 v2 stages."""

from __future__ import annotations

import builtins
import copy
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import scripts._m14_l049_v2_real_runtime as real_runtime
from scripts._m14_l049_v2_fixture import authoring_manifest_digest, generate_rows, read_rows
from scripts._m14_l049_v2_power import POWER_ASSUMPTIONS, frozen_power_result, power_digest, validate_power_result
from scripts._m14_l049_v2_promotion import build_promotion_record, validate_promotion_record
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


def test_current_incident_assessment_binds_retained_evidence_without_secrets() -> None:
    sidecar = json.loads(CURRENT_INCIDENT_ASSESSMENT.read_bytes())
    assert canonical_digest(sidecar, "sidecar_sha256") == sidecar["sidecar_sha256"]
    assert sidecar["source"] == {
        "commit_sha": "13bf46e7b748f6fa64bf5f44cd80c194d1ca889d",
        "use_case": "L049V2StageA",
        "exact_source_verified": True,
    }
    assert sidecar["status"] == "pending"
    assert sidecar["assessment"]["promotion"] is False
    assert sidecar["assessment"]["finalization"] is False
    assert sidecar["execution"] == {
        "completed_payloads": 1,
        "ssh_launches_reported": 2,
        "second_launch_aborted": True,
        "second_remote_reach": "uncertain",
        "invocation_invariant": "not_satisfied",
        "evidence_limitation": sidecar["execution"]["evidence_limitation"],
    }
    assert all(sidecar["evidence"][name]["path"].startswith("artifacts/") for name in ("raw_capture", "audit"))
    for item in (sidecar["evidence"]["raw_capture"], sidecar["evidence"]["audit"]):
        path = ROOT / item["path"]
        payload = path.read_bytes()
        assert len(payload) == item["bytes"]
        assert hashlib.sha256(payload).hexdigest() == item["sha256"]
    for item in sidecar["evidence"]["triad"].values():
        path = ROOT / item["path"]
        payload = path.read_bytes()
        assert len(payload) == item["bytes"]
        assert hashlib.sha256(payload).hexdigest() == item["sha256"]
    serialized = json.dumps(sidecar, sort_keys=True)
    assert all(secret not in serialized for secret in ("traceback", "holdout_plaintext", "BEGIN PRIVATE"))


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
    assert "exception" not in json.dumps(peak).lower()


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
        "finalizer_rejection_code": "finalizer_not_mapping"
        if finalizer_result is None
        else "finalizer_top_level_fields",
    }
    assert artifact["runtime_attestation"]["cleanup_hook_count"] == 1
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
    assert artifact["resources"]["cleanup"]["finalizer_rejection_code"] == "finalizer_top_level_fields"
    assert validate_stage_a(artifact, rows, addendum) == []


def test_stage_a_finalizer_rejection_uses_production_runtime_closure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows, addendum, _ = _base()

    class FakeIntegration:
        def __init__(self, **_kwargs: Any) -> None:
            pass

    margin_calls = 0

    def fake_forward(*_args: Any, **_kwargs: Any) -> tuple[Any, dict[int, np.ndarray]]:
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
    resources["finalize"] = lambda: {"stage": "cleanup"}

    artifact = run_real_stage_a(
        rows, addendum, source_sha256="a" * 64, runtime={"score": scorer, "resources": resources}
    )
    assert artifact["resources"]["operation_counts"]["candidate_evaluations"] > 0
    assert artifact["resources"]["cleanup"]["reason"] == "finalizer_invalid_result"
    assert artifact["resources"]["cleanup"]["finalizer_rejection_code"] == "finalizer_top_level_fields"
    assert artifact["evidence_level"] == "D0"
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

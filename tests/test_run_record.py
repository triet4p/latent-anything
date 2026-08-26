"""Offline contract tests for Sprint 62 run evidence."""

from __future__ import annotations

import hashlib
import inspect
import json
import pickle
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import pytest

from latent_anything.integrations.lerobot import LeRobotEvaluationResult
from latent_anything.integrations.lerobot_recording import (
    record_lerobot_dataset_inspection,
    record_lerobot_evaluation,
    record_lerobot_intervention,
    record_lerobot_policy_capture,
    supported_capture_points,
)
from latent_anything.run_record import (
    DuplicateRunError,
    FileSystemRunRecorder,
    RunRecord,
    build_comparison_report,
)
from latent_anything.runtime import RuntimeProfiler


def _start(recorder: FileSystemRunRecorder, *, name: str = "fixture") -> RunRecord:
    return recorder.start(
        name,
        config={"strength": 0.0},
        code_version="test-code",
        framework_version="test-framework",
        model_revisions={"policy": "model@abc"},
        dataset_revisions={"dataset": "data@def"},
        seeds=(1, 2),
        environment={"device": "cpu"},
        theory_evidence_ids=("THY-T05-CAUSAL-INTERVENTION-VS-OBSERVATIONAL-STUDY",),
    )


def test_run_record_round_trip_contains_reproducibility_fields() -> None:
    record = RunRecord.create(
        "fixture",
        config={"alpha": 1},
        code_version="git:test",
        framework_version="0.1.0b1",
        model_revisions={"policy": "repo@rev"},
        dataset_revisions={"dataset": "repo@rev"},
        seeds=(7,),
        environment={"python": "3.12"},
        theory_evidence_ids=("THY-EXAMPLE",),
    )

    restored = RunRecord.from_dict(record.to_dict())

    assert restored == record
    assert restored.identity == record.identity
    assert restored.run_id == record.identity[:16]


def test_identity_excludes_lifecycle_timestamps_and_status() -> None:
    first = RunRecord.create("fixture", config={"x": 1})
    second = first.transition("completed", metrics={"score": 1.0})

    assert first.identity == second.identity
    assert first.run_id == second.run_id


def test_run_record_freezes_nested_inputs_and_keeps_serialized_identity_equivalent() -> None:
    limits = {"max_steps": 3}
    value_item = {"enabled": True}
    values: list[object] = [1, value_item]
    nested: dict[str, object] = {"limits": limits, "values": values}
    record = RunRecord.create("fixture", config=nested)
    identity = record.identity

    limits["max_steps"] = 99
    value_item["enabled"] = False

    assert record.identity == identity
    restored_limits = cast(Mapping[str, object], record.config["limits"])
    assert restored_limits["max_steps"] == 3
    restored = RunRecord.from_dict(record.to_dict())
    assert restored.identity == identity
    assert restored.to_dict() == record.to_dict()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), object()])
def test_run_record_rejects_non_finite_and_unsupported_inputs(value: object) -> None:
    with pytest.raises((TypeError, ValueError), match="finite|unsupported"):
        RunRecord.create("fixture", config={"value": value})


def test_filesystem_recorder_writes_atomic_content_addressed_artifacts_and_recovers(tmp_path: Path) -> None:
    recorder = FileSystemRunRecorder(tmp_path / "runs")
    record = _start(recorder)
    reference = recorder.add_artifact(record.run_id, b"hello", name="hello.txt", media_type="text/plain")
    recovered = recorder.recover_interrupted()

    assert reference.digest == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
    assert recorder.read_artifact(reference) == b"hello"
    assert recovered[0].status == "interrupted"
    payload = json.loads((tmp_path / "runs" / "runs" / f"{record.run_id}.json").read_text(encoding="utf-8"))
    assert payload["status"] == "interrupted"


def test_artifact_reference_cannot_escape_recorder_artifacts_directory(tmp_path: Path) -> None:
    recorder = FileSystemRunRecorder(tmp_path)
    record = _start(recorder)
    reference = recorder.add_artifact(record.run_id, b"hello", name="hello.txt")

    with pytest.raises(ValueError, match="relative_path"):
        type(reference)(
            name=reference.name,
            digest=reference.digest,
            size_bytes=reference.size_bytes,
            relative_path=f"artifacts/{reference.digest}/../../outside",
        )


def test_duplicate_identity_reuses_existing_run_and_conflict_is_rejected(tmp_path: Path) -> None:
    recorder = FileSystemRunRecorder(tmp_path)
    first = _start(recorder)
    second = _start(recorder)

    assert second.run_id == first.run_id
    conflicting = RunRecord.create("other", config={"x": 2})
    conflicting_payload = conflicting.to_dict()
    conflicting_payload["run_id"] = first.run_id
    (recorder.runs_dir / f"{first.run_id}.json").write_text(json.dumps(conflicting_payload), encoding="utf-8")
    with pytest.raises(DuplicateRunError):
        recorder.save(first)


def test_legacy_record_migrates_to_schema_v1() -> None:
    restored = RunRecord.from_dict({"id": "legacy-id", "run_name": "old", "config": {"seed": 4}, "metrics": {"x": 2}})

    assert restored.schema_version == 1
    assert restored.run_id == "legacy-id"
    assert restored.name == "old"
    assert restored.status == "completed"


def test_filesystem_recorder_loads_windows_written_schema_v1_artifact_reference(tmp_path: Path) -> None:
    recorder = FileSystemRunRecorder(tmp_path)
    record = _start(recorder).transition("completed")
    digest = "a" * 64
    payload = record.to_dict()
    payload["artifacts"] = [
        {
            "name": "legacy.bin",
            "digest": digest,
            "size_bytes": 3,
            "relative_path": f"artifacts\\{digest}",
            "media_type": "application/octet-stream",
        }
    ]
    (recorder.runs_dir / f"{record.run_id}.json").write_text(json.dumps(payload), encoding="utf-8")

    restored = recorder.get(record.run_id)

    assert restored.artifacts[0].relative_path == f"artifacts/{digest}"

    invalid_payload = dict(payload)
    invalid_payload["artifacts"] = [
        {
            "name": "legacy.bin",
            "digest": digest,
            "size_bytes": 3,
            "relative_path": f"artifacts\\{digest}\\extra",
            "media_type": "application/octet-stream",
        }
    ]
    with pytest.raises(ValueError, match="relative_path"):
        RunRecord.from_dict(invalid_payload)


def test_lerobot_record_helpers_attach_profile_theory_and_parent_metadata(tmp_path: Path) -> None:
    recorder = FileSystemRunRecorder(tmp_path)
    profiler = RuntimeProfiler()
    profiler.record("encode", 0.25, component="fixture")
    capture = record_lerobot_policy_capture(
        recorder,
        {"capture": "action_expert"},
        config={"strength": 0.0},
        model_revisions={"policy": "repo@rev"},
        environment={"device": "cpu"},
        code_version="git:capture",
        runtime_profile=profiler.snapshot(),
        theory_evidence_ids=("THY-T05",),
    )
    intervention = record_lerobot_intervention(
        recorder,
        {"action_change": 0.2},
        config={"strength": 1.0},
        model_revisions={"policy": "repo@rev"},
        environment={"device": "cpu"},
        code_version="git:intervention",
        parent_run_ids=(capture.run_id,),
    )
    evaluation = record_lerobot_evaluation(
        recorder,
        LeRobotEvaluationResult(episodes=2, success_rate=0.5, metrics={"mean_return": 1.25}),
        config={"condition": "targeted"},
        model_revisions={"policy": "repo@rev"},
        environment={"device": "cpu"},
        code_version="git:evaluation",
        parent_run_ids=(intervention.run_id,),
    )

    assert capture.runtime_profile["stage_totals"] == {"encode": 0.25}
    assert capture.theory_evidence_ids == ("THY-T05",)
    assert capture.environment == {"device": "cpu"}
    assert capture.code_version == "git:capture"
    assert intervention.parent_run_ids == (capture.run_id,)
    assert intervention.code_version == "git:intervention"
    assert evaluation.metrics["success_rate"] == 0.5
    assert evaluation.code_version == "git:evaluation"
    assert len(evaluation.artifacts) == 1


def test_lerobot_dataset_record_helper_preserves_environment_and_code_version(tmp_path: Path) -> None:
    recorder = FileSystemRunRecorder(tmp_path)
    record = record_lerobot_dataset_inspection(
        recorder,
        {"episodes": 2},
        config={"repo_id": "fixture/dataset"},
        dataset_revisions={"fixture/dataset": "rev"},
        environment={"device": "cpu"},
        code_version="git:dataset",
    )

    assert record.environment == {"device": "cpu"}
    assert record.code_version == "git:dataset"


def test_capture_points_cover_existing_policy_seams() -> None:
    points = supported_capture_points()

    assert {point.policy for point in points} == {"act", "diffusion", "smolvla"}
    assert any(point.supports_intervention for point in points)
    assert supported_capture_points("act")[0].location == "model.decoder"


def test_comparison_report_requires_two_runs_and_reports_deltas(tmp_path: Path) -> None:
    recorder = FileSystemRunRecorder(tmp_path)
    first = recorder.complete(_start(recorder, name="baseline").run_id, metrics={"success_rate": 0.5, "return": 1.0})
    second = recorder.complete(
        recorder.start("intervention", config={"strength": 1.0}, model_revisions={"policy": "model@abc"}).run_id,
        metrics={"success_rate": 0.75, "return": 1.5},
    )

    report = build_comparison_report((first, second))

    assert report.baseline_run_id == first.run_id
    assert report.metric_deltas[second.run_id] == {"success_rate": 0.25, "return": 0.5}
    with pytest.raises(ValueError, match="at least two"):
        build_comparison_report((first,))


def test_schema_v1_canonical_json_and_migration_digests_are_stable() -> None:
    payload = {
        "schema_version": 1,
        "run_id": "fixture-id",
        "identity": "",
        "name": "fixture",
        "status": "completed",
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:01:00+00:00",
        "config": {"z": [2, 1], "a": "stable"},
        "code_version": "git:test",
        "framework_version": "0.9",
        "model_revisions": {"m": "rev"},
        "dataset_revisions": {"d": "rev"},
        "seeds": [3, 7],
        "environment": {"device": "cpu"},
        "metrics": {"score": 1.25},
        "artifacts": [],
        "parent_run_ids": [],
        "child_run_ids": [],
        "runtime_profile": {},
        "theory_evidence_ids": ["THY-1"],
        "metadata": {"x": True},
        "error": None,
    }
    record = RunRecord.from_dict(payload)
    encoded = json.dumps(record.to_dict(), sort_keys=True, separators=(",", ":")).encode()
    assert hashlib.sha256(encoded).hexdigest() == "25a8bc21cf19a67ce9a553d469236e06f19aed0b96760f9b97e3d2ed3b3c4964"

    legacy = {
        "id": "legacy-id",
        "run_name": "old",
        "config": {"seed": 4},
        "metrics": {"x": 2},
        "created_at": "2026-01-01T00:00:00+00:00",
        "artifacts": [
            {
                "name": "x",
                "digest": "a" * 64,
                "size_bytes": 1,
                "relative_path": f"artifacts\\{'a' * 64}",
            }
        ],
    }
    from latent_anything.run_record import migrate_run_record

    migrated = migrate_run_record(legacy)
    migration_bytes = json.dumps(migrated, sort_keys=True, separators=(",", ":")).encode()
    assert hashlib.sha256(migration_bytes).hexdigest() == (
        "fa243e2d7ca35695c6d381940d844312fe8df8476c8d88ef4da5f063406f5083"
    )


def test_public_run_record_types_keep_signatures_and_pickle_identity() -> None:
    from latent_anything.run_record import ArtifactRef, FileSystemRunRecorder, RunComparisonReport

    assert ArtifactRef.__module__ == "latent_anything.run_record"
    assert RunRecord.__module__ == "latent_anything.run_record"
    assert FileSystemRunRecorder.__module__ == "latent_anything.run_record"
    assert RunComparisonReport.__module__ == "latent_anything.run_record"
    assert tuple(inspect.signature(RunRecord.create).parameters) == (
        "name",
        "config",
        "code_version",
        "framework_version",
        "model_revisions",
        "dataset_revisions",
        "seeds",
        "environment",
        "parent_run_ids",
        "runtime_profile",
        "theory_evidence_ids",
        "metadata",
        "status",
    )

    artifact = ArtifactRef(name="x", digest="a" * 64, size_bytes=1, relative_path=f"artifacts/{'a' * 64}")
    assert pickle.loads(pickle.dumps(artifact)) == artifact
    report = RunComparisonReport("fixture", "run", (), {})
    assert pickle.loads(pickle.dumps(report)) == report


def test_recorder_rejects_tampered_and_symlinked_artifacts(tmp_path: Path) -> None:
    recorder = FileSystemRunRecorder(tmp_path / "runs")
    record = _start(recorder)
    reference = recorder.add_artifact(record.run_id, b"hello", name="hello.txt")
    artifact_path = recorder.artifacts_dir / reference.digest
    artifact_path.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="digest mismatch"):
        recorder.read_artifact(reference)

    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"hello")
    artifact_path.unlink()
    try:
        artifact_path.symlink_to(outside)
    except OSError as error:
        pytest.skip(f"symlink creation unavailable: {error}")
    with pytest.raises(ValueError, match="outside"):
        recorder.read_artifact(reference)


def test_saved_record_is_readable_in_a_fresh_process(tmp_path: Path) -> None:
    recorder = FileSystemRunRecorder(tmp_path / "runs")
    record = _start(recorder)
    script = (
        "import sys; "
        "from latent_anything.run_record import FileSystemRunRecorder; "
        "print(FileSystemRunRecorder(sys.argv[1]).get(sys.argv[2]).identity)"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path / "runs"), record.run_id],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip() == record.identity

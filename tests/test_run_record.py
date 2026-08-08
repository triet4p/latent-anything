"""Offline contract tests for Sprint 62 run evidence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from latent_anything.integrations.lerobot import LeRobotEvaluationResult
from latent_anything.integrations.lerobot_recording import (
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


def test_lerobot_record_helpers_attach_profile_theory_and_parent_metadata(tmp_path: Path) -> None:
    recorder = FileSystemRunRecorder(tmp_path)
    profiler = RuntimeProfiler()
    profiler.record("encode", 0.25, component="fixture")
    capture = record_lerobot_policy_capture(
        recorder,
        {"capture": "action_expert"},
        config={"strength": 0.0},
        model_revisions={"policy": "repo@rev"},
        runtime_profile=profiler.snapshot(),
        theory_evidence_ids=("THY-T05",),
    )
    intervention = record_lerobot_intervention(
        recorder,
        {"action_change": 0.2},
        config={"strength": 1.0},
        model_revisions={"policy": "repo@rev"},
        parent_run_ids=(capture.run_id,),
    )
    evaluation = record_lerobot_evaluation(
        recorder,
        LeRobotEvaluationResult(episodes=2, success_rate=0.5, metrics={"mean_return": 1.25}),
        config={"condition": "targeted"},
        model_revisions={"policy": "repo@rev"},
        parent_run_ids=(intervention.run_id,),
    )

    assert capture.runtime_profile["stage_totals"] == {"encode": 0.25}
    assert capture.theory_evidence_ids == ("THY-T05",)
    assert intervention.parent_run_ids == (capture.run_id,)
    assert evaluation.metrics["success_rate"] == 0.5
    assert len(evaluation.artifacts) == 1


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

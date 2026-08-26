"""Contract and local-adapter tests for optional experiment recorders."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Literal

import pytest

from latent_anything.experiment_recorder import (
    ExperimentRecorder,
    ExperimentRun,
    LocalExperimentRecorder,
    RecorderArtifact,
    RecorderContractError,
    RecorderRunInfo,
    canonical_recorder_json,
    compute_recorder_identity,
    read_recorder_artifact,
    validate_recorder_artifact_name,
)


def test_recorder_public_import_and_schema_snapshot() -> None:
    assert RecorderArtifact.__module__ == "latent_anything.experiment_recorder"
    assert RecorderRunInfo.__module__ == "latent_anything.experiment_recorder"
    assert ExperimentRun.__module__ == "latent_anything.experiment_recorder"
    assert tuple(inspect.signature(LocalExperimentRecorder.start_run).parameters) == (
        "self",
        "name",
        "config",
        "tags",
        "parent_run_id",
        "resume_run_id",
        "code_version",
        "framework_version",
        "model_revisions",
        "dataset_revisions",
        "seeds",
        "environment",
        "metadata",
    )
    assert canonical_recorder_json({"z": [2, 1], "a": "stable"}) == b'{"a":"stable","z":[2,1]}'
    assert compute_recorder_identity(name="run", config={"b": 2, "a": 1}) == compute_recorder_identity(
        name="run", config={"a": 1, "b": 2}
    )


def test_local_recorder_contract_preserves_identity_artifact_and_parent_child(tmp_path: Path) -> None:
    recorder = LocalExperimentRecorder(tmp_path)
    assert isinstance(recorder, ExperimentRecorder)
    parent = recorder.start_run(
        "fixture",
        config={"alpha": 1},
        tags={"lane": "offline"},
        code_version="git:test",
        framework_version="test-framework",
        parent_run_id=None,
    )
    assert parent.info.identity == compute_recorder_identity(
        name="fixture",
        config={"alpha": 1},
        tags={"lane": "offline"},
        code_version="git:test",
        framework_version="test-framework",
    )

    parent.log_params({"batch": 4})
    parent.log_metrics({"loss": 1.0}, step=0)
    parent.set_tags({"condition": "baseline"})
    artifact = parent.log_artifact(b"portable-evidence", name="evidence.bin")
    child = parent.child("child", config={"alpha": 2})
    child.log_metrics({"loss": 0.5}, step=1)
    child.finish()
    parent.finish()

    restored = recorder.get_record(parent.info.run_id)
    assert restored.status == "completed"
    assert restored.metrics == {"loss": 1.0}
    assert restored.child_run_ids == (child.info.run_id,)
    assert restored.artifacts[-1].digest == artifact.digest
    assert recorder.read_artifact(restored.artifacts[-1]) == b"portable-evidence"
    child_record = recorder.get_record(child.info.run_id)
    assert child_record.parent_run_ids == (parent.info.run_id,)


def test_local_recorder_resumes_metric_history_and_rejects_invalid_steps(tmp_path: Path) -> None:
    recorder = LocalExperimentRecorder(tmp_path)
    run = recorder.start_run("resume", config={"seed": 3})
    run.log_metrics({"score": 1.0}, step=2)
    with pytest.raises(RecorderContractError, match="non-decreasing"):
        run.log_metrics({"score": 0.0}, step=1)

    resumed = recorder.start_run("resume", resume_run_id=run.info.run_id)
    resumed.log_metrics({"score": 0.5}, step=3)
    resumed.finish()
    assert recorder.get_record(run.info.run_id).metrics == {"score": 0.5}


_ResumeMismatchField = Literal[
    "name",
    "config",
    "tags",
    "parent_run_id",
    "code_version",
    "framework_version",
    "model_revisions",
    "dataset_revisions",
    "seeds",
    "environment",
    "metadata",
]


def _start_run_with_mismatch(recorder: LocalExperimentRecorder, run_id: str, field: _ResumeMismatchField) -> None:
    if field == "name":
        recorder.start_run("other", resume_run_id=run_id)
    elif field == "config":
        recorder.start_run("resume-matrix", resume_run_id=run_id, config={"seed": 4})
    elif field == "tags":
        recorder.start_run("resume-matrix", resume_run_id=run_id, tags={"lane": "online"})
    elif field == "parent_run_id":
        recorder.start_run("resume-matrix", resume_run_id=run_id, parent_run_id="other-parent")
    elif field == "code_version":
        recorder.start_run("resume-matrix", resume_run_id=run_id, code_version="other-code")
    elif field == "framework_version":
        recorder.start_run("resume-matrix", resume_run_id=run_id, framework_version="other-framework")
    elif field == "model_revisions":
        recorder.start_run("resume-matrix", resume_run_id=run_id, model_revisions={"model": "other"})
    elif field == "dataset_revisions":
        recorder.start_run("resume-matrix", resume_run_id=run_id, dataset_revisions={"dataset": "other"})
    elif field == "seeds":
        recorder.start_run("resume-matrix", resume_run_id=run_id, seeds=(8,))
    elif field == "environment":
        recorder.start_run("resume-matrix", resume_run_id=run_id, environment={"platform": "other"})
    else:
        recorder.start_run("resume-matrix", resume_run_id=run_id, metadata={"owner": "other"})


@pytest.mark.parametrize(
    "field",
    [
        "name",
        "config",
        "tags",
        "parent_run_id",
        "code_version",
        "framework_version",
        "model_revisions",
        "dataset_revisions",
        "seeds",
        "environment",
        "metadata",
    ],
)
def test_local_resume_rejects_each_explicit_identity_mismatch(tmp_path: Path, field: _ResumeMismatchField) -> None:
    recorder = LocalExperimentRecorder(tmp_path / field)
    run = recorder.start_run(
        "resume-matrix",
        config={"seed": 3},
        tags={"lane": "offline"},
        parent_run_id="parent",
        code_version="code",
        framework_version="framework",
        model_revisions={"model": "r1"},
        dataset_revisions={"dataset": "r1"},
        seeds=(7,),
        environment={"platform": "cpu"},
        metadata={"owner": "test"},
    )
    with pytest.raises(RecorderContractError, match="local resume"):
        _start_run_with_mismatch(recorder, run.info.run_id, field)


def test_local_recorder_rejects_mutable_parameter_changes_nonfinite_metrics_and_double_finish(tmp_path: Path) -> None:
    recorder = LocalExperimentRecorder(tmp_path)
    run = recorder.start_run("validation", config={"value": 1})
    with pytest.raises(RecorderContractError, match="cannot change"):
        run.log_params({"value": 2})
    with pytest.raises(RecorderContractError, match="finite"):
        run.log_metrics({"score": float("nan")}, step=0)
    run.finish()
    with pytest.raises(RecorderContractError, match="completed"):
        run.finish()


def test_local_recorder_rejects_unsafe_names_and_oversized_config(tmp_path: Path) -> None:
    recorder = LocalExperimentRecorder(tmp_path)
    with pytest.raises(RecorderContractError, match="path separators"):
        recorder.start_run("../escape")
    with pytest.raises(RecorderContractError, match="strings exceed"):
        recorder.start_run("large", config={"payload": "x" * 300_000})


@pytest.mark.parametrize(
    "name", ["../escape", "..\\escape", "C:/escape", "foo:bar", "/absolute", "a//b", "a/./b", "%2e%2e/x"]
)
def test_artifact_names_are_canonical_posix_relative_paths(name: str) -> None:
    with pytest.raises(RecorderContractError, match="canonical POSIX|safe relative"):
        validate_recorder_artifact_name(name)


def test_recorder_metadata_rejects_secret_like_nested_values_and_unbounded_shapes(tmp_path: Path) -> None:
    recorder = LocalExperimentRecorder(tmp_path)
    cases = [
        {"api_key": "not-recorded"},
        {"nested": {"password": "not-recorded"}},
        {"value": "Bearer abc"},
        {"nested": {"items": list(range(4097))}},
    ]
    for config in cases:
        with pytest.raises(RecorderContractError):
            recorder.start_run("safe", config=config)


def test_recorder_validation_normalizes_bad_mapping_and_metric_types(tmp_path: Path) -> None:
    recorder = LocalExperimentRecorder(tmp_path)
    with pytest.raises(RecorderContractError):
        recorder.start_run("bad-key", config={1: "value"})  # type: ignore[dict-item]
    run = recorder.start_run("bad-metric")
    with pytest.raises(RecorderContractError):
        run.log_metrics({"score": object()}, step=0)  # type: ignore[dict-item]


def test_artifact_reader_rejects_unbounded_memoryview_before_copy() -> None:
    payload = bytearray(16 * 1024 * 1024 + 1)
    with pytest.raises(RecorderContractError, match="exceeds"):
        read_recorder_artifact(memoryview(payload))


def test_artifact_reader_rejects_arbitrary_bytes_protocol() -> None:
    class Trap:
        def __bytes__(self) -> bytes:
            raise AssertionError("__bytes__ must not be called")

    with pytest.raises(RecorderContractError, match="must be bytes"):
        read_recorder_artifact(Trap())  # type: ignore[arg-type]


def test_artifact_reader_rejects_oversized_file_before_return(tmp_path: Path) -> None:
    path = tmp_path / "oversized.bin"
    with path.open("wb") as handle:
        handle.seek(16 * 1024 * 1024)
        handle.write(b"x")
    with pytest.raises(RecorderContractError, match="exceeds|bounded regular"):
        read_recorder_artifact(path)

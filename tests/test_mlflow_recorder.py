"""Offline contract tests for the optional MLflow adapter."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from latent_anything.experiment_recorder import RecorderContractError
from latent_anything.integrations.mlflow_recorder import MLflowRecorder


class FakeMLflow:
    def __init__(self) -> None:
        self.tracking_uri = ""
        self.experiment = ""
        self.runs: dict[str, SimpleNamespace] = {}
        self.calls: list[tuple[str, object, dict[str, object]]] = []
        self.next_id = 0
        self.active: SimpleNamespace | None = None

    def set_tracking_uri(self, value: str) -> None:
        self.tracking_uri = value

    def set_experiment(self, value: str) -> None:
        self.experiment = value

    def start_run(self, **kwargs: object) -> SimpleNamespace:
        run_id = str(kwargs.get("run_id", f"mlflow-{self.next_id}"))
        self.next_id += 1
        if run_id in self.runs:
            run = self.runs[run_id]
        else:
            tags = dict(cast(Mapping[str, str], kwargs.get("tags", {})))
            run = SimpleNamespace(info=SimpleNamespace(run_id=run_id), data=SimpleNamespace(tags=tags))
            self.runs[run_id] = run
        self.active = run
        self.calls.append(("start_run", run_id, kwargs))
        return run

    def log_params(self, values: dict[str, str]) -> None:
        self.calls.append(("log_params", values, {}))

    def log_metrics(self, values: dict[str, float], *, step: int) -> None:
        self.calls.append(("log_metrics", values, {"step": step}))

    def set_tags(self, values: dict[str, str]) -> None:
        self.calls.append(("set_tags", values, {}))
        if self.active is not None:
            self.active.data.tags.update(values)

    def set_tag(self, key: str, value: str) -> None:
        self.set_tags({key: value})

    def log_artifact(self, path: str) -> None:
        self.calls.append(("log_artifact", Path(path).read_bytes(), {}))

    def end_run(self, *, status: str) -> None:
        self.calls.append(("end_run", status, {}))
        self.active = None


def test_mlflow_local_adapter_maps_contract_and_parent_child(tmp_path: Path) -> None:
    sdk = FakeMLflow()
    recorder = MLflowRecorder(tmp_path / "mlruns", experiment_name="fixture", _sdk=sdk)
    parent = recorder.start_run("parent", config={"seed": 1}, tags={"lane": "offline"})
    parent.log_metrics({"score": 1.0}, step=0)
    with pytest.raises(RecorderContractError, match="non-decreasing"):
        parent.log_metrics({"score": 0.5}, step=-1)
    artifact = parent.log_artifact(b"hello", name="nested/evidence.bin", media_type="application/octet-stream")
    child = parent.child("child", config={"seed": 2})
    child.finish()
    parent.finish()

    assert artifact.size_bytes == 5
    assert artifact.uri == f"runs:/{parent.info.run_id}/nested/evidence.bin"
    start_call = next(call for call in sdk.calls if call[0] == "start_run")
    parent_kwargs = cast(dict[str, object], start_call[2])
    parent_tags = cast(dict[str, str], parent_kwargs["tags"])
    assert parent_tags["latent_anything.identity"] == parent.info.identity
    assert "nested" not in parent_kwargs
    child_start = [call for call in sdk.calls if call[0] == "start_run"][1]
    child_kwargs = cast(dict[str, object], child_start[2])
    assert child_kwargs["nested"] is True
    assert any(call[0] == "log_metrics" and call[2]["step"] == 0 for call in sdk.calls)
    assert any(call[0] == "log_artifact" and call[1] == b"hello" for call in sdk.calls)


def test_mlflow_resume_requires_identity_and_rejects_remote_uri(tmp_path: Path) -> None:
    sdk = FakeMLflow()
    recorder = MLflowRecorder(tmp_path / "mlruns", _sdk=sdk)
    run = recorder.start_run("resume", config={"x": 1})
    resumed = recorder.start_run("resume", config={"x": 1}, resume_run_id=run.info.run_id)
    resumed.log_metrics({"value": 2.0}, step=1)
    resumed.finish()
    assert resumed.info.status == "completed"
    with pytest.raises(RecorderContractError, match="identity mismatch"):
        recorder.start_run("resume", config={"x": 2}, resume_run_id=run.info.run_id)

    with pytest.raises(RecorderContractError, match="local file"):
        MLflowRecorder("https://tracking.example", _sdk=sdk)

    for uri in (
        "file://remote/share",
        "file:///tmp/%2e%2e/escape",
        "file:///tmp/%2Fetc",
        "\\\\server\\share",
    ):
        with pytest.raises(RecorderContractError):
            MLflowRecorder(uri, _sdk=sdk)
    if os.name != "nt":
        with pytest.raises(RecorderContractError, match="drive-qualified"):
            MLflowRecorder("file:///C:/tracking", _sdk=sdk)
    else:
        assert MLflowRecorder("file:///C:/tracking", _sdk=sdk)._tracking_uri == "file:///C:/tracking"  # type: ignore[attr-defined]


def test_mlflow_rejects_symlink_tracking_root_before_sdk_setup(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    try:
        link.symlink_to(target, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation is unavailable on this platform")
    sdk = FakeMLflow()
    with pytest.raises(RecorderContractError, match="symlinks or reparse"):
        MLflowRecorder(link, _sdk=sdk)


@pytest.mark.parametrize(
    "value",
    [
        Path(r"\\server\share"),
        Path(r"\\?\C:\tracking"),
        Path(r"C:\tmp\%2e2e\escape"),
        Path(r"C:\tmp\artifact:stream"),
        Path(r"C:\tmp\..\escape"),
        Path("https://tracking.example"),
    ],
)
def test_mlflow_rejects_ambiguous_path_objects_before_sdk_setup(tmp_path: Path, value: Path) -> None:
    del tmp_path
    sdk = FakeMLflow()
    with pytest.raises(RecorderContractError):
        MLflowRecorder(value, _sdk=sdk)


def test_mlflow_drive_path_string_matches_path_and_file_uri_forms(tmp_path: Path) -> None:
    del tmp_path
    sdk = FakeMLflow()
    if os.name == "nt":
        assert MLflowRecorder("C:/tracking", _sdk=sdk)._tracking_uri == "file:///C:/tracking"  # type: ignore[attr-defined]
        assert MLflowRecorder(r"C:\tracking", _sdk=sdk)._tracking_uri == "file:///C:/tracking"  # type: ignore[attr-defined]
    else:
        for value in ("C:/tracking", r"C:\tracking"):
            with pytest.raises(RecorderContractError, match="drive-qualified"):
                MLflowRecorder(value, _sdk=sdk)
    for value in ("C:relative", "C:/tmp/%2e2/escape", "C:/tmp/artifact:stream", "C:/tmp/../escape"):
        with pytest.raises(RecorderContractError):
            MLflowRecorder(value, _sdk=sdk)


def test_mlflow_rejects_artifact_path_traversal_and_double_finish(tmp_path: Path) -> None:
    sdk = FakeMLflow()
    run = MLflowRecorder(tmp_path / "mlruns", _sdk=sdk).start_run("secure")
    for name in ("../escape", "..\\escape", "C:/escape", "a//b", "%2e%2e/x"):
        with pytest.raises(RecorderContractError, match="canonical POSIX|safe relative"):
            run.log_artifact(b"x", name=name)
    run.finish()
    with pytest.raises(RecorderContractError, match="completed"):
        run.finish()


def test_mlflow_external_state_enforces_cumulative_parameter_bound(tmp_path: Path) -> None:
    run = MLflowRecorder(tmp_path / "mlruns", _sdk=FakeMLflow()).start_run("bounded")
    run.log_params({f"first-{index}": "x" * 4_000 for index in range(40)})
    with pytest.raises(RecorderContractError, match="cumulative"):
        run.log_params({f"second-{index}": "x" * 4_000 for index in range(40)})


def test_mlflow_provider_failures_do_not_commit_local_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sdk = FakeMLflow()
    run = MLflowRecorder(tmp_path / "mlruns", _sdk=sdk).start_run("atomic")
    original_params = sdk.log_params

    def fail_params(_values: dict[str, str]) -> None:
        raise RuntimeError("provider params failed")

    monkeypatch.setattr(sdk, "log_params", fail_params)
    with pytest.raises(RuntimeError, match="provider params"):
        run.log_params({"retry": 1})
    monkeypatch.setattr(sdk, "log_params", original_params)
    run.log_params({"retry": 2})

    original_metrics = sdk.log_metrics

    def fail_metrics(_values: dict[str, float], *, step: int) -> None:
        del step
        raise RuntimeError("provider metrics failed")

    monkeypatch.setattr(sdk, "log_metrics", fail_metrics)
    with pytest.raises(RuntimeError, match="provider metrics"):
        run.log_metrics({"score": 1.0}, step=1)
    monkeypatch.setattr(sdk, "log_metrics", original_metrics)
    run.log_metrics({"score": 0.5}, step=0)


def test_mlflow_resume_requires_provider_id_continuity_and_cleans_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sdk = FakeMLflow()
    recorder = MLflowRecorder(tmp_path / "mlruns", _sdk=sdk)
    original_start = sdk.start_run
    first = recorder.start_run("provider-id", config={"x": 1})

    def ignore_resume_id(**kwargs: object) -> SimpleNamespace:
        if "run_id" in kwargs:
            kwargs = {key: value for key, value in kwargs.items() if key != "run_id"}
        return original_start(**kwargs)

    monkeypatch.setattr(sdk, "start_run", ignore_resume_id)
    with pytest.raises(RecorderContractError, match="provider ID mismatch"):
        recorder.start_run("provider-id", config={"x": 1}, resume_run_id=first.info.run_id)
    assert any(call[0] == "end_run" and call[1] == "FAILED" for call in sdk.calls)

    monkeypatch.setattr(sdk, "start_run", original_start)
    resumed = recorder.start_run("provider-id", config={"x": 1}, resume_run_id=first.info.run_id)
    assert resumed.info.run_id == first.info.run_id


def test_mlflow_resume_id_mismatch_reports_cleanup_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sdk = FakeMLflow()
    recorder = MLflowRecorder(tmp_path / "mlruns", _sdk=sdk)
    original_start = sdk.start_run
    first = recorder.start_run("provider-id-cleanup", config={"x": 1})

    def ignore_resume_id(**kwargs: object) -> SimpleNamespace:
        kwargs = {key: value for key, value in kwargs.items() if key != "run_id"}
        return original_start(**kwargs)

    def fail_end(*, status: str) -> None:
        del status
        raise RuntimeError("cleanup failed")

    monkeypatch.setattr(sdk, "start_run", ignore_resume_id)
    monkeypatch.setattr(sdk, "end_run", fail_end)
    with pytest.raises(RecorderContractError, match="failed to clean up"):
        recorder.start_run("provider-id-cleanup", config={"x": 1}, resume_run_id=first.info.run_id)

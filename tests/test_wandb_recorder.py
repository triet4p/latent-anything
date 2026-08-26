"""Offline contract tests for the optional W&B adapter."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from _typeshed import SupportsKeysAndGetItem

import pytest

from latent_anything.experiment_recorder import RecorderContractError
from latent_anything.integrations.wandb_recorder import WandbRecorder


class _FakeArtifact:
    def __init__(self, *, name: str, type: str) -> None:
        self.name = name
        self.type = type
        self.files: list[tuple[str, str]] = []

    def add_file(self, path: str, *, name: str) -> None:
        self.files.append((name, Path(path).read_bytes().decode("utf-8")))


class _FakeRun:
    def __init__(self, run_id: str, tags: list[str]) -> None:
        self.id = run_id
        self.tags = tuple(tags)
        self.config: dict[str, object] = {}
        self.summary: dict[str, str] = {}
        self.logs: list[tuple[dict[str, float], int]] = []
        self.artifacts: list[_FakeArtifact] = []
        self.finished: list[int | None] = []

    def log(self, values: dict[str, float], *, step: int) -> None:
        self.logs.append((values, step))

    def log_artifact(self, artifact: _FakeArtifact) -> None:
        self.artifacts.append(artifact)

    def finish(self, *, exit_code: int | None = None) -> None:
        self.finished.append(exit_code)


class FakeWandb:
    Artifact = _FakeArtifact

    def __init__(self) -> None:
        self.runs: dict[str, _FakeRun] = {}
        self.next_id = 0
        self.init_calls: list[dict[str, object]] = []

    def init(self, **kwargs: object) -> _FakeRun:
        run_id = str(kwargs.get("id", f"wandb-{self.next_id}"))
        self.next_id += 1
        run = self.runs.get(run_id)
        if run is None:
            run = _FakeRun(run_id, list(cast(list[str], kwargs.get("tags", []))))
            run.config.update(cast(dict[str, object], kwargs.get("config", {})))
            self.runs[run_id] = run
        self.init_calls.append(kwargs)
        return run


def test_wandb_offline_adapter_maps_contract_and_parent_group(tmp_path: Path) -> None:
    del tmp_path
    sdk = FakeWandb()
    recorder = WandbRecorder("fixture", mode="offline", _sdk=sdk)
    parent = recorder.start_run("parent", config={"seed": 1}, tags={"lane": "offline"})
    parent.set_tags({"long": "x" * 100})
    parent.log_metrics({"score": 1.0}, step=0)
    artifact = parent.log_artifact(b"hello", name="nested/evidence.bin")
    child = parent.child("child", config={"seed": 2})
    child.finish()
    parent.finish()

    assert artifact.size_bytes == 5
    assert artifact.uri == f"wandb-artifact://{parent.info.run_id}/nested/evidence.bin"
    child_kwargs = sdk.init_calls[1]
    assert child_kwargs["group"] == parent.info.run_id
    assert child_kwargs["reinit"] == "create_new"
    assert sdk.runs[parent.info.run_id].logs == [({"score": 1.0}, 0)]
    assert '"long"' in str(sdk.runs[parent.info.run_id].config["latent_anything.tags"])
    assert sdk.runs[parent.info.run_id].artifacts[0].files == [("nested/evidence.bin", "hello")]
    assert sdk.runs[parent.info.run_id].finished == [None]


def test_wandb_resume_requires_identity_and_rejects_online(tmp_path: Path) -> None:
    del tmp_path
    sdk = FakeWandb()
    recorder = WandbRecorder("fixture", mode="disabled", _sdk=sdk)
    run = recorder.start_run("resume", config={"x": 1})
    resumed = recorder.start_run("resume", config={"x": 1}, resume_run_id=run.info.run_id)
    resumed.log_metrics({"value": 2.0}, step=1)
    resumed.finish()
    assert resumed.info.status == "completed"
    with pytest.raises(RecorderContractError, match="identity mismatch"):
        recorder.start_run("resume", config={"x": 2}, resume_run_id=run.info.run_id)

    with pytest.raises(RecorderContractError, match="offline or disabled"):
        WandbRecorder("fixture", mode="online", _sdk=sdk)


@pytest.mark.parametrize("stored_identity", [None, "", "not-a-digest", 42])
def test_wandb_resume_rejects_missing_or_malformed_identity(stored_identity: object) -> None:
    sdk = FakeWandb()
    recorder = WandbRecorder("fixture", mode="disabled", _sdk=sdk)
    run = recorder.start_run("resume-provenance", config={"x": 1})
    config = sdk.runs[run.info.run_id].config
    if stored_identity is None:
        config.pop("latent_anything.identity", None)
    else:
        config["latent_anything.identity"] = stored_identity
    with pytest.raises(RecorderContractError, match="identity provenance"):
        recorder.start_run("resume-provenance", config={"x": 1}, resume_run_id=run.info.run_id)


def test_wandb_rejects_artifact_path_traversal_and_double_finish() -> None:
    run = WandbRecorder("fixture", _sdk=FakeWandb()).start_run("secure")
    for name in ("../escape", "..\\escape", "C:/escape", "a//b", "%2e%2e/x"):
        with pytest.raises(RecorderContractError, match="canonical POSIX|safe relative"):
            run.log_artifact(b"x", name=name)
    run.finish()
    with pytest.raises(RecorderContractError, match="completed"):
        run.finish()


def test_wandb_failure_diagnostic_is_recorded_in_provider_summary() -> None:
    sdk = FakeWandb()
    run = WandbRecorder("fixture", mode="disabled", _sdk=sdk).start_run("failure")
    run.fail("offline failure")
    assert sdk.runs[run.info.run_id].summary["latent_anything.error"] == "offline failure"


def test_wandb_provider_failures_do_not_commit_local_state(monkeypatch: pytest.MonkeyPatch) -> None:
    sdk = FakeWandb()
    run = WandbRecorder("fixture", mode="disabled", _sdk=sdk).start_run("atomic")
    sdk_run = sdk.runs[run.info.run_id]
    original_log = sdk_run.log

    def fail_log(_values: dict[str, float], *, step: int) -> None:
        del step
        raise RuntimeError("provider metrics failed")

    monkeypatch.setattr(sdk_run, "log", fail_log)
    with pytest.raises(RuntimeError, match="provider metrics"):
        run.log_metrics({"score": 1.0}, step=1)
    monkeypatch.setattr(sdk_run, "log", original_log)
    run.log_metrics({"score": 0.5}, step=0)

    class FailingConfig(dict[str, object]):
        fail = True

        def update(
            self,
            values: SupportsKeysAndGetItem[str, object] | Iterable[tuple[str, object]] = (),
            /,
            **kwargs: object,
        ) -> None:
            if self.fail:
                raise RuntimeError("provider params failed")
            super().update(values, **kwargs)

    sdk_run.config = FailingConfig(sdk_run.config)
    with pytest.raises(RuntimeError, match="provider params"):
        run.log_params({"retry": 1})
    sdk_run.config.fail = False
    run.log_params({"retry": 2})


def test_wandb_resume_requires_provider_id_continuity_and_cleans_up(monkeypatch: pytest.MonkeyPatch) -> None:
    sdk = FakeWandb()
    recorder = WandbRecorder("fixture", mode="disabled", _sdk=sdk)
    original_init = sdk.init
    first = recorder.start_run("provider-id", config={"x": 1})

    def ignore_resume_id(**kwargs: object) -> object:
        if "id" in kwargs:
            kwargs = {key: value for key, value in kwargs.items() if key != "id"}
        return original_init(**kwargs)

    monkeypatch.setattr(sdk, "init", ignore_resume_id)
    with pytest.raises(RecorderContractError, match="provider ID mismatch"):
        recorder.start_run("provider-id", config={"x": 1}, resume_run_id=first.info.run_id)
    wrong_run = sdk.runs["wandb-1"]
    assert wrong_run.finished == [1]

    monkeypatch.setattr(sdk, "init", original_init)
    resumed = recorder.start_run("provider-id", config={"x": 1}, resume_run_id=first.info.run_id)
    assert resumed.info.run_id == first.info.run_id


def test_wandb_resume_id_mismatch_reports_cleanup_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    sdk = FakeWandb()
    recorder = WandbRecorder("fixture", mode="disabled", _sdk=sdk)
    original_init = sdk.init
    first = recorder.start_run("provider-id-cleanup", config={"x": 1})

    def ignore_resume_id(**kwargs: object) -> object:
        kwargs = {key: value for key, value in kwargs.items() if key != "id"}
        wrong_run = original_init(**kwargs)

        def fail_finish(*, exit_code: int | None = None) -> None:
            del exit_code
            raise RuntimeError("cleanup failed")

        wrong_run.finish = fail_finish
        return wrong_run

    monkeypatch.setattr(sdk, "init", ignore_resume_id)
    with pytest.raises(RecorderContractError, match="failed to clean up"):
        recorder.start_run("provider-id-cleanup", config={"x": 1}, resume_run_id=first.info.run_id)

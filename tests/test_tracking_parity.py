"""Three-backend contract parity and opt-in local SDK smoke tests."""

from __future__ import annotations

import hashlib
import json
import socket
import threading
import time
import urllib.request
from pathlib import Path

import numpy as np
import pytest

from latent_anything.experiment_recorder import LocalExperimentRecorder, RecorderContractError
from latent_anything.integrations.mlflow_recorder import MLflowRecorder
from latent_anything.integrations.wandb_recorder import WandbRecorder
from tests.test_mlflow_recorder import FakeMLflow
from tests.test_wandb_recorder import FakeWandb


def test_local_mlflow_wandb_preserve_identity_metrics_and_artifact_digest(tmp_path: Path) -> None:
    local = LocalExperimentRecorder(tmp_path / "local")
    mlflow = MLflowRecorder(tmp_path / "mlruns", _sdk=FakeMLflow())
    wandb = WandbRecorder("fixture", mode="offline", _sdk=FakeWandb())
    backends = [local, mlflow, wandb]

    states = np.array([[0.0, 0.0], [1.0, 0.5], [2.0, 1.0], [3.0, 1.5]])
    actions = np.ones((3, 2), dtype=np.float64)
    predicted = states[:-1] + actions
    expected = states[1:]
    rollout_mse = float(np.mean((predicted - expected) ** 2))
    evidence = json.dumps(
        {"fixture": "deterministic-affine-world-model", "states": states.tolist(), "actions": actions.tolist()},
        sort_keys=True,
    ).encode("utf-8")
    config = {"seed": 7, "fixture": "deterministic-affine-world-model", "horizon": 3}
    runs = [backend.start_run("world-model-fixture", config=config, tags={"lane": "offline"}) for backend in backends]
    assert len({run.info.identity for run in runs}) == 1
    expected_digest: str | None = None
    for run in runs:
        run.log_metrics({"rollout_mse": rollout_mse}, step=3)
        artifact = run.log_artifact(evidence, name="world-model-evidence.json", media_type="application/json")
        if expected_digest is None:
            expected_digest = artifact.digest
        assert artifact.digest == expected_digest
        child = run.child("child", config={"seed": 8})
        assert child.info.parent_run_id == run.info.run_id
        child.finish()
        run.finish()
        assert run.info.status == "completed"


@pytest.mark.integration
def test_mlflow_real_local_file_store_when_extra_is_enabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sdk = pytest.importorskip("mlflow")
    # MLflow 3.x requires an explicit opt-in for its local file-store mode;
    # this test never starts a server or contacts a remote endpoint.
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    recorder = MLflowRecorder(tmp_path / "mlruns", _sdk=sdk)
    parent = recorder.start_run("real-local", config={"seed": 1})
    child = parent.child("real-child", config={"seed": 2})
    child.finish()
    payload = b"offline"
    artifact = parent.log_artifact(payload, name="evidence.txt", media_type="text/plain")
    parent.finish()
    assert artifact.digest == hashlib.sha256(payload).hexdigest()
    downloaded = Path(sdk.artifacts.download_artifacts(run_id=parent.info.run_id, artifact_path="evidence.txt"))
    downloaded_bytes = downloaded.read_bytes()
    assert downloaded_bytes == payload
    assert hashlib.sha256(downloaded_bytes).hexdigest() == artifact.digest
    resumed = recorder.start_run("real-local", config={"seed": 1}, resume_run_id=parent.info.run_id)
    assert resumed.info.identity == parent.info.identity
    resumed.finish()
    assert child.info.parent_run_id == parent.info.run_id


@pytest.mark.integration
def test_wandb_real_offline_when_extra_is_enabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))
    sdk = pytest.importorskip("wandb")

    def denied_urlopen(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("real offline W&B lane attempted network access")

    requests = pytest.importorskip("requests")
    monkeypatch.setattr(urllib.request, "urlopen", denied_urlopen)
    monkeypatch.setattr(requests.sessions.Session, "request", denied_urlopen)
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex
    original_create_connection = socket.create_connection

    def denied_remote_connect(sock: socket.socket, address: object) -> object:
        # W&B offline still uses loopback/IPC for its local service; only
        # non-local network destinations are forbidden by this evidence lane.
        host = address[0] if isinstance(address, tuple) and address else None
        if isinstance(address, str) or host in {"127.0.0.1", "localhost", "::1"}:
            return original_connect(sock, address)  # type: ignore[arg-type]
        raise AssertionError("real offline W&B lane attempted remote socket access")

    def denied_remote_connect_ex(sock: socket.socket, address: object) -> object:
        host = address[0] if isinstance(address, tuple) and address else None
        if isinstance(address, str) or host in {"127.0.0.1", "localhost", "::1"}:
            return original_connect_ex(sock, address)  # type: ignore[arg-type]
        raise AssertionError("real offline W&B lane attempted remote socket access")

    def denied_remote_create_connection(address: object, *args: object, **kwargs: object) -> object:
        host = address[0] if isinstance(address, tuple) and address else None
        if isinstance(address, str) or host in {"127.0.0.1", "localhost", "::1"}:
            return original_create_connection(address, *args, **kwargs)  # type: ignore[arg-type]
        raise AssertionError("real offline W&B lane attempted remote socket access")

    monkeypatch.setattr(socket.socket, "connect", denied_remote_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", denied_remote_connect_ex)
    monkeypatch.setattr(socket, "create_connection", denied_remote_create_connection)
    baseline_threads = {id(thread) for thread in threading.enumerate()}
    recorder = WandbRecorder("latent-anything-test", mode="offline", _sdk=sdk)
    parent = recorder.start_run("real-offline", config={"seed": 1})
    child = parent.child("real-child", config={"seed": 2})
    assert child.info.run_id != parent.info.run_id
    child.finish()
    payload = b"offline"
    artifact = parent.log_artifact(payload, name="evidence.txt", media_type="text/plain")
    parent.finish()
    assert artifact.digest == hashlib.sha256(payload).hexdigest()
    provider_path = Path(parent._sdk_run.dir) / "latent_anything_artifacts" / "evidence.txt"  # type: ignore[attr-defined]
    provider_bytes = provider_path.read_bytes()
    assert provider_bytes == payload
    assert hashlib.sha256(provider_bytes).hexdigest() == artifact.digest
    with pytest.raises(RecorderContractError, match="identity provenance"):
        recorder.start_run("real-offline", config={"seed": 1}, resume_run_id=parent.info.run_id)
    assert child.info.parent_run_id == parent.info.run_id
    assert provider_path.exists()
    sdk.teardown()
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and any(
        id(thread) not in baseline_threads and thread.is_alive() for thread in threading.enumerate()
    ):
        time.sleep(0.01)
    assert not any(id(thread) not in baseline_threads and thread.is_alive() for thread in threading.enumerate())
    assert not any(path.suffix == ".lock" for path in tmp_path.rglob("*"))

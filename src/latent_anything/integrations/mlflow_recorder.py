"""Optional MLflow recorder adapter with a local file-store boundary."""

from __future__ import annotations

import os
import re
import stat
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType
from urllib.parse import unquote, urlsplit

from latent_anything.experiment_recorder import (
    RecorderArtifact,
    RecorderContractError,
    RecorderRunInfo,
    canonical_recorder_json,
    read_recorder_artifact,
    recorder_artifact_from_bytes,
    safe_recorder_artifact_path,
    validate_recorder_artifact_name,
    validate_recorder_name,
)
from latent_anything.integrations import require_optional
from latent_anything.integrations._tracking_common import ExternalRunState, prepare_run


def _call(sdk: object, method: str, *args: object, **kwargs: object) -> object:
    function = getattr(sdk, method, None)
    if not callable(function):
        raise RecorderContractError(f"MLflow SDK does not expose {method}()")
    return function(*args, **kwargs)


def _local_tracking_uri(value: str | Path) -> str:
    if isinstance(value, Path):
        raw_path = str(value.expanduser())
        _validate_local_path_lexical(raw_path)
        path = value.expanduser()
    else:
        # ``urlsplit`` treats ``C:/...`` as a URI with scheme ``c``. Detect
        # absolute drive paths first so the string and Path forms agree.
        if re.match(r"^[A-Za-z]:[\\/]", value):
            _validate_local_path_lexical(value)
            path = Path(value)
        else:
            parsed = urlsplit(value)
            if parsed.scheme and parsed.scheme.lower() != "file":
                raise RecorderContractError("MLflow recorder only permits a local file tracking URI")
            if parsed.scheme.lower() == "file":
                if parsed.netloc or parsed.query or parsed.fragment or "%" in value:
                    raise RecorderContractError("MLflow file URI must have an empty authority and no encoding")
                raw_path = parsed.path
                if not raw_path or "\\" in raw_path or "\x00" in raw_path:
                    raise RecorderContractError("MLflow file URI must be a canonical local path")
                decoded = unquote(raw_path)
                if decoded != raw_path:
                    raise RecorderContractError("MLflow file URI must not contain encoded path components")
                if len(raw_path) >= 3 and raw_path[0] == "/" and raw_path[2] == ":":
                    if os.name != "nt":
                        raise RecorderContractError("MLflow file URI has an ambiguous drive-qualified path")
                    raw_path = raw_path[1:]
                _validate_local_path_lexical(raw_path)
                path = Path(raw_path)
            else:
                if "\\" in value or "\x00" in value:
                    raise RecorderContractError("MLflow tracking path must be a canonical local path")
                _validate_local_path_lexical(value)
                path = Path(value)
    if _has_reparse_component(path):
        raise RecorderContractError("MLflow tracking path must not traverse symlinks or reparse points")
    try:
        canonical = path.resolve(strict=False)
    except OSError as error:
        raise RecorderContractError("MLflow tracking path cannot be resolved safely") from error
    if _has_reparse_component(canonical):
        raise RecorderContractError("MLflow tracking path must not traverse symlinks or reparse points")
    return canonical.as_uri()


def _validate_local_path_lexical(raw_path: str) -> None:
    """Reject ambiguous path syntax before filesystem resolution or SDK setup."""

    if not raw_path or "\x00" in raw_path or "%" in raw_path:
        raise RecorderContractError("MLflow tracking path must use unencoded local syntax")
    normalized = raw_path.replace("\\", "/")
    if normalized.startswith("//") or normalized.startswith("/?/") or normalized.startswith("/./"):
        raise RecorderContractError("MLflow tracking path must not be UNC or device syntax")
    drive = re.match(r"^[A-Za-z]:", normalized)
    if drive:
        if os.name != "nt" or len(normalized) < 3 or normalized[2] != "/":
            raise RecorderContractError("MLflow tracking path has an ambiguous drive-qualified form")
        if ":" in normalized[2:]:
            raise RecorderContractError("MLflow tracking path must not contain URI or ADS syntax")
    elif ":" in normalized:
        raise RecorderContractError("MLflow tracking path must not contain URI or ADS syntax")
    if os.name != "nt" and "\\" in raw_path:
        raise RecorderContractError("MLflow tracking path must use POSIX separators")
    parts = normalized.split("/")
    if any(part in {".", ".."} for part in parts):
        raise RecorderContractError("MLflow tracking path must not contain traversal or dot segments")


def _has_reparse_component(path: Path) -> bool:
    """Reject existing symlink/reparse components before provider setup."""

    candidate = path if path.is_absolute() else (Path.cwd() / path)
    current = Path(candidate.anchor) if candidate.anchor else Path()
    for part in candidate.parts[1:] if candidate.anchor else candidate.parts:
        current = current / part
        try:
            info = os.stat(current, follow_symlinks=False)
        except FileNotFoundError:
            continue
        except OSError as error:
            raise RecorderContractError("MLflow tracking path cannot be inspected safely") from error
        if stat.S_ISLNK(info.st_mode) or bool(getattr(info, "st_file_attributes", 0) & 0x400):
            return True
    return False


def _validate_external_id(value: str | None, *, field: str) -> str | None:
    if value is None:
        return None
    if not value or len(value) > 256 or any(character in value for character in ("\x00", "\n", "\r")):
        raise RecorderContractError(f"{field} must be a bounded identifier")
    return value


def _param_value(value: object) -> str:
    if isinstance(value, str):
        return value
    return canonical_recorder_json(value).decode("utf-8")


class MLflowRecorder:
    """Implement the recorder contract through MLflow's local file store."""

    backend_name = "mlflow"

    def __init__(
        self,
        tracking_uri: str | Path,
        *,
        experiment_name: str = "latent-anything",
        _sdk: ModuleType | object | None = None,
    ) -> None:
        self._tracking_uri = _local_tracking_uri(tracking_uri)
        self._experiment_name = validate_recorder_name(experiment_name, field="experiment_name")
        self._sdk = _sdk if _sdk is not None else require_optional("mlflow", extra="tracking-mlflow")
        _call(self._sdk, "set_tracking_uri", self._tracking_uri)
        _call(self._sdk, "set_experiment", self._experiment_name)

    def start_run(
        self,
        name: str,
        *,
        config: Mapping[str, object] | None = None,
        tags: Mapping[str, str] | None = None,
        parent_run_id: str | None = None,
        resume_run_id: str | None = None,
        code_version: str = "",
        framework_version: str = "",
        model_revisions: Mapping[str, str] | None = None,
        dataset_revisions: Mapping[str, str] | None = None,
        seeds: Sequence[int] = (),
        environment: Mapping[str, object] | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> MLflowExperimentRun:
        prepared = prepare_run(
            name,
            config=config,
            tags=tags,
            parent_run_id=parent_run_id,
            code_version=code_version,
            framework_version=framework_version,
            model_revisions=model_revisions,
            dataset_revisions=dataset_revisions,
            seeds=seeds,
            environment=environment,
            metadata=metadata,
        )
        resolved_resume = _validate_external_id(resume_run_id, field="resume_run_id")
        if resolved_resume is None:
            kwargs: dict[str, object] = {
                "run_name": prepared.name,
                "tags": prepared.provider_tags,
            }
            if prepared.parent_run_id is not None:
                kwargs.update({"nested": True, "parent_run_id": prepared.parent_run_id})
        else:
            kwargs = {"run_id": resolved_resume, "tags": prepared.provider_tags}
        sdk_run = _call(self._sdk, "start_run", **kwargs)
        run_id = str(getattr(getattr(sdk_run, "info", None), "run_id", ""))
        if not run_id:
            raise RecorderContractError("MLflow start_run() returned no run id")
        if resolved_resume is not None:
            if run_id != resolved_resume:
                try:
                    _call(self._sdk, "end_run", status="FAILED")
                except Exception as cleanup_error:
                    raise RecorderContractError(
                        "MLflow resume provider ID mismatch; failed to clean up unexpected provider run"
                    ) from cleanup_error
                raise RecorderContractError("MLflow resume provider ID mismatch")
            data_tags = getattr(getattr(sdk_run, "data", None), "tags", None)
            stored_identity = data_tags.get("latent_anything.identity") if isinstance(data_tags, Mapping) else None
            if stored_identity != prepared.identity:
                _call(self._sdk, "end_run", status="FAILED")
                raise RecorderContractError("MLflow resume identity mismatch")
        state = ExternalRunState(
            info=RecorderRunInfo(
                run_id=run_id,
                identity=prepared.identity,
                name=prepared.name,
                status="running",
                backend=self.backend_name,
                parent_run_id=prepared.parent_run_id,
            ),
            params={},
            tags=dict(prepared.provider_tags),
        )
        handle = MLflowExperimentRun(self, sdk_run, state)
        if prepared.config:
            handle.log_params(prepared.config)
        return handle

    def _call_provider(self, method: str, *args: object, **kwargs: object) -> object:
        """Invoke one adapter-owned provider operation."""

        return _call(self._sdk, method, *args, **kwargs)


class MLflowExperimentRun:
    """One MLflow-backed implementation of ``ExperimentRun``."""

    def __init__(self, owner: MLflowRecorder, sdk_run: object, state: ExternalRunState) -> None:
        self._owner = owner
        self._sdk_run = sdk_run
        self._state = state

    @property
    def info(self) -> RecorderRunInfo:
        return self._state.info

    def log_params(self, params: Mapping[str, object]) -> None:
        values, candidate = self._state.prepare_params(params)
        if values:
            self._owner._call_provider(  # pyright: ignore[reportPrivateUsage]
                "log_params", {key: _param_value(value) for key, value in values.items()}
            )
        self._state.commit_params(candidate)

    def log_metrics(self, metrics: Mapping[str, float], *, step: int) -> None:
        values, candidate, resolved_step, metric_events = self._state.prepare_metrics(metrics, step)
        if values:
            self._owner._call_provider("log_metrics", values, step=step)  # pyright: ignore[reportPrivateUsage]
        self._state.commit_metrics(candidate, resolved_step, metric_events)

    def set_tags(self, tags: Mapping[str, str]) -> None:
        values, candidate = self._state.prepare_tags(tags)
        if values:
            self._owner._call_provider("set_tags", values)  # pyright: ignore[reportPrivateUsage]
        self._state.commit_tags(candidate)

    def log_artifact(
        self,
        content: bytes | bytearray | memoryview | str | Path,
        *,
        name: str,
        media_type: str = "application/octet-stream",
    ) -> RecorderArtifact:
        self._state.require_running()
        safe_name = validate_recorder_artifact_name(name)
        data = read_recorder_artifact(content)
        reference = recorder_artifact_from_bytes(
            data, name=safe_name, media_type=media_type, uri=f"runs:/{self.info.run_id}/{safe_name}"
        )
        with tempfile.TemporaryDirectory(prefix="latent-anything-mlflow-") as temporary:
            path = safe_recorder_artifact_path(temporary, safe_name)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            self._owner._call_provider("log_artifact", str(path))  # pyright: ignore[reportPrivateUsage]
        return reference

    def child(self, name: str, *, config: Mapping[str, object] | None = None) -> MLflowExperimentRun:
        self._state.require_running()
        return self._owner.start_run(name, config=config, parent_run_id=self.info.run_id)

    def finish(self) -> RecorderRunInfo:
        self._state.require_running()
        self._owner._call_provider("end_run", status="FINISHED")  # pyright: ignore[reportPrivateUsage]
        return self._state.finish(status="completed")

    def fail(self, error: str) -> RecorderRunInfo:
        self._state.require_running()
        if not error or len(error) > 4096:
            raise RecorderContractError("failure diagnostic must be a bounded non-empty string")
        self._owner._call_provider("set_tag", "latent_anything.error", error)  # pyright: ignore[reportPrivateUsage]
        self._owner._call_provider("end_run", status="FAILED")  # pyright: ignore[reportPrivateUsage]
        return self._state.finish(status="failed")


__all__ = ["MLflowExperimentRun", "MLflowRecorder"]

"""Optional offline W&B recorder adapter.

The adapter intentionally supports only W&B's local ``offline`` and
``disabled`` modes.  The public surface remains the validated recorder
contract; W&B SDK values never cross that boundary.
"""

from __future__ import annotations

import hashlib
import re
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any, cast

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


def _call(value: object, method: str, *args: object, **kwargs: object) -> object:
    function = getattr(value, method, None)
    if not callable(function):
        raise RecorderContractError(f"W&B object does not expose {method}()")
    return function(*args, **kwargs)


def _param_value(value: object) -> object:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return canonical_recorder_json(value).decode("utf-8")


def _tag(key: str, value: str) -> str:
    raw = f"{key}={value}"
    if len(raw) <= 64:
        return raw
    return f"la:{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:16]}"


class WandbRecorder:
    """Implement the recorder contract through W&B offline/disabled mode."""

    backend_name = "wandb"

    def __init__(
        self,
        project: str,
        *,
        mode: str = "offline",
        entity: str | None = None,
        _sdk: ModuleType | object | None = None,
    ) -> None:
        self._project = validate_recorder_name(project, field="project")
        if mode not in {"offline", "disabled"}:
            raise RecorderContractError("W&B recorder permits only offline or disabled mode")
        self._mode = mode
        self._entity = None if entity is None else validate_recorder_name(entity, field="entity")
        self._sdk = _sdk if _sdk is not None else require_optional("wandb", extra="tracking-wandb")

    def _create_artifact(self, *, name: str, type: str) -> object:
        factory = getattr(self._sdk, "Artifact", None)
        if not callable(factory):
            raise RecorderContractError("W&B SDK does not expose Artifact()")
        return factory(name=name, type=type)

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
    ) -> WandbExperimentRun:
        """Start or resume an offline/disabled W&B run with provenance config."""
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
        resume = resume_run_id
        if resume is not None and (not resume or len(resume) > 256):
            raise RecorderContractError("resume_run_id must be a bounded identifier")
        if any(key in prepared.config for key in ("latent_anything.identity", "latent_anything.tags")):
            raise RecorderContractError("reserved W&B provenance config keys are not user-configurable")
        provider_config = {
            **{key: _param_value(value) for key, value in prepared.config.items()},
            "latent_anything.identity": prepared.identity,
            "latent_anything.tags": canonical_recorder_json(prepared.tags).decode("utf-8"),
        }
        kwargs: dict[str, object] = {
            "project": self._project,
            "name": prepared.name,
            "config": provider_config,
            # W&B tags are strings rather than a mapping; key=value retains
            # the validated provenance values without silently dropping them.
            "tags": [_tag(key, value) for key, value in prepared.provider_tags.items()],
            "mode": self._mode,
        }
        if resume is None:
            # W&B otherwise returns the currently active run, which collapses
            # provider-side child runs in offline mode.
            kwargs["reinit"] = "create_new"
        if self._entity is not None:
            kwargs["entity"] = self._entity
        if parent_run_id is not None:
            # W&B has no provider-neutral nested-run contract; group plus an
            # explicit tag preserves the relationship in offline files.
            kwargs["group"] = parent_run_id
        if resume is not None:
            kwargs.update({"id": resume, "resume": "allow"})
        sdk_run = _call(self._sdk, "init", **kwargs)
        run_id = str(getattr(sdk_run, "id", ""))
        if not run_id:
            raise RecorderContractError("W&B init() returned no run id")
        if resume is not None:
            if run_id != resume:
                try:
                    _call(sdk_run, "finish", exit_code=1)
                except Exception as cleanup_error:
                    raise RecorderContractError(
                        "W&B resume provider ID mismatch; failed to clean up unexpected provider run"
                    ) from cleanup_error
                raise RecorderContractError("W&B resume provider ID mismatch")
            existing_config = getattr(sdk_run, "config", {})
            stored_identity = (
                existing_config.get("latent_anything.identity") if isinstance(existing_config, Mapping) else None
            )
            if type(stored_identity) is not str or not re.fullmatch(r"[0-9a-f]{64}", stored_identity):
                _call(sdk_run, "finish", exit_code=1)
                raise RecorderContractError("W&B resume identity provenance is missing or malformed")
            if stored_identity != prepared.identity:
                _call(sdk_run, "finish", exit_code=1)
                raise RecorderContractError("W&B resume identity mismatch")
        state = ExternalRunState(
            info=RecorderRunInfo(
                run_id=run_id,
                identity=prepared.identity,
                name=prepared.name,
                status="running",
                backend=self.backend_name,
                parent_run_id=prepared.parent_run_id,
            ),
            params=dict(prepared.config),
            tags=dict(prepared.provider_tags),
        )
        return WandbExperimentRun(self, sdk_run, state)


class WandbExperimentRun:
    """One W&B-backed implementation of ``ExperimentRun``."""

    def __init__(self, owner: WandbRecorder, sdk_run: object, state: ExternalRunState) -> None:
        self._owner = owner
        self._sdk_run = sdk_run
        self._state = state

    @property
    def info(self) -> RecorderRunInfo:
        """Return this provider run's identity, backend, and lifecycle status."""
        return self._state.info

    def log_params(self, params: Mapping[str, object]) -> None:
        """Validate and update parameters in the offline W&B config."""
        values, candidate = self._state.prepare_params(params)
        if values:
            config = getattr(self._sdk_run, "config", None)
            if config is None:
                raise RecorderContractError("W&B run does not expose config")
            _call(config, "update", {key: _param_value(value) for key, value in values.items()})
        self._state.commit_params(candidate)

    def log_metrics(self, metrics: Mapping[str, float], *, step: int) -> None:
        """Validate and log step metrics to the offline W&B run."""
        values, candidate, resolved_step, metric_events = self._state.prepare_metrics(metrics, step)
        if values:
            _call(self._sdk_run, "log", values, step=step)
        self._state.commit_metrics(candidate, resolved_step, metric_events)

    def set_tags(self, tags: Mapping[str, str]) -> None:
        """Validate and merge tags into the provider run when mutable."""
        values, candidate = self._state.prepare_tags(tags)
        if not values:
            self._state.commit_tags(candidate)
            return
        current = getattr(self._sdk_run, "tags", ())
        current_values = cast(Sequence[object], current) if isinstance(current, (list, tuple, set)) else ()
        current_tags: list[str] = [item for item in current_values if isinstance(item, str)]
        current_keys = {item.split("=", 1)[0] for item in current_tags if "=" in item}
        current_tags.extend(_tag(key, value) for key, value in values.items() if key not in current_keys)
        try:
            cast(Any, self._sdk_run).tags = tuple(current_tags)
        except (AttributeError, TypeError) as error:
            raise RecorderContractError("W&B run tags are not mutable") from error
        config = getattr(self._sdk_run, "config", None)
        if config is not None:
            _call(config, "update", {"latent_anything.tags": canonical_recorder_json(candidate).decode("utf-8")})
        self._state.commit_tags(candidate)

    def log_artifact(
        self,
        content: bytes | bytearray | memoryview | str | Path,
        *,
        name: str,
        media_type: str = "application/octet-stream",
    ) -> RecorderArtifact:
        """Safely stage bytes, log the artifact, and return its local reference."""
        self._state.require_running()
        safe_name = validate_recorder_artifact_name(name)
        data = read_recorder_artifact(content)
        reference = recorder_artifact_from_bytes(
            data, name=safe_name, media_type=media_type, uri=f"wandb-artifact://{self.info.run_id}/{safe_name}"
        )
        artifact_name = f"{self.info.run_id}-{safe_name.replace('/', '__')}"
        with tempfile.TemporaryDirectory(prefix="latent-anything-wandb-") as temporary:
            path = safe_recorder_artifact_path(temporary, safe_name)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            artifact = self._owner._create_artifact(  # pyright: ignore[reportPrivateUsage]
                name=artifact_name, type=media_type
            )
            _call(artifact, "add_file", str(path), name=safe_name)
            _call(self._sdk_run, "log_artifact", artifact)
        provider_dir = getattr(self._sdk_run, "dir", None)
        if isinstance(provider_dir, str) and provider_dir:
            mirror_root = Path(provider_dir) / "latent_anything_artifacts"
            mirror_root.mkdir(parents=True, exist_ok=True)
            mirror_path = safe_recorder_artifact_path(mirror_root, safe_name)
            mirror_path.parent.mkdir(parents=True, exist_ok=True)
            mirror_path.write_bytes(data)
        return reference

    def child(self, name: str, *, config: Mapping[str, object] | None = None) -> WandbExperimentRun:
        """Start a child run grouped under this running W&B handle."""
        self._state.require_running()
        return self._owner.start_run(name, config=config, parent_run_id=self.info.run_id)

    def finish(self) -> RecorderRunInfo:
        """Finish the provider run and return its completed recorder info."""
        self._state.require_running()
        _call(self._sdk_run, "finish")
        return self._state.finish(status="completed")

    def fail(self, error: str) -> RecorderRunInfo:
        """Record a bounded failure diagnostic and close the provider run."""
        self._state.require_running()
        if not error or len(error) > 4096:
            raise RecorderContractError("failure diagnostic must be a bounded non-empty string")
        summary = getattr(self._sdk_run, "summary", None)
        if summary is None:
            raise RecorderContractError("W&B run does not expose a mutable summary")
        updater = getattr(summary, "update", None)
        try:
            if callable(updater):
                updater({"latent_anything.error": error})
            elif isinstance(summary, Mapping):
                summary["latent_anything.error"] = error  # type: ignore[index]
            else:
                raise RecorderContractError("W&B run summary is not mutable")
        except (TypeError, AttributeError) as exc:
            raise RecorderContractError("W&B run summary is not mutable") from exc
        _call(self._sdk_run, "finish", exit_code=1)
        return self._state.finish(status="failed")


__all__ = ["WandbExperimentRun", "WandbRecorder"]

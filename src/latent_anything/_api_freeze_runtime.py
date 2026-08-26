"""Shared normalization and observed beta/runtime contract inventory."""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import importlib
import inspect
import io
import json
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import numpy as np


def signature(value: object) -> str:
    try:
        return str(inspect.signature(cast(Callable[..., object], value)))
    except (TypeError, ValueError):
        return "<no-signature>"


def json_value(value: object) -> object:
    if value is dataclasses.MISSING:
        return "<required>"
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, tuple | list):
        return [json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    return f"<{type(value).__module__}.{type(value).__name__}>"


def symbol(name: str, value: object, *, import_path: str) -> dict[str, str]:
    return {
        "name": name,
        "import_path": import_path,
        "module": getattr(value, "__module__", "<unknown>"),
        "object_name": getattr(value, "__name__", type(value).__name__),
        "signature": signature(value),
    }


def _is_subparsers(action: argparse.Action) -> bool:
    return type(action).__name__ == "_SubParsersAction"


def _is_help(action: argparse.Action) -> bool:
    return type(action).__name__ == "_HelpAction"


def aliases() -> dict[str, object]:
    package = importlib.import_module("latent_anything")

    def resolve(name: str) -> object:
        if hasattr(package, name):
            return getattr(package, name)
        for module_name in ("latent_anything.methods.protocols", "latent_anything.methods.b_protocols"):
            module = importlib.import_module(module_name)
            if hasattr(module, name):
                return getattr(module, name)
        raise AttributeError(name)

    symbol_rows = [
        {
            "canonical": canonical,
            "legacy": legacy,
            "deadline": "0.9.0",
            "planned_rfc_window": "0.2.0 (planned; never released)",
            "actual_state": "Unreleased / Sprint78.29 (metadata 0.1.0b1)",
            "policy": "retain-exact-alias-through-beta",
            "observed_identity": resolve(canonical) is resolve(legacy),
        }
        for canonical, legacy in (
            ("AnalysisMethod", "Method"),
            ("Intervention", "BMethod"),
            ("InterventionPipeline", "ManipulationPipeline"),
        )
    ]
    registry_aliases = importlib.import_module("latent_anything.registry_aliases")
    with warnings.catch_warnings(record=True) as captured_a:
        warnings.simplefilter("always")
        normalized_a = registry_aliases.canonical_kind("method_a", warn=True)
    with warnings.catch_warnings(record=True) as captured_b:
        warnings.simplefilter("always")
        normalized_b = registry_aliases.canonical_kind("method_b", warn=True)
    registry_rows = [
        {
            "canonical": normalized_a,
            "legacy": "method_a",
            "constant": "KIND_METHOD_A",
            "deadline": "0.9.0",
            "planned_rfc_window": "0.2.0 (planned; never released)",
            "actual_state": "Unreleased / Sprint31 (metadata 0.1.0b1)",
            "observed_warning": bool(captured_a),
        },
        {
            "canonical": normalized_b,
            "legacy": "method_b",
            "constant": "KIND_METHOD_B",
            "deadline": "0.9.0",
            "planned_rfc_window": "0.2.0 (planned; never released)",
            "actual_state": "Unreleased / Sprint31 (metadata 0.1.0b1)",
            "observed_warning": bool(captured_b),
        },
    ]
    cli = importlib.import_module("latent_anything.cli")
    parser = cli._parser()
    subparsers = cast(Any, next(action for action in parser._actions if _is_subparsers(action)))
    cli_rows = [
        {
            "canonical": canonical,
            "legacy": legacy,
            "policy": "retain-through-beta",
            "observed_same_parser": subparsers.choices[canonical] is subparsers.choices[legacy],
        }
        for canonical, legacy in (("capture-points", "list-capture-points"), ("replay-run", "replay-run-config"))
    ]
    from latent_anything.mppi import MPPIConfig

    config_rows = [
        {
            "canonical": "temperature",
            "legacy": "lambda",
            "also_accepted": "lambda_",
            "model": "MPPIConfig",
            "deadline": "none",
            "observed_values": [
                cast(Any, MPPIConfig)(
                    horizon=1, action_dim=1, lower_bounds=(-1,), upper_bounds=(1,), **{key: 2.0}
                ).temperature
                for key in ("lambda", "lambda_")
            ],
        }
    ]
    result_rows = []
    for model_name, canonical, legacy in (
        ("CEMPlanResult", "actions", "selected_actions"),
        ("MPPIPlanResult", "actions", "selected_actions"),
        ("RolloutResult", "trajectory", "states"),
    ):
        model = getattr(package, model_name)
        declared = {field.name for field in dataclasses.fields(model)} if dataclasses.is_dataclass(model) else set()
        result_rows.append(
            {
                "canonical": canonical,
                "legacy": legacy,
                "models": [model_name],
                "observed_presence": all(name in declared or hasattr(model, name) for name in (canonical, legacy)),
            }
        )
    transition_rows = []
    from latent_anything._transition_deterministic import DeterministicLatentTransition
    from latent_anything._transition_stochastic import StochasticGaussianLatentTransition
    from latent_anything._transition_types import (
        GaussianPrediction,
        StochasticOneStepMetrics,
        StochasticRolloutMetrics,
    )

    for model, canonical, legacy, kind in (
        (DeterministicLatentTransition, "step", "predict", "method"),
        (StochasticGaussianLatentTransition, "step", "predict", "method"),
        (GaussianPrediction, "scale", "std", "property"),
        (StochasticOneStepMetrics, "negative_log_likelihood", "nll", "property"),
        (StochasticRolloutMetrics, "negative_log_likelihood_by_horizon", "nll_by_horizon", "property"),
        (StochasticRolloutMetrics, "mean_error_by_horizon", "errors_by_horizon", "property"),
    ):
        declared = {field.name for field in dataclasses.fields(model)} if dataclasses.is_dataclass(model) else set()
        transition_rows.append(
            {
                "canonical": canonical,
                "legacy": legacy,
                "owner": f"{model.__module__}.{model.__name__}",
                "kind": kind,
                "deadline": "none",
                "warning": "none",
                "observed_presence": all(name in declared or hasattr(model, name) for name in (canonical, legacy)),
                "observed_method_identity": (
                    getattr(model, canonical, None) is getattr(model, legacy, None) if kind == "method" else None
                ),
            }
        )
    return {
        "symbol_aliases": symbol_rows,
        "registry_kind_aliases": registry_rows,
        "cli_aliases": cli_rows,
        "config_aliases": config_rows,
        "result_property_aliases": result_rows,
        "transition_aliases": transition_rows,
        "schema_migrations": [
            {
                "legacy": "result-envelope-v0",
                "canonical": "result-envelope-v1",
                "path": "latent_anything.portable_results.decode_result_envelope",
                "warning": "none; explicit local migration",
            },
            {
                "legacy": "pre-versioned run record / Windows artifact paths",
                "canonical": "run-record schema-v1",
                "path": "latent_anything.run_record.migrate_run_record",
                "warning": "none; explicit local migration",
            },
        ],
    }


def cli_contract() -> dict[str, object]:
    cli = importlib.import_module("latent_anything.cli")
    parser = cast(argparse.ArgumentParser, cli._parser())
    subparsers = cast(Any, next(action for action in parser._actions if _is_subparsers(action)))
    command_parsers: dict[int, argparse.ArgumentParser] = {}
    aliases_by_id: dict[int, list[str]] = {}
    for alias, command_parser in subparsers.choices.items():
        command_parsers.setdefault(id(command_parser), command_parser)
        aliases_by_id.setdefault(id(command_parser), []).append(alias)

    def action_row(action: argparse.Action) -> dict[str, object]:
        default = str(action.default) if isinstance(action.default, Path) else action.default
        choices = None if action.choices is None else [json_value(choice) for choice in action.choices]
        return {
            "option_strings": list(action.option_strings),
            "dest": action.dest,
            "nargs": action.nargs,
            "required": action.required,
            "choices": choices,
            "default": json_value(default),
            "type": None if action.type is None else getattr(action.type, "__name__", str(action.type)),
        }

    commands = []
    for command_parser in sorted(command_parsers.values(), key=lambda item: item.prog):
        name = command_parser.prog.rsplit(" ", 1)[-1]
        commands.append(
            {
                "name": name,
                "aliases": sorted(alias for alias in aliases_by_id[id(command_parser)] if alias != name),
                "actions": [action_row(action) for action in command_parser._actions if not _is_help(action)],
            }
        )
    malformed_exit = None
    with contextlib.redirect_stderr(io.StringIO()):
        try:
            cli.main(["inspect-policy"])
        except SystemExit as exc:
            malformed_exit = exc.code
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        success_exit = cli.main(["capture-points", "--policy", "act"])
    return {
        "commands": commands,
        "parser_malformed_exit": malformed_exit,
        "main_success_exit": success_exit,
        "runtime_failure": "typed domain exception, process non-zero",
    }


def serialization() -> dict[str, object]:
    portable = importlib.import_module("latent_anything.portable")
    portable_payload = portable.encode_portable({"array": [1, 2, 3]})
    from latent_anything.artifact_store import ArtifactStore
    from latent_anything.cem import CEMIteration, CEMPlanResult
    from latent_anything.portable_results import encode_result_envelope
    from latent_anything.run_record import RunRecord, migrate_run_record
    from latent_anything.runtime import CacheKey
    from latent_anything.runtime.disk_cache import make_disk_cache_key
    from latent_anything.runtime.profiling import ProfileEvent, RuntimeProfile

    iteration = CEMIteration(0, 4, 1, 1.0, 0.2, 1.5, 1.2, np.array([0.1, 0.2]), np.array([0.3, 0.4]))
    result = CEMPlanResult(
        np.array([[0.1, 0.2]], dtype=np.float64),
        1.5,
        (iteration,),
        (1.0, 1.5),
        RuntimeProfile(events=(ProfileEvent("planning", 0.25, {"seed": 7}),)),
        7,
    )
    result_payload = encode_result_envelope(
        result,
        provenance={"plugin": "builtin-cem", "version": "0.1.0b1"},
        behavior_state={"config": {"horizon": 1, "action_dim": 2}, "checkpoint": "none"},
    )
    record_payload = {
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
    legacy_payload = {
        "id": "legacy-id",
        "run_name": "old",
        "config": {"seed": 4},
        "metrics": {"x": 2},
        "created_at": "2026-01-01T00:00:00+00:00",
        "artifacts": [{"name": "x", "digest": "a" * 64, "size_bytes": 1, "relative_path": f"artifacts\\{'a' * 64}"}],
    }
    record = RunRecord.from_dict(record_payload)
    key = CacheKey("runtime", "encode", "component", "a" * 64, "b" * 64, "c" * 64, "0.1")
    return {
        "portable": {
            "version": portable._SCHEMA_VERSION,
            "header_version_key": portable._VERSION_KEY.decode("ascii"),
            "fixture_digest": hashlib.sha256(portable_payload).hexdigest(),
        },
        "result_envelope": {
            "version": importlib.import_module("latent_anything.portable_results")._SCHEMA_VERSION,
            "migrated_from": ["result-envelope-v0"],
            "fixture_digest": "815ea47a07cecc202c5312d4c9ff4de5441c9ce74d4315e647615f483ba050ea",
            "observed_fixture_digest": hashlib.sha256(result_payload).hexdigest(),
            "golden_fixture_digest": "815ea47a07cecc202c5312d4c9ff4de5441c9ce74d4315e647615f483ba050ea",
            "fixture_note": "Golden digest retained; observed builder digest is recorded to expose fixture drift.",
        },
        "artifact_envelope": {
            "version": importlib.import_module("latent_anything.artifact_store")._SCHEMA_VERSION,
            "magic": importlib.import_module("latent_anything.artifact_store")._MAGIC.decode("ascii"),
            "fixture_identity": ArtifactStore._identity(  # pyright: ignore[reportPrivateUsage]
                "latent-value", hashlib.sha256(b"portable-bytes").hexdigest(), {"v": 1}
            ),
        },
        "run_record": {
            "version": (
                f"schema-v{importlib.import_module('latent_anything._run_record_codec').RUN_RECORD_SCHEMA_VERSION}"
            ),
            "canonical_fixture_digest": hashlib.sha256(
                json.dumps(record.to_dict(), sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "migration_fixture_digest": hashlib.sha256(
                json.dumps(migrate_run_record(legacy_payload), sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
        },
        "disk_cache": {
            "version": importlib.import_module("latent_anything.runtime.disk_cache")._SCHEMA_VERSION,
            "bound_key_fixture_digest": make_disk_cache_key(
                key, plugin_identity="plugin@1", checkpoint_identity="ckpt-a", behavior_state_identity="state-a"
            ),
        },
    }


def async_pairs() -> dict[str, object]:
    package = importlib.import_module("latent_anything")
    classes = {id(value): value for name in package.__all__ if inspect.isclass(value := getattr(package, name))}
    pairs = []
    for value in classes.values():
        for async_name in dir(value):
            if not async_name.endswith("_async") or async_name.startswith("_"):
                continue
            async_value = getattr(value, async_name)
            sync_name = async_name[:-6]
            sync_value = getattr(value, sync_name, None)
            if (
                sync_name.startswith("_")
                or not (inspect.iscoroutinefunction(async_value) or inspect.isasyncgenfunction(async_value))
                or not callable(sync_value)
            ):
                continue
            pairs.append(
                {
                    "class": f"{value.__module__}.{value.__name__}",
                    "sync": f"{value.__name__}.{sync_name}",
                    "async": f"{value.__name__}.{async_name}",
                    "sync_signature": signature(sync_value),
                    "async_signature": signature(async_value),
                    "sync_is_coroutine": inspect.iscoroutinefunction(sync_value),
                    "async_is_coroutine": inspect.iscoroutinefunction(async_value),
                    "async_is_async_generator": inspect.isasyncgenfunction(async_value),
                }
            )
    pairs.sort(key=lambda item: (str(item["class"]), str(item["sync"])))
    return {"count": len(pairs), "pairs": pairs}


def exceptions() -> dict[str, object]:
    modules = [
        "latent_anything",
        "latent_anything.artifact_store",
        "latent_anything.runtime.disk_cache",
        "latent_anything.run_record",
        "latent_anything.portable",
        "latent_anything.portable_results",
        "latent_anything.plugin_discovery",
        "latent_anything.experiment_recorder",
    ]
    classes: dict[int, type[BaseException]] = {}
    for module_name in modules:
        module = importlib.import_module(module_name)
        for name in getattr(module, "__all__", []):
            value = getattr(module, name, None)
            if inspect.isclass(value) and issubclass(value, BaseException) and name.endswith(("Error", "Exception")):
                classes.setdefault(id(value), cast(type[BaseException], value))
    entries = []
    for value in classes.values():
        base = next(
            (item.__name__ for item in value.__mro__[1:] if item in (ValueError, RuntimeError, Exception)), "Exception"
        )
        entries.append({"name": value.__name__, "module": value.__module__, "base": base})
    entries.sort(key=lambda item: (str(item["module"]), str(item["name"])))
    return {"count": len(entries), "entries": entries}

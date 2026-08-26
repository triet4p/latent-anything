"""Dynamic public API-freeze inventory sections A and C through H."""

from __future__ import annotations

import dataclasses
import importlib
import inspect
import tomllib
from pathlib import Path
from types import ModuleType
from typing import cast

from latent_anything._api_freeze_runtime import json_value, symbol

ROOT = Path(__file__).resolve().parents[2]
CANONICAL_ADDITIONS = ("AnalysisMethod", "Intervention", "InterventionPipeline")
LEGACY_REPLACEMENTS = {
    "Method": "AnalysisMethod",
    "BMethod": "Intervention",
    "ManipulationPipeline": "InterventionPipeline",
}


def public_surface() -> dict[str, object]:
    package = importlib.import_module("latent_anything")
    current = [symbol(name, getattr(package, name), import_path="latent_anything") for name in package.__all__]
    baseline = [item for item in current if item["name"] not in CANONICAL_ADDITIONS]
    canonical = [dict(item, name=LEGACY_REPLACEMENTS.get(str(item["name"]), str(item["name"]))) for item in baseline]
    return {
        "current_top_level": current,
        "current_count": len(current),
        "canonical_stable_surface": canonical,
        "canonical_stable_count": len(canonical),
        "canonical_top_level_additions": list(CANONICAL_ADDITIONS),
        "note": (
            "BMethod was not a top-level beta export; its canonical replacement is frozen through the methods "
            "submodule and additive top-level export."
        ),
    }


def _module_exports(module_name: str, module: ModuleType, names: list[str] | None = None) -> dict[str, object]:
    exported = list(module.__all__) if hasattr(module, "__all__") else (names or [])
    return {"module": module_name, "exports": exported, "count": len(exported)}


def submodule_surface() -> dict[str, object]:
    modules = [
        ("latent_anything.methods", None),
        ("latent_anything.pipeline", None),
        ("latent_anything.adapters", None),
        ("latent_anything.runtime", None),
        ("latent_anything.registry", None),
        ("latent_anything.methods.protocols", ["AnalysisMethod", "Method"]),
        ("latent_anything.methods.b_protocols", ["Intervention", "BMethod"]),
        ("latent_anything.manipulation_pipeline", ["InterventionPipeline", "ManipulationPipeline"]),
    ]
    return {"modules": [_module_exports(name, importlib.import_module(name), names) for name, names in modules]}


def registry() -> dict[str, object]:
    from latent_anything.registry import GLOBAL_REGISTRY

    rows = [
        {
            "kind": entry.kind,
            "name": entry.name,
            "factory_module": getattr(entry.factory, "__module__", "<unknown>"),
            "factory_name": getattr(entry.factory, "__name__", type(entry.factory).__name__),
        }
        for entry in sorted(GLOBAL_REGISTRY.list(), key=lambda item: (item.kind, item.name))
    ]
    return {"count": len(rows), "rows": rows}


def plugin_groups() -> dict[str, object]:
    from latent_anything.plugin_groups import CANONICAL_ENTRY_POINT_GROUPS

    return {
        "api_version": "1",
        "groups": list(CANONICAL_ENTRY_POINT_GROUPS),
        "count": len(CANONICAL_ENTRY_POINT_GROUPS),
    }


def profiles() -> dict[str, object]:
    with (ROOT / "pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream)["project"]
    names = list(project["optional-dependencies"])
    return {"profiles": names, "count": len(names), "order_policy": "pyproject declaration order"}


def model_fields(model: type[object]) -> list[dict[str, object]]:
    pydantic_fields = getattr(model, "model_fields", None)
    if isinstance(pydantic_fields, dict):
        return [
            {
                "name": name,
                "required": bool(field.is_required()),
                "default": json_value(field.default),
                "default_factory": "<factory>" if field.default_factory is not None else None,
            }
            for name, field in pydantic_fields.items()
        ]
    if dataclasses.is_dataclass(model):
        return [
            {
                "name": field.name,
                "required": field.default is dataclasses.MISSING and field.default_factory is dataclasses.MISSING,
                "default": json_value(field.default),
                "default_factory": "<factory>" if field.default_factory is not dataclasses.MISSING else None,
            }
            for field in dataclasses.fields(model)
        ]
    return []


def config_schemas() -> dict[str, object]:
    package = importlib.import_module("latent_anything")
    names = [name for name in package.__all__ if name.endswith(("Config", "Spec", "Limits"))]
    schemas = [
        {"name": name, "module": getattr(package, name).__module__, "fields": model_fields(getattr(package, name))}
        for name in names
    ]
    benchmark = importlib.import_module("latent_anything.integrations.lerobot_benchmark").SimulationBenchmarkConfig
    schemas.append(
        {"name": "SimulationBenchmarkConfig", "module": benchmark.__module__, "fields": model_fields(benchmark)}
    )
    return {"count": len(schemas), "schemas": schemas}


def dataclass_schemas() -> dict[str, object]:
    package = importlib.import_module("latent_anything")
    schemas = []
    for name in package.__all__:
        value = getattr(package, name)
        if inspect.isclass(value) and dataclasses.is_dataclass(value):
            schemas.append(
                {"name": name, "module": value.__module__, "fields": model_fields(cast(type[object], value))}
            )
    return {"count": len(schemas), "schemas": schemas}

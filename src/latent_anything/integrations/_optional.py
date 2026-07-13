"""Actionable lazy imports for optional integration extras."""

from __future__ import annotations

from importlib import import_module
from types import ModuleType


def require_optional(module_name: str, *, extra: str) -> ModuleType:
    """Import an optional backend or explain the exact extra required."""

    try:
        return import_module(module_name)
    except ModuleNotFoundError as error:
        if error.name != module_name:
            raise
        msg = f"Optional backend {module_name!r} is unavailable. Install with: uv sync --extra {extra}"
        raise ImportError(msg) from error

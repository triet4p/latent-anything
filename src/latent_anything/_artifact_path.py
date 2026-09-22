"""Resolve frozen repository artifacts from any installation layout.

The frozen schema and taxonomy documents live in the repository ``artifacts/``
directory. The library must locate them both in the developer source layout
(package under ``<root>/src``) and in an installed layout (package inside a
project-local virtual environment under ``<root>/.venv``), so clean-environment
execution never depends on the editable import path or on the process working
directory. Loaders still validate file content, so resolving to the wrong
document fails closed on schema identity instead of silently trusting it.
"""

from __future__ import annotations

from pathlib import Path

_PACKAGE_DIR = Path(__file__).resolve().parent


def resolve_artifact_path(filename: str) -> Path:
    """Return the frozen artifact path for ``filename``.

    Walks upward from the installed package until a directory containing
    ``artifacts/<filename>`` exists. When no such directory exists (for example
    a wheel installed outside any project checkout), the historical
    developer-layout location is returned so the loader raises its fail-closed
    "cannot be loaded" error against the same path it always has.
    """
    for directory in (_PACKAGE_DIR, *_PACKAGE_DIR.parents):
        candidate = directory / "artifacts" / filename
        if candidate.is_file():
            return candidate
    return _PACKAGE_DIR.parents[1] / "artifacts" / filename

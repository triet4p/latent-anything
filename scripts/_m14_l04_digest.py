"""Canonical hashing and provenance helpers for L04 envelopes."""

from __future__ import annotations

import hashlib
import platform
import re
import subprocess
from collections.abc import Mapping
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from scripts._m14_l04_contract_common import canonical_json_bytes

SHA1_RE = re.compile(r"^[0-9a-f]{40}$")


def canonical_digest(value: dict[str, Any], field: str) -> str:
    unsigned = dict(value)
    unsigned.pop(field, None)
    return hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()


def code_sha() -> str:
    """Return the current committed SHA or fail closed."""
    try:
        value = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("committed code SHA is unavailable") from error
    if not SHA1_RE.fullmatch(value):
        raise RuntimeError("committed code SHA is not a 40-character hexadecimal value")
    return value


def source_digests() -> dict[str, Any]:
    """Digest every L04 implementation/contract Python file deterministically."""
    root = Path(__file__).resolve().parent
    names = sorted(path.name for path in root.glob("*m14_l04_*.py"))
    names.append("m14_l04_explanations.py")
    files = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in sorted(set(names))}
    implementation = source_map_digest(files)
    return {
        "runner_source_sha256": files["m14_l04_explanations.py"],
        "contract_source_sha256": files["m14_l04_contract.py"],
        "implementation_source_sha256": implementation,
        "implementation_source_files": files,
    }


def source_map_digest(files: Mapping[str, str]) -> str:
    """Hash the canonical filename-to-SHA256 map, independent of live bytes."""
    return hashlib.sha256(canonical_json_bytes(dict(files))).hexdigest()


def runtime_versions() -> dict[str, str]:
    result = {"python": platform.python_version(), "platform": platform.platform()}
    for package in ("numpy", "torch", "transformers", "tokenizers"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = "not-installed"
    return result

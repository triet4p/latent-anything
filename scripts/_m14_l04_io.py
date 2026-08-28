"""Atomic caller-directory JSON persistence for L04 artifacts."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any


def safe_write(path: Path, value: dict[str, Any]) -> None:
    """Atomically write one UTF-8 JSON object below the caller's directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n").encode()
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
        handle.write(payload)
        temporary = Path(handle.name)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)

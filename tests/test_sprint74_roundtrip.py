"""Cross-process Sprint 74 portable artifact reproduction test."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_cross_process_portable_behavior_parity_reproduction() -> None:
    script = Path(__file__).parents[1] / "scripts" / "sprint74_portable_roundtrip.py"
    result = subprocess.run([sys.executable, str(script)], check=True, capture_output=True, text=True)
    report = json.loads(result.stdout)
    assert report["status"] == "pass"
    assert report["child"]["cache_hit"] is True
    assert report["artifact_bytes"] > 0
    assert report["elapsed_seconds"] >= 0.0

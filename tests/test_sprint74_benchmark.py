"""Smoke test for the offline Sprint 74 benchmark contract."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_offline_artifact_benchmark_reports_declared_metrics() -> None:
    script = Path(__file__).parents[1] / "scripts" / "sprint74_artifact_benchmark.py"
    result = subprocess.run([sys.executable, str(script)], check=True, capture_output=True, text=True)
    report = json.loads(result.stdout)
    expected = {
        "payload_bytes",
        "stored_artifact_bytes",
        "arrow_encode_us",
        "arrow_decode_us",
        "artifact_write_us",
        "artifact_read_us",
        "cache_set_us",
        "cache_get_us",
        "in_memory_copy_us",
    }
    assert set(report) == expected
    assert report["payload_bytes"] > 0
    assert report["stored_artifact_bytes"] > report["payload_bytes"]
    assert all(report[name] >= 0 for name in expected if name.endswith("_us"))

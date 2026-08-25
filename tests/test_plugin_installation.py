"""Integration proof for a separately installed hello-world plugin."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "sprint73_hello_plugin"


def test_separately_installed_hello_plugin_is_discovered_and_constructed(tmp_path: Path) -> None:
    """Install the fixture distribution, then exercise it in a clean child."""

    uv = shutil.which("uv")
    assert uv is not None, "Sprint 73 installation proof requires uv"
    fixture_copy = tmp_path / "fixture-source"
    shutil.copytree(FIXTURE_DIR, fixture_copy)
    target = tmp_path / "installed-plugin"
    install = subprocess.run(
        [
            uv,
            "pip",
            "install",
            "--offline",
            "--no-build-isolation",
            "--target",
            str(target),
            "--no-deps",
            str(fixture_copy),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert install.returncode == 0, install.stdout + install.stderr
    assert list(target.glob("latent_anything_hello_plugin-*.dist-info"))

    child_code = """
import json
import sys

from latent_anything.config import ObjectSpec, build_from_config
from latent_anything.plugin_discovery import list_entry_points, load_entry_points
from latent_anything.registry import Registry

before_listing = "latent_anything_hello" in sys.modules
listing = list_entry_points()
after_listing = "latent_anything_hello" in sys.modules
registry = Registry("fixture-child")
report = load_entry_points(registry=registry)
after_loading = "latent_anything_hello" in sys.modules
adapter = build_from_config(
    ObjectSpec(kind="adapter", name="hello-world", params={"prefix": "hi"}),
    registry=registry,
)
entry = registry.lookup("hello-world")
fixture_listing = [item for item in listing if item.name == "hello-world"]
print(json.dumps({
    "before_listing": before_listing,
    "after_listing": after_listing,
    "after_loading": after_loading,
    "listing_count": len(fixture_listing),
    "name": fixture_listing[0].name,
    "group": fixture_listing[0].group,
    "distribution": fixture_listing[0].distribution,
    "version": fixture_listing[0].version,
    "loaded": [item.name for item in report.loaded],
    "issues": [issue.reason for issue in report.issues],
    "result": adapter("world"),
    "source": entry.metadata["source"],
    "entry_point_group": entry.metadata["entry_point_group"],
    "entry_point_value": entry.metadata["entry_point_value"],
    "plugin_api_version": entry.metadata["plugin_api_version"],
    "version_metadata": entry.metadata["version"],
}))
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((str(target), str(Path(__file__).parents[1] / "src")))
    env["PYTHONNOUSERSITE"] = "1"
    completed = subprocess.run(
        [sys.executable, "-c", child_code],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = json.loads(completed.stdout)
    assert result == {
        "before_listing": False,
        "after_listing": False,
        "after_loading": True,
        "listing_count": 1,
        "name": "hello-world",
        "group": "latent_anything.adapter",
        "distribution": "latent-anything-hello-plugin",
        "version": "0.1.0",
        "loaded": ["hello-world"],
        "issues": [],
        "result": "hi:world",
        "source": "external",
        "entry_point_group": "latent_anything.adapter",
        "entry_point_value": "latent_anything_hello:HelloAdapter",
        "plugin_api_version": "1",
        "version_metadata": "0.1.0",
    }

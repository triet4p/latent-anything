"""Build and verify the deterministic Sprint 78 API-freeze snapshot."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path
from typing import cast

from latent_anything._api_freeze_inventory import (
    config_schemas,
    dataclass_schemas,
    plugin_groups,
    profiles,
    public_surface,
    registry,
    submodule_surface,
)
from latent_anything._api_freeze_runtime import aliases, async_pairs, cli_contract, exceptions, serialization

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_PATH = ROOT / "artifacts" / "api_freeze_snapshot_0.1.0b1.json"


def build_snapshot() -> dict[str, object]:
    sections = {
        "A_public_surface": public_surface(),
        "B_beta_compatibility": aliases(),
        "C_submodule_reexports": submodule_surface(),
        "D_registry": registry(),
        "E_plugin_groups": plugin_groups(),
        "F_optional_profiles": profiles(),
        "G_config_schemas": config_schemas(),
        "H_dataclass_schemas": dataclass_schemas(),
        "I_cli": cli_contract(),
        "J_serialization": serialization(),
        "K_sync_async": async_pairs(),
        "L_exceptions": exceptions(),
    }
    payload: dict[str, object] = {
        "schema_version": 1,
        "package_version": importlib.import_module("latent_anything").__version__,
        "generated_by": "scripts/api_freeze_snapshot.py",
        "normalization": (
            "sorted JSON keys; declaration order only where contractually public; no repr/address/object identity"
        ),
        "regenerate": "uv run python scripts/api_freeze_snapshot.py --write",
        "verify": "uv run python scripts/api_freeze_snapshot.py --check",
        "sections": sections,
    }
    payload["section_digests"] = {
        name: hashlib.sha256(encoded(section).encode("utf-8")).hexdigest() for name, section in sections.items()
    }
    encoded_payload = json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    payload["snapshot_sha256"] = hashlib.sha256(encoded_payload.encode("utf-8")).hexdigest()
    return payload


def encoded(snapshot: dict[str, object]) -> str:
    return json.dumps(snapshot, ensure_ascii=True, indent=2, sort_keys=True) + "\n"


def _diff_paths(expected: object, current: object, path: str) -> list[str]:
    if isinstance(expected, dict) and isinstance(current, dict):
        differences: list[str] = []
        for key in sorted(set(expected) | set(current)):
            if key == "snapshot_sha256" and path == "snapshot":
                continue
            child = f"{path}.{key}" if path else str(key)
            if key not in expected or key not in current:
                differences.append(child)
            else:
                differences.extend(_diff_paths(expected[key], current[key], child))
        return differences
    if isinstance(expected, list) and isinstance(current, list):
        differences = []
        for index in range(max(len(expected), len(current))):
            child = f"{path}[{index}]"
            if index >= len(expected) or index >= len(current):
                differences.append(child)
            else:
                differences.extend(_diff_paths(expected[index], current[index], child))
        return differences
    return [] if expected == current else [path]


def compare_snapshot(expected: dict[str, object], current: dict[str, object]) -> list[str]:
    """Return stable section/name paths for reviewed snapshot drift."""

    return _diff_paths(expected, current, "snapshot")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--write", action="store_true")
    action.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    current = build_snapshot()
    if args.write:
        SNAPSHOT_PATH.write_text(encoded(current), encoding="utf-8")
        print(f"wrote {SNAPSHOT_PATH.relative_to(ROOT)} ({current['snapshot_sha256']})")
        return 0
    if not SNAPSHOT_PATH.exists():
        print(f"missing snapshot: {SNAPSHOT_PATH}", file=sys.stderr)
        return 2
    expected = cast(dict[str, object], json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8")))
    differences = compare_snapshot(expected, current)
    if differences:
        for difference in differences:
            print(f"drift: {difference}", file=sys.stderr)
        return 1
    print(f"snapshot clean ({current['snapshot_sha256']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Report canonical registry kinds for repository-owned JSON-compatible config files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from latent_anything.registry_aliases import migration_record


def _walk(value: Any, records: list[dict[str, str | bool]]) -> None:
    if isinstance(value, dict):
        if isinstance(value.get("kind"), str) and isinstance(value.get("name"), str):
            records.append(migration_record(value["kind"]))
        for child in value.values():
            _walk(child, records)
    elif isinstance(value, list):
        for child in value:
            _walk(child, records)


def main() -> int:
    """Emit a JSON migration report without modifying config files."""

    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path, nargs="+", help="JSON config files to inspect")
    args = parser.parse_args()
    records: list[dict[str, str | bool]] = []
    for path in args.config:
        _walk(json.loads(path.read_text(encoding="utf-8")), records)
    print(json.dumps({"records": records, "migrations": sum(record["migrated"] for record in records)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

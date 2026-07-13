"""Report canonical registry kinds for repository-owned JSON-compatible config files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

from latent_anything.registry_aliases import migration_record

JsonValue = dict[str, "JsonValue"] | list["JsonValue"] | str | int | float | bool | None


def _walk(value: JsonValue, records: list[dict[str, str | bool]]) -> None:
    if isinstance(value, dict):
        kind = value.get("kind")
        name = value.get("name")
        if isinstance(kind, str) and isinstance(name, str):
            records.append(migration_record(kind))
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
        _walk(cast(JsonValue, json.loads(path.read_text(encoding="utf-8"))), records)
    migrated = sum(1 for record in records if record["migrated"] is True)
    print(json.dumps({"records": records, "migrations": migrated}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

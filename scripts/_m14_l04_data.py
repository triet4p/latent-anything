"""Offline fixture metadata for the L04 dispatch envelope."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from scripts._m14_l04_fixture_contract import fixture_digests


def fixture_metadata(plan: Mapping[str, Any], raw: bytes, rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "path": str(plan["fixture"]["path"]),
        "rows": len(rows),
        **fixture_digests(raw, rows),
    }

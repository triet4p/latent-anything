"""Canonical offline checker facade for the M14 L04 explanation lane."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.m14_l04_contract import FIXTURE_PATH, PLAN_PATH, load_and_validate


def main(argv: list[str] | None = None) -> None:
    """Validate the frozen contract; real execution is intentionally not implemented here."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="run deterministic offline plan/fixture validation")
    parser.add_argument("--plan", type=Path, default=PLAN_PATH)
    parser.add_argument("--fixture", type=Path, default=FIXTURE_PATH)
    args = parser.parse_args(argv)
    if not args.check:
        parser.error("only --check is available until the separately authorized real-run preflight")
    print(json.dumps(load_and_validate(args.plan, args.fixture), sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()

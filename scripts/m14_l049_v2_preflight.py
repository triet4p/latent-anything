"""Validate canonical Stage B inputs without evaluating the holdout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts._m14_l049_v2_inputs import validate_canonical_stage_b_inputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--require-tracked", action="store_true")
    args = parser.parse_args(argv)
    errors = validate_canonical_stage_b_inputs(args.repo_root, require_tracked=args.require_tracked)
    if errors:
        print(json.dumps({"stage": "stage_b_input_preflight", "status": "FAIL", "error_codes": errors}))
        return 65
    print(json.dumps({"stage": "stage_b_input_preflight", "status": "PASS", "evaluation": "not_run"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

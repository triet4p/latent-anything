"""Stable offline contract facade for the frozen M14 L04 explanation plan.

The implementation is split by responsibility into private plan, fixture, and
tokenization modules. Keep imports from this module stable for the checker and
its callers; real tokenizer/model resolution remains outside the offline lane.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from scripts._m14_l04_contract_common import (
    ContractValidationError,
    canonical_json_bytes,
    digest_bytes,
    load_json_bytes,
)
from scripts._m14_l04_fixture_contract import (
    FIXTURE_PAIR_SCHEMA,
    FIXTURE_PATH,
    FIXTURE_ROW_KEYS,
    FIXTURE_SPLIT_SCHEMA,
    content_digest,
    fixture_digests,
    pair_digest,
    read_fixture,
    split_digest,
    validate_fixture,
)
from scripts._m14_l04_plan_contract import (
    EXPECTED_PLAN_KEYS,
    EXPECTED_RECORD_IDS,
    EXPECTED_USE_CASES,
    PLAN_PATH,
    SCHEMA_VERSION,
    plan_digest,
    validate_plan,
)
from scripts._m14_l04_token_contract import validate_target_tokens


def load_plan(path: Path = PLAN_PATH) -> dict[str, Any]:
    """Load and validate the frozen plan without optional dependencies."""
    plan = load_json_bytes(path.read_bytes(), source="plan")
    errors = validate_plan(plan)
    if errors:
        raise ContractValidationError("; ".join(errors))
    return plan


def load_fixture(path: Path = FIXTURE_PATH) -> tuple[bytes, list[dict[str, Any]]]:
    """Load the raw UTF-8/LF fixture and its parsed rows."""
    return read_fixture(path)


def load_and_validate(plan_path: Path = PLAN_PATH, fixture_path: Path = FIXTURE_PATH) -> dict[str, str]:
    """Read the frozen plan/fixture without writing, downloading, or tokenizing.

    The offline check validates only the declared target strings and expected
    one-token count. It does not establish actual GPT-2 token cardinality;
    that belongs to the future real-run preflight via ``validate_target_tokens``.
    """
    plan = load_json_bytes(plan_path.read_bytes(), source="plan")
    errors = validate_plan(plan)
    raw, rows = read_fixture(fixture_path)
    errors.extend(validate_fixture(plan, raw, rows))
    if errors:
        raise ContractValidationError("; ".join(errors))
    return {"plan_sha256": plan_digest(plan), **fixture_digests(raw, rows)}


def check_plan(plan_path: Path = PLAN_PATH, fixture_path: Path = FIXTURE_PATH) -> dict[str, str]:
    """Compatibility facade for the canonical side-effect-free check."""
    return load_and_validate(plan_path, fixture_path)


__all__ = [
    "ContractValidationError",
    "EXPECTED_PLAN_KEYS",
    "EXPECTED_RECORD_IDS",
    "EXPECTED_USE_CASES",
    "FIXTURE_PAIR_SCHEMA",
    "FIXTURE_PATH",
    "FIXTURE_ROW_KEYS",
    "FIXTURE_SPLIT_SCHEMA",
    "PLAN_PATH",
    "SCHEMA_VERSION",
    "canonical_json_bytes",
    "check_plan",
    "content_digest",
    "digest_bytes",
    "fixture_digests",
    "load_and_validate",
    "load_fixture",
    "load_plan",
    "pair_digest",
    "plan_digest",
    "split_digest",
    "validate_fixture",
    "validate_plan",
    "validate_target_tokens",
]

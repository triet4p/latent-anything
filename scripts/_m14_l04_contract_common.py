"""Shared primitives for the private M14 L04 contract validators."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any


class ContractValidationError(ValueError):
    """Raised when a frozen L04 contract is malformed or inconsistent."""


def reject_non_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value!r}")


def load_json_bytes(raw: bytes, *, source: str) -> dict[str, Any]:
    if raw.startswith(b"\xef\xbb\xbf"):
        raise ContractValidationError(f"{source} must not contain a UTF-8 BOM")
    try:
        value = json.loads(raw.decode("utf-8"), parse_constant=reject_non_json_constant)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ContractValidationError(f"{source} is not strict UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ContractValidationError(f"{source} must be a JSON object")
    return value


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    """Serialize a JSON object using the plan's immutable canonical encoding."""
    try:
        encoded = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ContractValidationError("value cannot be canonically serialized") from exc
    return (encoded + "\n").encode("utf-8")


def digest_bytes(value: bytes) -> str:
    """Return a lowercase SHA-256 digest."""
    return hashlib.sha256(value).hexdigest()


def mapping(value: object, name: str, errors: list[str]) -> Mapping[str, Any] | None:
    if not isinstance(value, Mapping):
        errors.append(f"{name} must be an object")
        return None
    return value


def exact_keys(value: Mapping[str, Any], expected: frozenset[str], name: str, errors: list[str]) -> None:
    actual = frozenset(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        details = []
        if missing:
            details.append(f"missing {missing}")
        if extra:
            details.append(f"unexpected {extra}")
        errors.append(f"{name} schema mismatch ({'; '.join(details)})")

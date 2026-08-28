"""Validation rules for the frozen M14 L04 prompt-factor fixture."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from scripts._m14_l04_contract_common import (
    ContractValidationError,
    digest_bytes,
    mapping,
    reject_non_json_constant,
)

FIXTURE_PATH = Path(__file__).resolve().parents[1] / "artifacts/m14/l04-prompt-factor-fixture.jsonl"
FIXTURE_SPLIT_SCHEMA = "l04-fixture-split-v1"
FIXTURE_PAIR_SCHEMA = "l04-fixture-pair-v1"
FIXTURE_ROW_KEYS = frozenset(
    {"row_id", "group_id", "causal_pair_id", "condition", "split", "task", "prompt", "target_text", "factor_labels"}
)


def read_fixture(path: Path) -> tuple[bytes, list[dict[str, Any]]]:
    raw = path.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        raise ContractValidationError("fixture must not contain a UTF-8 BOM")
    if b"\r" in raw:
        raise ContractValidationError("fixture must use LF-only line endings")
    if not raw.endswith(b"\n"):
        raise ContractValidationError("fixture must end with exactly one LF per row")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ContractValidationError("fixture is not UTF-8") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.split("\n")[:-1], start=1):
        if not line:
            raise ContractValidationError(f"fixture row {line_number} is empty")
        try:
            value = json.loads(line, parse_constant=reject_non_json_constant)
        except (json.JSONDecodeError, ValueError) as exc:
            raise ContractValidationError(f"fixture row {line_number} is invalid JSON") from exc
        if not isinstance(value, dict):
            raise ContractValidationError(f"fixture row {line_number} must be an object")
        rows.append(value)
    return raw, rows


def _canonical_fixture_payload_bytes(value: Mapping[str, Any]) -> bytes:
    """Encode split/pair payloads while preserving their explicitly listed key order."""
    try:
        encoded = json.dumps(value, ensure_ascii=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ContractValidationError("fixture payload cannot be canonically serialized") from exc
    return (encoded + "\n").encode("utf-8")


def content_digest(raw: bytes) -> str:
    """Digest the exact fixture bytes, without parsing or normalization."""
    return digest_bytes(raw)


def _group_by(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key))].append(row)
    return grouped


def split_digest(rows: Sequence[Mapping[str, Any]]) -> str:
    """Digest the canonical split payload with its declared field order."""
    split_rows = [
        {"row_id": row["row_id"], "group_id": row["group_id"], "split": row["split"]}
        for row in sorted(rows, key=lambda item: (str(item["group_id"]), str(item["row_id"])))
    ]
    return digest_bytes(_canonical_fixture_payload_bytes({"schema": FIXTURE_SPLIT_SCHEMA, "rows": split_rows}))


def pair_digest(rows: Sequence[Mapping[str, Any]]) -> str:
    """Digest the canonical one-clean/one-corrupted causal-pair payload."""
    pair_rows: list[dict[str, Any]] = []
    for pair_id, pair in sorted(_group_by(rows, "causal_pair_id").items()):
        clean = next(row for row in pair if row.get("condition") == "clean")
        pair_rows.append(
            {
                "causal_pair_id": pair_id,
                "group_id": clean["group_id"],
                "clean_row_id": clean["row_id"],
                "corrupted_row_id": next(row["row_id"] for row in pair if row.get("condition") == "corrupted"),
                "split": clean["split"],
            }
        )
    return digest_bytes(_canonical_fixture_payload_bytes({"schema": FIXTURE_PAIR_SCHEMA, "pairs": pair_rows}))


def fixture_digests(raw: bytes, rows: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """Recompute content, split, and causal-pair digests independently."""
    return {
        "content_sha256": content_digest(raw),
        "split_sha256": split_digest(rows),
        "pair_sha256": pair_digest(rows),
    }


def validate_fixture(plan: Mapping[str, Any], raw: bytes, rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Validate fixture rows, splits, pairs, labels, and all three digests."""
    errors: list[str] = []
    fixture = mapping(plan.get("fixture"), "fixture", errors)
    if fixture is None:
        return errors
    if len(rows) != 24:
        errors.append("fixture must contain exactly 24 records")
    if len({row.get("row_id") for row in rows}) != len(rows):
        errors.append("fixture row_id values must be unique")
    if any(frozenset(row) != FIXTURE_ROW_KEYS for row in rows):
        errors.append("fixture rows must use the exact frozen schema")
    for index, row in enumerate(rows):
        if not all(
            isinstance(row.get(key), str) and row.get(key)
            for key in ("row_id", "group_id", "causal_pair_id", "condition", "split", "task", "prompt", "target_text")
        ):
            errors.append(f"fixture row {index} has missing/non-string identity or text")
        labels = row.get("factor_labels")
        if (
            not isinstance(labels, Mapping)
            or set(labels) != {"animal_cat", "tone_positive"}
            or any(
                not isinstance(value, int) or isinstance(value, bool) or value not in (0, 1)
                for value in labels.values()
            )
        ):
            errors.append(f"fixture row {index} has invalid factor labels")
            continue
        condition = row.get("condition")
        expected_target = " true" if condition == "clean" else " false" if condition == "corrupted" else None
        if expected_target is None or row.get("target_text") != expected_target:
            errors.append(f"fixture row {index} condition/target mismatch")
        if labels.get("tone_positive") != (1 if condition == "clean" else 0 if condition == "corrupted" else -1):
            errors.append(f"fixture row {index} condition/tone_positive mismatch")

    groups = _group_by(rows, "group_id")
    pairs = _group_by(rows, "causal_pair_id")
    if len(groups) != 12 or len(pairs) != 12:
        errors.append("fixture must contain exactly 12 groups and 12 causal pairs")
    train_groups = {f"g{i:02d}" for i in range(1, 9)}
    holdout_groups = {f"g{i:02d}" for i in range(9, 13)}
    for group_id, group_rows in groups.items():
        splits = {row.get("split") for row in group_rows}
        pair_ids = {row.get("causal_pair_id") for row in group_rows}
        if len(splits) != 1 or len(pair_ids) != 1:
            errors.append(f"group {group_id!r} crosses split or causal pair")
        expected_split = "train" if group_id in train_groups else "holdout" if group_id in holdout_groups else None
        if expected_split is None or splits != {expected_split}:
            errors.append(f"group {group_id!r} has an undeclared split")
    for pair_id, pair_rows in pairs.items():
        conditions = [str(row.get("condition")) for row in pair_rows]
        if len(pair_rows) != 2 or sorted(conditions) != ["clean", "corrupted"]:
            errors.append(f"pair {pair_id!r} must have exactly one clean and one corrupted row")
            continue
        if (
            len({row.get("group_id") for row in pair_rows}) != 1
            or len({row.get("split") for row in pair_rows}) != 1
            or len({row.get("task") for row in pair_rows}) != 1
        ):
            errors.append(f"pair {pair_id!r} crosses group, split, or task")

    try:
        digests = fixture_digests(raw, rows)
    except (KeyError, StopIteration, TypeError) as exc:
        errors.append(f"fixture digest payload cannot be constructed: {exc}")
    else:
        for key in ("content_sha256", "split_sha256", "pair_sha256"):
            if digests[key] != fixture.get(key):
                errors.append(f"fixture {key} does not match its canonical recomputation")
    return errors

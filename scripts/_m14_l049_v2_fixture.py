"""Deterministic v2 fixture authoring and validation.

The train fixture is public and checked in. The holdout fixture is generated
with the same function outside the repository and is supplied only to Stage B.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from scripts._m14_l049_v2_schema import (
    AUTHORING_MANIFEST_DIGEST_FIELD,
    HOLDOUT_GROUP_COUNT,
    PUBLIC_TRAIN_SEED,
    ROWS_PER_GROUP,
    TRAIN_GROUP_COUNT,
    V2_ROW_KEYS,
    canonical_fixture_bytes,
    canonical_json_bytes,
    digest_bytes,
)

TRAIN_FIXTURE_PATH = Path(__file__).resolve().parents[1] / "artifacts/m14/l04-l049-v2-train.jsonl"


def generate_rows(split: str, group_count: int, seed: int) -> list[dict[str, Any]]:
    """Generate distinct pair rows; ``seed`` is public for train only."""
    if split not in {"train", "holdout"} or group_count <= 0:
        raise ValueError("split and group_count are invalid")
    prefix = "trn" if split == "train" else "hld"
    rows: list[dict[str, Any]] = []
    for index in range(group_count):
        group = f"v2-{prefix}-g{index + 1:03d}"
        pair = f"v2-{prefix}-p{index + 1:03d}"
        family = f"{prefix}-family-{index + 1:03d}"
        vocabulary = f"{prefix}lex{seed:x}{index:03d}"
        base = f"classify {family} {vocabulary} premise{index:03d}"
        for condition, target, label in (("clean", " true", 1), ("corrupted", " false", 0)):
            rows.append(
                {
                    "row_id": f"{pair}-{condition}",
                    "group_id": group,
                    "causal_pair_id": pair,
                    "condition": condition,
                    "split": split,
                    "prompt_family": family,
                    "prompt": f"{base} condition-{condition} answer",
                    "target_text": target,
                    "factor_labels": {"clean_label": label, "tone_positive": label},
                }
            )
    return rows


def train_rows() -> list[dict[str, Any]]:
    return generate_rows("train", TRAIN_GROUP_COUNT, PUBLIC_TRAIN_SEED)


def holdout_rows(seed: int) -> list[dict[str, Any]]:
    return generate_rows("holdout", HOLDOUT_GROUP_COUNT, int(seed))


def read_rows(path: Path) -> tuple[bytes, list[dict[str, Any]]]:
    raw = path.read_bytes()
    if b"\r" in raw or not raw.endswith(b"\n"):
        raise ValueError("v2 fixture must be LF-terminated")
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines()]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError("v2 fixture rows must be objects")
    return raw, rows


def fixture_digest(rows: Sequence[Mapping[str, Any]]) -> str:
    return digest_bytes(canonical_fixture_bytes(rows))


def authoring_manifest_digest(manifest: Mapping[str, Any]) -> str:
    unsigned = dict(manifest)
    unsigned.pop(AUTHORING_MANIFEST_DIGEST_FIELD, None)
    return hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()


def validate_rows(rows: Sequence[Mapping[str, Any]], split: str, expected_groups: int) -> list[str]:
    errors: list[str] = []
    if len(rows) != expected_groups * ROWS_PER_GROUP:
        errors.append("v2 fixture row count is invalid")
    if any(not isinstance(row, Mapping) or list(row) != list(V2_ROW_KEYS) for row in rows):  # pyright: ignore[reportUnnecessaryIsInstance]
        errors.append("v2 fixture row key order is invalid")
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        if not isinstance(row, Mapping):  # pyright: ignore[reportUnnecessaryIsInstance]
            continue
        groups.setdefault(str(row.get("group_id")), []).append(row)
        if row.get("split") != split or not isinstance(row.get("prompt"), str) or not row.get("prompt"):
            errors.append("v2 fixture split or prompt is invalid")
        if not all(
            isinstance(row.get(key), str) and row.get(key)
            for key in ("row_id", "group_id", "causal_pair_id", "prompt_family", "target_text")
        ):
            errors.append("v2 fixture identifiers/text are invalid")
        labels = row.get("factor_labels")
        if (
            not isinstance(labels, Mapping)
            or set(labels) != {"clean_label", "tone_positive"}
            or any(
                isinstance(labels.get(key), bool) or labels.get(key) not in {0, 1}
                for key in ("clean_label", "tone_positive")
            )
        ):
            errors.append("v2 fixture labels are invalid")
    if len(groups) != expected_groups or any(len(pair) != ROWS_PER_GROUP for pair in groups.values()):
        errors.append("v2 fixture group cardinality is invalid")
    for group_rows in groups.values():
        if {row.get("condition") for row in group_rows} != {"clean", "corrupted"}:
            errors.append("v2 fixture group must contain one clean and one corrupted row")
        if len({row.get("causal_pair_id") for row in group_rows}) != 1:
            errors.append("v2 fixture group must contain one causal pair")
        by_condition = {row.get("condition"): row for row in group_rows}
        if set(by_condition) == {"clean", "corrupted"}:
            clean, corrupted = by_condition["clean"], by_condition["corrupted"]
            if clean.get("target_text") != " true" or corrupted.get("target_text") != " false":
                errors.append("v2 fixture target labels are inconsistent")
            if clean.get("factor_labels") != {"clean_label": 1, "tone_positive": 1} or corrupted.get(
                "factor_labels"
            ) != {"clean_label": 0, "tone_positive": 0}:
                errors.append("v2 fixture clean/corrupted label relationship is invalid")
            if clean.get("prompt_family") != corrupted.get("prompt_family"):
                errors.append("v2 fixture causal pair prompt family is invalid")
            if clean.get("causal_pair_id") != corrupted.get("causal_pair_id"):
                errors.append("v2 fixture causal pair linkage is invalid")
    if len({row.get("row_id") for row in rows if isinstance(row, Mapping)}) != len(rows):  # pyright: ignore[reportUnnecessaryIsInstance]
        errors.append("v2 fixture row IDs are duplicated")
    if len({row.get("causal_pair_id") for row in rows if isinstance(row, Mapping)}) != expected_groups:  # pyright: ignore[reportUnnecessaryIsInstance]
        errors.append("v2 fixture pair IDs are duplicated")
    prompt_texts = [row.get("prompt") for row in rows if isinstance(row, Mapping)]  # pyright: ignore[reportUnnecessaryIsInstance]
    prompt_hashes = [hashlib.sha256(str(prompt).encode("utf-8")).hexdigest() for prompt in prompt_texts]
    if len(set(prompt_texts)) != len(prompt_texts) or len(set(prompt_hashes)) != len(prompt_hashes):
        errors.append("v2 fixture prompt text/hash values are not unique")
    if len({str(row.get("prompt_family")) for row in rows}) != expected_groups:
        errors.append("v2 fixture prompt families must be unique")
    return errors


def validate_fixture(train: Sequence[Mapping[str, Any]], holdout: Sequence[Mapping[str, Any]]) -> list[str]:
    """Validate both splits and the frozen cross-split near-duplicate policy."""
    errors = validate_rows(train, "train", TRAIN_GROUP_COUNT) + validate_rows(holdout, "holdout", HOLDOUT_GROUP_COUNT)
    train_ids = {row.get(key) for row in train for key in ("row_id", "group_id", "causal_pair_id")}
    holdout_ids = {row.get(key) for row in holdout for key in ("row_id", "group_id", "causal_pair_id")}
    if train_ids & holdout_ids:
        errors.append("v2 train/holdout identifiers are not disjoint")
    train_families = {row.get("prompt_family") for row in train}
    holdout_families = {row.get("prompt_family") for row in holdout}
    if train_families & holdout_families:
        errors.append("v2 train/holdout prompt families are not disjoint")
    token_sets = []
    for rows in (train, holdout):
        token_sets.append(
            {
                token
                for row in rows
                if isinstance(row.get("prompt"), str)
                for token in re.findall(r"[A-Za-z0-9_-]+", row["prompt"])
                if "family" in token or "lex" in token
            }
        )
    if token_sets[0] & token_sets[1]:
        errors.append("v2 train/holdout vocabulary is not disjoint")
    prompts = [row.get("prompt") for row in (*train, *holdout)]
    if len(set(prompts)) != len(prompts):
        errors.append("v2 train/holdout prompt text is not unique")
    return errors


__all__ = [
    "TRAIN_FIXTURE_PATH",
    "authoring_manifest_digest",
    "fixture_digest",
    "generate_rows",
    "holdout_rows",
    "read_rows",
    "train_rows",
    "validate_rows",
    "validate_fixture",
]

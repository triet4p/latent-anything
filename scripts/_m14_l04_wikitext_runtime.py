"""Pinned, bounded WikiText provisioning for the tuned-lens runner."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, cast

from scripts.m14_l04_wikitext_manifest import (
    FROZEN_CONTRACT,
    SPLITS,
    ManifestContract,
    ManifestError,
    validate_manifest,
)


def read_manifest(path: Path, *, contract: ManifestContract = FROZEN_CONTRACT) -> tuple[dict[str, Any], str]:
    """Read and validate a manifest, returning its exact-file SHA-256."""
    try:
        raw = path.read_bytes()
        manifest = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ManifestError("tuned-lens manifest is not valid UTF-8 JSON") from exc
    if not isinstance(manifest, dict):
        raise ManifestError("tuned-lens manifest must be an object")
    validate_manifest(manifest, contract=contract)
    return manifest, hashlib.sha256(raw).hexdigest()


def _indexed_split(value: object, split: str, expected_length: int) -> tuple[int, Callable[[int], object]]:
    """Require a bounded indexed dataset and never materialize an entire split."""
    if isinstance(value, (str, bytes, bytearray, Mapping)) or not hasattr(value, "__len__"):
        raise ManifestError(f"loader returned an unexpected {split} split shape")
    try:
        length = len(cast(Any, value))
    except (TypeError, ValueError) as exc:
        raise ManifestError(f"loader returned an un-sized {split} split") from exc
    if length != expected_length:
        raise ManifestError(f"{split} official row count mismatch: expected {expected_length}, got {length}")

    def get(index: int) -> object:
        try:
            return cast(Any, value)[index]
        except (IndexError, KeyError, TypeError) as exc:
            raise ManifestError(f"{split}[{index}] is unavailable") from exc

    return length, get


def load_selected_rows(
    manifest_path: Path,
    *,
    dataset_loader: Callable[..., object],
    contract: ManifestContract = FROZEN_CONTRACT,
) -> dict[str, list[dict[str, str]]]:
    """Scan each pinned split by index, retaining only selected text rows."""
    manifest, _manifest_sha256 = read_manifest(manifest_path, contract=contract)
    split_values = cast(Mapping[str, object], manifest["splits"])
    selected_rows: dict[str, list[dict[str, str]]] = {}
    expected_official = cast(Mapping[str, int], contract.official_rows)
    for split in SPLITS:
        dataset = dataset_loader(contract.dataset_id, contract.config, split=split, revision=contract.revision)
        length, get = _indexed_split(dataset, split, expected_official[split])
        split_mapping = cast(Mapping[str, object], split_values[split])
        selected = cast(list[Mapping[str, object]], split_mapping["selected"])
        expected_hashes = {int(cast(int, item["index"])): str(item["text_sha256"]) for item in selected}
        selected_by_index: dict[int, dict[str, str]] = {}
        selected_indices = set(expected_hashes)
        nonblank_count = 0
        for index in range(length):
            row = get(index)
            if not isinstance(row, Mapping):
                raise ManifestError(f"loader returned a non-object row in {split}[{index}]")
            text = row.get("text")
            if not isinstance(text, str):
                raise ManifestError(f"{split}[{index}] text is not a string")
            if text.strip():
                nonblank_count += 1
            if index in selected_indices:
                expected = expected_hashes[index]
                observed = hashlib.sha256(text.encode("utf-8", errors="strict")).hexdigest()
                if observed != expected:
                    raise ManifestError(f"{split}[{index}] text hash does not match manifest")
                selected_by_index[index] = {
                    "split": split,
                    "index": str(index),
                    "row_id": f"{split}:{index}",
                    "text": text,
                    "text_sha256": observed,
                }
        if nonblank_count != int(cast(Any, split_mapping["nonblank_rows"])):
            raise ManifestError(f"{split} nonblank row count does not match manifest")
        if set(selected_by_index) != selected_indices or len(selected_by_index) != int(
            cast(Any, split_mapping["selected_rows"])
        ):
            raise ManifestError(f"{split} selected row coverage does not match manifest")
        selected_rows[split] = [selected_by_index[int(cast(int, item["index"]))] for item in selected]
    return selected_rows


__all__ = ["load_selected_rows", "read_manifest"]

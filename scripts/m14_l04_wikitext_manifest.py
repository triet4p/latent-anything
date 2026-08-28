#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "datasets==4.8.5",
# ]
# ///
"""Acquire and validate the sanitized WikiText-2 manifest for M14 L04.

The acquisition entry point is deliberately network-gated and is not run by
the offline tests.  Raw corpus text is consumed only long enough to compute a
SHA-256 value; it is never placed in the manifest.  The two digest algorithms
are deliberately independent:

* ``content_sha256`` hashes compact UTF-8 JSON with non-blank counts and one
  row per selected item, sorted by ``(split order, original index)`` and
  containing only split, index, and ``text_sha256``.
* ``split_sha256`` hashes compact UTF-8 JSON with official/non-blank counts and
  one row per selected item, sorted the same way and containing only split and
  original index.

Both payloads use ``ensure_ascii=True``, separators ``(',', ':')``, and one
trailing LF.  The manifest contains metadata, counts, selected indices, and
text hashes only; it is not a corpus cache or a redistribution artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import tempfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict, cast

SCHEMA_VERSION = "m14-l04-wikitext-2-manifest-v1"
DATASET_ID = "Salesforce/wikitext"
CONFIG = "wikitext-2-raw-v1"
REVISION = "f776294184f13b8ff2337b3841cf9269a6216d1e"
SOURCE_URL = "https://huggingface.co/datasets/Salesforce/wikitext/tree/f776294184f13b8ff2337b3841cf9269a6216d1e/wikitext-2-raw-v1"
LICENSE = "CC BY-SA 3.0 and GFDL"
LICENSE_ACCESS = (
    "CC BY-SA 3.0 and GFDL as shown by the authoritative Salesforce/wikitext dataset card at the pinned revision; "
    "retain attribution and license metadata."
)
LICENSE_URL = "https://huggingface.co/datasets/Salesforce/wikitext"
FROZEN_PLAN_SHA256 = "f3c315e356af0ee54d4196cc365ee22bd997b069d18a3e72c6b479f94e0b3e1a"
PLAN_PATH = Path(__file__).resolve().parents[1] / "artifacts/m14/l04-explanations.plan.json"
SEED = 79
MAX_TOKENS_PER_ROW = 128
TRAIN_ROWS = 36_718
VALIDATION_ROWS = 3_760
TRAIN_SELECTION = 8_192
VALIDATION_SELECTION = 2_048
DATASETS_VERSION = "4.8.5"
SPLITS = ("train", "validation")


class ManifestError(ValueError):
    """Raised when acquisition data or a sanitized manifest is unsafe."""


class SelectedRow(TypedDict):
    index: int
    text_sha256: str


class SplitManifest(TypedDict):
    official_rows: int
    nonblank_rows: int
    selected_rows: int
    selected: list[SelectedRow]


@dataclass(frozen=True)
class ManifestContract:
    """Expected identity, sizes, and bounded selection for one manifest."""

    dataset_id: str = DATASET_ID
    config: str = CONFIG
    revision: str = REVISION
    license: str = LICENSE
    license_url: str = LICENSE_URL
    official_rows: Mapping[str, int] | None = None
    selection_rows: Mapping[str, int] | None = None
    seed: int = SEED
    max_tokens_per_row: int = MAX_TOKENS_PER_ROW

    def __post_init__(self) -> None:
        if self.official_rows is None:
            object.__setattr__(self, "official_rows", {"train": TRAIN_ROWS, "validation": VALIDATION_ROWS})
        if self.selection_rows is None:
            object.__setattr__(self, "selection_rows", {"train": TRAIN_SELECTION, "validation": VALIDATION_SELECTION})
        assert self.official_rows is not None
        assert self.selection_rows is not None
        if set(self.official_rows) != set(SPLITS) or set(self.selection_rows) != set(SPLITS):
            raise ManifestError("contract must declare exactly train and validation splits")
        if any(value < 0 for value in (*self.official_rows.values(), *self.selection_rows.values())):
            raise ManifestError("contract row counts must be non-negative")
        if any(self.selection_rows[name] > self.official_rows[name] for name in SPLITS):
            raise ManifestError("selection cannot exceed official split size")


FROZEN_CONTRACT = ManifestContract()


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_payload(payload: Mapping[str, object]) -> bytes:
    return (json.dumps(payload, ensure_ascii=True, separators=(",", ":")) + "\n").encode("utf-8")


def verify_frozen_plan(path: Path = PLAN_PATH) -> None:
    """Reject acquisition if the immutable L04 plan no longer has its digest."""
    try:
        plan = json.loads(path.read_bytes().decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ManifestError("frozen L04 plan cannot be read as UTF-8 JSON") from exc
    if not isinstance(plan, dict) or plan.get("plan_sha256") != FROZEN_PLAN_SHA256:
        raise ManifestError("frozen L04 plan binding is missing or altered")
    unsigned = dict(plan)
    unsigned.pop("plan_sha256")
    try:
        digest = _sha256(
            (json.dumps(unsigned, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        )
    except (TypeError, ValueError) as exc:
        raise ManifestError("frozen L04 plan cannot be canonically serialized") from exc
    if digest != FROZEN_PLAN_SHA256:
        raise ManifestError("frozen L04 plan canonical digest does not match its binding")


def _text_hash(text: object) -> str:
    if not isinstance(text, str):
        raise ManifestError("dataset row text must be a string")
    try:
        encoded = text.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise ManifestError("dataset row text is not valid UTF-8") from exc
    return _sha256(encoded)


def _is_blank(text: object) -> bool:
    return isinstance(text, str) and not text.strip()


def _selected_payload_rows(splits: Mapping[str, SplitManifest], *, include_hash: bool) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for split in SPLITS:
        for selected in splits[split]["selected"]:
            row: dict[str, object] = {"split": split, "index": selected["index"]}
            if include_hash:
                row["text_sha256"] = selected["text_sha256"]
            rows.append(row)
    return sorted(rows, key=lambda row: (SPLITS.index(cast(str, row["split"])), cast(int, row["index"])))


def content_sha256(splits: Mapping[str, SplitManifest]) -> str:
    """Return the canonical digest binding selected indices to text hashes."""
    return _sha256(
        _canonical_payload(
            {
                "schema": "m14-wikitext-content-v1",
                "nonblank_counts": {split: splits[split]["nonblank_rows"] for split in SPLITS},
                "rows": _selected_payload_rows(splits, include_hash=True),
            }
        )
    )


def split_sha256(splits: Mapping[str, SplitManifest]) -> str:
    """Return the canonical digest binding selected split/index membership."""
    return _sha256(
        _canonical_payload(
            {
                "schema": "m14-wikitext-split-v1",
                "official_counts": {split: splits[split]["official_rows"] for split in SPLITS},
                "nonblank_counts": {split: splits[split]["nonblank_rows"] for split in SPLITS},
                "rows": _selected_payload_rows(splits, include_hash=False),
            }
        )
    )


def _select_split(rows: Sequence[Mapping[str, object]], *, split: str, contract: ManifestContract) -> SplitManifest:
    candidates: list[tuple[str, int]] = []
    for index, row in enumerate(rows):
        if "text" not in row:
            raise ManifestError(f"{split}[{index}] is missing its text field")
        if _is_blank(row["text"]):
            continue
        candidates.append((_text_hash(row["text"]), index))
    candidates.sort(key=lambda item: (item[0], item[1]))
    selected = candidates[: cast(Mapping[str, int], contract.selection_rows)[split]]
    return {
        "official_rows": len(rows),
        "nonblank_rows": len(candidates),
        "selected_rows": len(selected),
        "selected": [{"index": index, "text_sha256": text_hash} for text_hash, index in selected],
    }


def _source_metadata(contract: ManifestContract) -> dict[str, object]:
    return {
        "dataset_id": contract.dataset_id,
        "config": contract.config,
        "revision": contract.revision,
        "source_url": SOURCE_URL,
        "license": contract.license,
        "license_access": LICENSE_ACCESS,
        "license_source_url": contract.license_url,
    }


def build_manifest(
    rows_by_split: Mapping[str, Sequence[Mapping[str, object]]],
    *,
    contract: ManifestContract = FROZEN_CONTRACT,
    datasets_version: str = DATASETS_VERSION,
    acquisition_tool: str = "datasets.load_dataset",
) -> dict[str, object]:
    """Build a sanitized manifest from rows without retaining any raw text."""
    if set(rows_by_split) != set(SPLITS):
        raise ManifestError("dataset must expose exactly train and validation splits")
    expected = cast(Mapping[str, int], contract.official_rows)
    splits = {split: _select_split(rows_by_split[split], split=split, contract=contract) for split in SPLITS}
    for split in SPLITS:
        if splits[split]["official_rows"] != expected[split]:
            raise ManifestError(f"{split} row count does not match the pinned official size")
    metadata = _source_metadata(contract)
    manifest: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "plan_sha256": FROZEN_PLAN_SHA256,
        "source": metadata,
        "selection": {
            "blank_policy": "drop rows where text.strip() is empty; retain all other text bytes exactly",
            "sort_key": "sha256(UTF-8(text)), then original split-local index",
            "selected_counts": dict(cast(Mapping[str, int], contract.selection_rows)),
            "seed": contract.seed,
            "seed_scope": "downstream tuned-lens training provenance; selection is deterministic hash/index sorting",
            "max_tokens_per_row": contract.max_tokens_per_row,
        },
        "tool": {"name": acquisition_tool, "datasets_version": datasets_version, "script_version": SCHEMA_VERSION},
        "splits": splits,
        "content_sha256": content_sha256(splits),
        "split_sha256": split_sha256(splits),
    }
    validate_manifest(manifest, contract=contract)
    return manifest


def _exact_keys(value: object, expected: set[str], label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ManifestError(f"{label} schema mismatch")
    return value


def _validate_selected(selected: object, *, split: str) -> list[SelectedRow]:
    if not isinstance(selected, list):
        raise ManifestError(f"{split}.selected must be a list")
    result: list[SelectedRow] = []
    seen: set[int] = set()
    for position, row in enumerate(selected):
        mapping = _exact_keys(row, {"index", "text_sha256"}, f"{split}.selected[{position}]")
        index = mapping["index"]
        digest = mapping["text_sha256"]
        if not isinstance(index, int) or isinstance(index, bool) or index < 0 or index in seen:
            raise ManifestError(f"{split}.selected contains duplicate or invalid indices")
        if not isinstance(digest, str) or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ManifestError(f"{split}.selected[{position}] has an invalid UTF-8 text hash")
        seen.add(index)
        result.append({"index": index, "text_sha256": digest})
    return result


def validate_manifest(manifest: Mapping[str, object], *, contract: ManifestContract = FROZEN_CONTRACT) -> None:
    """Fail closed on identity, leakage, counts, selected rows, and digests."""
    _exact_keys(
        manifest,
        {"schema_version", "plan_sha256", "source", "selection", "tool", "splits", "content_sha256", "split_sha256"},
        "manifest",
    )
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise ManifestError("manifest schema version mismatch")
    if manifest["plan_sha256"] != FROZEN_PLAN_SHA256:
        raise ManifestError("plan_sha256 does not match the frozen L04 plan")
    source = _exact_keys(
        manifest["source"],
        {"dataset_id", "config", "revision", "source_url", "license", "license_access", "license_source_url"},
        "source",
    )
    expected_source = {
        "dataset_id": contract.dataset_id,
        "config": contract.config,
        "revision": contract.revision,
        "source_url": SOURCE_URL,
        "license": contract.license,
        "license_access": LICENSE_ACCESS,
        "license_source_url": contract.license_url,
    }
    if dict(source) != expected_source:
        raise ManifestError("source identity, revision, or license metadata mismatch")
    selection = _exact_keys(
        manifest["selection"],
        {"blank_policy", "sort_key", "selected_counts", "seed", "seed_scope", "max_tokens_per_row"},
        "selection",
    )
    if (
        selection["blank_policy"] != "drop rows where text.strip() is empty; retain all other text bytes exactly"
        or selection["sort_key"] != "sha256(UTF-8(text)), then original split-local index"
        or selection["selected_counts"] != dict(cast(Mapping[str, int], contract.selection_rows))
        or selection["seed"] != contract.seed
        or selection["seed_scope"]
        != "downstream tuned-lens training provenance; selection is deterministic hash/index sorting"
        or selection["max_tokens_per_row"] != contract.max_tokens_per_row
    ):
        raise ManifestError("selection contract mismatch")
    tool = _exact_keys(manifest["tool"], {"name", "datasets_version", "script_version"}, "tool")
    if (
        not isinstance(tool["name"], str)
        or not tool["name"]
        or not isinstance(tool["datasets_version"], str)
        or tool["datasets_version"] != DATASETS_VERSION
        or tool["script_version"] != SCHEMA_VERSION
    ):
        raise ManifestError("tool provenance is invalid")
    splits_value = _exact_keys(manifest["splits"], set(SPLITS), "splits")
    splits: dict[str, SplitManifest] = {}
    expected_official = cast(Mapping[str, int], contract.official_rows)
    expected_selected = cast(Mapping[str, int], contract.selection_rows)
    for split in SPLITS:
        split_mapping = _exact_keys(
            splits_value[split], {"official_rows", "nonblank_rows", "selected_rows", "selected"}, split
        )
        official_rows = split_mapping["official_rows"]
        nonblank_rows = split_mapping["nonblank_rows"]
        selected_rows = split_mapping["selected_rows"]
        selected = _validate_selected(split_mapping["selected"], split=split)
        if (
            not isinstance(official_rows, int)
            or isinstance(official_rows, bool)
            or not isinstance(nonblank_rows, int)
            or isinstance(nonblank_rows, bool)
            or not isinstance(selected_rows, int)
            or isinstance(selected_rows, bool)
        ):
            raise ManifestError(f"{split} counts must be integers")
        if (
            official_rows != expected_official[split]
            or nonblank_rows < selected_rows
            or selected_rows != len(selected)
            or selected_rows != expected_selected[split]
        ):
            raise ManifestError(f"{split} counts do not match the frozen selection contract")
        if any(row["index"] >= official_rows for row in selected):
            raise ManifestError(f"{split} selected index is outside the official split")
        if selected != sorted(selected, key=lambda row: (row["text_sha256"], row["index"])):
            raise ManifestError(f"{split} selected rows are not in canonical hash/index order")
        splits[split] = {
            "official_rows": official_rows,
            "nonblank_rows": nonblank_rows,
            "selected_rows": selected_rows,
            "selected": selected,
        }
    content = manifest["content_sha256"]
    split_digest = manifest["split_sha256"]
    if not isinstance(content, str) or content != content_sha256(splits):
        raise ManifestError("content_sha256 does not match selected text hashes")
    if not isinstance(split_digest, str) or split_digest != split_sha256(splits):
        raise ManifestError("split_sha256 does not match selected indices")


def write_manifest(path: Path, manifest: Mapping[str, object], *, contract: ManifestContract = FROZEN_CONTRACT) -> None:
    """Atomically write a validated LF-only JSON manifest."""
    validate_manifest(manifest, contract=contract)
    payload = (json.dumps(manifest, ensure_ascii=True, separators=(",", ":"), sort_keys=True) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _materialize_split_rows(value: object, *, split: str) -> list[Mapping[str, object]]:
    if isinstance(value, (str, bytes, bytearray, Mapping)):
        raise ManifestError(f"loader returned an unexpected {split} split shape")
    try:
        raw_rows = list(cast(Iterable[object], value))
    except TypeError as exc:
        raise ManifestError(f"loader returned an unexpected {split} split shape") from exc
    rows: list[Mapping[str, object]] = []
    for index, row in enumerate(raw_rows):
        if not isinstance(row, Mapping):
            raise ManifestError(f"loader returned a non-object row in {split}[{index}]")
        rows.append(cast(Mapping[str, object], row))
    return rows


def acquire_manifest(
    *,
    output: Path,
    contract: ManifestContract = FROZEN_CONTRACT,
    dataset_loader: Callable[..., object] | None = None,
) -> dict[str, object]:
    """Download the pinned dataset exactly once and write its sanitized manifest."""
    verify_frozen_plan()
    installed_version = DATASETS_VERSION
    if dataset_loader is None:
        datasets = importlib.import_module("datasets")
        from importlib.metadata import PackageNotFoundError, version

        try:
            installed_version = version("datasets")
        except PackageNotFoundError as exc:
            raise ManifestError("the isolated acquisition environment is missing datasets") from exc
        if installed_version != DATASETS_VERSION:
            raise ManifestError(f"datasets must be pinned to {DATASETS_VERSION}, got {installed_version}")
        dataset_loader = cast(Callable[..., object], datasets.load_dataset)
    rows = {
        split: _materialize_split_rows(
            dataset_loader(contract.dataset_id, contract.config, split=split, revision=contract.revision), split=split
        )
        for split in SPLITS
    }
    manifest = build_manifest(rows, contract=contract, datasets_version=installed_version)
    write_manifest(output, manifest, contract=contract)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acquire", action="store_true", help="enable the one-time pinned dataset download")
    parser.add_argument("--output", type=Path, default=Path("artifacts/m14/l04-wikitext-2-manifest.json"))
    args = parser.parse_args(argv)
    if not args.acquire:
        parser.error("acquisition is disabled by default; pass --acquire only in the owner-approved run")
    acquire_manifest(output=args.output)
    print(f"wrote sanitized manifest: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

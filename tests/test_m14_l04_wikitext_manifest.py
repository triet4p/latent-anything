"""Offline tests for the sanitized WikiText-2 acquisition contract."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import cast

import pytest

from scripts.m14_l04_wikitext_manifest import (
    CONFIG,
    DATASET_ID,
    LICENSE,
    LICENSE_ACCESS,
    REVISION,
    ManifestContract,
    ManifestError,
    acquire_manifest,
    build_manifest,
    content_sha256,
    main,
    split_sha256,
    validate_manifest,
    verify_frozen_plan,
    write_manifest,
)


def _contract() -> ManifestContract:
    return ManifestContract(official_rows={"train": 5, "validation": 4}, selection_rows={"train": 2, "validation": 2})


def _rows() -> dict[str, list[dict[str, str]]]:
    return {
        "train": [{"text": "zulu"}, {"text": "   "}, {"text": "đặc biệt"}, {"text": "alpha"}, {"text": ""}],
        "validation": [{"text": "beta"}, {"text": "gamma"}, {"text": "\u2003"}, {"text": "delta"}],
    }


def test_build_manifest_drops_blank_rows_and_retains_only_hashes() -> None:
    manifest = build_manifest(_rows(), contract=_contract())

    assert manifest["source"] == {
        "dataset_id": DATASET_ID,
        "config": CONFIG,
        "revision": REVISION,
        "source_url": "https://huggingface.co/datasets/Salesforce/wikitext/tree/f776294184f13b8ff2337b3841cf9269a6216d1e/wikitext-2-raw-v1",
        "license": LICENSE,
        "license_access": LICENSE_ACCESS,
        "license_source_url": "https://huggingface.co/datasets/Salesforce/wikitext",
    }
    splits = cast(Mapping[str, Mapping[str, object]], manifest["splits"])
    train = splits["train"]
    assert train["official_rows"] == 5
    assert train["nonblank_rows"] == 3
    assert train["selected_rows"] == 2
    assert all(set(row) == {"index", "text_sha256"} for row in train["selected"])
    assert '"text":' not in json.dumps(manifest, ensure_ascii=True)
    validate_manifest(manifest, contract=_contract())


def test_content_and_split_digests_are_independent() -> None:
    manifest = build_manifest(_rows(), contract=_contract())
    splits = cast(Mapping[str, Mapping[str, object]], manifest["splits"])
    assert manifest["content_sha256"] == content_sha256(splits)
    assert manifest["split_sha256"] == split_sha256(splits)
    changed = copy.deepcopy(manifest)
    changed_splits = cast(dict[str, dict[str, object]], changed["splits"])
    changed_selected = cast(list[dict[str, object]], cast(dict[str, object], changed_splits["train"])["selected"])
    changed_selected[0]["text_sha256"] = "0" * 64
    assert changed["content_sha256"] != content_sha256(changed_splits)
    assert changed["split_sha256"] == split_sha256(changed_splits)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value["source"].update({"revision": "bad"}), "source identity"),
        (lambda value: value["source"].update({"license": "MIT"}), "source identity"),
        (lambda value: value["splits"]["train"].update({"official_rows": 6}), "counts"),
        (
            lambda value: value["splits"]["train"]["selected"].__setitem__(0, {"index": 0, "text": "leak"}),
            "schema mismatch",
        ),
        (
            lambda value: value["splits"]["train"]["selected"].append(value["splits"]["train"]["selected"][0]),
            "duplicate",
        ),
    ],
)
def test_manifest_tampering_fails_closed(mutation: Callable[[dict[str, object]], None], message: str) -> None:
    manifest = build_manifest(_rows(), contract=_contract())
    mutated = copy.deepcopy(manifest)
    mutation(mutated)

    with pytest.raises(ManifestError, match=message):
        validate_manifest(mutated, contract=_contract())


def test_wrong_digest_fails_closed() -> None:
    manifest = build_manifest(_rows(), contract=_contract())
    manifest["content_sha256"] = "0" * 64

    with pytest.raises(ManifestError, match="content_sha256"):
        validate_manifest(manifest, contract=_contract())


def test_altered_nonblank_count_fails_digest_binding() -> None:
    manifest = build_manifest(_rows(), contract=_contract())
    splits = cast(dict[str, dict[str, object]], manifest["splits"])
    splits["train"]["nonblank_rows"] = 4

    with pytest.raises(ManifestError, match="content_sha256"):
        validate_manifest(manifest, contract=_contract())


def test_frozen_plan_binding_fails_closed() -> None:
    manifest = build_manifest(_rows(), contract=_contract())
    manifest["plan_sha256"] = "0" * 64

    with pytest.raises(ManifestError, match="plan_sha256"):
        validate_manifest(manifest, contract=_contract())


def test_frozen_plan_digest_is_verified_without_mutating_the_plan() -> None:
    verify_frozen_plan()


def test_altered_frozen_plan_fails_before_acquisition(tmp_path: Path) -> None:
    altered = tmp_path / "plan.json"
    altered.write_bytes(b'{"plan_sha256":"' + b"0" * 64 + b'"}\n')

    with pytest.raises(ManifestError, match="plan binding"):
        verify_frozen_plan(altered)


def test_invalid_utf8_surrogate_fails_before_manifest_creation() -> None:
    rows = _rows()
    rows["train"][0]["text"] = "bad\ud800"

    with pytest.raises(ManifestError, match="UTF-8"):
        build_manifest(rows, contract=_contract())


def test_wrong_split_shape_and_counts_fail_closed() -> None:
    rows = _rows()
    rows.pop("validation")

    with pytest.raises(ManifestError, match="exactly train and validation"):
        build_manifest(rows, contract=_contract())


def test_write_manifest_is_lf_only_and_does_not_write_text(tmp_path: Path) -> None:
    manifest = build_manifest(_rows(), contract=_contract())
    output = tmp_path / "manifest.json"
    write_manifest(output, manifest, contract=_contract())
    payload = output.read_bytes()

    assert b"\r" not in payload
    assert payload.endswith(b"\n")
    assert b'"text":' not in payload
    assert hashlib.sha256(payload).hexdigest()


def test_acquisition_cli_requires_explicit_network_guard() -> None:
    with pytest.raises(SystemExit, match="2"):
        main([])


def test_acquisition_requests_only_pinned_train_and_validation_splits(tmp_path: Path) -> None:
    calls: list[tuple[str, str, str, str]] = []
    rows = _rows()

    def fake_loader(dataset_id: str, config: str, *, split: str, revision: str) -> object:
        calls.append((dataset_id, config, split, revision))
        return rows[split]

    output = tmp_path / "manifest.json"
    manifest = acquire_manifest(output=output, contract=_contract(), dataset_loader=fake_loader)

    assert calls == [
        (DATASET_ID, CONFIG, "train", REVISION),
        (DATASET_ID, CONFIG, "validation", REVISION),
    ]
    assert output.exists()
    validate_manifest(manifest, contract=_contract())


@pytest.mark.parametrize("bad_value", [None, {"train": []}, ["validation", "train"]])
def test_acquisition_rejects_missing_reordered_or_unexpected_split_shapes(tmp_path: Path, bad_value: object) -> None:
    def fake_loader(_dataset_id: str, _config: str, *, split: str, revision: str) -> object:
        del revision
        if split == "validation":
            return bad_value
        return _rows()["train"]

    with pytest.raises(ManifestError, match="shape|row count|non-object"):
        acquire_manifest(output=tmp_path / "manifest.json", contract=_contract(), dataset_loader=fake_loader)

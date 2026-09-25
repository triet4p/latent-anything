"""Versioned generic target-label evidence (Sprint 80.24 contract amendment).

Frozen taxonomy v1 requires ``sample``/``feature``/``label`` capture axes for
``separability_probe_leakage``, but supervised target labels are not an
activation-array axis: the accepted 80.7 binder has no ``label`` axis and real
bound captures truthfully carry ``slice``/``feature`` (or ``sample``/``feature``).
Aliasing a label as a feature axis or forging capture provenance is forbidden.

This module defines the explicit version boundary:

- ``diagnostic-evidence-v1`` (legacy): capture axes alone must satisfy the
  frozen taxonomy ``required_axes``. Truthful transformer captures fail closed
  here with ``taxonomy applicability mismatch``; that behavior is preserved.
- ``diagnostic-evidence-v2`` (amended): real capture axes stay truthful (they
  must never contain ``label``/``split``/``seed``/``sequence_or_time``), and a
  separately content-addressed target record binds sample identities, split
  membership, and the predeclared target rule. The independent validator
  requires both sources before persistence.

The record is generic: any supervised probe target (transformer section-header
attribute, encoder brightness bin, future probe tasks) uses the same fields.
No architecture-specific code enters this module.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from numbers import Integral
from typing import cast

from latent_anything._run_record_codec import canonical_json

TARGET_EVIDENCE_SCHEMA_VERSION = "diagnostic-target-evidence-v1"
"""Schema identity for one content-addressed target-label record."""

_TARGET_RECORD_FIELDS = frozenset(
    {
        "capacity",
        "dataset_split_identity",
        "eval_indices",
        "eval_split_identity",
        "label_digest",
        "labels",
        "representation_identity",
        "rule",
        "rule_digest",
        "rule_kind",
        "sample_digest",
        "sample_ids",
        "schema_version",
        "target_id",
        "train_indices",
        "train_split_identity",
    }
)

EVIDENCE_CONTRACT_V1 = "diagnostic-evidence-v1"
"""Legacy contract: taxonomy required axes must all appear as capture axes."""

EVIDENCE_CONTRACT_V2 = "diagnostic-evidence-v2"
"""Amended contract: truthful capture axes plus aligned target evidence."""

EVIDENCE_CONTRACTS = frozenset({EVIDENCE_CONTRACT_V1, EVIDENCE_CONTRACT_V2})

ACTIVATION_AXES = frozenset({"sample", "slice", "token", "time", "checkpoint", "layer", "module", "feature"})
"""Names the 80.7 binder understands. Anything else is target/split provenance."""

PROVENANCE_TOKENS = frozenset({"label", "split", "seed", "sequence_or_time"})
"""Taxonomy axis tokens that are never activation-array axes."""


class TargetEvidenceError(ValueError):
    """Raised when target-label evidence is malformed, tampered, or misaligned."""


def _non_empty_string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TargetEvidenceError(f"{name} must be a non-empty string")
    return value


def _require_digest(value: object, *, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise TargetEvidenceError(f"{name} must be a 64-character hex digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise TargetEvidenceError(f"{name} must be a 64-character hex digest") from exc
    return value


def _string_tuple(value: object, *, name: str, minimum: int = 0) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TargetEvidenceError(f"{name} must be a sequence of strings")
    items = tuple(value)
    if len(items) < minimum:
        raise TargetEvidenceError(f"{name} must contain at least {minimum} item(s)")
    for item in items:
        if not isinstance(item, str) or not item:
            raise TargetEvidenceError(f"{name} must contain non-empty strings")
    return cast(tuple[str, ...], items)


def _int_tuple(value: object, *, name: str, minimum: int = 0) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TargetEvidenceError(f"{name} must be a sequence of integers")
    items = tuple(value)
    if len(items) < minimum:
        raise TargetEvidenceError(f"{name} must contain at least {minimum} item(s)")
    if any(isinstance(item, bool) or not isinstance(item, Integral) for item in items):
        raise TargetEvidenceError(f"{name} must contain integers")
    return tuple(int(item) for item in items)


def _digest_of(data: bytes) -> str:
    return sha256(data).hexdigest()


def _canonical_value_bytes(value: object) -> bytes:
    try:
        return canonical_json(value) + b"\n"
    except Exception as exc:  # noqa: BLE001 - any serialization gap fails closed
        raise TargetEvidenceError(f"target value is not canonical JSON: {exc}") from exc


def _canonical_record_bytes(record: Mapping[str, object]) -> bytes:
    return _canonical_value_bytes(dict(record))


def target_label_digest(labels: Sequence[int]) -> str:
    """Digest integer target labels with the evidence contract's canonical encoding."""
    clean_labels = _int_tuple(labels, name="labels", minimum=1)
    return _digest_of(_canonical_value_bytes(list(clean_labels)))


def target_sample_digest(sample_ids: Sequence[str]) -> str:
    """Digest sample identities with the evidence contract's canonical encoding."""
    clean_samples = _string_tuple(sample_ids, name="sample_ids", minimum=1)
    return _digest_of(_canonical_value_bytes(list(clean_samples)))


def build_target_record(
    *,
    target_id: str,
    rule: str,
    rule_kind: str,
    representation_identity: str,
    dataset_split_identity: str,
    train_split_identity: str,
    eval_split_identity: str,
    sample_ids: Sequence[str],
    labels: Sequence[int],
    train_indices: Sequence[int],
    eval_indices: Sequence[int],
    capacity: str,
) -> dict[str, object]:
    """Build one target-label record from real caller-owned batch data.

    The caller (proof composition) supplies the actual evaluated labels,
    sample identities, and split membership plus the predeclared target rule.
    Nothing is inferred from activation axes. The returned mapping is the
    canonical content-addressed payload; its digest binds every field.
    """
    clean_target = _non_empty_string(target_id, name="target_id")
    clean_rule = _non_empty_string(rule, name="target rule")
    clean_kind = _non_empty_string(rule_kind, name="rule_kind")
    clean_representation = _non_empty_string(representation_identity, name="representation_identity")
    clean_split = _non_empty_string(dataset_split_identity, name="dataset_split_identity")
    clean_train_split = _non_empty_string(train_split_identity, name="train_split_identity")
    clean_eval_split = _non_empty_string(eval_split_identity, name="eval_split_identity")
    clean_capacity = _non_empty_string(capacity, name="capacity")
    clean_samples = _string_tuple(sample_ids, name="sample_ids", minimum=2)
    clean_labels = _int_tuple(labels, name="labels", minimum=2)
    clean_train = _int_tuple(train_indices, name="train_indices", minimum=1)
    clean_eval = _int_tuple(eval_indices, name="eval_indices", minimum=1)
    # Fail-closed structural checks now so tampered inputs never produce a record.
    _check_alignment(clean_samples, clean_labels, clean_train, clean_eval, clean_train_split, clean_eval_split)
    record: dict[str, object] = {
        "capacity": clean_capacity,
        "dataset_split_identity": clean_split,
        "eval_indices": list(clean_eval),
        "eval_split_identity": clean_eval_split,
        "label_digest": target_label_digest(clean_labels),
        "labels": list(clean_labels),
        "representation_identity": clean_representation,
        "rule": clean_rule,
        "rule_digest": _digest_of(clean_rule.encode("utf-8")),
        "rule_kind": clean_kind,
        "sample_digest": target_sample_digest(clean_samples),
        "sample_ids": list(clean_samples),
        "schema_version": TARGET_EVIDENCE_SCHEMA_VERSION,
        "target_id": clean_target,
        "train_indices": list(clean_train),
        "train_split_identity": clean_train_split,
    }
    # The digests above must round-trip through validation before release.
    validate_target_record(record)
    return record


def _check_alignment(
    sample_ids: tuple[str, ...],
    labels: tuple[int, ...],
    train_indices: tuple[int, ...],
    eval_indices: tuple[int, ...],
    train_split_identity: str,
    eval_split_identity: str,
) -> None:
    n = len(sample_ids)
    if len(labels) != n:
        raise TargetEvidenceError(f"labels cover {len(labels)} samples but sample_ids cover {n}")
    if len(set(sample_ids)) != n:
        raise TargetEvidenceError("sample identities must be unique within one target record")
    if len(set(labels)) < 2:
        raise TargetEvidenceError("labels must contain at least two classes")
    for name, indices in (("train_indices", train_indices), ("eval_indices", eval_indices)):
        if any(item < 0 or item >= n for item in indices):
            raise TargetEvidenceError(f"{name} holds a position outside the target of {n} samples")
        if len(set(indices)) != len(indices):
            raise TargetEvidenceError(f"{name} must not repeat a sample position")
    if set(train_indices) & set(eval_indices):
        raise TargetEvidenceError("train/eval split leaks: a sample position appears on both sides")
    assigned = set(train_indices) | set(eval_indices)
    if assigned != set(range(n)):
        raise TargetEvidenceError("train/eval splits must assign every target sample exactly once")
    train_ids = {sample_ids[item] for item in train_indices}
    eval_ids = {sample_ids[item] for item in eval_indices}
    if train_ids & eval_ids:
        raise TargetEvidenceError("train/eval split leaks: a sample identity appears on both sides")
    if train_split_identity == eval_split_identity:
        raise TargetEvidenceError("train/eval split identities must be distinct")


def validate_target_record(record: object) -> None:
    """Validate one target-label record fail-closed (internal consistency)."""
    if not isinstance(record, Mapping):
        raise TargetEvidenceError("target record must be a mapping")
    record_map = cast(Mapping[str, object], record)
    if frozenset(record_map) != _TARGET_RECORD_FIELDS:
        raise TargetEvidenceError("target record fields do not match its schema version")
    if record_map.get("schema_version") != TARGET_EVIDENCE_SCHEMA_VERSION:
        raise TargetEvidenceError("unsupported target evidence schema version")
    _non_empty_string(record_map.get("target_id"), name="target_id")
    rule = _non_empty_string(record_map.get("rule"), name="target rule")
    _non_empty_string(record_map.get("rule_kind"), name="rule_kind")
    _non_empty_string(record_map.get("representation_identity"), name="representation_identity")
    _non_empty_string(record_map.get("dataset_split_identity"), name="dataset_split_identity")
    train_split = _non_empty_string(record_map.get("train_split_identity"), name="train_split_identity")
    eval_split = _non_empty_string(record_map.get("eval_split_identity"), name="eval_split_identity")
    _non_empty_string(record_map.get("capacity"), name="capacity")
    samples = _string_tuple(record_map.get("sample_ids"), name="sample_ids", minimum=2)
    labels = _int_tuple(record_map.get("labels"), name="labels", minimum=2)
    train = _int_tuple(record_map.get("train_indices"), name="train_indices", minimum=1)
    eval_indices = _int_tuple(record_map.get("eval_indices"), name="eval_indices", minimum=1)
    _check_alignment(samples, labels, train, eval_indices, train_split, eval_split)
    rule_digest = _require_digest(record_map.get("rule_digest"), name="rule_digest")
    if rule_digest != _digest_of(rule.encode("utf-8")):
        raise TargetEvidenceError("target rule digest does not match the declared rule")
    label_digest = _require_digest(record_map.get("label_digest"), name="label_digest")
    if label_digest != target_label_digest(labels):
        raise TargetEvidenceError("target label digest does not match the declared labels")
    sample_digest = _require_digest(record_map.get("sample_digest"), name="sample_digest")
    if sample_digest != target_sample_digest(samples):
        raise TargetEvidenceError("target sample digest does not match the declared sample identities")


def target_record_digest(record: Mapping[str, object]) -> str:
    """Return the content address of one validated target record."""
    validate_target_record(record)
    return _digest_of(_canonical_record_bytes(record))


def bind_target_from_batch(
    batch: object,
    *,
    target_id: str,
    rule: str,
    rule_kind: str,
    representation_identity: str,
    dataset_split_identity: str,
) -> dict[str, object]:
    """Build a target record from the real evaluated ``LabeledBatch``.

    Reads labels, sample identities, split membership, and capacity directly
    from the batch the detector evaluated, so the record cannot drift from
    the measured evidence. The rule itself stays caller-declared (predeclared
    in the proof composition before any evidence).
    """
    from latent_anything._redundancy_separability_detection import LabeledBatch as _LabeledBatch

    if not isinstance(batch, _LabeledBatch):
        raise TargetEvidenceError("target binding requires a LabeledBatch with sample identities and splits")
    return build_target_record(
        target_id=target_id,
        rule=rule,
        rule_kind=rule_kind,
        representation_identity=representation_identity,
        dataset_split_identity=dataset_split_identity,
        train_split_identity=str(batch.train_split_identity),
        sample_ids=cast(Sequence[str], batch.sample_ids),
        labels=cast(Sequence[int], batch.labels),
        train_indices=cast(Sequence[int], batch.train_indices),
        eval_indices=cast(Sequence[int], batch.eval_indices),
        eval_split_identity=str(batch.eval_split_identity),
        capacity=str(batch.capacity),
    )


@dataclass(frozen=True)
class TargetLink:
    """Detect-payload linkage binding one evaluation to its target record."""

    target_id: str
    record_digest: str
    rule_digest: str
    label_digest: str
    sample_digest: str


def link_for_record(record: Mapping[str, object]) -> dict[str, object]:
    """Return the minimal detect-payload linkage block for one target record."""
    validate_target_record(record)
    return {
        "label_digest": str(record["label_digest"]),
        "record_digest": target_record_digest(record),
        "rule_digest": str(record["rule_digest"]),
        "sample_digest": str(record["sample_digest"]),
        "target_id": str(record["target_id"]),
    }


__all__ = [
    "ACTIVATION_AXES",
    "EVIDENCE_CONTRACT_V1",
    "EVIDENCE_CONTRACT_V2",
    "EVIDENCE_CONTRACTS",
    "PROVENANCE_TOKENS",
    "TARGET_EVIDENCE_SCHEMA_VERSION",
    "TargetEvidenceError",
    "TargetLink",
    "bind_target_from_batch",
    "build_target_record",
    "link_for_record",
    "target_record_digest",
    "target_label_digest",
    "target_sample_digest",
    "validate_target_record",
]

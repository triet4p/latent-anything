"""Registry-backed provenance for the historical L04.9 v2 CLI binding.

The context is deliberately an in-process capability, not a cryptographic
token.  It prevents accidental or ordinary data-only forgery by requiring the
exact object issued by the promotion path to remain registered and unchanged.
It cannot protect against arbitrary code already executing in this Python
process: such code can inspect or mutate module state.  The real security
boundary is the independently loaded, byte-pinned promotion policy and its
tracked source tree.
"""

from __future__ import annotations

import copy
import os
import re
import threading
import weakref
from dataclasses import dataclass
from typing import Any, Final, SupportsIndex, cast

_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_OID = re.compile(r"[0-9a-f]{40}\Z")
_CONTEXT_SEAL: Final = object()


@dataclass(frozen=True, slots=True)
class _ContextSnapshot:
    """Immutable fields retained by the registry instead of the object."""

    d1_cli_sha256: str
    d2_cli_sha256: str
    source_commit_sha: str
    source_tree_algorithm: str
    source_tree_oid: str


@dataclass(frozen=True, slots=True)
class _RegistryEntry:
    """Identity and provenance retained for one live context object."""

    reference: weakref.ReferenceType[_PolicyBoundValidationContext]
    snapshot: _ContextSnapshot
    issuing_pid: int


@dataclass(frozen=True, slots=True, weakref_slot=True, init=False)
class _PolicyBoundValidationContext:
    """Opaque, weak-referenceable context accepted by the validation path.

    The constructor is intentionally not a public construction API.  Even a
    correctly shaped object made through ``object.__new__`` is invalid because
    it has no corresponding registry entry.
    """

    d1_cli_sha256: str
    d2_cli_sha256: str
    source_commit_sha: str
    source_tree_algorithm: str
    source_tree_oid: str

    def __init__(
        self,
        *,
        seal: object,
        d1_cli_sha256: str,
        d2_cli_sha256: str,
        source_commit_sha: str,
        source_tree_algorithm: str,
        source_tree_oid: str,
    ) -> None:
        if seal is not _CONTEXT_SEAL:
            raise TypeError("policy-bound validation context is private")
        object.__setattr__(self, "d1_cli_sha256", d1_cli_sha256)
        object.__setattr__(self, "d2_cli_sha256", d2_cli_sha256)
        object.__setattr__(self, "source_commit_sha", source_commit_sha)
        object.__setattr__(self, "source_tree_algorithm", source_tree_algorithm)
        object.__setattr__(self, "source_tree_oid", source_tree_oid)

    def __copy__(self) -> _PolicyBoundValidationContext:
        """Return an unregistered copy, which therefore cannot validate."""
        clone = object.__new__(type(self))
        for field in _CONTEXT_FIELDS:
            object.__setattr__(clone, field, object.__getattribute__(self, field))
        return clone

    def __deepcopy__(self, memo: dict[int, object]) -> _PolicyBoundValidationContext:
        """Return an unregistered deepcopy, which therefore cannot validate."""
        del memo
        return copy.copy(self)

    def __reduce__(self) -> str | tuple[Any, ...]:
        raise TypeError("policy-bound validation context is not picklable")

    def __reduce_ex__(self, protocol: SupportsIndex) -> str | tuple[Any, ...]:
        del protocol
        raise TypeError("policy-bound validation context is not picklable")


_CONTEXT_FIELDS: Final = (
    "d1_cli_sha256",
    "d2_cli_sha256",
    "source_commit_sha",
    "source_tree_algorithm",
    "source_tree_oid",
)
_CONTEXT_REGISTRY: dict[int, _RegistryEntry] = {}
_REGISTRY_LOCK = threading.RLock()


def _snapshot_is_well_formed(snapshot: _ContextSnapshot) -> bool:
    return (
        type(snapshot.d1_cli_sha256) is str
        and _DIGEST.fullmatch(snapshot.d1_cli_sha256) is not None
        and type(snapshot.d2_cli_sha256) is str
        and _DIGEST.fullmatch(snapshot.d2_cli_sha256) is not None
        and type(snapshot.source_commit_sha) is str
        and _OID.fullmatch(snapshot.source_commit_sha) is not None
        and type(snapshot.source_tree_algorithm) is str
        and snapshot.source_tree_algorithm == "sha1"
        and type(snapshot.source_tree_oid) is str
        and _OID.fullmatch(snapshot.source_tree_oid) is not None
    )


def _entry_snapshot(value: object) -> _ContextSnapshot | None:
    """Return a snapshot only for a live, exact, unchanged registry member."""
    if type(value) is not _PolicyBoundValidationContext:
        return None
    context_id = id(value)
    with _REGISTRY_LOCK:
        entry = _CONTEXT_REGISTRY.get(context_id)
        if entry is None or entry.reference() is not value:
            return None
        if type(entry.issuing_pid) is not int or entry.issuing_pid != os.getpid():
            return None
        snapshot = entry.snapshot
        if not _snapshot_is_well_formed(snapshot):
            return None
        try:
            current = tuple(object.__getattribute__(value, field) for field in _CONTEXT_FIELDS)
        except (AttributeError, TypeError):
            return None
        expected = (
            snapshot.d1_cli_sha256,
            snapshot.d2_cli_sha256,
            snapshot.source_commit_sha,
            snapshot.source_tree_algorithm,
            snapshot.source_tree_oid,
        )
        return snapshot if current == expected else None


def context_is_valid(value: object) -> bool:
    """Return whether ``value`` is the currently registered context."""
    return _entry_snapshot(value) is not None


def context_cli_sha256(value: object, stage: str) -> str | None:
    """Read a stage CLI digest exclusively from the registry snapshot."""
    snapshot = _entry_snapshot(value)
    if snapshot is None:
        return None
    if stage == "stage_a_train_selection":
        return snapshot.d1_cli_sha256
    if stage == "stage_b_holdout_evaluation":
        return snapshot.d2_cli_sha256
    return None


def _issue_policy_bound_validation_context(  # pyright: ignore[reportUnusedFunction]
    *,
    d1_cli_sha256: object,
    d2_cli_sha256: object,
    source_commit_sha: object,
    source_tree_algorithm: object,
    source_tree_oid: object,
) -> _PolicyBoundValidationContext:
    """Issue one context for the canonical promotion implementation only."""
    values = (d1_cli_sha256, d2_cli_sha256, source_commit_sha, source_tree_algorithm, source_tree_oid)
    if not all(type(value) is str for value in values):
        raise ValueError("policy-bound validation context is malformed")
    snapshot = _ContextSnapshot(*(cast(str, value) for value in values))
    if not _snapshot_is_well_formed(snapshot):
        raise ValueError("policy-bound validation context is malformed")
    context = _PolicyBoundValidationContext(
        seal=_CONTEXT_SEAL,
        d1_cli_sha256=snapshot.d1_cli_sha256,
        d2_cli_sha256=snapshot.d2_cli_sha256,
        source_commit_sha=snapshot.source_commit_sha,
        source_tree_algorithm=snapshot.source_tree_algorithm,
        source_tree_oid=snapshot.source_tree_oid,
    )
    context_id = id(context)

    def _cleanup(
        reference: weakref.ReferenceType[_PolicyBoundValidationContext], *, identity: int = context_id
    ) -> None:
        with _REGISTRY_LOCK:
            entry = _CONTEXT_REGISTRY.get(identity)
            if entry is not None and entry.reference is reference:
                del _CONTEXT_REGISTRY[identity]

    reference = weakref.ref(context, _cleanup)
    with _REGISTRY_LOCK:
        _CONTEXT_REGISTRY[context_id] = _RegistryEntry(reference, snapshot, os.getpid())
    return context


__all__ = ["context_cli_sha256", "context_is_valid"]

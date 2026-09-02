"""Focused provenance tests for the L04.9 v2 validation context."""

from __future__ import annotations

import copy
import gc
import pickle
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pytest

import scripts._m14_l049_v2_promotion as promotion
import scripts._m14_l049_v2_validation_context as context

ROOT = Path(__file__).resolve().parents[1]


def _issued_context() -> context._PolicyBoundValidationContext:
    return context._issue_policy_bound_validation_context(
        d1_cli_sha256="1" * 64,
        d2_cli_sha256="2" * 64,
        source_commit_sha="3" * 40,
        source_tree_algorithm="sha1",
        source_tree_oid="4" * 40,
    )


def test_canonical_context_is_registered_and_uses_both_pinned_cli_digests() -> None:
    value = promotion.canonical_validation_context(ROOT)

    assert context.context_is_valid(value)
    assert (
        context.context_cli_sha256(value, "stage_a_train_selection")
        == promotion.load_real_promotion_policy(ROOT).d1_cli_sha256
    )
    assert (
        context.context_cli_sha256(value, "stage_b_holdout_evaluation")
        == promotion.load_real_promotion_policy(ROOT).cli_sha256
    )
    assert context.context_cli_sha256(value, "unknown") is None


@pytest.mark.parametrize("factory", [lambda: object.__new__(context._PolicyBoundValidationContext)])
def test_unregistered_or_directly_constructed_context_is_invalid(factory: object) -> None:
    value = factory()  # type: ignore[operator]
    assert not context.context_is_valid(value)
    sealed_raw = context._PolicyBoundValidationContext(
        seal=context._CONTEXT_SEAL,
        d1_cli_sha256="1" * 64,
        d2_cli_sha256="2" * 64,
        source_commit_sha="3" * 40,
        source_tree_algorithm="sha1",
        source_tree_oid="4" * 40,
    )
    assert not context.context_is_valid(sealed_raw)
    assert "bind_policy_bound_validation_context" not in context.__all__
    with pytest.raises(TypeError, match="private|missing"):
        context._PolicyBoundValidationContext(
            d1_cli_sha256="1" * 64,
            d2_cli_sha256="2" * 64,
            source_commit_sha="3" * 40,
            source_tree_algorithm="sha1",
            source_tree_oid="4" * 40,
        )


def test_copy_deepcopy_pickle_and_mutation_fail_closed() -> None:
    value = _issued_context()

    assert not context.context_is_valid(copy.copy(value))
    assert not context.context_is_valid(copy.deepcopy(value))
    with pytest.raises(TypeError, match="not picklable"):
        pickle.dumps(value)

    object.__setattr__(value, "d1_cli_sha256", "f" * 64)
    assert not context.context_is_valid(value)
    assert context.context_cli_sha256(value, "stage_a_train_selection") is None


def test_registry_entry_is_removed_after_context_collection() -> None:
    before = len(context._CONTEXT_REGISTRY)
    value = _issued_context()
    reference = weakref.ref(value)
    assert len(context._CONTEXT_REGISTRY) == before + 1

    del value
    gc.collect()

    assert reference() is None
    assert len(context._CONTEXT_REGISTRY) == before


def test_policy_type_equality_and_source_commitment_are_rechecked() -> None:
    policy = promotion.load_real_promotion_policy(ROOT)
    errors, value = promotion._real_validation_context(
        ROOT,
        replace(policy, cli_sha256="f" * 64),
        source_commit_sha=policy.source_commit_sha,
        source_tree_algorithm=policy.source_tree_algorithm,
        source_tree_oid=policy.source_tree_oid,
    )
    assert errors == ["real promotion policy is not independently canonical"]
    assert value is None

    errors, value = promotion._real_validation_context(
        ROOT,
        policy,
        source_commit_sha="f" * 40,
        source_tree_algorithm=policy.source_tree_algorithm,
        source_tree_oid=policy.source_tree_oid,
    )
    assert errors == ["real promotion source commitment differs from owner policy"]
    assert value is None


def test_registry_issue_and_validation_are_safe_concurrently() -> None:
    barrier = threading.Barrier(8)

    def issue(index: int) -> tuple[bool, str | None]:
        barrier.wait()
        value = context._issue_policy_bound_validation_context(
            d1_cli_sha256=f"{index + 1:064x}",
            d2_cli_sha256=f"{index + 2:064x}",
            source_commit_sha="a" * 40,
            source_tree_algorithm="sha1",
            source_tree_oid="b" * 40,
        )
        return context.context_is_valid(value), context.context_cli_sha256(value, "stage_b_holdout_evaluation")

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(issue, range(8)))
    assert all(valid for valid, _digest in results)
    assert [digest for _valid, digest in results] == [f"{index + 2:064x}" for index in range(8)]


def test_stale_process_id_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    value = _issued_context()
    monkeypatch.setattr(context.os, "getpid", lambda: 999_999_999)
    assert not context.context_is_valid(value)

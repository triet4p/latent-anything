"""Consumer-observable tests for the Sprint 80.7 capture-axis binding."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from latent_anything._capture_binding import (
    CaptureBindingError,
    bind_selection,
    bound_trajectory,
    resolve_capture,
    resolve_captures,
)
from latent_anything.capture import CapturedActivation, CaptureMetadata
from latent_anything.diagnostics import CaptureSelection
from latent_anything.latent_value import LatentValue
from latent_anything.trajectory import Trajectory

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"


def _load(name: str) -> dict[str, object]:
    return dict(json.loads((ARTIFACTS / name).read_text(encoding="utf-8")))


def _encoder_manifest() -> dict[str, object]:
    return _load("benchmark_manifest_sprint80_encoder_autoencoder_v1.json")


def _transformer_manifest() -> dict[str, object]:
    return _load("benchmark_manifest_sprint80_transformer_hidden_state_v1.json")


def _encoder_selection() -> CaptureSelection:
    return CaptureSelection(
        capture_id="capture-encoder",
        representation_identity="conv_vae_8x8:bottleneck-mu:latent_dim=4",
        axes=("sample", "feature", "slice"),
    )


def _transformer_selection() -> CaptureSelection:
    return CaptureSelection(
        capture_id="capture-transformer",
        representation_identity="openai-community-gpt2:hidden-states:layers0-11:dim768",
        axes=("slice", "token", "feature", "layer"),
    )


def _captured(
    values: np.ndarray,
    *,
    location: str,
    call_index: int,
    version: str,
) -> CapturedActivation:
    metadata = CaptureMetadata(
        location=location,
        call_index=call_index,
        shape=tuple(int(size) for size in values.shape),
        batch_axis=0 if values.ndim >= 1 else None,
        sequence_axis=None,
        device="cpu",
        dtype=str(values.dtype),
        source_model_version=version,
    )
    return CapturedActivation(values=values, metadata=metadata)


def test_both_core_models_share_one_generic_binding_path() -> None:
    encoder_plan = bind_selection(_encoder_selection(), manifest=_encoder_manifest(), request_id="req-80-7")
    transformer_plan = bind_selection(_transformer_selection(), manifest=_transformer_manifest(), request_id="req-80-7")
    assert type(encoder_plan) is type(transformer_plan)

    rng = np.random.default_rng(80)
    encoder_values = rng.normal(size=(360, 4)).astype(np.float64)
    transformer_values = rng.normal(size=(4, 16, 768)).astype(np.float64)

    encoder_cap = _captured(
        encoder_values, location="bottleneck.mu", call_index=0, version="conv-vae-8x8-latent4-seed0-epochs5"
    )
    transformer_cap = _captured(
        transformer_values,
        location="transformer.h.11",
        call_index=0,
        version="e7da7f221d5bf496a48136c0cd264e630fe9fcc8",
    )

    encoder_bound, encoder_value = resolve_capture(encoder_plan, encoder_cap, order_index=0, total=1)
    transformer_bound, transformer_value = resolve_capture(transformer_plan, transformer_cap, order_index=0, total=1)

    assert isinstance(encoder_value, LatentValue)
    assert isinstance(transformer_value, LatentValue)
    assert encoder_value.shape == (360, 4)
    assert transformer_value.shape == (4, 16, 768)

    # Axis alignment: feature is always last, batch/sequence from rank alone.
    encoder_index = {axis.name: axis.axis_index for axis in encoder_bound.axes}
    assert encoder_index["feature"] == 1
    assert encoder_index["sample"] == 0
    transformer_index = {axis.name: axis.axis_index for axis in transformer_bound.axes}
    assert transformer_index == {"slice": 0, "token": 1, "feature": 2, "layer": None}

    # Provenance survives through the existing primitive metadata.
    for bound, value in ((encoder_bound, encoder_value), (transformer_bound, transformer_value)):
        assert value.identity != ""
        metadata = value.metadata
        assert metadata["capture_identity"] == bound.capture_identity
        assert metadata["representation_identity"] == bound.representation_identity
        assert metadata["model_revision"] == bound.model_revision
        assert metadata["split_identity"] == bound.split_identity
        assert tuple(metadata["shape"]) == tuple(bound.shape or ())
        assert metadata["dtype"] == bound.dtype
        assert metadata["device"] == bound.device
        assert dict(metadata["capture_point"]) == {"location": bound.location, "call_index": bound.call_index}
        assert dict(metadata["ordering"]) == {"order_index": 0, "total": 1}


def test_capture_identity_is_deterministic_and_canonical() -> None:
    first = bind_selection(_encoder_selection(), manifest=_encoder_manifest(), request_id="req-80-7")
    second = bind_selection(_encoder_selection(), manifest=_encoder_manifest(), request_id="req-80-7")
    assert first.capture_identity == second.capture_identity

    reordered = CaptureSelection(
        capture_id="capture-encoder",
        representation_identity="conv_vae_8x8:bottleneck-mu:latent_dim=4",
        axes=("slice", "feature", "sample"),
    )
    permuted = bind_selection(reordered, manifest=_encoder_manifest(), request_id="req-80-7")
    assert permuted.capture_identity == first.capture_identity

    rng = np.random.default_rng(7)
    values = rng.normal(size=(360, 4)).astype(np.float64)
    captured = _captured(values, location="bottleneck.mu", call_index=0, version="conv-vae-8x8-latent4-seed0-epochs5")
    bound_a, _ = resolve_capture(first, captured, order_index=0, total=1)
    bound_b, _ = resolve_capture(second, captured, order_index=0, total=1)
    assert bound_a.capture_identity == bound_b.capture_identity
    assert bound_a.capture_identity != first.capture_identity


def test_absent_axes_are_explicit_and_trajectory_gates_sequence_axes() -> None:
    encoder_plan = bind_selection(_encoder_selection(), manifest=_encoder_manifest(), request_id="req-80-7")
    assert "token" in encoder_plan.absent_axes
    assert "time" in encoder_plan.absent_axes
    assert "checkpoint" in encoder_plan.absent_axes

    transformer_plan = bind_selection(_transformer_selection(), manifest=_transformer_manifest(), request_id="req-80-7")
    assert "sample" in transformer_plan.absent_axes
    assert "checkpoint" in transformer_plan.absent_axes
    assert "time" in transformer_plan.absent_axes

    rng = np.random.default_rng(11)
    encoder_values = rng.normal(size=(8, 4)).astype(np.float64)
    encoder_cap = _captured(
        encoder_values, location="bottleneck.mu", call_index=0, version="conv-vae-8x8-latent4-seed0-epochs5"
    )
    encoder_bound, encoder_value = resolve_capture(encoder_plan, encoder_cap, order_index=0, total=1)
    with pytest.raises(CaptureBindingError, match="non-applicable"):
        bound_trajectory(encoder_bound, encoder_value, axis="sample")

    transformer_values = rng.normal(size=(4, 16, 768)).astype(np.float64)
    transformer_cap = _captured(
        transformer_values,
        location="transformer.h.11",
        call_index=0,
        version="e7da7f221d5bf496a48136c0cd264e630fe9fcc8",
    )
    transformer_bound, transformer_value = resolve_capture(transformer_plan, transformer_cap, order_index=0, total=1)
    sliced = transformer_value[0]
    assert sliced.shape == (16, 768)
    trajectory = bound_trajectory(transformer_bound, sliced, axis="token")
    assert isinstance(trajectory, Trajectory)
    assert trajectory.shape == (16, 768)
    assert trajectory.metadata["capture_identity"] == transformer_bound.capture_identity
    assert trajectory.metadata["trajectory_axis"] == "token"
    with pytest.raises(CaptureBindingError, match="non-applicable"):
        bound_trajectory(transformer_bound, sliced, axis="checkpoint")


def test_resolve_preserves_ordering_and_rejects_mixed_shapes() -> None:
    plan = bind_selection(_encoder_selection(), manifest=_encoder_manifest(), request_id="req-80-7")
    rng = np.random.default_rng(13)
    first = _captured(
        rng.normal(size=(8, 4)).astype(np.float64),
        location="bottleneck.mu",
        call_index=0,
        version="conv-vae-8x8-latent4-seed0-epochs5",
    )
    second = _captured(
        rng.normal(size=(8, 4)).astype(np.float64),
        location="bottleneck.mu",
        call_index=1,
        version="conv-vae-8x8-latent4-seed0-epochs5",
    )
    bounds, values = resolve_captures(plan, (first, second))
    assert [bound.order_index for bound in bounds] == [0, 1]
    assert [bound.total for bound in bounds] == [2, 2]
    assert [bound.call_index for bound in bounds] == [0, 1]
    assert values[0].shape == (8, 4) == values[1].shape
    assert bounds[0].capture_identity != bounds[1].capture_identity

    odd = _captured(
        rng.normal(size=(8, 5)).astype(np.float64),
        location="bottleneck.mu",
        call_index=2,
        version="conv-vae-8x8-latent4-seed0-epochs5",
    )
    with pytest.raises(CaptureBindingError, match="incompatible shape"):
        resolve_captures(plan, (first, odd))


def test_duplicate_and_ambiguous_axes_reject() -> None:
    manifest = _encoder_manifest()
    with pytest.raises(ValueError, match="duplicates"):
        CaptureSelection(
            capture_id="capture-encoder",
            representation_identity="conv_vae_8x8:bottleneck-mu:latent_dim=4",
            axes=("sample", "sample"),
        )
    ambiguous = CaptureSelection(
        capture_id="capture-encoder",
        representation_identity="conv_vae_8x8:bottleneck-mu:latent_dim=4",
        axes=("layer", "module"),
    )
    with pytest.raises(CaptureBindingError, match="ambiguous"):
        bind_selection(ambiguous, manifest=manifest, request_id="req-80-7")
    unknown = CaptureSelection(
        capture_id="capture-encoder",
        representation_identity="conv_vae_8x8:bottleneck-mu:latent_dim=4",
        axes=("sample", "head"),
    )
    with pytest.raises(CaptureBindingError, match="unsupported axis"):
        bind_selection(unknown, manifest=manifest, request_id="req-80-7")


def test_missing_identity_and_provenance_mismatch_reject() -> None:
    manifest = _encoder_manifest()
    wrong_identity = CaptureSelection(
        capture_id="capture-encoder",
        representation_identity="openai-community-gpt2:hidden-states:layers0-11:dim768",
        axes=("sample", "feature"),
    )
    with pytest.raises(CaptureBindingError, match="provenance mismatch"):
        bind_selection(wrong_identity, manifest=manifest, request_id="req-80-7")

    undeclared = CaptureSelection(
        capture_id="capture-encoder",
        representation_identity="conv_vae_8x8:bottleneck-mu:latent_dim=4",
        axes=("token", "feature"),
    )
    with pytest.raises(CaptureBindingError, match="not declared"):
        bind_selection(undeclared, manifest=manifest, request_id="req-80-7")

    plan = bind_selection(_encoder_selection(), manifest=manifest, request_id="req-80-7")
    rng = np.random.default_rng(17)
    values = rng.normal(size=(8, 4)).astype(np.float64)
    wrong_version = _captured(values, location="bottleneck.mu", call_index=0, version="some-other-revision")
    with pytest.raises(CaptureBindingError, match="provenance mismatch"):
        resolve_capture(plan, wrong_version, order_index=0, total=1)

    mismatched_values = values
    mismatched = CapturedActivation(
        values=mismatched_values,
        metadata=CaptureMetadata(
            location="bottleneck.mu",
            call_index=0,
            shape=(8, 5),
            batch_axis=0,
            sequence_axis=None,
            device="cpu",
            dtype=str(values.dtype),
            source_model_version="conv-vae-8x8-latent4-seed0-epochs5",
        ),
    )
    with pytest.raises(CaptureBindingError, match="incompatible shape"):
        resolve_capture(plan, mismatched, order_index=0, total=1)


def test_nondeterministic_and_incompatible_shapes_reject() -> None:
    plan = bind_selection(_encoder_selection(), manifest=_encoder_manifest(), request_id="req-80-7")
    with pytest.raises(CaptureBindingError, match="non-empty sequence"):
        resolve_captures(plan, ())
    rng = np.random.default_rng(19)
    first = _captured(
        rng.normal(size=(8, 4)).astype(np.float64),
        location="bottleneck.mu",
        call_index=0,
        version="conv-vae-8x8-latent4-seed0-epochs5",
    )
    with pytest.raises(CaptureBindingError, match="duplicate capture point"):
        resolve_captures(plan, (first, first))
    flat = _captured(
        rng.normal(size=(4,)).astype(np.float64),
        location="bottleneck.mu",
        call_index=0,
        version="conv-vae-8x8-latent4-seed0-epochs5",
    )
    with pytest.raises(CaptureBindingError, match="incompatible shape"):
        resolve_capture(plan, flat, order_index=0, total=1)

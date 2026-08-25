"""Focused Sprint 74 Task 02 tests for typed envelopes and allowlisting."""

from __future__ import annotations

import numpy as np
import pytest

from latent_anything.cem import CEMIteration, CEMPlanResult
from latent_anything.config import ObjectSpec
from latent_anything.mppi import MPPIConfig
from latent_anything.portable import encode_portable
from latent_anything.portable_results import (
    PortableResultError,
    decode_result_envelope,
    encode_result_envelope,
)
from latent_anything.runtime.profiling import ProfileEvent, RuntimeProfile


def test_allowlisted_result_round_trip_restores_arrays_nested_dataclasses_and_state() -> None:
    profile = RuntimeProfile(events=(ProfileEvent("planning", 0.25, {"seed": 7}),))
    iteration = CEMIteration(
        iteration=0,
        population_size=4,
        elite_count=1,
        mean_return=1.0,
        std_return=0.2,
        best_return=1.5,
        elite_threshold=1.2,
        mean_action=np.array([0.1, 0.2]),
        std_action=np.array([0.3, 0.4]),
    )
    result = CEMPlanResult(
        actions=np.array([[0.1, 0.2]], dtype=np.float64),
        predicted_return=1.5,
        candidate_statistics=(iteration,),
        convergence_history=(1.0, 1.5),
        runtime_profile=profile,
        seed=7,
    )
    provenance = {"plugin": "builtin-cem", "version": "0.1.0b1"}
    behavior_state = {"config": {"horizon": 1, "action_dim": 2}, "checkpoint": "none"}

    envelope = decode_result_envelope(
        encode_result_envelope(result, provenance=provenance, behavior_state=behavior_state)
    )

    assert isinstance(envelope.value, CEMPlanResult)
    np.testing.assert_array_equal(envelope.value.actions, result.actions)
    assert isinstance(envelope.value.candidate_statistics, tuple)
    assert isinstance(envelope.value.convergence_history, tuple)
    assert isinstance(envelope.value.runtime_profile.events, tuple)
    assert envelope.value.candidate_statistics[0].mean_action.flags.writeable is False
    assert envelope.provenance == provenance
    assert envelope.behavior_state == behavior_state
    assert envelope.identity


def test_allowlisted_pydantic_config_round_trip_and_unknown_types_are_rejected() -> None:
    spec = ObjectSpec(kind="method_a", name="pca", params={"n_components": 2})

    restored = decode_result_envelope(encode_result_envelope(spec, behavior_state={"seed": 3}))

    assert isinstance(restored.value, ObjectSpec)
    assert restored.value == spec
    with pytest.raises(PortableResultError, match="allowlist"):
        encode_result_envelope(object())


def test_identity_mismatch_and_unallowlisted_type_markers_fail_closed() -> None:
    known_type = "latent_anything.cem:CEMConfig"
    malformed = encode_portable(
        {
            "schema_version": "result-envelope-v1",
            "type_id": known_type,
            "identity": "wrong",
            "provenance": {},
            "behavior_state": {},
            "value": {"kind": "pydantic", "type_id": known_type, "fields": {}},
        }
    )
    with pytest.raises(PortableResultError, match="identity"):
        decode_result_envelope(malformed)

    unknown = encode_portable(
        {
            "schema_version": "result-envelope-v1",
            "type_id": "evil.module:Payload",
            "identity": "x",
            "provenance": {},
            "behavior_state": {},
            "value": {},
        }
    )
    with pytest.raises(PortableResultError, match="allowlisted"):
        decode_result_envelope(unknown)


def test_nested_provenance_and_behavior_state_are_immutable() -> None:
    restored = decode_result_envelope(
        encode_result_envelope(
            ObjectSpec(kind="analysis", name="pca", params={}),
            provenance={"plugin": {"name": "demo", "tags": ["one"]}},
            behavior_state={"config": {"seed": 7}},
        )
    )

    with pytest.raises(TypeError):
        restored.provenance["plugin"]["name"] = "changed"  # type: ignore[index]
    with pytest.raises(TypeError):
        restored.behavior_state["config"]["seed"] = 8  # type: ignore[index]
    with pytest.raises(AttributeError):
        restored.provenance["plugin"]["tags"].append("two")  # type: ignore[union-attr]


def test_explicit_v0_envelope_migration_reconstructs_allowlisted_config() -> None:
    migrated = encode_portable(
        {
            "schema_version": "result-envelope-v0",
            "type_id": "latent_anything.config:ObjectSpec",
            "provenance": {},
            "behavior_state": {},
            "value": {
                "kind": "pydantic",
                "type_id": "latent_anything.config:ObjectSpec",
                "fields": {"kind": "analysis", "name": "pca", "params": {}},
            },
        }
    )

    restored = decode_result_envelope(migrated)

    assert isinstance(restored.value, ObjectSpec)
    assert restored.value.name == "pca"


def test_allowlisted_mppi_config_preserves_tuple_fields() -> None:
    config = MPPIConfig(horizon=2, action_dim=2, lower_bounds=(-1.0, -1.0), upper_bounds=(1.0, 1.0))

    restored = decode_result_envelope(encode_result_envelope(config))

    assert isinstance(restored.value, MPPIConfig)
    assert isinstance(restored.value.lower_bounds, tuple)
    assert isinstance(restored.value.upper_bounds, tuple)
    assert isinstance(restored.value.noise_std, tuple)

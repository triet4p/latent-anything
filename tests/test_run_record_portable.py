"""Sprint 74 Task 07 integration with run records and plugin provenance."""

from __future__ import annotations

import numpy as np

from latent_anything.cem import CEMIteration, CEMPlanResult
from latent_anything.portable_results import decode_result_envelope, encode_result_envelope
from latent_anything.run_record import FileSystemRunRecorder
from latent_anything.runtime.profiling import ProfileEvent, RuntimeProfile


def test_run_recorder_attaches_and_validates_portable_plugin_artifact(tmp_path: object) -> None:
    recorder = FileSystemRunRecorder(str(tmp_path))
    record = recorder.start(
        "portable-plugin-run",
        config={"planner": "cem", "action_dim": 2},
        metadata={"plugin": {"name": "hello", "version": "1.0"}},
    )
    iteration = CEMIteration(0, 4, 1, 1.0, 0.1, 1.2, 1.1, np.array([0.1, 0.2]), np.array([0.3, 0.4]))
    result = CEMPlanResult(
        np.array([[0.1, 0.2]]),
        1.2,
        (iteration,),
        (1.0, 1.2),
        RuntimeProfile((ProfileEvent("planning", 0.01, {}),)),
        7,
    )
    payload = encode_result_envelope(
        result,
        provenance={"plugin": "hello", "plugin_version": "1.0", "entry_point": "hello:build"},
        behavior_state={"config_identity": "cfg-1", "checkpoint_identity": "ckpt-1"},
    )

    reference = recorder.add_portable_artifact(
        record.run_id,
        payload,
        name="cem-result.la",
        artifact_type="cem-result",
        metadata={
            "plugin": {"name": "hello", "version": "1.0", "entry_point": "hello:build"},
            "config_identity": "cfg-1",
            "checkpoint_identity": "ckpt-1",
        },
    )
    restored = recorder.read_portable_artifact(reference)
    envelope = decode_result_envelope(restored.payload)
    current = recorder.get(record.run_id)

    assert reference in current.artifacts
    assert restored.artifact_type == "cem-result"
    assert restored.metadata["plugin"] == {"name": "hello", "version": "1.0", "entry_point": "hello:build"}
    assert isinstance(envelope.value, CEMPlanResult)
    np.testing.assert_array_equal(envelope.value.actions, result.actions)

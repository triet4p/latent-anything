"""Consumer-observable tests for the Sprint 80.8 diagnostic workflow state machine."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import pytest

from latent_anything._diagnostic_workflow import (
    WORKFLOW_STAGES,
    DiagnosticWorkflow,
    StageContractError,
    StageInvocation,
    StageOutput,
    WorkflowCheckpoint,
    WorkflowError,
)
from latent_anything.diagnostics import (
    CaptureSelection,
    ControlSelection,
    DiagnosticRequest,
    DiagnosticSelection,
    OutputSelection,
)

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"


def _manifest() -> dict[str, object]:
    path = ARTIFACTS / "benchmark_manifest_sprint80_encoder_autoencoder_v1.json"
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _request(request_id: str = "request-80-8") -> DiagnosticRequest:
    return DiagnosticRequest(
        request_id=request_id,
        manifest_id="sprint80-core-encoder-autoencoder-collapse-v1",
        capture=CaptureSelection(
            capture_id="capture-encoder",
            representation_identity="conv_vae_8x8:bottleneck-mu:latent_dim=4",
            axes=("sample", "feature"),
        ),
        diagnostics=DiagnosticSelection(family_ids=("collapse_rank_loss",)),
        controls=ControlSelection(
            control_ids=("control-healthy-counterexample", "control-benign-low-variance"),
            metric_ids=("bottleneck-effective-rank", "bottleneck-singular-spread"),
        ),
        output=OutputSelection(output_location="artifacts/diagnostics/request-80-8"),
    )


def _versions() -> dict[str, str]:
    return {stage: f"{stage}-executor-v1" for stage in WORKFLOW_STAGES}


def _success_executors(
    calls: list[str],
    outcomes: Mapping[str, str] | None = None,
) -> dict[str, object]:
    resolved = dict(outcomes or {})

    def _make(stage: str) -> object:
        def _run(invocation: StageInvocation) -> StageOutput:
            calls.append(stage)
            assert invocation.stage == stage
            assert tuple(item.stage for item in invocation.prior) == WORKFLOW_STAGES[: WORKFLOW_STAGES.index(stage)]
            assert invocation.request.request_id == "request-80-8"
            assert invocation.workflow_identity == invocation.workflow_identity.strip()
            payload: dict[str, object] = {"stage": stage, "request": invocation.request.request_id}
            if stage == "report":
                payload["report_id"] = "report-80-8"
            return StageOutput(
                stage=stage,
                outcome=resolved.get(stage, "completed"),  # type: ignore[arg-type]
                payload=payload,
                artifact_refs=(f"{stage}-artifact-1",),
            )

        return _run

    return {stage: _make(stage) for stage in WORKFLOW_STAGES}


def _workflow(calls: list[str], outcomes: Mapping[str, str] | None = None) -> DiagnosticWorkflow:
    return DiagnosticWorkflow(_success_executors(calls, outcomes), _versions())  # type: ignore[arg-type]


def test_full_seven_stage_flow_has_explicit_contracts() -> None:
    calls: list[str] = []
    workflow = _workflow(calls)
    result, checkpoint = workflow.run(_request(), _manifest())

    assert calls == list(WORKFLOW_STAGES)
    assert result.status == "completed"
    assert result.completed_stages == WORKFLOW_STAGES
    assert result.report_id == "report-80-8"
    assert result.artifact_refs == tuple(f"{stage}-artifact-1" for stage in WORKFLOW_STAGES)
    assert checkpoint.status == "completed"
    assert checkpoint.next_stage is None
    assert checkpoint.failure is None
    assert checkpoint.completed_stages == WORKFLOW_STAGES
    assert checkpoint.artifact_refs == result.artifact_refs
    assert checkpoint.output_digests and len(checkpoint.output_digests) == len(WORKFLOW_STAGES)
    assert result.stage_results["workflow_identity"] == checkpoint.workflow_identity
    assert result.stage_results["manifest_digest"] == checkpoint.manifest_digest
    assert result.stage_results["request_digest"] == checkpoint.request_digest
    assert result.stage_results["config_digest"] == checkpoint.config_digest
    assert set(result.stage_results["stages"]) == set(WORKFLOW_STAGES)


def test_stop_and_resume_is_equivalent_to_uninterrupted_run() -> None:
    first_calls: list[str] = []
    workflow = _workflow(first_calls)
    request = _request()
    manifest = _manifest()
    partial_result, partial = workflow.run(request, manifest, stop_after="detect")

    assert partial_result.status == "running"
    assert partial.completed_stages == ("capture", "detect")
    assert partial.next_stage == "localize"
    assert first_calls == ["capture", "detect"]

    resumed_calls: list[str] = []
    resumed_workflow = _workflow(resumed_calls)
    resumed_result, resumed = resumed_workflow.resume(partial, request, manifest)
    assert resumed_calls == ["localize", "explain", "intervene", "compare", "report"]

    direct_calls: list[str] = []
    direct_result, direct = _workflow(direct_calls).run(request, manifest)
    assert resumed_result == direct_result
    assert resumed == direct
    assert WorkflowCheckpoint.from_dict(resumed.to_dict()) == resumed


def test_handlers_are_independent_per_invocation() -> None:
    seen: list[tuple[str, int]] = []

    def _capture(invocation: StageInvocation) -> StageOutput:
        seen.append(("capture", len(invocation.prior)))
        return StageOutput(stage="capture", outcome="completed", payload={"fresh": "yes"}, artifact_refs=())

    executors = _success_executors([])
    executors["capture"] = _capture
    workflow = DiagnosticWorkflow(executors, _versions())  # type: ignore[arg-type]
    _, first = workflow.run(_request(), _manifest())
    _, second = workflow.run(_request(), _manifest())
    assert seen == [("capture", 0), ("capture", 0)]
    assert first == second
    assert first.outputs[0].payload == {"fresh": "yes"}


def test_stage_failure_preserves_boundary_and_failure_metadata() -> None:
    calls: list[str] = []

    def _failing(invocation: StageInvocation) -> StageOutput:
        calls.append(invocation.stage)
        raise RuntimeError("detect boom")

    executors = _success_executors(calls)
    executors["detect"] = _failing
    workflow = DiagnosticWorkflow(executors, _versions())  # type: ignore[arg-type]
    result, checkpoint = workflow.run(_request(), _manifest())

    assert calls == ["capture", "detect"]
    assert result.status == "failed"
    assert result.completed_stages == ("capture",)
    assert result.artifact_refs == ("capture-artifact-1",)
    assert "detect boom" in result.message
    assert checkpoint.status == "failed"
    assert checkpoint.completed_stages == ("capture",)
    assert checkpoint.next_stage == "detect"
    assert checkpoint.failure is not None
    assert checkpoint.failure.stage == "detect"
    assert checkpoint.failure.error_type == "RuntimeError"
    assert WorkflowCheckpoint.from_dict(checkpoint.to_dict()) == checkpoint


def test_unsupported_outcome_is_recorded_not_promoted() -> None:
    calls: list[str] = []
    workflow = _workflow(calls, {"intervene": "unsupported"})
    result, checkpoint = workflow.run(_request(), _manifest())

    assert result.status == "completed"
    assert result.report_id == "report-80-8"
    assert result.completed_stages == WORKFLOW_STAGES
    assert result.stage_results["stages"]["intervene"]["outcome"] == "unsupported"
    assert result.stage_results["failure"] is None
    assert checkpoint.status == "completed"


def test_resume_rejects_mismatched_request_manifest_config_and_order() -> None:
    calls: list[str] = []
    workflow = _workflow(calls)
    request = _request()
    manifest = _manifest()
    _, partial = workflow.run(request, manifest, stop_after="capture")

    with pytest.raises(WorkflowError, match="request_id"):
        workflow.resume(partial, _request("other-request"), manifest)
    mutated = dict(manifest)
    mutated["manifest_id"] = "other-manifest"
    with pytest.raises(WorkflowError, match="manifest"):
        workflow.resume(partial, request, mutated)
    other_versions = dict(_versions())
    other_versions["detect"] = "detect-executor-v2"
    with pytest.raises(WorkflowError, match="config_digest"):
        DiagnosticWorkflow(_success_executors([]), other_versions).resume(  # type: ignore[arg-type]
            partial, request, manifest
        )
    bad_outputs = (StageOutput(stage="detect", outcome="completed", payload={}, artifact_refs=()),)
    with pytest.raises(WorkflowError, match="exact prefix"):
        WorkflowCheckpoint(
            request_id=partial.request_id,
            manifest_id=partial.manifest_id,
            request_digest=partial.request_digest,
            manifest_digest=partial.manifest_digest,
            config_digest=partial.config_digest,
            workflow_identity=partial.workflow_identity,
            completed_stages=("detect",),
            outputs=bad_outputs,
            output_digests=(bad_outputs[0].digest(partial.workflow_identity),),
            artifact_refs=(),
            next_stage="localize",
            status="running",
            failure=None,
        )
    with pytest.raises(WorkflowError, match="stop_after must not precede"):
        workflow.resume(partial, request, manifest, stop_after="capture")


def _boom(_invocation: StageInvocation) -> StageOutput:
    raise RuntimeError("boom")


def test_failed_checkpoint_requires_restart() -> None:
    request = _request()
    manifest = _manifest()
    executors = _success_executors([])
    executors["detect"] = _boom
    _, failed = DiagnosticWorkflow(executors, _versions()).run(request, manifest)  # type: ignore[arg-type]
    assert failed.status == "failed"
    with pytest.raises(WorkflowError, match="restart=True"):
        _workflow([]).resume(failed, request, manifest)
    with pytest.raises(WorkflowError, match="unknown stop_after"):
        _workflow([]).run(request, manifest, stop_after="not_a_stage")
    with pytest.raises(WorkflowError, match="must cover exactly"):
        DiagnosticWorkflow(
            {"capture": lambda _invocation: StageOutput(stage="capture", outcome="completed", payload={})}, _versions()
        )


def test_wrong_stage_output_and_bad_executor_fail_the_run() -> None:
    def _wrong_stage(_invocation: StageInvocation) -> StageOutput:
        return StageOutput(stage="detect", outcome="completed", payload={}, artifact_refs=())

    executors = _success_executors([])
    executors["capture"] = _wrong_stage
    result, checkpoint = DiagnosticWorkflow(executors, _versions()).run(_request(), _manifest())  # type: ignore[arg-type]
    assert checkpoint.status == "failed"
    assert checkpoint.next_stage == "capture"

    executors = _success_executors([])
    executors["capture"] = lambda _invocation: {"not": "a stage output"}  # type: ignore[assignment]
    result, checkpoint = DiagnosticWorkflow(executors, _versions()).run(_request(), _manifest())  # type: ignore[arg-type]
    assert result.status == "failed"
    assert "StageOutput" in result.message


def test_stage_output_contract_rejects_bad_shapes() -> None:
    with pytest.raises(StageContractError, match="unknown workflow stage"):
        StageOutput(stage="not_a_stage", outcome="completed", payload={}, artifact_refs=())
    with pytest.raises(StageContractError, match="unsupported stage outcome"):
        StageOutput(stage="capture", outcome="succeeded", payload={}, artifact_refs=())  # type: ignore[arg-type]
    with pytest.raises(StageContractError, match="non-empty strings"):
        StageOutput(stage="capture", outcome="completed", payload={}, artifact_refs=("",))
    with pytest.raises(WorkflowError, match="stage output fields are invalid"):
        StageOutput.from_dict({"stage": "capture"})
    with pytest.raises(WorkflowError, match="checkpoint fields are invalid"):
        WorkflowCheckpoint.from_dict({"request_id": "x"})


def test_identities_are_deterministic_and_request_bound() -> None:
    first_calls: list[str] = []
    second_calls: list[str] = []
    first_result, first = _workflow(first_calls).run(_request(), _manifest())
    second_result, second = _workflow(second_calls).run(_request(), _manifest())
    assert first.workflow_identity == second.workflow_identity
    assert first_result == second_result
    assert first == second

    other_result, other = _workflow([]).run(_request("request-other"), _manifest())
    assert other.workflow_identity != first.workflow_identity
    assert other_result != first_result

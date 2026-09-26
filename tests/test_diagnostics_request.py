"""Consumer-observable tests for the Sprint 80.6 diagnostic request/result API."""

from __future__ import annotations

import pytest

import latent_anything
from latent_anything.diagnostics import (
    CaptureSelection,
    ComparisonRequest,
    ControlSelection,
    DiagnosticRequest,
    DiagnosticRequestError,
    DiagnosticResult,
    DiagnosticSelection,
    InterventionRequest,
    OutputSelection,
)


def _request() -> DiagnosticRequest:
    return DiagnosticRequest(
        request_id="request-80-6",
        manifest_id="sprint80-core-encoder-autoencoder-collapse-v1",
        capture=CaptureSelection(
            capture_id="capture-1",
            representation_identity="conv_vae_8x8:bottleneck-mu:latent_dim=4",
            axes=("sample", "feature"),
        ),
        diagnostics=DiagnosticSelection(family_ids=("collapse_rank_loss",)),
        controls=ControlSelection(
            control_ids=("control-healthy-counterexample", "control-benign-low-variance"),
            metric_ids=("bottleneck-effective-rank", "bottleneck-singular-spread"),
        ),
        interventions=(
            InterventionRequest(
                intervention_id="ablate-diagnosed-direction",
                target="bottleneck-mu-dim-lowest-variance",
                control_ids=("control-healthy-counterexample",),
            ),
        ),
        comparisons=(
            ComparisonRequest(
                comparison_id="run-a-vs-b",
                baseline_run="run-a",
                candidate_run="run-b",
                metric_ids=("bottleneck-effective-rank",),
            ),
        ),
        output=OutputSelection(output_location="artifacts/diagnostics/request-80-6"),
    )


def test_valid_request_constructs_and_round_trips() -> None:
    request = _request()
    rebuilt = DiagnosticRequest.from_dict(request.to_dict())

    assert rebuilt == request
    assert rebuilt.interventions[0].target == "bottleneck-mu-dim-lowest-variance"
    assert rebuilt.comparisons[0].baseline_run == "run-a"


def test_result_carries_resumable_state_and_round_trips() -> None:
    result = DiagnosticResult(
        request_id="request-80-6",
        manifest_id="sprint80-core-encoder-autoencoder-collapse-v1",
        status="running",
        completed_stages=("capture", "detect"),
        artifact_refs=("capture-artifact-1",),
        message="detect complete",
        stage_results={"capture": {"capture_id": "capture-1"}},
    )
    rebuilt = DiagnosticResult.from_dict(result.to_dict())

    assert rebuilt == result
    assert rebuilt.completed_stages == ("capture", "detect")


def test_api_uses_domain_terms_without_architecture_fields() -> None:
    request = _request()
    payload = request.to_dict()

    assert set(payload) == {
        "request_id",
        "manifest_id",
        "capture",
        "diagnostics",
        "controls",
        "interventions",
        "comparisons",
        "output",
    }
    field_names = str(sorted(payload.keys())).lower()
    assert "transformer_layer" not in field_names
    assert "attention_head" not in field_names
    assert "vae_latent_dim" not in field_names
    assert "hidden_size" not in field_names
    assert set(request.capture.axes) == {"sample", "feature"}


def test_public_surface_exports_request_and_result() -> None:
    for name in (
        "DiagnosticRequest",
        "DiagnosticResult",
        "DiagnosticRequestError",
        "CaptureSelection",
        "DiagnosticSelection",
        "ControlSelection",
        "InterventionRequest",
        "ComparisonRequest",
        "OutputSelection",
    ):
        assert name in latent_anything.__all__
        assert getattr(latent_anything, name) is getattr(
            __import__("latent_anything.diagnostics", fromlist=[name]), name
        )


def test_request_has_no_duplicate_objectspec_path() -> None:
    request = _request()

    assert not hasattr(request, "kind")
    assert not hasattr(request, "name")
    assert not hasattr(request, "params")
    assert type(request).__module__ == "latent_anything.diagnostics"


def test_unknown_family_is_rejected() -> None:
    with pytest.raises(DiagnosticRequestError, match="unknown diagnostic families"):
        _request().to_dict() and DiagnosticSelection(family_ids=("not_a_family",))


def test_duplicate_intervention_identifiers_are_rejected() -> None:
    request = _request()

    with pytest.raises(DiagnosticRequestError, match="intervention identifiers must be unique"):
        DiagnosticRequest(
            request_id="request-dup",
            manifest_id="sprint80-core-encoder-autoencoder-collapse-v1",
            capture=request.capture,
            diagnostics=request.diagnostics,
            controls=request.controls,
            interventions=(request.interventions[0], request.interventions[0]),
            comparisons=(),
            output=request.output,
        )


def test_intervention_with_undeclared_control_is_rejected() -> None:
    request = _request()

    with pytest.raises(DiagnosticRequestError, match="undeclared controls"):
        DiagnosticRequest(
            request_id="request-bad-control",
            manifest_id=request.manifest_id,
            capture=request.capture,
            diagnostics=request.diagnostics,
            controls=request.controls,
            interventions=(
                InterventionRequest(
                    intervention_id="bad-intervention",
                    target="bottleneck-mu-dim-lowest-variance",
                    control_ids=("control-missing",),
                ),
            ),
            comparisons=(),
            output=request.output,
        )


def test_comparison_metrics_can_extend_beyond_detector_selection() -> None:
    base = _request()
    task_metric = "heldout-task-accuracy"
    comparison = ComparisonRequest(
        comparison_id="utility-comparison",
        baseline_run="run-a",
        candidate_run="run-b",
        metric_ids=(base.controls.metric_ids[0], task_metric),
    )

    request = DiagnosticRequest(
        request_id="request-task-utility",
        manifest_id=base.manifest_id,
        capture=base.capture,
        diagnostics=base.diagnostics,
        controls=base.controls,
        interventions=(),
        comparisons=(comparison,),
        output=base.output,
    )

    assert request.controls.metric_ids == base.controls.metric_ids
    assert request.comparisons[0].metric_ids == (base.controls.metric_ids[0], task_metric)


def test_identical_comparison_runs_are_rejected() -> None:
    with pytest.raises(DiagnosticRequestError, match="must differ"):
        ComparisonRequest(
            comparison_id="same-run",
            baseline_run="run-a",
            candidate_run="run-a",
            metric_ids=("bottleneck-effective-rank",),
        )


def test_absolute_output_location_is_rejected() -> None:
    with pytest.raises(DiagnosticRequestError, match="repository-relative"):
        OutputSelection(output_location="/tmp/outside-repo")


def test_parent_escaping_output_location_is_rejected() -> None:
    with pytest.raises(DiagnosticRequestError, match="must not escape"):
        OutputSelection(output_location="artifacts/../outside")


def test_unknown_result_stage_is_rejected() -> None:
    with pytest.raises(DiagnosticRequestError, match="unknown workflow stages"):
        DiagnosticResult(
            request_id="request-80-6",
            manifest_id="sprint80-core-encoder-autoencoder-collapse-v1",
            status="running",
            completed_stages=("detect-typo",),
        )


def test_malformed_request_mapping_is_rejected() -> None:
    request = _request().to_dict()
    request.pop("output")

    with pytest.raises(DiagnosticRequestError, match="request fields are invalid"):
        DiagnosticRequest.from_dict(request)

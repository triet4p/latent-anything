"""Consumer-observable tests for Sprint 80.13 layer/slice localization."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from latent_anything._benchmark_manifest import manifest_digest, validate_manifest
from latent_anything._diagnostic_workflow import (
    WORKFLOW_STAGES,
    DiagnosticWorkflow,
    StageInvocation,
    StageOutput,
)
from latent_anything._layer_slice_localization import (
    DetectionEvidence,
    LayerCell,
    LocalizationError,
    LocalizationInput,
    SampleCell,
    SliceDefinition,
    evaluate_localization,
    evidence_from_manifest,
    localization_payload,
    localize_findings,
    make_localize_executor,
)
from latent_anything._portable_contract import canonical_json
from latent_anything.diagnostics import (
    CaptureSelection,
    ControlSelection,
    DiagnosticRequest,
    DiagnosticSelection,
    OutputSelection,
)

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"
MANIFEST_ID = "manifest-80-13-localization-v1"
REPRESENTATION = "R-80-13"
FAMILY = "collapse_rank_loss"
METRIC = "layer-health"


def _manifest(manifest_id: str = MANIFEST_ID) -> dict[str, object]:
    manifest: dict[str, object] = {
        "schema_version": "benchmark-manifest-schema-v1",
        "manifest_id": manifest_id,
        "status": "predeclared",
        "model": {
            "id": "m",
            "revision": "r",
            "revision_kind": "digest",
            "revision_immutable": True,
            "artifact_digest": "a" * 64,
        },
        "dataset": {"id": "d", "revision": "r1", "split": "test", "split_identity": "s", "data_digest": "b" * 64},
        "representation": {
            "identity": REPRESENTATION,
            "axes": [
                {"name": "sample", "selection": "all", "identity": "ax-s"},
                {"name": "feature", "selection": "all", "identity": "ax-f"},
                {"name": "layer", "selection": "l0-l2", "identity": "ax-l"},
                {"name": "slice", "selection": "front-back", "identity": "ax-sl"},
            ],
        },
        "defect": {"classification": "none", "declaration": "n/a", "predeclared": True, "source": "not_applicable"},
        "metrics": [
            {
                "id": METRIC,
                "taxonomy_family_id": FAMILY,
                "direction": "higher_is_better",
                "unit": "rank",
                "estimator": "declared",
                "aggregation": "mean-with-interval",
            },
        ],
        "seeds": {"training": [11], "evaluation": [5], "controls": [7], "independent": True},
        "controls": [
            {
                "id": "control-a",
                "kind": "counterexample",
                "metric_ids": [METRIC],
                "required": True,
                "expected_behavior": "healthy reference passes",
            },
            {
                "id": "control-b",
                "kind": "negative",
                "metric_ids": [METRIC],
                "required": True,
                "expected_behavior": "benign negative stays healthy",
            },
        ],
        "uncertainty": {
            "method": "bootstrap",
            "confidence_level": 0.95,
            "repetitions": 20,
            "unit_of_analysis": "sample",
            "predeclared": True,
        },
        "causal_expectation": {
            "applicable": False,
            "expectation": "not_applicable",
            "target_metric_ids": [],
            "falsification_rule": "n/a",
            "non_applicable_reason": "no causal trial in 80.13",
        },
        "thresholds": [
            {"metric_id": METRIC, "comparator": ">=", "value": 3.0, "tolerance": 0.1, "predeclared": True},
        ],
        "commitment": {
            "locked": True,
            "declared_at": "2026-09-21T00:00:00Z",
            "manifest_sha256": "x",
            "canonicalization": "json-sort-keys-no-whitespace-utf8",
        },
    }
    unsigned = deepcopy(manifest)
    commitment = dict(unsigned["commitment"])  # type: ignore[union-attr]
    commitment.pop("manifest_sha256", None)
    unsigned["commitment"] = commitment
    manifest["commitment"]["manifest_sha256"] = hashlib.sha256(  # type: ignore[index]
        json.dumps(unsigned, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    validate_manifest(manifest)
    return manifest


def _request(manifest_id: str = MANIFEST_ID) -> DiagnosticRequest:
    return DiagnosticRequest(
        request_id="request-80-13",
        manifest_id=manifest_id,
        capture=CaptureSelection(
            capture_id="capture-80-13",
            representation_identity=REPRESENTATION,
            axes=("sample", "feature", "layer", "slice"),
        ),
        diagnostics=DiagnosticSelection(family_ids=(FAMILY,)),
        controls=ControlSelection(
            control_ids=("control-a", "control-b"),
            metric_ids=(METRIC,),
        ),
        output=OutputSelection(output_location="artifacts/diagnostics/request-80-13"),
    )


def _evidence(**overrides: object) -> DetectionEvidence:
    base: dict[str, object] = {
        "manifest_id": MANIFEST_ID,
        "family_id": FAMILY,
        "metric_id": METRIC,
        "comparator": ">=",
        "threshold_value": 3.0,
        "tolerance": 0.1,
        "direction": "higher_is_better",
        "affected_when": "threshold_fail",
        "required_controls": ("control-a", "control-b"),
        "control_outcomes": {"control-a": "passed", "control-b": "passed"},
        "outcome": "supported",
        "claim_allowed": True,
        "representation_identity": REPRESENTATION,
    }
    base.update(overrides)
    return DetectionEvidence(**base)  # type: ignore[arg-type]


def _positive_input(global_score: float | None = 3.4) -> LocalizationInput:
    return LocalizationInput(
        manifest_id=MANIFEST_ID,
        evidence=_evidence(),
        layer_order=("layer-0", "layer-1", "layer-2"),
        layer_cells=(
            LayerCell("layer-0", 3.5, REPRESENTATION),
            LayerCell("layer-1", 1.2, REPRESENTATION),
            LayerCell("layer-2", 0.8, REPRESENTATION),
        ),
        sample_cells=(
            SampleCell("s0", 3.8, REPRESENTATION),
            SampleCell("s1", 3.9, REPRESENTATION),
            SampleCell("s2", 3.7, REPRESENTATION),
            SampleCell("s3", 1.0, REPRESENTATION),
            SampleCell("s4", 1.1, REPRESENTATION),
            SampleCell("s5", 3.8, REPRESENTATION),
        ),
        declared_slice_ids=("slice-front", "slice-back"),
        slices=(
            SliceDefinition("slice-front", "samples s0-s2", ("s0", "s1", "s2")),
            SliceDefinition("slice-back", "samples s3-s5", ("s3", "s4", "s5")),
        ),
        global_score=global_score,
    )


def test_multi_layer_defect_yields_correct_earliest_layer() -> None:
    result = localize_findings(_positive_input())

    assert result.verdict == "localized"
    assert result.earliest_layer == "layer-1"
    assert result.affected_layers == ("layer-1", "layer-2")


def test_affected_samples_and_slice_identified_despite_benign_global() -> None:
    source = _positive_input(global_score=3.4)
    result = localize_findings(source)

    assert result.affected_samples == ("s3", "s4")
    # slice-front mean ~3.8 (healthy); slice-back mean ~1.97 (affected).
    assert result.affected_slices == ("slice-back",)
    payload = localization_payload(result, source)
    assert payload["global_score"] == 3.4
    assert payload["global_score_ignored"] is True
    assert payload["slice_means"]["slice-front"] > 3.0
    assert payload["slice_means"]["slice-back"] < 3.0
    assert result.report_localization[0]["axis"] == "layer"
    assert result.report_localization[0]["selection"] == "layer-1"
    # confidence=1.0 is deterministic selection certainty under the qualified
    # predeclared rule, not a probability or statistical interval: every
    # emitted row carries it, and evidence strength travels separately in the
    # qualified detection evidence plus control outcomes.
    assert all(row["confidence"] == 1.0 for row in result.report_localization)
    assert all(row["confidence"] == 1.0 for row in payload["report_localization"])


def test_shuffled_input_order_does_not_change_localization() -> None:
    ordered = _positive_input()
    shuffled = LocalizationInput(
        manifest_id=MANIFEST_ID,
        evidence=_evidence(),
        layer_order=("layer-0", "layer-1", "layer-2"),
        layer_cells=(
            LayerCell("layer-2", 0.8, REPRESENTATION),
            LayerCell("layer-0", 3.5, REPRESENTATION),
            LayerCell("layer-1", 1.2, REPRESENTATION),
        ),
        sample_cells=tuple(reversed(ordered.sample_cells)),
        declared_slice_ids=("slice-back", "slice-front"),
        slices=(
            SliceDefinition("slice-back", "samples s3-s5", ("s3", "s4", "s5")),
            SliceDefinition("slice-front", "samples s0-s2", ("s0", "s1", "s2")),
        ),
        global_score=3.4,
    )
    first = localize_findings(ordered)
    second = localize_findings(shuffled)

    assert second.earliest_layer == first.earliest_layer == "layer-1"
    assert second.affected_layers == first.affected_layers
    assert second.affected_samples == first.affected_samples
    assert set(second.affected_slices) == set(first.affected_slices)


def test_control_blocked_or_below_threshold_evidence_cannot_localize() -> None:
    blocked = LocalizationInput(
        manifest_id=MANIFEST_ID,
        evidence=_evidence(control_outcomes={"control-a": "failed", "control-b": "passed"}),
        layer_order=("layer-0", "layer-1"),
        layer_cells=(
            LayerCell("layer-0", 3.5, REPRESENTATION),
            LayerCell("layer-1", 1.0, REPRESENTATION),
        ),
        sample_cells=(
            SampleCell("s0", 3.5, REPRESENTATION),
            SampleCell("s1", 1.0, REPRESENTATION),
        ),
        declared_slice_ids=("slice-all",),
        slices=(SliceDefinition("slice-all", "all samples", ("s0", "s1")),),
    )
    with pytest.raises(LocalizationError, match="failed required controls block"):
        localize_findings(blocked)

    unqualified = LocalizationInput(
        manifest_id=MANIFEST_ID,
        evidence=_evidence(outcome="inconclusive", claim_allowed=False),
        layer_order=("layer-0", "layer-1"),
        layer_cells=(
            LayerCell("layer-0", 3.5, REPRESENTATION),
            LayerCell("layer-1", 1.0, REPRESENTATION),
        ),
        sample_cells=(
            SampleCell("s0", 3.5, REPRESENTATION),
            SampleCell("s1", 1.0, REPRESENTATION),
        ),
        declared_slice_ids=("slice-all",),
        slices=(SliceDefinition("slice-all", "all samples", ("s0", "s1")),),
    )
    with pytest.raises(LocalizationError, match="unqualified"):
        localize_findings(unqualified)


def test_benign_negative_produces_no_location() -> None:
    source = LocalizationInput(
        manifest_id=MANIFEST_ID,
        evidence=_evidence(),
        layer_order=("layer-0", "layer-1"),
        layer_cells=(
            LayerCell("layer-0", 3.5, REPRESENTATION),
            LayerCell("layer-1", 3.6, REPRESENTATION),
        ),
        sample_cells=(
            SampleCell("s0", 3.5, REPRESENTATION),
            SampleCell("s1", 3.6, REPRESENTATION),
        ),
        declared_slice_ids=("slice-all",),
        slices=(SliceDefinition("slice-all", "all samples", ("s0", "s1")),),
    )
    result, payload = evaluate_localization(source)

    assert result.verdict == "negative"
    assert result.earliest_layer is None
    assert result.report_localization == ()
    assert payload["verdict"] == "negative"
    assert payload["earliest_layer"] is None


def test_undeclared_slices_and_identity_alignment_faults_reject() -> None:
    base = _positive_input()
    with pytest.raises(LocalizationError, match="undeclared slices"):
        LocalizationInput(
            manifest_id=base.manifest_id,
            evidence=base.evidence,
            layer_order=base.layer_order,
            layer_cells=base.layer_cells,
            sample_cells=base.sample_cells,
            declared_slice_ids=("slice-front",),
            slices=base.slices,
        )
    with pytest.raises(LocalizationError, match="duplicate"):
        LocalizationInput(
            manifest_id=base.manifest_id,
            evidence=base.evidence,
            layer_order=("layer-0", "layer-0", "layer-1"),
            layer_cells=base.layer_cells,
            sample_cells=base.sample_cells,
            declared_slice_ids=base.declared_slice_ids,
            slices=base.slices,
        )
    with pytest.raises(LocalizationError, match="misaligned"):
        LocalizationInput(
            manifest_id=base.manifest_id,
            evidence=base.evidence,
            layer_order=base.layer_order,
            layer_cells=(
                LayerCell("layer-0", 3.5, "WRONG-REP"),
                LayerCell("layer-1", 1.2, REPRESENTATION),
                LayerCell("layer-2", 0.8, REPRESENTATION),
            ),
            sample_cells=base.sample_cells,
            declared_slice_ids=base.declared_slice_ids,
            slices=base.slices,
        )
    with pytest.raises(LocalizationError, match="exactly the declared layer_order"):
        LocalizationInput(
            manifest_id=base.manifest_id,
            evidence=base.evidence,
            layer_order=("layer-0", "layer-1", "layer-9"),
            layer_cells=base.layer_cells,
            sample_cells=base.sample_cells,
            declared_slice_ids=base.declared_slice_ids,
            slices=base.slices,
        )
    with pytest.raises(LocalizationError, match="misaligned with sample identities"):
        LocalizationInput(
            manifest_id=base.manifest_id,
            evidence=base.evidence,
            layer_order=base.layer_order,
            layer_cells=base.layer_cells,
            sample_cells=base.sample_cells,
            declared_slice_ids=base.declared_slice_ids,
            slices=(
                SliceDefinition("slice-front", "samples s0-s2", ("s0", "s1", "ghost")),
                SliceDefinition("slice-back", "samples s3-s5", ("s3", "s4", "s5")),
            ),
        )


def test_global_only_evidence_is_insufficient() -> None:
    with pytest.raises(LocalizationError, match="global score alone"):
        LocalizationInput(
            manifest_id=MANIFEST_ID,
            evidence=_evidence(),
            layer_order=("layer-0",),
            layer_cells=(),
            sample_cells=(SampleCell("s0", 1.0, REPRESENTATION),),
            declared_slice_ids=("slice-all",),
            slices=(SliceDefinition("slice-all", "one sample", ("s0",)),),
            global_score=1.0,
        )
    with pytest.raises(LocalizationError, match="global score alone"):
        LocalizationInput(
            manifest_id=MANIFEST_ID,
            evidence=_evidence(),
            layer_order=("layer-0",),
            layer_cells=(LayerCell("layer-0", 1.0, REPRESENTATION),),
            sample_cells=(),
            declared_slice_ids=("slice-all",),
            slices=(SliceDefinition("slice-all", "one sample", ("s0",)),),
            global_score=1.0,
        )


def test_threshold_direction_mismatch_rejects() -> None:
    with pytest.raises(LocalizationError, match="threshold-direction mismatch"):
        _evidence(comparator="<=", direction="higher_is_better")


def test_ambiguous_tolerance_tie_rejects() -> None:
    source = LocalizationInput(
        manifest_id=MANIFEST_ID,
        evidence=_evidence(),
        layer_order=("layer-0", "layer-1"),
        layer_cells=(
            LayerCell("layer-0", 3.5, REPRESENTATION),
            LayerCell("layer-1", 2.95, REPRESENTATION),
        ),
        sample_cells=(
            SampleCell("s0", 3.5, REPRESENTATION),
            SampleCell("s1", 2.95, REPRESENTATION),
        ),
        declared_slice_ids=("slice-all",),
        slices=(SliceDefinition("slice-all", "all samples", ("s0", "s1")),),
    )
    with pytest.raises(LocalizationError, match="ambiguous"):
        localize_findings(source)


def test_unsupported_evidence_is_honest_not_success() -> None:
    source = LocalizationInput(
        manifest_id=MANIFEST_ID,
        evidence=_evidence(outcome="unsupported", claim_allowed=False),
        layer_order=("layer-0",),
        layer_cells=(LayerCell("layer-0", 1.0, REPRESENTATION),),
        sample_cells=(SampleCell("s0", 1.0, REPRESENTATION),),
        declared_slice_ids=("slice-all",),
        slices=(SliceDefinition("slice-all", "one sample", ("s0",)),),
    )
    result, payload = evaluate_localization(source)

    assert result.verdict == "unsupported"
    assert result.earliest_layer is None
    assert "unsupported_axes_80_14" not in payload
    assert "unsupported_axes_80_14" not in result.to_dict()
    executor = make_localize_executor(source)
    invocation = StageInvocation(
        stage="localize",
        request=_request(),
        manifest=_manifest(),
        prior=tuple(
            StageOutput(stage=stage, outcome="completed", payload={}, artifact_refs=())
            for stage in ("capture", "detect")
        ),
        workflow_identity="0" * 64,
        request_digest="1" * 64,
        manifest_digest="2" * 64,
        config_digest="3" * 64,
    )
    output = executor(invocation)
    assert output.outcome == "unsupported"
    assert executor.localizer_version == "layer-slice-localizer-v1"  # type: ignore[attr-defined]


def test_output_is_deterministic_and_canonical() -> None:
    first_result, first_payload = evaluate_localization(_positive_input())
    second_result, second_payload = evaluate_localization(_positive_input())

    assert first_result == second_result
    assert first_payload == second_payload
    canonical_json(first_payload)
    canonical_json([first_result.to_dict(), _positive_input().to_dict()])


def test_workflow_integration_completes_localize_without_coordinator_algorithms() -> None:
    manifest = _manifest()
    request = _request()
    source = _positive_input()
    localize = make_localize_executor(source)

    def _passthrough(invocation: StageInvocation) -> StageOutput:
        if invocation.stage == "localize":
            return localize(invocation)
        payload: dict[str, object] = {"stage": invocation.stage}
        if invocation.stage == "report":
            payload["report_id"] = "report-80-13"
        return StageOutput(stage=invocation.stage, outcome="completed", payload=payload, artifact_refs=())

    executors = {stage: _passthrough for stage in WORKFLOW_STAGES}
    versions = {stage: "stage-v1" for stage in WORKFLOW_STAGES}
    workflow = DiagnosticWorkflow(executors, versions)  # type: ignore[arg-type]
    result, _ = workflow.run(request, manifest)

    assert result.status == "completed"
    localize_stage = result.stage_results["stages"]["localize"]  # type: ignore[index]
    assert localize_stage["outcome"] == "completed"
    assert localize_stage["payload"]["earliest_layer"] == "layer-1"
    assert localize_stage["payload"]["affected_slices"] == ["slice-back"]
    assert localize_stage["payload"]["global_score_ignored"] is True
    assert "DiagnosticWorkflow" not in str(type(localize))


def test_evidence_binds_to_frozen_encoder_manifest_wiring() -> None:
    path = ARTIFACTS / "benchmark_manifest_sprint80_encoder_autoencoder_v1.json"
    manifest = dict(json.loads(path.read_text(encoding="utf-8")))
    request = DiagnosticRequest(
        request_id="request-80-13-encoder",
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
        output=OutputSelection(output_location="artifacts/diagnostics/request-80-13"),
    )
    evidence = evidence_from_manifest(
        request,
        manifest,
        family_id="collapse_rank_loss",
        metric_id="bottleneck-effective-rank",
        affected_when="threshold_fail",
        control_outcomes={
            "control-healthy-counterexample": "passed",
            "control-benign-low-variance": "passed",
        },
        outcome="supported",
        claim_allowed=True,
        representation_identity="conv_vae_8x8:bottleneck-mu:latent_dim=4",
    )

    assert evidence.comparator == ">="
    assert evidence.threshold_value == 3.0
    assert evidence.direction == "higher_is_better"
    assert evidence.passes(3.5) is True
    assert evidence.is_affected(1.2) is True

    tampered = deepcopy(manifest)
    tampered["manifest_id"] = "other"
    with pytest.raises(LocalizationError, match="does not match"):
        evidence_from_manifest(
            request,
            tampered,
            family_id="collapse_rank_loss",
            metric_id="bottleneck-effective-rank",
            affected_when="threshold_fail",
            control_outcomes={"control-healthy-counterexample": "passed"},
            outcome="supported",
            claim_allowed=True,
            representation_identity="conv_vae_8x8:bottleneck-mu:latent_dim=4",
        )


def test_public_surface_and_frozen_artifacts_unchanged() -> None:
    import latent_anything as la

    expected = {
        "CaptureSelection",
        "ComparisonRequest",
        "ControlSelection",
        "DiagnosticRequest",
        "DiagnosticRequestError",
        "DiagnosticResult",
        "DiagnosticSelection",
        "InterventionRequest",
        "OutputSelection",
    }
    names = {name for name in la.__all__ if "Selection" in name or "Request" in name or "Result" in name}
    assert expected <= names
    assert expected <= set(la.__all__)
    assert len([name for name in la.__all__ if name in expected]) == 9
    for leaked in (
        "DetectionEvidence",
        "LocalizationInput",
        "localize_findings",
        "make_localize_executor",
        "LayerCell",
    ):
        assert not hasattr(la, leaked)

    for name in (
        "benchmark_manifest_schema_v1.json",
        "diagnostic_report_schema_v1.json",
        "representation_problem_taxonomy_v1.json",
        "benchmark_manifest_sprint80_encoder_autoencoder_v1.json",
        "benchmark_manifest_sprint80_transformer_hidden_state_v1.json",
    ):
        path = ARTIFACTS / name
        assert path.exists()
    encoder = dict(
        json.loads((ARTIFACTS / "benchmark_manifest_sprint80_encoder_autoencoder_v1.json").read_text(encoding="utf-8"))
    )
    assert encoder["commitment"]["manifest_sha256"] == manifest_digest(encoder)

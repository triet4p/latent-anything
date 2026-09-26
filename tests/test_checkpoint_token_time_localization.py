"""Consumer-observable tests for Sprint 80.14 checkpoint/token/time localization."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import cast

import pytest

from latent_anything._benchmark_manifest import manifest_digest
from latent_anything._diagnostic_workflow import (
    WORKFLOW_STAGES,
    DiagnosticWorkflow,
    StageInvocation,
    StageOutput,
)
from latent_anything._layer_slice_localization import (
    AxialLocalizationInput,
    CheckpointAxisDeclaration,
    CheckpointCell,
    DetectionEvidence,
    FeatureAxisDeclaration,
    FeatureCell,
    LayerCell,
    LocalizationError,
    LocalizationInput,
    SampleCell,
    SliceDefinition,
    TimeAxisDeclaration,
    TimeCell,
    TokenAxisDeclaration,
    TokenCell,
    axial_localize,
    evaluate_axial_localization,
    evaluate_localization,
    evidence_from_manifest,
    localize_findings,
    make_axial_localize_executor,
    make_localize_executor,
    manifest_axis_lookup,
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
MANIFEST_ID = "manifest-80-14-axial-v1"
REPRESENTATION = "R-80-14"
FAMILY = "sequence_trajectory_drift"
METRIC = "axial-health"


def _manifest(manifest_id: str = MANIFEST_ID) -> dict[str, object]:
    manifest: dict[str, object] = {
        "schema_version": "benchmark-manifest-schema-v1",
        "manifest_id": manifest_id,
        "status": "predeclared",
        "model": {
            "id": "m",
            "revision": "rev-7",
            "revision_kind": "digest",
            "revision_immutable": True,
            "artifact_digest": "a" * 64,
        },
        "dataset": {
            "id": "d",
            "revision": "r1",
            "split": "test",
            "split_identity": "slice-A-config-v3",
            "data_digest": "b" * 64,
        },
        "representation": {
            "identity": REPRESENTATION,
            "axes": [
                {"name": "sample", "selection": "all", "identity": "ax-s"},
                {"name": "feature", "selection": "all", "identity": "ax-f"},
                {"name": "checkpoint", "selection": "ckpt0-ckpt2", "identity": "ax-ckpt"},
                {"name": "token", "selection": "tok0-tok3", "identity": "ax-tok"},
                {"name": "time", "selection": "step0-step3", "identity": "ax-time"},
            ],
        },
        "defect": {"classification": "none", "declaration": "n/a", "predeclared": True, "source": "not_applicable"},
        "metrics": [
            {
                "id": METRIC,
                "taxonomy_family_id": FAMILY,
                "direction": "higher_is_better",
                "unit": "score",
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
            "non_applicable_reason": "no causal trial in 80.14",
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
    return manifest


def _request(manifest_id: str = MANIFEST_ID) -> DiagnosticRequest:
    return DiagnosticRequest(
        request_id="request-80-14",
        manifest_id=manifest_id,
        capture=CaptureSelection(
            capture_id="capture-80-14",
            representation_identity=REPRESENTATION,
            axes=("sample", "feature", "checkpoint", "token", "time"),
        ),
        diagnostics=DiagnosticSelection(family_ids=(FAMILY,)),
        controls=ControlSelection(
            control_ids=("control-a", "control-b"),
            metric_ids=(METRIC,),
        ),
        output=OutputSelection(output_location="artifacts/diagnostics/request-80-14"),
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


def _axes() -> dict[str, dict[str, str]]:
    return manifest_axis_lookup(_manifest())


def _checkpoint_source(**overrides: object) -> AxialLocalizationInput:
    base: dict[str, object] = {
        "manifest_id": MANIFEST_ID,
        "manifest_axes": _axes(),
        "evidence": _evidence(),
        "requested_axes": ("checkpoint",),
        "checkpoint_declaration": CheckpointAxisDeclaration(
            checkpoint_order=("ckpt-0", "ckpt-1", "ckpt-2"),
            checkpoint_index={"ckpt-0": 0, "ckpt-1": 1, "ckpt-2": 2},
            dataset_slice_id="slice-A",
            dataset_configuration="config-v3",
            model_identity="model-rev-7",
            representation_identity=REPRESENTATION,
        ),
        "checkpoint_cells": (
            CheckpointCell("ckpt-0", 3.6, REPRESENTATION, "model-rev-7", "slice-A", "config-v3", "layer-2"),
            CheckpointCell("ckpt-1", 1.4, REPRESENTATION, "model-rev-7", "slice-A", "config-v3", "layer-2"),
            CheckpointCell("ckpt-2", 0.9, REPRESENTATION, "model-rev-7", "slice-A", "config-v3", "layer-2"),
        ),
        "global_score": 3.4,
    }
    base.update(overrides)
    return AxialLocalizationInput(**base)  # type: ignore[arg-type]


def _token_source(**overrides: object) -> AxialLocalizationInput:
    base: dict[str, object] = {
        "manifest_id": MANIFEST_ID,
        "manifest_axes": _axes(),
        "evidence": _evidence(),
        "requested_axes": ("token",),
        "token_declaration": TokenAxisDeclaration(
            sample_id="sample-7",
            sequence_id="seq-7",
            token_order=("tok-0", "tok-1", "tok-2", "tok-3"),
            token_positions={"tok-0": 0, "tok-1": 1, "tok-2": 2, "tok-3": 3},
            tokenization_identity="tokenizer-bpe-v2",
            preprocessing_identity="lowercase-strip-v1",
            representation_identity=REPRESENTATION,
        ),
        "token_cells": (
            TokenCell("tok-0", 3.8, REPRESENTATION, "sample-7", "seq-7", "tokenizer-bpe-v2", "lowercase-strip-v1"),
            TokenCell("tok-1", 3.7, REPRESENTATION, "sample-7", "seq-7", "tokenizer-bpe-v2", "lowercase-strip-v1"),
            TokenCell("tok-2", 1.1, REPRESENTATION, "sample-7", "seq-7", "tokenizer-bpe-v2", "lowercase-strip-v1"),
            TokenCell("tok-3", 0.8, REPRESENTATION, "sample-7", "seq-7", "tokenizer-bpe-v2", "lowercase-strip-v1"),
        ),
    }
    base.update(overrides)
    return AxialLocalizationInput(**base)  # type: ignore[arg-type]


def _time_source(**overrides: object) -> AxialLocalizationInput:
    base: dict[str, object] = {
        "manifest_id": MANIFEST_ID,
        "manifest_axes": _axes(),
        "evidence": _evidence(),
        "requested_axes": ("time",),
        "time_declaration": TimeAxisDeclaration(
            trajectory_id="traj-3",
            step_order=("step-0", "step-1", "step-2", "step-3"),
            step_index={"step-0": 0, "step-1": 1, "step-2": 2, "step-3": 3},
            ordering="time-increasing",
            representation_identity=REPRESENTATION,
        ),
        "time_cells": (
            TimeCell("step-0", 3.9, REPRESENTATION, "traj-3"),
            TimeCell("step-1", 3.8, REPRESENTATION, "traj-3"),
            TimeCell("step-2", 1.2, REPRESENTATION, "traj-3"),
            TimeCell("step-3", 0.7, REPRESENTATION, "traj-3"),
        ),
    }
    base.update(overrides)
    return AxialLocalizationInput(**base)  # type: ignore[arg-type]


def _feature_cells(values: tuple[float, ...] = (0.0, 1.05, 0.97, 1.02)) -> tuple[FeatureCell, ...]:
    order = ("feature-0", "feature-1", "feature-2", "feature-3")
    return tuple(
        FeatureCell(feature_id, index, value, REPRESENTATION, "ax-f")
        for index, (feature_id, value) in enumerate(zip(order, values, strict=True))
    )


def _feature_source(**overrides: object) -> AxialLocalizationInput:
    order = ("feature-0", "feature-1", "feature-2", "feature-3")
    base: dict[str, object] = {
        "manifest_id": MANIFEST_ID,
        "manifest_axes": _axes(),
        "evidence": _evidence(threshold_value=0.1, tolerance=0.01),
        "requested_axes": ("feature",),
        "feature_declaration": FeatureAxisDeclaration(
            feature_order=order,
            feature_indices={feature_id: index for index, feature_id in enumerate(order)},
            axis_identity="ax-f",
            selection="all",
            representation_identity=REPRESENTATION,
        ),
        "feature_cells": _feature_cells(),
    }
    base.update(overrides)
    return AxialLocalizationInput(**base)  # type: ignore[arg-type]


def test_aligned_checkpoint_defect_localized_under_declared_order() -> None:
    result = axial_localize(_checkpoint_source())

    assert result.verdict == "localized"
    checkpoint = result.axes[0]
    assert checkpoint.axis == "checkpoint"
    assert checkpoint.status == "localized"
    assert checkpoint.to_dict()["status"] == "localized"
    assert checkpoint.earliest == "ckpt-1"
    assert checkpoint.affected == ("ckpt-1", "ckpt-2")
    assert result.report_localization[0]["axis"] == "checkpoint"
    assert result.report_localization[0]["selection"] == "ckpt-1"


def test_affected_token_and_time_coordinates_localized_with_identities() -> None:
    token = axial_localize(_token_source())
    assert token.verdict == "localized"
    assert token.axes[0].affected == ("tok-2", "tok-3")
    assert token.axes[0].earliest == "tok-2"

    timed = axial_localize(_time_source())
    assert timed.verdict == "localized"
    assert timed.axes[0].affected == ("step-2", "step-3")
    assert timed.axes[0].earliest == "step-2"

    _, payload = evaluate_axial_localization(_time_source())
    assert payload["global_score_ignored"] is True
    assert payload["requested_axes"] == ["time"]


def test_declared_feature_axis_localizes_and_binds_every_coordinate() -> None:
    result, payload = evaluate_axial_localization(_feature_source())

    assert result.verdict == "localized"
    feature = result.axes[0]
    assert feature.axis == "feature"
    assert feature.status == "localized"
    assert feature.earliest == "feature-0"
    assert feature.affected == ("feature-0",)
    assert result.report_localization[0]["selection"] == "feature-0"
    feature_declaration = payload["feature_declaration"]
    feature_cells = payload["feature_cells"]
    assert isinstance(feature_declaration, dict)
    assert feature_declaration["axis_identity"] == "ax-f"
    assert isinstance(feature_cells, list) and len(feature_cells) == 4

    reordered = _feature_source(feature_cells=tuple(reversed(_feature_cells())))
    assert axial_localize(reordered) == result


@pytest.mark.parametrize("invalid_index", [True, 1.5, -1])
def test_feature_cell_rejects_invalid_indices(invalid_index: object) -> None:
    with pytest.raises(LocalizationError, match="feature_index must be a non-negative integer"):
        FeatureCell("feature-0", cast(int, invalid_index), 0.8, REPRESENTATION, "ax-f")


def test_feature_axis_negative_boundary_alignment_and_control_fail_closed() -> None:
    healthy = axial_localize(_feature_source(feature_cells=_feature_cells((0.8, 0.8, 0.8, 0.8))))
    assert healthy.verdict == "negative"
    assert healthy.report_localization == ()

    with pytest.raises(LocalizationError, match="within tolerance"):
        axial_localize(_feature_source(feature_cells=_feature_cells((0.095, 0.8, 0.8, 0.8))))

    bad_axes = dict(_axes())
    bad_axes["feature"] = {**bad_axes["feature"], "selection": "subset"}
    with pytest.raises(LocalizationError, match="feature selection does not match"):
        _feature_source(manifest_axes=bad_axes)

    with pytest.raises(LocalizationError, match="cover exactly"):
        _feature_source(feature_cells=_feature_cells()[:-1])

    with pytest.raises(LocalizationError, match="failed required controls block"):
        axial_localize(
            _feature_source(
                evidence=_evidence(
                    threshold_value=0.1,
                    tolerance=0.01,
                    control_outcomes={"control-a": "failed", "control-b": "passed"},
                )
            )
        )


def test_input_order_invariance_across_checkpoint_token_time() -> None:
    checkpoint = _checkpoint_source()
    shuffled_checkpoint = AxialLocalizationInput(
        manifest_id=checkpoint.manifest_id,
        manifest_axes=checkpoint.manifest_axes,
        evidence=checkpoint.evidence,
        requested_axes=checkpoint.requested_axes,
        checkpoint_declaration=checkpoint.checkpoint_declaration,
        checkpoint_cells=tuple(reversed(checkpoint.checkpoint_cells)),
        global_score=3.4,
    )
    assert axial_localize(shuffled_checkpoint).axes[0].earliest == "ckpt-1"

    token = _token_source()
    shuffled_token = AxialLocalizationInput(
        manifest_id=token.manifest_id,
        manifest_axes=token.manifest_axes,
        evidence=token.evidence,
        requested_axes=token.requested_axes,
        token_declaration=token.token_declaration,
        token_cells=tuple(reversed(token.token_cells)),
    )
    assert axial_localize(shuffled_token).axes[0].affected == ("tok-2", "tok-3")

    timed = _time_source()
    shuffled_time = AxialLocalizationInput(
        manifest_id=timed.manifest_id,
        manifest_axes=timed.manifest_axes,
        evidence=timed.evidence,
        requested_axes=timed.requested_axes,
        time_declaration=timed.time_declaration,
        time_cells=tuple(reversed(timed.time_cells)),
    )
    assert axial_localize(shuffled_time).axes[0].earliest == "step-2"


def test_absent_axis_is_explicit_non_applicability_including_mixed() -> None:
    axes = dict(_axes())
    axes.pop("time")
    mixed = AxialLocalizationInput(
        manifest_id=MANIFEST_ID,
        manifest_axes=axes,
        evidence=_evidence(),
        requested_axes=("checkpoint", "time"),
        checkpoint_declaration=CheckpointAxisDeclaration(
            checkpoint_order=("ckpt-0", "ckpt-1"),
            checkpoint_index={"ckpt-0": 0, "ckpt-1": 1},
            dataset_slice_id="slice-A",
            dataset_configuration="config-v3",
            model_identity="model-rev-7",
            representation_identity=REPRESENTATION,
        ),
        checkpoint_cells=(
            CheckpointCell("ckpt-0", 3.6, REPRESENTATION, "model-rev-7", "slice-A", "config-v3"),
            CheckpointCell("ckpt-1", 1.1, REPRESENTATION, "model-rev-7", "slice-A", "config-v3"),
        ),
    )
    result = axial_localize(mixed)

    assert result.verdict == "localized"
    by_axis = {item.axis: item for item in result.axes}
    assert by_axis["checkpoint"].status == "localized"
    assert by_axis["checkpoint"].earliest == "ckpt-1"
    assert by_axis["time"].status == "not_applicable"
    assert by_axis["time"].earliest is None
    assert all(row["axis"] == "checkpoint" for row in result.report_localization)

    only_absent = AxialLocalizationInput(
        manifest_id=MANIFEST_ID,
        manifest_axes=axes,
        evidence=_evidence(),
        requested_axes=("time",),
    )
    absent_result = axial_localize(only_absent)
    assert absent_result.verdict == "not_applicable"
    assert absent_result.report_localization == ()
    assert absent_result.axes[0].status == "not_applicable"


def test_misaligned_checkpoint_dataset_rejects() -> None:
    with pytest.raises(LocalizationError, match="dataset slice is misaligned"):
        AxialLocalizationInput(
            manifest_id=MANIFEST_ID,
            manifest_axes=_axes(),
            evidence=_evidence(),
            requested_axes=("checkpoint",),
            checkpoint_declaration=CheckpointAxisDeclaration(
                checkpoint_order=("ckpt-0", "ckpt-1"),
                checkpoint_index={"ckpt-0": 0, "ckpt-1": 1},
                dataset_slice_id="slice-A",
                dataset_configuration="config-v3",
                model_identity="model-rev-7",
                representation_identity=REPRESENTATION,
            ),
            checkpoint_cells=(
                CheckpointCell("ckpt-0", 3.6, REPRESENTATION, "model-rev-7", "slice-A", "config-v3"),
                CheckpointCell("ckpt-1", 1.1, REPRESENTATION, "model-rev-7", "slice-B", "config-v3"),
            ),
        )


def test_misaligned_token_and_time_inputs_reject() -> None:
    token = _token_source()
    drifted = tuple(
        TokenCell(
            cell.token_id,
            cell.metric_value,
            cell.representation_identity,
            cell.sample_id,
            cell.sequence_id,
            "tokenizer-word-v9",
            cell.preprocessing_identity,
        )
        for cell in token.token_cells
    )
    with pytest.raises(LocalizationError, match="tokenization identity drifted"):
        AxialLocalizationInput(
            manifest_id=token.manifest_id,
            manifest_axes=token.manifest_axes,
            evidence=token.evidence,
            requested_axes=token.requested_axes,
            token_declaration=token.token_declaration,
            token_cells=drifted,
        )

    timed = _time_source()
    crossed = tuple(
        TimeCell(cell.step_id, cell.metric_value, cell.representation_identity, "traj-9") for cell in timed.time_cells
    )
    with pytest.raises(LocalizationError, match="trajectory identity is misaligned"):
        AxialLocalizationInput(
            manifest_id=timed.manifest_id,
            manifest_axes=timed.manifest_axes,
            evidence=timed.evidence,
            requested_axes=timed.requested_axes,
            time_declaration=timed.time_declaration,
            time_cells=crossed,
        )

    with pytest.raises(LocalizationError, match="monotonic"):
        TimeAxisDeclaration(
            trajectory_id="traj-3",
            step_order=("step-0", "step-1"),
            step_index={"step-0": 1, "step-1": 0},
            ordering="time-increasing",
            representation_identity=REPRESENTATION,
        )


def test_control_blocked_boundary_and_global_only_cannot_localize() -> None:
    with pytest.raises(LocalizationError, match="failed required controls block"):
        axial_localize(
            _time_source(evidence=_evidence(control_outcomes={"control-a": "failed", "control-b": "passed"}))
        )

    with pytest.raises(LocalizationError, match="unqualified"):
        axial_localize(_token_source(evidence=_evidence(outcome="inconclusive", claim_allowed=False)))

    boundary = AxialLocalizationInput(
        manifest_id=MANIFEST_ID,
        manifest_axes=_axes(),
        evidence=_evidence(),
        requested_axes=("checkpoint",),
        checkpoint_declaration=CheckpointAxisDeclaration(
            checkpoint_order=("ckpt-0", "ckpt-1"),
            checkpoint_index={"ckpt-0": 0, "ckpt-1": 1},
            dataset_slice_id="slice-A",
            dataset_configuration="config-v3",
            model_identity="model-rev-7",
            representation_identity=REPRESENTATION,
        ),
        checkpoint_cells=(
            CheckpointCell("ckpt-0", 3.6, REPRESENTATION, "model-rev-7", "slice-A", "config-v3"),
            CheckpointCell("ckpt-1", 2.95, REPRESENTATION, "model-rev-7", "slice-A", "config-v3"),
        ),
    )
    with pytest.raises(LocalizationError, match="within tolerance"):
        axial_localize(boundary)

    with pytest.raises(LocalizationError, match="global score alone"):
        AxialLocalizationInput(
            manifest_id=MANIFEST_ID,
            manifest_axes=_axes(),
            evidence=_evidence(),
            requested_axes=("time",),
            time_declaration=TimeAxisDeclaration(
                trajectory_id="traj-3",
                step_order=("step-0",),
                step_index={"step-0": 0},
                ordering="time-increasing",
                representation_identity=REPRESENTATION,
            ),
            time_cells=(),
            global_score=0.5,
        )

    with pytest.raises(LocalizationError, match="declared feature axis requires a feature declaration"):
        AxialLocalizationInput(
            manifest_id=MANIFEST_ID,
            manifest_axes=_axes(),
            evidence=_evidence(),
            requested_axes=("feature",),
        )


def test_benign_negative_yields_explicit_negative() -> None:
    timed = _time_source()
    healthy = tuple(
        TimeCell(cell.step_id, 3.8, cell.representation_identity, cell.trajectory_id) for cell in timed.time_cells
    )
    result, payload = evaluate_axial_localization(
        AxialLocalizationInput(
            manifest_id=timed.manifest_id,
            manifest_axes=timed.manifest_axes,
            evidence=timed.evidence,
            requested_axes=timed.requested_axes,
            time_declaration=timed.time_declaration,
            time_cells=healthy,
        )
    )

    assert result.verdict == "negative"
    assert result.report_localization == ()
    assert payload["verdict"] == "negative"
    executor = make_axial_localize_executor(_time_source())
    assert executor.localizer_version == "axial-localizer-v2"  # type: ignore[attr-defined]


def test_report_workflow_canonical_and_deterministic() -> None:
    first, first_payload = evaluate_axial_localization(_checkpoint_source())
    second, second_payload = evaluate_axial_localization(_checkpoint_source())
    assert first == second
    assert first_payload == second_payload
    canonical_json(first_payload)

    manifest = _manifest()
    request = _request()
    executor = make_axial_localize_executor(_token_source())

    def _passthrough(invocation: StageInvocation) -> StageOutput:
        if invocation.stage == "localize":
            return executor(invocation)
        payload: dict[str, object] = {"stage": invocation.stage}
        if invocation.stage == "report":
            payload["report_id"] = "report-80-14"
        return StageOutput(stage=invocation.stage, outcome="completed", payload=payload, artifact_refs=())

    workflow = DiagnosticWorkflow(
        {stage: _passthrough for stage in WORKFLOW_STAGES},  # type: ignore[arg-type]
        {stage: "stage-v1" for stage in WORKFLOW_STAGES},
    )
    result, _ = workflow.run(request, manifest)
    assert result.status == "completed"
    stage = result.stage_results["stages"]["localize"]  # type: ignore[index]
    assert stage["outcome"] == "completed"
    assert stage["payload"]["requested_axes"] == ["token"]
    assert "DiagnosticWorkflow" not in str(type(executor))

    absent_only = AxialLocalizationInput(
        manifest_id=MANIFEST_ID,
        manifest_axes={axis: value for axis, value in _axes().items() if axis != "checkpoint"},
        evidence=_evidence(),
        requested_axes=("checkpoint",),
    )
    missing_executor = make_axial_localize_executor(absent_only)
    invocation = StageInvocation(
        stage="localize",
        request=request,
        manifest=manifest,
        prior=tuple(
            StageOutput(stage=stage, outcome="completed", payload={}, artifact_refs=())
            for stage in ("capture", "detect")
        ),
        workflow_identity="0" * 64,
        request_digest="1" * 64,
        manifest_digest="2" * 64,
        config_digest="3" * 64,
    )
    assert missing_executor(invocation).outcome == "not_applicable"


def test_manifest_axis_binding_and_80_13_regression() -> None:
    table = manifest_axis_lookup(_manifest())
    assert table["checkpoint"]["selection"] == "ckpt0-ckpt2"
    assert table["token"]["identity"] == "ax-tok"

    source = LocalizationInput(
        manifest_id="manifest-80-13-regression",
        evidence=DetectionEvidence(
            manifest_id="manifest-80-13-regression",
            family_id="collapse_rank_loss",
            metric_id="layer-health",
            comparator=">=",
            threshold_value=3.0,
            tolerance=0.1,
            direction="higher_is_better",
            affected_when="threshold_fail",
            required_controls=("control-a",),
            control_outcomes={"control-a": "passed"},
            outcome="supported",
            claim_allowed=True,
            representation_identity="R-80-13",
        ),
        layer_order=("layer-0", "layer-1"),
        layer_cells=(
            LayerCell("layer-0", 3.5, "R-80-13"),
            LayerCell("layer-1", 1.2, "R-80-13"),
        ),
        sample_cells=(
            SampleCell("s0", 3.5, "R-80-13"),
            SampleCell("s1", 1.1, "R-80-13"),
        ),
        declared_slice_ids=("slice-all",),
        slices=(SliceDefinition("slice-all", "all samples", ("s0", "s1")),),
    )
    result, _ = evaluate_localization(source)
    assert result.verdict == "localized"
    assert result.earliest_layer == "layer-1"
    assert localize_findings(source).affected_slices == ("slice-all",)
    assert make_localize_executor(source).localizer_version == "layer-slice-localizer-v1"  # type: ignore[attr-defined]


def test_evidence_binds_to_frozen_transformer_manifest_wiring() -> None:
    path = ARTIFACTS / "benchmark_manifest_sprint80_transformer_hidden_state_v1.json"
    manifest = dict(json.loads(path.read_text(encoding="utf-8")))
    table = manifest_axis_lookup(manifest)
    assert table["token"]["identity"] == "wikitext-token-axis-128"
    assert table["slice"]["selection"] == "validation-selected-2048"
    request = DiagnosticRequest(
        request_id="request-80-14-transformer",
        manifest_id="sprint80-core-transformer-hidden-state-probe-v1",
        capture=CaptureSelection(
            capture_id="capture-transformer",
            representation_identity="openai-community-gpt2:hidden-states:layers0-11:dim768",
            axes=("sample", "feature", "label"),
        ),
        diagnostics=DiagnosticSelection(family_ids=("separability_probe_leakage",)),
        controls=ControlSelection(
            control_ids=("control-capacity", "control-label-randomization", "control-nonseparable-negative"),
            metric_ids=("heldout-probe-accuracy", "probe-leakage-gap"),
        ),
        output=OutputSelection(output_location="artifacts/diagnostics/request-80-14"),
    )
    evidence = evidence_from_manifest(
        request,
        manifest,
        family_id="separability_probe_leakage",
        metric_id="heldout-probe-accuracy",
        affected_when="threshold_pass",
        control_outcomes={
            "control-capacity": "passed",
            "control-label-randomization": "passed",
            "control-nonseparable-negative": "passed",
        },
        outcome="supported",
        claim_allowed=True,
        representation_identity="openai-community-gpt2:hidden-states:layers0-11:dim768",
    )
    assert evidence.comparator == ">="
    assert evidence.threshold_value == 0.7
    assert evidence.direction == "higher_is_better"
    assert evidence.passes(0.8) is True
    assert evidence.is_affected(0.8) is True


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
    assert expected <= set(la.__all__)
    assert len([name for name in la.__all__ if name in expected]) == 9
    for leaked in (
        "DetectionEvidence",
        "AxialLocalizationInput",
        "axial_localize",
        "make_axial_localize_executor",
        "CheckpointCell",
        "TokenCell",
        "TimeCell",
    ):
        assert not hasattr(la, leaked)

    for name in (
        "benchmark_manifest_schema_v1.json",
        "diagnostic_report_schema_v1.json",
        "representation_problem_taxonomy_v1.json",
        "benchmark_manifest_sprint80_encoder_autoencoder_v1.json",
        "benchmark_manifest_sprint80_transformer_hidden_state_v1.json",
    ):
        assert (ARTIFACTS / name).exists()
    transformer = dict(
        json.loads(
            (ARTIFACTS / "benchmark_manifest_sprint80_transformer_hidden_state_v1.json").read_text(encoding="utf-8")
        )
    )
    assert transformer["commitment"]["manifest_sha256"] == manifest_digest(transformer)

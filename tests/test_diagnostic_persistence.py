"""Consumer-observable tests for the Sprint 80.21 diagnostic artifact persistence."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from latent_anything._benchmark_manifest import manifest_digest
from latent_anything._capture_binding import bind_selection
from latent_anything._collapse_detection import (
    detection_config_from_manifest,
    evaluate_detection,
)
from latent_anything._diagnostic_artifact import (
    DIAGNOSTIC_ARTIFACT_SCHEMA,
    DiagnosticArtifactError,
    load_diagnostic_artifact,
    persist_diagnostic_artifact,
    registered_artifact_bytes,
)
from latent_anything._diagnostic_validator import validate_diagnostic_report
from latent_anything._diagnostic_workflow import (
    WORKFLOW_STAGES,
    DiagnosticWorkflow,
    StageInvocation,
    StageOutput,
)
from latent_anything._intervention_trials import (
    Measurement,
    TrialApplication,
    TrialControl,
    TrialSpec,
    intervention_report_items,
    make_intervene_executor,
)
from latent_anything._probe_tcav_ig_explanation import (
    ExplanationHypothesis,
    MethodInputs,
    make_explain_executor,
)
from latent_anything._run_comparison import (
    ComparisonApplication,
    ComparisonMeasurement,
    ComparisonSpec,
    RunSide,
    capture_axes_identity,
    comparison_report_items,
    detect_config_identity,
    make_compare_executor,
    manifest_seed_identity,
)
from latent_anything.diagnostics import (
    CaptureSelection,
    ComparisonRequest,
    ControlSelection,
    DiagnosticRequest,
    DiagnosticSelection,
    InterventionRequest,
    OutputSelection,
)
from latent_anything.latent_value import LatentValue

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"
MANIFEST_ID = "sprint80-core-encoder-autoencoder-collapse-v1"
REPRESENTATION = "conv_vae_8x8:bottleneck-mu:latent_dim=4"
FAMILY = "collapse_rank_loss"
REP_METRIC = "bottleneck-effective-rank"
TASK_METRIC = "bottleneck-singular-spread"
SLICE_ID = "slice-back"
HYPOTHESIS_ID = "h-collapse-probe"
_BASELINE_RUN = "run-a"
_CANDIDATE_RUN = "run-b"


def _manifest() -> dict[str, object]:
    return dict(
        json.loads((ARTIFACTS / "benchmark_manifest_sprint80_encoder_autoencoder_v1.json").read_text(encoding="utf-8"))
    )


def _batches() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    collapsed = np.column_stack(
        [
            rng.normal(size=360),
            np.full(360, 2.0),
            rng.normal(size=360) * 1e-9,
            rng.normal(size=360),
        ]
    )
    healthy = rng.normal(size=(360, 4))
    scaled = rng.normal(size=(360, 4)) * 0.05
    return {"collapsed": collapsed, "healthy": healthy, "scaled": scaled}


def _value(data: np.ndarray) -> LatentValue:
    from latent_anything.latent_space import LatentSpace

    space = LatentSpace(
        dim=4,
        source_model="test-80-21",
        metadata={
            "source_representation_identity": REPRESENTATION,
            "model_version": "conv-vae-8x8-latent4-seed0-epochs5",
        },
    )
    return LatentValue(np.asarray(data, dtype=np.float64), space)


def _detect_request() -> DiagnosticRequest:
    return DiagnosticRequest(
        request_id="request-80-21-detect",
        manifest_id=MANIFEST_ID,
        capture=CaptureSelection(
            capture_id="capture-encoder",
            representation_identity=REPRESENTATION,
            axes=("sample", "feature"),
        ),
        diagnostics=DiagnosticSelection(family_ids=(FAMILY,)),
        controls=ControlSelection(
            control_ids=("control-healthy-counterexample", "control-benign-low-variance"),
            metric_ids=(REP_METRIC, TASK_METRIC),
        ),
        output=OutputSelection(output_location="artifacts/diagnostics/request-80-21-detect"),
    )


def _ablation_spec(intervention_id: str) -> TrialSpec:
    return TrialSpec(
        intervention_id=intervention_id,
        kind="ablate",
        target="bottleneck-feature-mu-dim0",
        metric_id=REP_METRIC,
        hypothesis_id=HYPOTHESIS_ID,
        strength=1.0,
        strength_semantics="fraction-of-target-activation-zeroed",
        expected_effect="decrease",
        controls=(
            TrialControl(f"{intervention_id}-zero", "zero_strength", "identity rerun reproduces the baseline"),
            TrialControl(f"{intervention_id}-random", "random", "random intervention stays below the effect"),
            TrialControl(
                f"{intervention_id}-shuffled", "shuffled", "shuffled assignment leaves the spectrum unchanged"
            ),
            TrialControl(
                f"{intervention_id}-off",
                "off_target",
                "off-target intervention leaves the spectrum unchanged",
                target="bottleneck-feature-mu-dim1",
            ),
        ),
        provenance={"method": "ablation", "carrier": "test-seam"},
    )


def _ablation_specs() -> tuple[TrialSpec, ...]:
    return (_ablation_spec("iv-ablate"), _ablation_spec("iv-inert"))


def _intervene_measure(app: TrialApplication) -> Measurement:
    if app.role == "baseline" or app.strength == 0.0 or app.control_class == "zero_strength":
        value = 2.0
    elif app.role == "intervened":
        value = 1.0 if app.intervention_id == "iv-ablate" else 2.02
    elif app.control_class == "random":
        value = 2.05
    elif app.control_class == "shuffled":
        value = 2.03
    else:
        value = 2.01
    return Measurement(app.metric_id, value, app.inputs_digest)


def _probe_hypothesis(manifest: dict[str, object]) -> ExplanationHypothesis:
    dataset = manifest["dataset"]
    assert isinstance(dataset, dict)
    return ExplanationHypothesis(
        hypothesis_id=HYPOTHESIS_ID,
        symptom_id="symptom-collapse",
        family_id=FAMILY,
        target_id="bottleneck-mu",
        representation_id=REPRESENTATION,
        layer_id="bottleneck-mu",
        slice_id=SLICE_ID,
        method="probe",
        expected_direction="higher",
        dataset_id=str(dataset["split_identity"]),
        train_split_identity="split-train-A",
        eval_split_identity="split-eval-A",
        seeds=(7, 8),
        control_ids=(
            "capacity:control-probe-capacity",
            "randomized:control-probe-randomized",
            "negative:control-probe-negative",
        ),
        metric_ids=(REP_METRIC,),
        thresholds=(("heldout_accuracy", ">=", 0.7), ("leakage_gap", ">=", 0.15)),
        manifest_id=MANIFEST_ID,
        localization_bindings=(("slice", SLICE_ID),),
    )


def _probe_bundle() -> dict[str, object]:
    rng = np.random.default_rng(0)
    n, dim = 200, 6
    matrix = rng.normal(size=(n, dim))
    matrix[:100] += 1.5
    matrix[100:] -= 1.5
    labels = np.array([0] * 100 + [1] * 100)
    perm = np.random.default_rng(79).permutation(n)
    train = np.array([int(item) for item in perm[:140]])
    evaluation = np.array([int(item) for item in perm[140:]])
    neg = rng.normal(size=(n, dim))
    from latent_anything.probes import _fast_probe

    negative = float(_fast_probe(neg[train], labels[train], neg[evaluation], labels[evaluation], 7).accuracy)
    train_set = set(int(i) for i in train)
    eval_set = set(int(i) for i in evaluation)
    return {
        "train_matrix": matrix[train],
        "eval_matrix": matrix[evaluation],
        "train_labels": labels[train],
        "eval_labels": labels[evaluation],
        "train_ids": [f"sample-{i}" for i in range(n) if i in train_set],
        "eval_ids": [f"sample-{i}" for i in range(n) if i in eval_set],
        "capacity": "linear-logreg-C1.0-standardized",
        "negative_accuracy": negative,
    }


def _localize_payload() -> dict[str, object]:
    return {
        "affected_layers": [],
        "affected_slices": [SLICE_ID],
        "declared_slice_ids": [SLICE_ID],
        "earliest_layer": None,
        "family_id": FAMILY,
        "layer_order": [],
        "manifest_id": MANIFEST_ID,
        "metric_id": REP_METRIC,
        "report_localization": [],
        "representation_identity": REPRESENTATION,
        "verdict": "localized",
    }


def _request() -> DiagnosticRequest:
    interventions = tuple(
        InterventionRequest(
            intervention_id=spec.intervention_id,
            target=spec.target,
            control_ids=tuple(control.control_id for control in spec.controls),
        )
        for spec in _ablation_specs()
    )
    control_ids: list[str] = []
    for entry in interventions:
        control_ids.extend(entry.control_ids)
    return DiagnosticRequest(
        request_id="request-80-21",
        manifest_id=MANIFEST_ID,
        capture=CaptureSelection(
            capture_id="capture-encoder",
            representation_identity=REPRESENTATION,
            axes=("sample", "feature"),
        ),
        diagnostics=DiagnosticSelection(family_ids=(FAMILY,)),
        controls=ControlSelection(
            control_ids=tuple(control_ids),
            metric_ids=(REP_METRIC, TASK_METRIC),
        ),
        interventions=interventions,
        comparisons=(
            ComparisonRequest(
                comparison_id="cmp-both",
                baseline_run=_BASELINE_RUN,
                candidate_run=_CANDIDATE_RUN,
                metric_ids=(REP_METRIC, TASK_METRIC),
            ),
            ComparisonRequest(
                comparison_id="cmp-rep-only",
                baseline_run=_BASELINE_RUN,
                candidate_run=_CANDIDATE_RUN,
                metric_ids=(REP_METRIC, TASK_METRIC),
            ),
        ),
        output=OutputSelection(output_location="artifacts/diagnostics/request-80-21"),
    )


def _alignment(
    manifest: dict[str, object],
    capture_provenance: dict[str, object],
    detect_payload: dict[str, object],
) -> dict[str, str]:
    dataset = manifest["dataset"]
    model = manifest["model"]
    assert isinstance(dataset, dict) and isinstance(model, dict)
    return {
        "axes": capture_axes_identity(capture_provenance),
        "dataset_configuration": str(dataset["split_identity"]),
        "dataset_slice_id": SLICE_ID,
        "diagnostic_config": detect_config_identity(detect_payload),
        "layer_module_identity": REPRESENTATION,
        "manifest_identity": MANIFEST_ID,
        "model_identity": str(model["id"]),
        "preprocessing_identity": "digits-scale-01-v1",
        "representation_identity": REPRESENTATION,
        "representation_metric": REP_METRIC,
        "schema_identity": str(manifest["schema_version"]),
        "seeds": manifest_seed_identity(manifest),
        "taxonomy_identity": FAMILY,
        "task_metric": TASK_METRIC,
    }


def _comparison_specs(alignment: dict[str, str]) -> tuple[ComparisonSpec, ...]:
    return (
        ComparisonSpec(
            comparison_id="cmp-both",
            baseline=RunSide(run_id=_BASELINE_RUN, checkpoint_id="ckpt-000100", alignment=alignment),
            candidate=RunSide(run_id=_CANDIDATE_RUN, checkpoint_id="ckpt-000200", alignment=alignment),
            representation_metric_id=REP_METRIC,
            task_metric_id=TASK_METRIC,
            representation_tolerance=0.1,
            task_tolerance=0.05,
        ),
        ComparisonSpec(
            comparison_id="cmp-rep-only",
            baseline=RunSide(run_id=_BASELINE_RUN, checkpoint_id="ckpt-000100", alignment=alignment),
            candidate=RunSide(run_id=_CANDIDATE_RUN, checkpoint_id="ckpt-000200", alignment=alignment),
            representation_metric_id=REP_METRIC,
            task_metric_id=TASK_METRIC,
            representation_tolerance=0.1,
            task_tolerance=0.05,
        ),
    )


def _compare_measure(app: ComparisonApplication) -> ComparisonMeasurement:
    values = {
        ("cmp-both", "baseline", "representation"): 2.0,
        ("cmp-both", "candidate", "representation"): 2.6,
        ("cmp-both", "baseline", "task"): 0.40,
        ("cmp-both", "candidate", "task"): 0.50,
        ("cmp-rep-only", "baseline", "representation"): 2.0,
        ("cmp-rep-only", "candidate", "representation"): 2.5,
        ("cmp-rep-only", "baseline", "task"): 0.40,
        ("cmp-rep-only", "candidate", "task"): 0.42,
    }
    return ComparisonMeasurement(
        app.run_id, app.metric_id, values[(app.comparison_id, app.side, app.metric_role)], app.inputs_digest
    )


def _report(
    intervention_claims: list[dict[str, object]],
    comparison_rows: list[dict[str, object]],
    manifest: dict[str, object],
    *,
    report_id: str = "report-80-21",
) -> dict[str, object]:
    model = manifest["model"]
    dataset = manifest["dataset"]
    assert isinstance(model, dict) and isinstance(dataset, dict)
    return {
        "schema_version": "diagnostic-report-schema-v1",
        "report_id": report_id,
        "status": "declared",
        "capture_provenance": {
            "captures": [
                {
                    "artifact_refs": ["cap-1"],
                    "axes": ["sample", "feature"],
                    "capture_id": "capture-encoder",
                    "dataset_split": "heldout",
                    "model_revision": str(model["revision"]),
                    "representation_identity": REPRESENTATION,
                }
            ]
        },
        "symptoms": [
            {
                "description": "bottleneck rank collapse",
                "evidence_refs": ["o-1"],
                "id": "s-1",
                "metric_ids": [REP_METRIC],
                "status": "observed",
            }
        ],
        "localization": [
            {
                "axis": "slice",
                "confidence": 0.9,
                "evidence_refs": ["o-1"],
                "id": "l-1",
                "selection": SLICE_ID,
                "status": "observed",
            }
        ],
        "hypotheses": [
            {
                "alternatives": ["benign-low-variance"],
                "evidence_refs": ["o-1"],
                "id": HYPOTHESIS_ID,
                "statement": "the collapsed bottleneck direction causes the rank defect",
                "status": "supported",
            }
        ],
        "statistical_evidence": [
            {
                "control_refs": [
                    "control-healthy-counterexample",
                    "control-benign-low-variance",
                ],
                "estimate": 2.0,
                "evidence_refs": ["o-1"],
                "id": "e-1",
                "metric_id": REP_METRIC,
                "status": "observed",
                "uncertainty": {"kind": "interval", "lower": 1.9, "upper": 2.1},
            }
        ],
        "interventions": [],
        "comparisons": comparison_rows,
        "limitations": [
            {
                "affects": ["causal"],
                "blocking": False,
                "description": "the localize payload is a contract-shaped stand-in in tests",
                "id": "lim-1",
            }
        ],
        "next_action": {
            "action": "render the AI-engineer report",
            "rationale": "80.22 owns rendering",
            "required_evidence_refs": ["e-1", "o-1"],
        },
        "claims": [
            {
                "causal": False,
                "claim": "the bottleneck shows a rank-collapse symptom",
                "claim_allowed": True,
                "control_refs": [],
                "evidence_refs": ["cap-1"],
                "id": "o-1",
                "kind": "observation",
                "missing_evidence": [],
                "status": "observed",
            },
            *intervention_claims,
        ],
    }


def _completed_run(
    stop_after: str | None = None,
) -> tuple[DiagnosticRequest, dict[str, object], Any, Any, dict[str, object]]:
    manifest = _manifest()
    batches = _batches()
    request = _request()

    # Real detect evidence computed once through the production seam.
    detect_config = detection_config_from_manifest(_detect_request(), manifest)
    _, detect_payload = evaluate_detection(
        _value(batches["collapsed"]),
        detect_config,
        controls={
            "control-healthy-counterexample": _value(batches["healthy"]),
            "control-benign-low-variance": _value(batches["scaled"]),
        },
    )
    bound = bind_selection(request.capture, manifest=manifest, request_id=request.request_id)
    alignment = _alignment(manifest, bound.provenance(), detect_payload)

    explain = make_explain_executor((_probe_hypothesis(manifest),), MethodInputs(probe=_probe_bundle()))
    intervene = make_intervene_executor(_ablation_specs(), _intervene_measure)
    compare = make_compare_executor(_comparison_specs(alignment), _compare_measure)
    localize_payload = _localize_payload()

    def _stage(invocation: StageInvocation) -> StageOutput:
        if invocation.stage == "capture":
            return StageOutput(
                stage="capture",
                outcome="completed",
                payload={"capture": bound.provenance()},
                artifact_refs=(),
            )
        if invocation.stage == "detect":
            return StageOutput(stage="detect", outcome="completed", payload=dict(detect_payload), artifact_refs=())
        if invocation.stage == "localize":
            return StageOutput(stage="localize", outcome="completed", payload=localize_payload, artifact_refs=())
        if invocation.stage == "explain":
            return explain(invocation)
        if invocation.stage == "intervene":
            return intervene(invocation)
        if invocation.stage == "compare":
            return compare(invocation)
        payload: dict[str, object] = {"stage": invocation.stage}
        if invocation.stage == "report":
            payload["report_id"] = "report-80-21"
        return StageOutput(stage=invocation.stage, outcome="completed", payload=payload, artifact_refs=())

    executors = {stage: _stage for stage in WORKFLOW_STAGES}
    versions = {stage: "stage-v1" for stage in WORKFLOW_STAGES}
    workflow = DiagnosticWorkflow(executors, versions)  # type: ignore[arg-type]
    result, checkpoint = workflow.run(request, manifest, stop_after=stop_after)

    payload_by_stage = {output.stage: dict(output.payload) for output in checkpoint.outputs}
    intervention_claims = intervention_report_items(
        list(payload_by_stage.get("intervene", {}).get("trials", []))  # type: ignore[arg-type]
    )
    comparison_rows = comparison_report_items(
        list(payload_by_stage.get("compare", {}).get("comparisons", []))  # type: ignore[arg-type]
    )
    report = _report(intervention_claims, comparison_rows, manifest)
    return request, manifest, result, checkpoint, report


def test_persist_load_and_revalidate_complete_workflow(tmp_path: Path) -> None:
    request, manifest, result, checkpoint, report = _completed_run()
    handle = persist_diagnostic_artifact(
        tmp_path / "root",
        request=request,
        manifest=manifest,
        result=result,
        checkpoint=checkpoint,
        report=report,
    )
    assert len(handle.artifact_digest) == 64
    assert len(handle.report_digest) == 64

    loaded = load_diagnostic_artifact(tmp_path / "root", run_id=handle.run_id, manifest=manifest)
    assert loaded.artifact_digest == handle.artifact_digest
    document = loaded.document
    assert document["schema"] == DIAGNOSTIC_ARTIFACT_SCHEMA
    assert [stage["stage"] for stage in document["stages"]] == list(WORKFLOW_STAGES)  # type: ignore[index]
    assert document["result"]["status"] == "completed"  # type: ignore[index]
    assert document["validator_result"]["status"] == "passed"  # type: ignore[index]
    assert document["hashes"]["manifest_sha256"] == manifest_digest(manifest)  # type: ignore[index]

    # Registered rows:80.18 intervention claims and80.20 comparison rows resolve.
    registered = loaded.report
    intervention_ids = {row["id"] for row in registered["interventions"]}  # type: ignore[index]
    assert intervention_ids == {"intervention-iv-ablate-record", "intervention-iv-inert-record"}
    comparison_ids = {row["id"] for row in registered["comparisons"]}  # type: ignore[index]
    assert comparison_ids == {"cmp-both", "cmp-rep-only"}
    causal_claims = [
        claim
        for claim in registered["claims"]
        if claim["kind"] == "causal_result"  # type: ignore[index]
    ]
    assert {claim["status"] for claim in causal_claims} == {"supported", "falsified"}
    manifest_controls = {control["id"] for control in manifest["controls"]}  # type: ignore[index]
    for claim in causal_claims:
        assert set(claim["control_refs"]).issubset(manifest_controls)
    capture_row = registered["capture_provenance"]["captures"][0]  # type: ignore[index]
    assert capture_row["dataset_split"] == manifest["dataset"]["split_identity"]  # type: ignore[index]
    assert capture_row["artifact_refs"] == ["capture-capture-encoder-record"]

    # Independent revalidation with the persisted inputs and real bytes.
    blobs = registered_artifact_bytes(loaded, tmp_path / "root")
    validator_input = document["validator_input"]
    validate_diagnostic_report(
        registered,
        manifest,
        applicability=validator_input["applicability"],  # type: ignore[arg-type]
        family_evidence=validator_input["family_evidence"],  # type: ignore[arg-type]
        control_outcomes=validator_input["control_outcomes"],  # type: ignore[arg-type]
        artifacts=blobs,
        artifact_digests=validator_input["artifact_digests"],  # type: ignore[arg-type]
    )
    # Blob bytes are real content-addressed stage/evidence records.
    stage_record = json.loads(blobs["stage-record-capture"])
    assert stage_record["payload"]["capture"]["capture_id"] == "capture-encoder"
    trial_record = json.loads(blobs["stage-record-intervene-iv-ablate"])
    assert trial_record["intervention_id"] == "iv-ablate"
    assert json.loads(blobs["comparison-cmp-both-record"])["classification"] == "both"


def test_identical_input_yields_identical_bytes_and_idempotent_writes(tmp_path: Path) -> None:
    request, manifest, result, checkpoint, report = _completed_run()
    root_a = tmp_path / "root-a"
    root_b = tmp_path / "root-b"
    first = persist_diagnostic_artifact(
        root_a, request=request, manifest=manifest, result=result, checkpoint=checkpoint, report=report
    )
    second = persist_diagnostic_artifact(
        root_b, request=request, manifest=manifest, result=result, checkpoint=checkpoint, report=report
    )
    assert first.artifact_digest == second.artifact_digest
    assert first.run_id == second.run_id
    assert first.report_digest == second.report_digest

    def _tree_bytes(root: Path) -> dict[str, bytes]:
        return {str(path.relative_to(root)): path.read_bytes() for path in sorted(root.rglob("*")) if path.is_file()}

    def _run_bytes_without_timestamps(root: Path) -> dict[str, bytes]:
        stripped: dict[str, bytes] = {}
        for path in sorted((root / "runs").glob("*.json")):
            record = json.loads(path.read_text(encoding="utf-8"))
            record.pop("created_at", None)
            record.pop("updated_at", None)
            stripped[path.name] = json.dumps(record, sort_keys=True).encode("utf-8")
        return stripped

    tree_a = _tree_bytes(root_a)
    tree_b = _tree_bytes(root_b)
    assert set(tree_a) == set(tree_b)
    # Every artifact byte is identical; the run index differs only in
    # lifecycle timestamps, which never affect identity or run_id.
    for relative, data in tree_a.items():
        if relative.startswith("runs"):
            continue
        assert tree_b[relative] == data, relative
    assert _run_bytes_without_timestamps(root_a) == _run_bytes_without_timestamps(root_b)

    # Re-persisting identical input rewrites nothing.
    before = _tree_bytes(root_a)
    third = persist_diagnostic_artifact(
        root_a, request=request, manifest=manifest, result=result, checkpoint=checkpoint, report=report
    )
    assert third.artifact_digest == first.artifact_digest
    after = _tree_bytes(root_a)
    assert before == after


def test_fresh_process_load_uses_only_root_and_declared_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request, manifest, result, checkpoint, report = _completed_run()
    root = tmp_path / "root"
    handle = persist_diagnostic_artifact(
        root, request=request, manifest=manifest, result=result, checkpoint=checkpoint, report=report
    )
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    loaded = load_diagnostic_artifact(root, run_id=handle.run_id, manifest=manifest)
    assert loaded.artifact_digest == handle.artifact_digest
    assert loaded.report["report_id"] == "report-80-21"  # type: ignore[index]


def test_corrupt_missing_and_mis_hashed_blobs_reject(tmp_path: Path) -> None:
    request, manifest, result, checkpoint, report = _completed_run()
    root = tmp_path / "root"
    handle = persist_diagnostic_artifact(
        root, request=request, manifest=manifest, result=result, checkpoint=checkpoint, report=report
    )

    # Corrupt one blob in place (same path, wrong bytes).
    stage_path = root / "artifacts" / _stage_record_digest(checkpoint, "detect")
    stage_path.write_bytes(b"corrupted\n")
    with pytest.raises(DiagnosticArtifactError, match="rejected"):
        load_diagnostic_artifact(root, run_id=handle.run_id, manifest=manifest)
    # Restore by re-persisting would conflict; rebuild a clean root instead.
    clean_root = tmp_path / "clean-root"
    clean_handle = persist_diagnostic_artifact(
        clean_root, request=request, manifest=manifest, result=result, checkpoint=checkpoint, report=report
    )

    # Delete the report blob.
    (clean_root / "artifacts" / clean_handle.report_digest).unlink()
    with pytest.raises(DiagnosticArtifactError, match="missing"):
        load_diagnostic_artifact(clean_root, run_id=clean_handle.run_id, manifest=manifest)


def _stage_record_digest(checkpoint: Any, stage: str) -> str:
    from hashlib import sha256

    from latent_anything._run_record_codec import canonical_json

    for output in checkpoint.outputs:
        if output.stage == stage:
            return sha256(canonical_json(output.to_dict()) + b"\n").hexdigest()
    raise AssertionError(stage)


def test_schema_provenance_and_validator_disagreements_reject(tmp_path: Path) -> None:
    request, manifest, result, checkpoint, report = _completed_run()
    root = tmp_path / "root"
    handle = persist_diagnostic_artifact(
        root, request=request, manifest=manifest, result=result, checkpoint=checkpoint, report=report
    )

    # Manifest provenance disagreement: a different manifest hash rejects.
    other_manifest = copy.deepcopy(manifest)
    other_manifest["manifest_id"] = "other-manifest"
    with pytest.raises(DiagnosticArtifactError, match="manifest hash disagrees"):
        load_diagnostic_artifact(root, run_id=handle.run_id, manifest=other_manifest)
    # Same manifest_id but altered content also rejects (digest covers content).
    altered = copy.deepcopy(manifest)
    altered["status"] = "predeclared-but-different"
    with pytest.raises(DiagnosticArtifactError, match="manifest hash disagrees"):
        load_diagnostic_artifact(root, run_id=handle.run_id, manifest=altered)

    # Tampered document schema with updated run metadata still rejects.
    document_bytes = (root / "artifacts" / handle.artifact_digest).read_bytes()
    document = json.loads(document_bytes)
    document["schema"] = "diagnostic-artifact-v0-retired"
    tampered_bytes = json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
    from hashlib import sha256 as _sha256

    tampered_digest = _sha256(tampered_bytes).hexdigest()
    (root / "artifacts" / tampered_digest).write_bytes(tampered_bytes)
    _rewrite_run_metadata(
        root / "runs" / f"{handle.run_id}.json",
        digest=tampered_digest,
        size=len(tampered_bytes),
    )
    with pytest.raises(DiagnosticArtifactError, match="schema must be"):
        load_diagnostic_artifact(root, run_id=handle.run_id, manifest=manifest)

    # Validator-result disagreement: document claims a failed validation.
    honest_root = tmp_path / "honest-root"
    honest_handle = persist_diagnostic_artifact(
        honest_root, request=request, manifest=manifest, result=result, checkpoint=checkpoint, report=report
    )
    honest_bytes = (honest_root / "artifacts" / honest_handle.artifact_digest).read_bytes()
    honest_document = json.loads(honest_bytes)
    honest_document["validator_result"]["status"] = "failed"
    flipped_bytes = json.dumps(honest_document, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
    flipped_digest = _sha256(flipped_bytes).hexdigest()
    (honest_root / "artifacts" / flipped_digest).write_bytes(flipped_bytes)
    _rewrite_run_metadata(
        honest_root / "runs" / f"{honest_handle.run_id}.json",
        digest=flipped_digest,
        size=len(flipped_bytes),
    )
    with pytest.raises(DiagnosticArtifactError, match="validator result must be a passing"):
        load_diagnostic_artifact(honest_root, run_id=honest_handle.run_id, manifest=manifest)


def _rewrite_run_metadata(run_path: Path, *, digest: str, size: int) -> None:
    """Point a run record at a replacement document and keep its identity valid."""
    from latent_anything._run_record_codec import compute_run_identity

    record = json.loads(run_path.read_text(encoding="utf-8"))
    record["metadata"]["diagnostic_artifact_digest"] = digest
    record["metadata"]["diagnostic_artifact_size"] = size
    record["identity"] = compute_run_identity(
        name=record["name"],
        config=record["config"],
        code_version=record["code_version"],
        framework_version=record["framework_version"],
        model_revisions=record["model_revisions"],
        dataset_revisions=record["dataset_revisions"],
        seeds=record["seeds"],
        environment=record["environment"],
        parent_run_ids=record["parent_run_ids"],
        metadata=record["metadata"],
    )
    run_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_incomplete_workflow_rejects_before_any_write(tmp_path: Path) -> None:
    request, manifest, result, checkpoint, report = _completed_run(stop_after="intervene")
    root = tmp_path / "partial-root"
    with pytest.raises(DiagnosticArtifactError):
        persist_diagnostic_artifact(
            root, request=request, manifest=manifest, result=result, checkpoint=checkpoint, report=report
        )
    runs_dir = root / "runs"
    if runs_dir.exists():
        assert list(runs_dir.glob("*.json")) == []
    artifacts_dir = root / "artifacts"
    if artifacts_dir.exists():
        assert list(artifacts_dir.iterdir()) == []


def test_conflicting_blob_rejects_before_any_write(tmp_path: Path) -> None:
    request, manifest, result, checkpoint, report = _completed_run()
    root = tmp_path / "conflict-root"
    # Pre-seed one blob digest with conflicting bytes.
    from hashlib import sha256 as _sha256

    from latent_anything._run_record_codec import canonical_json

    detect_output = next(o for o in checkpoint.outputs if o.stage == "detect")
    digest = _sha256(canonical_json(detect_output.to_dict()) + b"\n").hexdigest()
    artifacts = root / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    (artifacts / digest).write_bytes(b"conflicting-bytes\n")
    with pytest.raises(DiagnosticArtifactError, match="conflicting blob"):
        persist_diagnostic_artifact(
            root, request=request, manifest=manifest, result=result, checkpoint=checkpoint, report=report
        )
    runs_dir = root / "runs"
    if runs_dir.exists():
        assert list(runs_dir.glob("*.json")) == []


def test_report_failing_independent_validation_persists_nothing(tmp_path: Path) -> None:
    request, manifest, result, checkpoint, report = _completed_run()
    bad_report = copy.deepcopy(report)
    observation = next(
        claim
        for claim in bad_report["claims"]
        if claim["id"] == "o-1"  # type: ignore[index]
    )
    observation["evidence_refs"] = ["ghost-record"]  # unknown evidence reference
    root = tmp_path / "bad-report-root"
    with pytest.raises(DiagnosticArtifactError, match="failed independent validation"):
        persist_diagnostic_artifact(
            root, request=request, manifest=manifest, result=result, checkpoint=checkpoint, report=bad_report
        )
    runs_dir = root / "runs"
    if runs_dir.exists():
        assert list(runs_dir.glob("*.json")) == []


def test_persisted_document_has_no_hidden_paths_or_secrets(tmp_path: Path) -> None:
    request, manifest, result, checkpoint, report = _completed_run()
    root = tmp_path / "root"
    handle = persist_diagnostic_artifact(
        root, request=request, manifest=manifest, result=result, checkpoint=checkpoint, report=report
    )
    document_bytes = (root / "artifacts" / handle.artifact_digest).read_bytes()
    document_text = document_bytes.decode("utf-8")
    for pattern in ("F:\\\\", "C:\\\\", "/home/", "/Users/", "\\\\appdata"):
        assert pattern not in document_text, pattern
    assert set(json.loads(document_text)["environment"]) == {"numpy", "python", "system"}
    for key, value in os.environ.items():
        if any(token in key for token in ("PATH", "SECRET", "TOKEN", "KEY", "HOME")) and len(value) >= 8:
            assert value not in document_text, key
    # All persisted blob paths are relative content-address references.
    for blob in json.loads(document_text)["blobs"]:
        assert blob["relative_path"] == f"artifacts/{blob['digest']}"


def _intervention_row(
    row_id: str, *, status: str = "supported", target: str = "bottleneck-feature-mu-dim0"
) -> dict[str, object]:
    return {
        "control_refs": [],
        "evidence_refs": ["cap-1"],
        "id": row_id,
        "intervention": "ablate",
        "outcome": {"conclusion": status, "record_digest": "0" * 64},
        "status": status,
        "target": target,
    }


def _with_intervention_rows(report: dict[str, object], rows: list[dict[str, object]]) -> dict[str, object]:
    mutated = copy.deepcopy(report)
    mutated["interventions"] = rows
    return mutated


def test_unbacked_duplicate_and_mismatched_intervention_rows_reject_before_write(
    tmp_path: Path,
) -> None:
    request, manifest, result, checkpoint, report = _completed_run()

    # Unbacked row: no recorded trial backs it.
    unbacked_root = tmp_path / "unbacked-root"
    with pytest.raises(DiagnosticArtifactError, match="not backed by a registered trial"):
        persist_diagnostic_artifact(
            unbacked_root,
            request=request,
            manifest=manifest,
            result=result,
            checkpoint=checkpoint,
            report=_with_intervention_rows(report, [_intervention_row("intervention-ghost-record")]),
        )
    assert not unbacked_root.exists() or not list((unbacked_root / "runs").glob("*.json"))

    # Duplicate row ids are rejected, never collapsed silently.
    duplicate_root = tmp_path / "duplicate-root"
    with pytest.raises(DiagnosticArtifactError, match="duplicate interventions row id"):
        persist_diagnostic_artifact(
            duplicate_root,
            request=request,
            manifest=manifest,
            result=result,
            checkpoint=checkpoint,
            report=_with_intervention_rows(
                report,
                [
                    _intervention_row("intervention-iv-ablate-record"),
                    _intervention_row("intervention-iv-ablate-record"),
                ],
            ),
        )
    assert not duplicate_root.exists() or not list((duplicate_root / "runs").glob("*.json"))

    # A supported row over a falsified recorded trial disagrees and rejects.
    mismatched_root = tmp_path / "mismatched-root"
    with pytest.raises(DiagnosticArtifactError, match="disagrees with the recorded trial conclusion"):
        persist_diagnostic_artifact(
            mismatched_root,
            request=request,
            manifest=manifest,
            result=result,
            checkpoint=checkpoint,
            report=_with_intervention_rows(
                report, [_intervention_row("intervention-iv-inert-record", status="supported")]
            ),
        )
    assert not mismatched_root.exists() or not list((mismatched_root / "runs").glob("*.json"))

    # Target disagreement also rejects.
    target_root = tmp_path / "target-root"
    with pytest.raises(DiagnosticArtifactError, match="target disagrees"):
        persist_diagnostic_artifact(
            target_root,
            request=request,
            manifest=manifest,
            result=result,
            checkpoint=checkpoint,
            report=_with_intervention_rows(
                report,
                [
                    _intervention_row(
                        "intervention-iv-ablate-record",
                        status="supported",
                        target="some-other-target",
                    )
                ],
            ),
        )
    assert not target_root.exists() or not list((target_root / "runs").glob("*.json"))


def test_capture_reference_collision_rejects_before_write(tmp_path: Path) -> None:
    request, manifest, result, checkpoint, report = _completed_run()
    for index, collision in enumerate(("stage-record-compare", "comparison-cmp-both-record", "stage-record-intervene")):
        root = tmp_path / f"collision-root-{index}"
        mutated = copy.deepcopy(report)
        mutated["capture_provenance"]["captures"][0]["artifact_refs"] = [  # type: ignore[index]
            "cap-1",
            collision,
        ]
        with pytest.raises(DiagnosticArtifactError, match="collides with a registered evidence name"):
            persist_diagnostic_artifact(
                root,
                request=request,
                manifest=manifest,
                result=result,
                checkpoint=checkpoint,
                report=mutated,
            )
        assert not root.exists() or not list((root / "runs").glob("*.json"))


def test_mid_write_failure_leaves_running_record_and_retry_completes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from latent_anything.run_record import FileSystemRunRecorder

    request, manifest, result, checkpoint, report = _completed_run()
    root = tmp_path / "root"
    original = FileSystemRunRecorder.add_artifact
    calls = {"count": 0}

    def _flaky(
        self: FileSystemRunRecorder,
        run_id: str,
        content: object,
        *,
        name: str,
        media_type: str = "application/octet-stream",
    ) -> object:
        calls["count"] += 1
        if calls["count"] == 3:
            raise RuntimeError("injected mid-write failure")
        return original(self, run_id, content, name=name, media_type=media_type)  # type: ignore[arg-type]

    monkeypatch.setattr(FileSystemRunRecorder, "add_artifact", _flaky)
    with pytest.raises(RuntimeError, match="injected mid-write failure"):
        persist_diagnostic_artifact(
            root, request=request, manifest=manifest, result=result, checkpoint=checkpoint, report=report
        )
    monkeypatch.undo()

    # A mid-write failure leaves a recoverable non-completed record that is
    # never surfaced as completed.
    run_files = list((root / "runs").glob("*.json"))
    assert len(run_files) == 1
    mid_state = json.loads(run_files[0].read_text(encoding="utf-8"))
    assert mid_state["status"] == "running"
    with pytest.raises(DiagnosticArtifactError, match="must be completed"):
        load_diagnostic_artifact(root, run_id=mid_state["run_id"], manifest=manifest)

    # Retry with identical input reuses the deterministic identity and
    # completes safely.
    handle = persist_diagnostic_artifact(
        root, request=request, manifest=manifest, result=result, checkpoint=checkpoint, report=report
    )
    assert handle.run_id == mid_state["run_id"]
    loaded = load_diagnostic_artifact(root, run_id=handle.run_id, manifest=manifest)
    assert loaded.artifact_digest == handle.artifact_digest
    completed = json.loads(run_files[0].read_text(encoding="utf-8"))
    assert completed["status"] == "completed"
    assert completed["identity"] == mid_state["identity"]

    # An already completed identical run is never downgraded or rewritten.
    bytes_before = run_files[0].read_bytes()
    again = persist_diagnostic_artifact(
        root, request=request, manifest=manifest, result=result, checkpoint=checkpoint, report=report
    )
    assert again.artifact_digest == handle.artifact_digest
    assert run_files[0].read_bytes() == bytes_before
    assert json.loads(run_files[0].read_text(encoding="utf-8"))["status"] == "completed"


def test_invalid_run_id_is_normalized_to_diagnostic_artifact_error(
    tmp_path: Path,
) -> None:
    request, manifest, _, _, _ = _completed_run()
    with pytest.raises(DiagnosticArtifactError, match="invalid run id"):
        load_diagnostic_artifact(tmp_path / "root", run_id="../evil", manifest=manifest)
    with pytest.raises(DiagnosticArtifactError, match="invalid run id"):
        load_diagnostic_artifact(tmp_path / "root", run_id="a/b", manifest=manifest)
    with pytest.raises(DiagnosticArtifactError, match="invalid run id"):
        load_diagnostic_artifact(tmp_path / "root", run_id="", manifest=manifest)
    del request


def completed_run_full_outcomes() -> tuple[DiagnosticRequest, dict[str, object], Any, Any, dict[str, object]]:
    """One real chain whose persisted records cover all four renderer outcomes.

    Interventions: supported (iv-ablate), falsified (iv-inert), and
    unsupported (iv-blocked, identity control broken). Comparisons: one
    observed change pair and one inconclusive (noisy uncertainty).
    """
    manifest = _manifest()
    batches = _batches()

    specs = (
        _ablation_spec("iv-ablate"),
        _ablation_spec("iv-inert"),
        _ablation_spec("iv-blocked"),
    )
    control_ids: list[str] = []
    interventions = []
    for spec in specs:
        ids = tuple(control.control_id for control in spec.controls)
        control_ids.extend(ids)
        interventions.append(
            InterventionRequest(intervention_id=spec.intervention_id, target=spec.target, control_ids=ids)
        )
    request = DiagnosticRequest(
        request_id="request-80-21-full",
        manifest_id=MANIFEST_ID,
        capture=CaptureSelection(
            capture_id="capture-encoder",
            representation_identity=REPRESENTATION,
            axes=("sample", "feature"),
        ),
        diagnostics=DiagnosticSelection(family_ids=(FAMILY,)),
        controls=ControlSelection(
            control_ids=tuple(control_ids),
            metric_ids=(REP_METRIC, TASK_METRIC),
        ),
        interventions=tuple(interventions),
        comparisons=(
            ComparisonRequest(
                comparison_id="cmp-both",
                baseline_run=_BASELINE_RUN,
                candidate_run=_CANDIDATE_RUN,
                metric_ids=(REP_METRIC, TASK_METRIC),
            ),
            ComparisonRequest(
                comparison_id="cmp-inconclusive",
                baseline_run=_BASELINE_RUN,
                candidate_run=_CANDIDATE_RUN,
                metric_ids=(REP_METRIC, TASK_METRIC),
            ),
        ),
        output=OutputSelection(output_location="artifacts/diagnostics/request-80-21-full"),
    )

    detect_config = detection_config_from_manifest(_detect_request(), manifest)
    _, detect_payload = evaluate_detection(
        _value(batches["collapsed"]),
        detect_config,
        controls={
            "control-healthy-counterexample": _value(batches["healthy"]),
            "control-benign-low-variance": _value(batches["scaled"]),
        },
    )
    bound = bind_selection(request.capture, manifest=manifest, request_id=request.request_id)
    alignment = _alignment(manifest, bound.provenance(), detect_payload)
    explain = make_explain_executor((_probe_hypothesis(manifest),), MethodInputs(probe=_probe_bundle()))

    def _measure(app: TrialApplication) -> Measurement:
        if app.role == "baseline":
            value = 2.0
        elif app.control_class == "zero_strength":
            value = 2.5 if app.intervention_id == "iv-blocked" else 2.0
        elif app.role == "intervened":
            value = {"iv-ablate": 1.0, "iv-inert": 2.02, "iv-blocked": 1.0}[app.intervention_id]
        elif app.control_class == "random":
            value = 2.05
        elif app.control_class == "shuffled":
            value = 2.03
        else:
            value = 2.01
        return Measurement(app.metric_id, value, app.inputs_digest)

    intervene = make_intervene_executor(specs, _measure)

    full_specs = (
        ComparisonSpec(
            comparison_id="cmp-both",
            baseline=RunSide(run_id=_BASELINE_RUN, checkpoint_id="ckpt-000100", alignment=alignment),
            candidate=RunSide(run_id=_CANDIDATE_RUN, checkpoint_id="ckpt-000200", alignment=alignment),
            representation_metric_id=REP_METRIC,
            task_metric_id=TASK_METRIC,
            representation_tolerance=0.1,
            task_tolerance=0.05,
        ),
        ComparisonSpec(
            comparison_id="cmp-inconclusive",
            baseline=RunSide(run_id=_BASELINE_RUN, checkpoint_id="ckpt-000100", alignment=alignment),
            candidate=RunSide(run_id=_CANDIDATE_RUN, checkpoint_id="ckpt-000200", alignment=alignment),
            representation_metric_id=REP_METRIC,
            task_metric_id=TASK_METRIC,
            representation_tolerance=0.1,
            task_tolerance=0.05,
        ),
    )
    compare_values: dict[tuple[str, str, str], float] = {
        ("cmp-both", "baseline", "representation"): 2.0,
        ("cmp-both", "candidate", "representation"): 2.6,
        ("cmp-both", "baseline", "task"): 0.40,
        ("cmp-both", "candidate", "task"): 0.50,
        ("cmp-inconclusive", "baseline", "representation"): 2.0,
        ("cmp-inconclusive", "candidate", "representation"): 2.6,
        ("cmp-inconclusive", "baseline", "task"): 0.40,
        ("cmp-inconclusive", "candidate", "task"): 0.42,
    }

    def _compare(app: ComparisonApplication) -> ComparisonMeasurement:
        if (
            app.rng is not None
            and app.comparison_id == "cmp-inconclusive"
            and app.side == "candidate"
            and app.metric_role == "representation"
        ):
            value = 2.6 + float(app.rng.normal(0.0, 0.5))
        else:
            value = compare_values[(app.comparison_id, app.side, app.metric_role)]
        return ComparisonMeasurement(app.run_id, app.metric_id, value, app.inputs_digest)

    compare = make_compare_executor(full_specs, _compare)
    localize_payload = _localize_payload()

    def _stage(invocation: StageInvocation) -> StageOutput:
        if invocation.stage == "capture":
            return StageOutput(
                stage="capture",
                outcome="completed",
                payload={"capture": bound.provenance()},
                artifact_refs=(),
            )
        if invocation.stage == "detect":
            return StageOutput(stage="detect", outcome="completed", payload=dict(detect_payload), artifact_refs=())
        if invocation.stage == "localize":
            return StageOutput(stage="localize", outcome="completed", payload=localize_payload, artifact_refs=())
        if invocation.stage == "explain":
            return explain(invocation)
        if invocation.stage == "intervene":
            return intervene(invocation)
        if invocation.stage == "compare":
            return compare(invocation)
        payload: dict[str, object] = {"stage": invocation.stage}
        if invocation.stage == "report":
            payload["report_id"] = "report-80-21-full"
        return StageOutput(stage=invocation.stage, outcome="completed", payload=payload, artifact_refs=())

    executors = {stage: _stage for stage in WORKFLOW_STAGES}
    versions = {stage: "stage-v1" for stage in WORKFLOW_STAGES}
    workflow = DiagnosticWorkflow(executors, versions)  # type: ignore[arg-type]
    result, checkpoint = workflow.run(request, manifest)
    payload_by_stage = {output.stage: dict(output.payload) for output in checkpoint.outputs}
    claims = intervention_report_items(
        list(payload_by_stage.get("intervene", {}).get("trials", []))  # type: ignore[arg-type]
    )
    rows = comparison_report_items(
        list(payload_by_stage.get("compare", {}).get("comparisons", []))  # type: ignore[arg-type]
    )
    report = _report(claims, rows, manifest, report_id="report-80-21-full")
    return request, manifest, result, checkpoint, report


def failed_control_run() -> tuple[DiagnosticRequest, dict[str, object], Any, Any, dict[str, object]]:
    """One real chain where a required manifest control failed.

    The detect-stage counterexample receives the defective batch so its
    control outcome is ``failed`` and the detection stays inconclusive;
    the report carries no promoted causal claims so the artifact may be
    persisted, letting a renderer show the failed control explicitly.
    """
    manifest = _manifest()
    batches = _batches()
    request = _request()

    detect_config = detection_config_from_manifest(_detect_request(), manifest)
    _, detect_payload = evaluate_detection(
        _value(batches["collapsed"]),
        detect_config,
        controls={
            "control-healthy-counterexample": _value(batches["collapsed"]),
            "control-benign-low-variance": _value(batches["scaled"]),
        },
    )
    families = detect_payload["families"]
    assert families[0]["outcome"] == "inconclusive"  # type: ignore[index]
    assert families[0]["control_outcomes"]["control-healthy-counterexample"] == "failed"  # type: ignore[index]

    bound = bind_selection(request.capture, manifest=manifest, request_id=request.request_id)
    alignment = _alignment(manifest, bound.provenance(), detect_payload)
    explain = make_explain_executor((_probe_hypothesis(manifest),), MethodInputs(probe=_probe_bundle()))
    intervene = make_intervene_executor(_ablation_specs(), _intervene_measure)
    compare = make_compare_executor(_comparison_specs(alignment), _compare_measure)
    localize_payload = _localize_payload()

    def _stage(invocation: StageInvocation) -> StageOutput:
        if invocation.stage == "capture":
            return StageOutput(
                stage="capture",
                outcome="completed",
                payload={"capture": bound.provenance()},
                artifact_refs=(),
            )
        if invocation.stage == "detect":
            return StageOutput(stage="detect", outcome="completed", payload=dict(detect_payload), artifact_refs=())
        if invocation.stage == "localize":
            return StageOutput(stage="localize", outcome="completed", payload=localize_payload, artifact_refs=())
        if invocation.stage == "explain":
            return explain(invocation)
        if invocation.stage == "intervene":
            return intervene(invocation)
        if invocation.stage == "compare":
            return compare(invocation)
        payload: dict[str, object] = {"stage": invocation.stage}
        if invocation.stage == "report":
            payload["report_id"] = "report-80-21-failed-control"
        return StageOutput(stage=invocation.stage, outcome="completed", payload=payload, artifact_refs=())

    executors = {stage: _stage for stage in WORKFLOW_STAGES}
    versions = {stage: "stage-v1" for stage in WORKFLOW_STAGES}
    workflow = DiagnosticWorkflow(executors, versions)  # type: ignore[arg-type]
    result, checkpoint = workflow.run(request, manifest)
    payload_by_stage = {output.stage: dict(output.payload) for output in checkpoint.outputs}
    rows = comparison_report_items(
        list(payload_by_stage.get("compare", {}).get("comparisons", []))  # type: ignore[arg-type]
    )
    report = _report([], rows, manifest, report_id="report-80-21-failed-control")
    return request, manifest, result, checkpoint, report

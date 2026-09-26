"""Consumer-observable tests for the Sprint 80.20 aligned comparison executor."""

from __future__ import annotations

import json
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path

import pytest

from latent_anything._benchmark_manifest import manifest_digest
from latent_anything._capture_binding import bind_selection
from latent_anything._diagnostic_report import validate_report_shape
from latent_anything._diagnostic_workflow import (
    WORKFLOW_STAGES,
    DiagnosticWorkflow,
    StageContractError,
    StageInvocation,
    StageOutput,
)
from latent_anything._run_comparison import (
    ALIGNMENT_FIELDS,
    COMPARISON_CLASSIFICATIONS,
    COMPARISON_VERSION,
    ComparisonApplication,
    ComparisonError,
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
    OutputSelection,
)

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
    path = ARTIFACTS / "benchmark_manifest_sprint80_encoder_autoencoder_v1.json"
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _detect_payload() -> dict[str, object]:
    return {
        "config": {
            "manifest_id": MANIFEST_ID,
            "metric_ids": [REP_METRIC, TASK_METRIC],
            "thresholds": [
                {"metric_id": REP_METRIC, "comparator": ">=", "value": 3.0, "tolerance": 0.1},
            ],
        },
        "families": [
            {
                "family_id": FAMILY,
                "observed_metrics": {REP_METRIC: 2.0, TASK_METRIC: 0.3},
                "threshold_pass": {REP_METRIC: False},
            }
        ],
    }


def _localize_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "affected_layers": [],
        "affected_slices": [SLICE_ID],
        "declared_slice_ids": [SLICE_ID, "slice-all"],
        "earliest_layer": None,
        "family_id": FAMILY,
        "layer_order": [],
        "manifest_id": MANIFEST_ID,
        "metric_id": REP_METRIC,
        "report_localization": [],
        "representation_identity": REPRESENTATION,
        "verdict": "localized",
    }
    payload.update(overrides)
    return payload


def _explain_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "hypotheses": [
            {
                "hypothesis_id": HYPOTHESIS_ID,
                "family_id": FAMILY,
                "metric_ids": [REP_METRIC],
                "representation_id": REPRESENTATION,
                "manifest_id": MANIFEST_ID,
            }
        ],
        "family_evidence": {
            HYPOTHESIS_ID: {
                "claim_allowed": True,
                "method": "probe",
                "outcome": "supported",
            }
        },
    }
    payload.update(overrides)
    return payload


def _intervene_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "conclusions": {"iv-ablate": "supported"},
        "intervention_version": "intervention-trials-v1",
        "trials": [{"intervention_id": "iv-ablate", "kind": "ablate"}],
    }
    payload.update(overrides)
    return payload


def _capture_payload(request: DiagnosticRequest, manifest: dict[str, object]) -> dict[str, object]:
    bound = bind_selection(request.capture, manifest=manifest, request_id=request.request_id)
    return {"capture": bound.provenance()}


def _alignment(
    request: DiagnosticRequest,
    manifest: dict[str, object],
    *,
    overrides: dict[str, str] | None = None,
) -> dict[str, str]:
    capture = _capture_payload(request, manifest)["capture"]
    assert isinstance(capture, dict)
    dataset = manifest["dataset"]
    assert isinstance(dataset, dict)
    model = manifest["model"]
    assert isinstance(model, dict)
    alignment = {
        "axes": capture_axes_identity(capture),
        "dataset_configuration": str(dataset["split_identity"]),
        "dataset_slice_id": SLICE_ID,
        "diagnostic_config": detect_config_identity(_detect_payload()),
        "layer_module_identity": REPRESENTATION,
        "manifest_identity": MANIFEST_ID,
        "model_identity": str(model["id"]),
        "preprocessing_identity": "standard-scale-v1",
        "representation_identity": REPRESENTATION,
        "representation_metric": REP_METRIC,
        "schema_identity": str(manifest["schema_version"]),
        "seeds": manifest_seed_identity(manifest),
        "taxonomy_identity": FAMILY,
        "task_metric": TASK_METRIC,
    }
    assert set(alignment) == set(ALIGNMENT_FIELDS)
    if overrides:
        alignment.update(overrides)
    return alignment


def _sides(request: DiagnosticRequest, manifest: dict[str, object]) -> tuple[RunSide, RunSide]:
    alignment = _alignment(request, manifest)
    return (
        RunSide(run_id=_BASELINE_RUN, checkpoint_id="ckpt-000100", alignment=alignment),
        RunSide(run_id=_CANDIDATE_RUN, checkpoint_id="ckpt-000200", alignment=alignment),
    )


def _spec(
    comparison_id: str = "cmp-both",
    *,
    request: DiagnosticRequest | None = None,
    manifest: dict[str, object] | None = None,
    representation_tolerance: float = 0.1,
    task_tolerance: float = 0.05,
    alignment_overrides: dict[str, str] | None = None,
    candidate_overrides: dict[str, str] | None = None,
) -> ComparisonSpec:
    request = request if request is not None else _request()
    manifest = manifest if manifest is not None else _manifest()
    baseline_alignment = _alignment(request, manifest, overrides=alignment_overrides)
    candidate_alignment = dict(baseline_alignment)
    if candidate_overrides:
        candidate_alignment.update(candidate_overrides)
    return ComparisonSpec(
        comparison_id=comparison_id,
        baseline=RunSide(run_id=_BASELINE_RUN, checkpoint_id="ckpt-000100", alignment=baseline_alignment),
        candidate=RunSide(run_id=_CANDIDATE_RUN, checkpoint_id="ckpt-000200", alignment=candidate_alignment),
        representation_metric_id=REP_METRIC,
        task_metric_id=TASK_METRIC,
        representation_tolerance=representation_tolerance,
        task_tolerance=task_tolerance,
    )


def _entry(comparison_id: str, *, metric_ids: tuple[str, ...] = (REP_METRIC, TASK_METRIC)) -> ComparisonRequest:
    return ComparisonRequest(
        comparison_id=comparison_id,
        baseline_run=_BASELINE_RUN,
        candidate_run=_CANDIDATE_RUN,
        metric_ids=metric_ids,
    )


def _request(
    comparisons: tuple[ComparisonRequest, ...] | None = None,
    *,
    metric_ids: tuple[str, ...] = (REP_METRIC, TASK_METRIC),
) -> DiagnosticRequest:
    return DiagnosticRequest(
        request_id="request-80-20",
        manifest_id=MANIFEST_ID,
        capture=CaptureSelection(
            capture_id="capture-encoder",
            representation_identity=REPRESENTATION,
            axes=("sample", "feature"),
        ),
        diagnostics=DiagnosticSelection(family_ids=(FAMILY,)),
        controls=ControlSelection(
            control_ids=("control-healthy-counterexample",),
            metric_ids=metric_ids,
        ),
        comparisons=comparisons or (_entry("cmp-both"),),
        output=OutputSelection(output_location="artifacts/diagnostics/request-80-20"),
    )


def _prior(
    request: DiagnosticRequest,
    manifest: dict[str, object],
    *,
    capture_payload: dict[str, object] | None = None,
    detect: dict[str, object] | None = None,
    localize: dict[str, object] | None = None,
    explain: dict[str, object] | None = None,
    intervene: dict[str, object] | None = None,
    intervene_outcome: str = "completed",
) -> tuple[StageOutput, ...]:
    return (
        StageOutput(
            stage="capture",
            outcome="completed",
            payload=capture_payload if capture_payload is not None else _capture_payload(request, manifest),
            artifact_refs=(),
        ),
        StageOutput(
            stage="detect",
            outcome="completed",
            payload=detect if detect is not None else _detect_payload(),
            artifact_refs=(),
        ),
        StageOutput(
            stage="localize",
            outcome="completed",
            payload=localize if localize is not None else _localize_payload(),
            artifact_refs=(),
        ),
        StageOutput(
            stage="explain",
            outcome="completed",
            payload=explain if explain is not None else _explain_payload(),
            artifact_refs=(),
        ),
        StageOutput(
            stage="intervene",
            outcome=intervene_outcome,
            payload=intervene if intervene is not None else _intervene_payload(),
            artifact_refs=(),
        ),
    )


def _invocation(
    request: DiagnosticRequest,
    manifest: dict[str, object],
    prior: tuple[StageOutput, ...],
    *,
    stage: str = "compare",
    prior_override: tuple[StageOutput, ...] | None = None,
) -> StageInvocation:
    return StageInvocation(
        stage=stage,
        request=request,
        manifest=manifest,
        prior=prior_override if prior_override is not None else prior,
        workflow_identity="a" * 64,
        request_digest="b" * 64,
        manifest_digest="c" * 64,
        config_digest="d" * 64,
    )


_Values = dict[tuple[str, str, str], float]


def _measure_factory(
    values: _Values,
    *,
    record: list[ComparisonApplication] | None = None,
    draw: Callable[[ComparisonApplication], float] | None = None,
) -> Callable[[ComparisonApplication], ComparisonMeasurement]:
    def _measure(app: ComparisonApplication) -> ComparisonMeasurement:
        if record is not None:
            record.append(app)
        if app.rng is not None and draw is not None:
            value = float(draw(app))
        else:
            value = float(values[(app.comparison_id, app.side, app.metric_role)])
        return ComparisonMeasurement(app.run_id, app.metric_id, value, app.inputs_digest)

    return _measure


def _run(
    specs: tuple[ComparisonSpec, ...],
    measure: Callable[[ComparisonApplication], ComparisonMeasurement],
    *,
    request: DiagnosticRequest | None = None,
    manifest: dict[str, object] | None = None,
    prior: tuple[StageOutput, ...] | None = None,
) -> StageOutput:
    manifest = manifest if manifest is not None else _manifest()
    request = request if request is not None else _request(tuple(_entry(spec.comparison_id) for spec in specs))
    executor = make_compare_executor(specs, measure)
    return executor(_invocation(request, manifest, prior if prior is not None else _prior(request, manifest)))


def test_aligned_comparison_records_both_change_with_full_provenance() -> None:
    spec = _spec("cmp-both")
    request = _request((_entry("cmp-both"),))
    manifest = _manifest()
    record: list[ComparisonApplication] = []
    values: _Values = {
        ("cmp-both", "baseline", "representation"): 2.0,
        ("cmp-both", "candidate", "representation"): 2.6,
        ("cmp-both", "baseline", "task"): 1.0,
        ("cmp-both", "candidate", "task"): 1.4,
    }
    output = _run(
        (spec,),
        _measure_factory(values, record=record),
        request=request,
        manifest=manifest,
    )

    assert output.stage == "compare"
    assert output.outcome == "completed"
    assert output.payload["comparison_version"] == COMPARISON_VERSION
    classifications = dict(output.payload["classifications"])  # type: ignore[arg-type]
    assert classifications == {"cmp-both": "both"}
    row = list(output.payload["comparisons"])[0]  # type: ignore[arg-type]
    assert row["classification"] == "both"
    assert set(row) == {
        "alignment",
        "baseline",
        "candidate",
        "classification",
        "comparison_id",
        "provenance",
        "reason",
        "representation",
        "task",
    }
    assert row["baseline"]["run_id"] == _BASELINE_RUN
    assert row["baseline"]["checkpoint_id"] == "ckpt-000100"
    assert row["candidate"]["run_id"] == _CANDIDATE_RUN
    assert row["candidate"]["checkpoint_id"] == "ckpt-000200"
    assert row["alignment"] == spec.baseline.alignment

    rep = row["representation"]
    assert rep["metric_id"] == REP_METRIC
    assert rep["signed_delta"] == pytest.approx(0.6)
    assert rep["absolute_delta"] == pytest.approx(0.6)
    assert rep["tolerance"] == pytest.approx(0.1)
    assert rep["changed"] is True and rep["resolved"] is True
    assert rep["delta_interval"] == pytest.approx([0.6, 0.6])
    assert rep["definition"]["direction"] == "higher_is_better"
    assert rep["definition"]["taxonomy_family_id"] == FAMILY
    assert rep["baseline"]["interval"]["repetitions"] == 200  # type: ignore[index]
    assert rep["candidate"]["interval"]["repetitions"] == 200  # type: ignore[index]
    task = row["task"]
    assert task["metric_id"] == TASK_METRIC
    assert task["signed_delta"] == pytest.approx(0.4)
    assert task["tolerance"] == pytest.approx(0.05)
    assert task["changed"] is True and task["resolved"] is True

    bound = bind_selection(request.capture, manifest=manifest, request_id=request.request_id)
    provenance = row["provenance"]
    assert provenance["capture_identity"] == bound.capture_identity  # type: ignore[index]
    assert provenance["detect_config_identity"] == detect_config_identity(_detect_payload())  # type: ignore[index]
    assert provenance["localize_family_id"] == FAMILY  # type: ignore[index]
    assert provenance["workflow_identity"] == "a" * 64  # type: ignore[index]
    assert provenance["upstream_blocked_by"] == []  # type: ignore[index]

    # Every callback saw the shared captured inputs and only declared metrics.
    assert {app.inputs_digest for app in record} == {bound.capture_identity}
    assert {app.metric_id for app in record} == {REP_METRIC, TASK_METRIC}
    assert {(app.side, app.metric_role) for app in record} == {
        ("baseline", "representation"),
        ("candidate", "representation"),
        ("baseline", "task"),
        ("candidate", "task"),
    }
    # 4 point measurements + 4 sides x 200 uncertainty draws.
    assert len(record) == 804
    point_seeds = {app.stream_seed for app in record if app.rng is None}
    assert len(point_seeds) == 4

    items = comparison_report_items([row])
    assert items == [
        {
            "alignment": dict(spec.baseline.alignment),
            "baseline": _BASELINE_RUN,
            "candidate": _CANDIDATE_RUN,
            "evidence_refs": ["comparison-cmp-both-record"],
            "id": "cmp-both",
            "metric_ids": [REP_METRIC, TASK_METRIC],
            "status": "observed",
        }
    ]


def test_representation_only_task_only_and_neither_are_distinct() -> None:
    rep_only = _run(
        (_spec("cmp-rep"),),
        _measure_factory(
            {
                ("cmp-rep", "baseline", "representation"): 2.0,
                ("cmp-rep", "candidate", "representation"): 2.5,
                ("cmp-rep", "baseline", "task"): 1.0,
                ("cmp-rep", "candidate", "task"): 1.02,
            }
        ),
    )
    row = list(rep_only.payload["comparisons"])[0]  # type: ignore[arg-type]
    assert row["classification"] == "representation_only"
    assert row["representation"]["changed"] is True
    assert row["task"]["changed"] is False  # no task change is inferred
    assert "no task-metric change is claimed or inferred" in str(row["reason"])
    items = comparison_report_items([row])
    assert items[0]["status"] == "observed"

    task_only = _run(
        (_spec("cmp-task"),),
        _measure_factory(
            {
                ("cmp-task", "baseline", "representation"): 2.0,
                ("cmp-task", "candidate", "representation"): 2.02,
                ("cmp-task", "baseline", "task"): 1.0,
                ("cmp-task", "candidate", "task"): 1.3,
            }
        ),
    )
    row = list(task_only.payload["comparisons"])[0]  # type: ignore[arg-type]
    assert row["classification"] == "task_only"
    assert row["representation"]["changed"] is False  # no representation change is inferred
    assert row["task"]["changed"] is True

    neither = _run(
        (_spec("cmp-neither"),),
        _measure_factory(
            {
                ("cmp-neither", "baseline", "representation"): 2.0,
                ("cmp-neither", "candidate", "representation"): 2.02,
                ("cmp-neither", "baseline", "task"): 1.0,
                ("cmp-neither", "candidate", "task"): 1.01,
            }
        ),
    )
    row = list(neither.payload["comparisons"])[0]  # type: ignore[arg-type]
    assert row["classification"] == "neither"
    assert row["representation"]["changed"] is False
    assert row["task"]["changed"] is False
    assert neither.outcome == "completed"


def test_unresolved_delta_interval_is_inconclusive() -> None:
    values: _Values = {
        ("cmp-uncertain", "baseline", "representation"): 2.0,
        ("cmp-uncertain", "candidate", "representation"): 2.6,
        ("cmp-uncertain", "baseline", "task"): 1.0,
        ("cmp-uncertain", "candidate", "task"): 1.4,
    }

    def _draw(app: ComparisonApplication) -> float:
        assert app.rng is not None
        if app.side == "candidate" and app.metric_role == "representation":
            return 2.6 + float(app.rng.normal(0.0, 0.5))
        return float(values[(app.comparison_id, app.side, app.metric_role)])

    output = _run(
        (_spec("cmp-uncertain"),),
        _measure_factory(values, draw=_draw),
    )
    row = list(output.payload["comparisons"])[0]  # type: ignore[arg-type]
    assert row["classification"] == "inconclusive"
    rep = row["representation"]
    assert rep["changed"] is True
    assert rep["resolved"] is False
    delta_lower, delta_upper = rep["delta_interval"]
    assert delta_lower < 0.0 < delta_upper  # type: ignore[operator]
    assert "delta interval spans zero" in str(row["reason"])
    items = comparison_report_items([row])
    assert items[0]["status"] == "inconclusive"


def test_upstream_unsupported_blocks_the_classification() -> None:
    manifest = _manifest()
    request = _request()
    prior = _prior(request, manifest, intervene_outcome="unsupported")
    values: _Values = {
        ("cmp-both", "baseline", "representation"): 2.0,
        ("cmp-both", "candidate", "representation"): 2.6,
        ("cmp-both", "baseline", "task"): 1.0,
        ("cmp-both", "candidate", "task"): 1.4,
    }
    record: list[ComparisonApplication] = []
    output = _run(
        (_spec("cmp-both", request=request, manifest=manifest),),
        _measure_factory(values, record=record),
        request=request,
        manifest=manifest,
        prior=prior,
    )
    assert output.outcome == "unsupported"
    classifications = dict(output.payload["classifications"])  # type: ignore[arg-type]
    assert classifications == {"cmp-both": "unsupported"}
    row = list(output.payload["comparisons"])[0]  # type: ignore[arg-type]
    assert row["provenance"]["upstream_blocked_by"] == ["intervene"]  # type: ignore[index]
    assert "unsupported outcome" in str(row["reason"])
    # Measurements are still recorded; the classification is never promoted.
    assert len(record) == 804
    assert row["representation"]["signed_delta"] == pytest.approx(0.6)
    items = comparison_report_items([row])
    assert items[0]["status"] == "unsupported"


def test_declaration_validation_fails_closed() -> None:
    manifest = _manifest()
    request = _request()

    with pytest.raises(ComparisonError, match="identical"):
        _spec("cmp-mismatch", request=request, manifest=manifest, candidate_overrides={"seeds": "training=9"})
    with pytest.raises(ComparisonError, match="missing"):
        RunSide(
            run_id="run-x",
            checkpoint_id="",
            alignment={key: "v" for key in ALIGNMENT_FIELDS if key != "seeds"},
        )
    with pytest.raises(ComparisonError, match="unexpected"):
        RunSide(
            run_id="run-x",
            checkpoint_id="",
            alignment={**{key: "v" for key in ALIGNMENT_FIELDS}, "extra": "v"},
        )
    with pytest.raises(ComparisonError, match="non-empty"):
        _spec("cmp-empty", request=request, manifest=manifest, alignment_overrides={"seeds": ""})
    with pytest.raises(ComparisonError, match="must differ"):
        ComparisonSpec(
            comparison_id="cmp-same-metric",
            baseline=_sides(request, manifest)[0],
            candidate=_sides(request, manifest)[1],
            representation_metric_id=REP_METRIC,
            task_metric_id=REP_METRIC,
            representation_tolerance=0.1,
            task_tolerance=0.05,
        )
    with pytest.raises(ComparisonError, match="different run identities"):
        alignment = _alignment(request, manifest)
        RunSide(run_id="run-same", checkpoint_id="", alignment=alignment)
        ComparisonSpec(
            comparison_id="cmp-same-run",
            baseline=RunSide(run_id="run-same", checkpoint_id="", alignment=alignment),
            candidate=RunSide(run_id="run-same", checkpoint_id="", alignment=alignment),
            representation_metric_id=REP_METRIC,
            task_metric_id=TASK_METRIC,
            representation_tolerance=0.1,
            task_tolerance=0.05,
        )
    with pytest.raises(ComparisonError, match="finite"):
        _spec("cmp-nan", request=request, manifest=manifest, representation_tolerance=float("nan"))
    with pytest.raises(ComparisonError, match="non-negative"):
        _spec("cmp-neg", request=request, manifest=manifest, task_tolerance=-0.1)
    with pytest.raises(ComparisonError, match="at least one comparison"):
        make_compare_executor((), _measure_factory({}))
    with pytest.raises(ComparisonError, match="unique"):
        make_compare_executor((_spec("cmp-dup"), _spec("cmp-dup")), _measure_factory({}))
    with pytest.raises(ComparisonError, match="ComparisonSpec"):
        make_compare_executor(("not-a-spec",), _measure_factory({}))  # type: ignore[arg-type]
    with pytest.raises(ComparisonError, match="callable"):
        make_compare_executor((_spec("cmp-x"),), "not-callable")  # type: ignore[arg-type]


def test_alignment_mismatch_and_prior_tamper_reject_before_callbacks() -> None:
    manifest = _manifest()
    request = _request()
    prior = _prior(request, manifest)
    record: list[ComparisonApplication] = []
    executor = make_compare_executor((_spec(),), _measure_factory({}, record=record))

    # Cross-check failures against prior/manifest evidence.
    bad_alignment_cases: list[dict[str, str]] = [
        {"dataset_configuration": "other-split-identity"},
        {"dataset_slice_id": "slice-nowhere"},
        {"seeds": "training=99"},
        {"diagnostic_config": "f" * 64},
        {"manifest_identity": "other-manifest"},
        {"taxonomy_identity": "anisotropy_inactive_dimensions"},
        {"schema_identity": "other-schema"},
        {"axes": "sample"},
        {"model_identity": "other-model"},
        {"representation_identity": "other:representation"},
        {"layer_module_identity": "somewhere-else"},
        {"representation_metric": "bottleneck-singular-spread"},
        {"task_metric": "bottleneck-effective-rank"},
    ]
    for overrides in bad_alignment_cases:
        bad_spec = _spec("cmp-both", request=request, manifest=manifest, alignment_overrides=overrides)
        bad_executor = make_compare_executor((bad_spec,), _measure_factory({}, record=record))
        with pytest.raises(StageContractError):
            bad_executor(_invocation(request, manifest, prior))
    assert record == []

    # Side-to-side alignment mismatch is rejected at declaration.
    with pytest.raises(ComparisonError, match="identical"):
        _spec("cmp-side", candidate_overrides={"preprocessing_identity": "other-preprocessing"})

    # Prior record tampering.
    capture = _capture_payload(request, manifest)
    capture_without_identity = {
        "capture": {
            key: value
            for key, value in dict(capture["capture"]).items()  # type: ignore[arg-type]
            if key != "capture_identity"
        }
    }
    capture_without_axes = {
        "capture": {**dict(capture["capture"]), "axes": []}  # type: ignore[arg-type]
    }
    detect_without_config = {"families": []}
    detect_without_task_metric = {
        "config": {"manifest_id": MANIFEST_ID, "metric_ids": [REP_METRIC]},
        "families": [{"family_id": FAMILY, "observed_metrics": {REP_METRIC: 2.0}}],
    }
    tampered_cases: list[tuple[StageOutput, ...]] = [
        _prior(request, manifest, capture_payload=capture_without_identity),
        _prior(request, manifest, capture_payload=capture_without_axes),
        _prior(request, manifest, detect=detect_without_config),
        _prior(request, manifest, detect=detect_without_task_metric),
        _prior(request, manifest, localize=_localize_payload(family_id="anisotropy_inactive_dimensions")),
        _prior(request, manifest, localize=_localize_payload(declared_slice_ids=[], affected_slices=[])),
        _prior(request, manifest, explain=_explain_payload(hypotheses=[])),
        _prior(request, manifest, explain=_explain_payload(family_evidence={})),
        _prior(
            request,
            manifest,
            explain=_explain_payload(family_evidence={HYPOTHESIS_ID: {"outcome": "omitted", "method": "probe"}}),
        ),
        _prior(request, manifest, intervene={"conclusions": {"iv-ablate": "supported"}}),
        _prior(request, manifest, intervene={"trials": [{"intervention_id": "iv-ablate"}]}),
    ]
    for tampered in tampered_cases:
        with pytest.raises(StageContractError):
            executor(_invocation(request, manifest, tampered))
    assert record == []

    # Request-declared comparison identity mismatches.
    wrong_run = _request((_entry("cmp-both"),))
    wrong_run_entry = ComparisonRequest(
        comparison_id="cmp-both",
        baseline_run="run-other",
        candidate_run=_CANDIDATE_RUN,
        metric_ids=(REP_METRIC, TASK_METRIC),
    )
    mismatched_request = _request((wrong_run_entry,))
    spec = _spec("cmp-both")
    mismatched_executor = make_compare_executor((spec,), _measure_factory({}, record=record))
    with pytest.raises(StageContractError, match="baseline run"):
        mismatched_executor(_invocation(mismatched_request, manifest, _prior(mismatched_request, manifest)))
    del wrong_run

    wrong_metrics = _request((_entry("cmp-both", metric_ids=(REP_METRIC,)),), metric_ids=(REP_METRIC, TASK_METRIC))
    with pytest.raises(StageContractError, match="metric set"):
        make_compare_executor((spec,), _measure_factory({}, record=record))(
            _invocation(wrong_metrics, manifest, _prior(wrong_metrics, manifest))
        )

    extra = _request((_entry("cmp-both"), _entry("cmp-extra")))
    with pytest.raises(StageContractError, match="match exactly"):
        make_compare_executor((spec,), _measure_factory({}, record=record))(
            _invocation(extra, manifest, _prior(extra, manifest))
        )

    # Wrong stage rejects before callbacks.
    report_prior = prior + (StageOutput(stage="compare", outcome="completed", payload={}, artifact_refs=()),)
    with pytest.raises(StageContractError, match="received stage"):
        executor(_invocation(request, manifest, prior, stage="report", prior_override=report_prior))
    assert record == []


def test_measurement_contract_violations_fail_closed() -> None:
    manifest = _manifest()
    request = _request()
    prior = _prior(request, manifest)
    invocation = _invocation(request, manifest, prior)
    spec = _spec("cmp-both", request=request, manifest=manifest)
    good: _Values = {
        ("cmp-both", "baseline", "representation"): 2.0,
        ("cmp-both", "candidate", "representation"): 2.6,
        ("cmp-both", "baseline", "task"): 1.0,
        ("cmp-both", "candidate", "task"): 1.4,
    }

    def _wrong_side(app: ComparisonApplication) -> ComparisonMeasurement:
        other = _CANDIDATE_RUN if app.side == "baseline" else _BASELINE_RUN
        return ComparisonMeasurement(other, app.metric_id, 2.0, app.inputs_digest)

    with pytest.raises(ComparisonError, match="returned side"):
        make_compare_executor((spec,), _wrong_side)(invocation)

    wrong_metric = make_compare_executor(
        (spec,),
        lambda app: ComparisonMeasurement(app.run_id, "bottleneck-singular-spread", 1.0, app.inputs_digest),
    )
    with pytest.raises(ComparisonError, match="domain mismatch"):
        wrong_metric(invocation)

    wrong_digest = make_compare_executor(
        (spec,),
        lambda app: ComparisonMeasurement(app.run_id, app.metric_id, 1.0, "f" * 64),
    )
    with pytest.raises(ComparisonError, match="inputs mismatch"):
        wrong_digest(invocation)

    non_finite = make_compare_executor(
        (spec,),
        lambda app: ComparisonMeasurement(app.run_id, app.metric_id, float("inf"), app.inputs_digest),
    )
    with pytest.raises(ComparisonError, match="finite"):
        non_finite(invocation)

    wrong_type = make_compare_executor(
        (spec,),
        lambda _app: {"run_id": _app.run_id},  # type: ignore[arg-type,return-value]
    )
    with pytest.raises(ComparisonError, match="must return a ComparisonMeasurement"):
        wrong_type(invocation)

    crashing = make_compare_executor((spec,), _boom)
    with pytest.raises(RuntimeError, match="comparison model exploded"):
        crashing(invocation)

    def _nan_draw(app: ComparisonApplication) -> float:
        if app.rng is not None and app.side == "candidate":
            return float("nan")
        return float(good[(app.comparison_id, app.side, app.metric_role)])

    nan_uncertainty = make_compare_executor(
        (spec,),
        _measure_factory(good, draw=_nan_draw),
    )
    with pytest.raises(ComparisonError, match="uncertainty failed"):
        nan_uncertainty(invocation)


def _boom(_app_unused: ComparisonApplication) -> ComparisonMeasurement:
    raise RuntimeError("comparison model exploded")


def test_deterministic_replay_reproduces_identical_payload_and_digest() -> None:
    manifest = _manifest()
    request = _request()
    spec = _spec("cmp-both", request=request, manifest=manifest)
    values: _Values = {
        ("cmp-both", "baseline", "representation"): 2.0,
        ("cmp-both", "candidate", "representation"): 2.6,
        ("cmp-both", "baseline", "task"): 1.0,
        ("cmp-both", "candidate", "task"): 1.4,
    }
    first = _run((spec,), _measure_factory(values), request=request, manifest=manifest)
    second = _run((spec,), _measure_factory(values), request=request, manifest=manifest)

    assert dict(first.payload) == dict(second.payload)  # type: ignore[arg-type]
    workflow_identity = "a" * 64
    assert first.digest(workflow_identity) == second.digest(workflow_identity)
    assert first.digest(workflow_identity) != first.digest("e" * 64)

    config = dict(first.payload["config"])  # type: ignore[arg-type]
    assert config["evaluation_seed"] == 42
    assert config["control_seed"] == 17
    assert config["repetitions"] == 200
    assert config["workflow_identity"] == workflow_identity
    assert config["manifest_id"] == MANIFEST_ID


def test_workflow_end_to_end_records_bounded_comparison_rows() -> None:
    ids = ("cmp-both", "cmp-rep", "cmp-task", "cmp-uncertain")
    specs = tuple(_spec(comparison_id) for comparison_id in ids)
    request = _request(tuple(_entry(comparison_id) for comparison_id in ids))
    manifest = _manifest()
    prior = _prior(request, manifest)
    payload_by_stage = {output.stage: dict(output.payload) for output in prior}
    values: _Values = {
        ("cmp-both", "baseline", "representation"): 2.0,
        ("cmp-both", "candidate", "representation"): 2.6,
        ("cmp-both", "baseline", "task"): 1.0,
        ("cmp-both", "candidate", "task"): 1.4,
        ("cmp-rep", "baseline", "representation"): 2.0,
        ("cmp-rep", "candidate", "representation"): 2.5,
        ("cmp-rep", "baseline", "task"): 1.0,
        ("cmp-rep", "candidate", "task"): 1.02,
        ("cmp-task", "baseline", "representation"): 2.0,
        ("cmp-task", "candidate", "representation"): 2.02,
        ("cmp-task", "baseline", "task"): 1.0,
        ("cmp-task", "candidate", "task"): 1.3,
        ("cmp-uncertain", "baseline", "representation"): 2.0,
        ("cmp-uncertain", "candidate", "representation"): 2.6,
        ("cmp-uncertain", "baseline", "task"): 1.0,
        ("cmp-uncertain", "candidate", "task"): 1.4,
    }

    def _draw(app: ComparisonApplication) -> float:
        assert app.rng is not None
        if app.comparison_id == "cmp-uncertain" and app.side == "candidate" and app.metric_role == "representation":
            return 2.6 + float(app.rng.normal(0.0, 0.5))
        return float(values[(app.comparison_id, app.side, app.metric_role)])

    executor = make_compare_executor(specs, _measure_factory(values, draw=_draw))

    def _stage(invocation: StageInvocation) -> StageOutput:
        if invocation.stage == "compare":
            return executor(invocation)
        payload = dict(payload_by_stage.get(invocation.stage, {"stage": invocation.stage}))
        if invocation.stage == "report":
            payload["report_id"] = "report-80-20"
        return StageOutput(stage=invocation.stage, outcome="completed", payload=payload, artifact_refs=())

    executors = {stage: _stage for stage in WORKFLOW_STAGES}
    versions = {stage: "stage-v1" for stage in WORKFLOW_STAGES}
    versions["compare"] = COMPARISON_VERSION
    workflow = DiagnosticWorkflow(executors, versions)  # type: ignore[arg-type]
    result, checkpoint = workflow.run(request, manifest)

    assert result.status == "completed"
    assert checkpoint.status == "completed"
    compare_stage = result.stage_results["stages"]["compare"]  # type: ignore[index]
    assert compare_stage["outcome"] == "completed"  # type: ignore[index]
    assert compare_stage["payload"]["classifications"] == {  # type: ignore[index]
        "cmp-both": "both",
        "cmp-rep": "representation_only",
        "cmp-task": "task_only",
        "cmp-uncertain": "inconclusive",
    }

    rows = list(compare_stage["payload"]["comparisons"])  # type: ignore[index]
    items = comparison_report_items(rows)
    assert {item["id"]: item["status"] for item in items} == {
        "cmp-both": "observed",
        "cmp-rep": "observed",
        "cmp-task": "observed",
        "cmp-uncertain": "inconclusive",
    }
    assert set(COMPARISON_CLASSIFICATIONS) == {
        "representation_only",
        "task_only",
        "both",
        "neither",
        "inconclusive",
        "unsupported",
    }
    for item in items:
        assert list(item) == [
            "alignment",
            "baseline",
            "candidate",
            "evidence_refs",
            "id",
            "metric_ids",
            "status",
        ]

    validate_report_shape(_report(items))


def _report(items: list[dict[str, object]]) -> dict[str, object]:
    return {
        "schema_version": "diagnostic-report-schema-v1",
        "report_id": "report-80-20",
        "status": "declared",
        "capture_provenance": {
            "captures": [
                {
                    "artifact_refs": ["cap-1"],
                    "axes": ["sample", "feature"],
                    "capture_id": "capture-encoder",
                    "dataset_split": "heldout",
                    "model_revision": "conv-vae-8x8-latent4-seed0-epochs5",
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
                "control_refs": [],
                "estimate": 2.0,
                "evidence_refs": ["o-1"],
                "id": "e-1",
                "metric_id": REP_METRIC,
                "status": "observed",
                "uncertainty": {"kind": "interval", "lower": 1.9, "upper": 2.1},
            }
        ],
        "interventions": [],
        "comparisons": items,
        "limitations": [
            {
                "affects": ["comparison"],
                "blocking": False,
                "description": "comparison sides are declared stand-ins in focused tests",
                "id": "lim-1",
            }
        ],
        "next_action": {
            "action": "persist the content-addressed artifact",
            "rationale": "80.21 owns persistence",
            "required_evidence_refs": [],
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
            }
        ],
    }


def test_manifest_declared_task_metric_can_be_compared_without_detection_selection() -> None:
    manifest = deepcopy(_manifest())
    task_metric_id = "heldout-brightness-bin-accuracy"
    metrics = manifest["metrics"]
    thresholds = manifest["thresholds"]
    assert isinstance(metrics, list)
    assert isinstance(thresholds, list)
    metrics.append(
        {
            "aggregation": "mean-with-interval",
            "direction": "higher_is_better",
            "estimator": "fixed-training-median-global-brightness-bin-accuracy",
            "id": task_metric_id,
            "taxonomy_family_id": FAMILY,
            "unit": "accuracy",
        }
    )
    thresholds.append(
        {
            "comparator": ">=",
            "metric_id": task_metric_id,
            "predeclared": True,
            "tolerance": 0.02,
            "value": 0.9,
        }
    )
    commitment = manifest["commitment"]
    assert isinstance(commitment, dict)
    commitment["manifest_sha256"] = manifest_digest(manifest)

    comparison_id = "cmp-task-only-declared-metric"
    entry = _entry(comparison_id, metric_ids=(REP_METRIC, task_metric_id))
    request = _request((entry,), metric_ids=(REP_METRIC,))
    alignment = _alignment(request, manifest, overrides={"task_metric": task_metric_id})
    spec = ComparisonSpec(
        comparison_id=comparison_id,
        baseline=RunSide(run_id=_BASELINE_RUN, checkpoint_id="ckpt-000100", alignment=alignment),
        candidate=RunSide(run_id=_CANDIDATE_RUN, checkpoint_id="ckpt-000200", alignment=alignment),
        representation_metric_id=REP_METRIC,
        task_metric_id=task_metric_id,
        representation_tolerance=0.1,
        task_tolerance=0.05,
    )
    values: _Values = {
        (comparison_id, "baseline", "representation"): 2.0,
        (comparison_id, "candidate", "representation"): 1.8,
        (comparison_id, "baseline", "task"): 0.95,
        (comparison_id, "candidate", "task"): 0.5,
    }

    output = _run(
        (spec,),
        _measure_factory(values),
        request=request,
        manifest=manifest,
        prior=_prior(request, manifest),
    )
    payload = output.payload
    records = payload["comparisons"]
    assert isinstance(records, list)
    record = records[0]
    assert isinstance(record, dict)
    task = record["task"]
    assert isinstance(task, dict)
    assert record["classification"] == "both"
    assert task["metric_id"] == task_metric_id
    assert task["changed"] is True
    assert task["signed_delta"] == pytest.approx(-0.45)
    assert task_metric_id not in request.controls.metric_ids
    assert task_metric_id not in _detect_payload()["config"]["metric_ids"]

"""High-level representation-diagnostic request/result API (Sprint 80.6).

This module is the smallest public configuration surface for the Sprint 80
diagnostic workflow. It declares *what* to diagnose, not how to execute it:
capture selection, diagnostic family selection, declared controls,
intervention requests, aligned comparisons, manifest identity, and output
location/artifact intent, plus a resumable result carrier.

No architecture-specific fields are exposed here: there are no transformer
layer indices, attention heads, VAE latent dimensions, or model-internal
knobs. Execution, orchestration, detectors, and interventions belong to
later tasks (80.7+). This module only constructs and validates requests
and carries results.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

_DIAGNOSTIC_FAMILIES = frozenset(
    {
        "collapse_rank_loss",
        "anisotropy_inactive_dimensions",
        "redundancy_superposition",
        "separability_probe_leakage",
        "density_ood_distribution_drift",
        "sparse_feature_instability",
        "sequence_trajectory_drift",
    }
)

_WORKFLOW_STAGES = (
    "capture",
    "detect",
    "localize",
    "explain",
    "intervene",
    "compare",
    "report",
)

_RESULT_STATUSES = frozenset({"pending", "running", "completed", "failed", "interrupted"})

ResultStatus = Literal["pending", "running", "completed", "failed", "interrupted"]


class DiagnosticRequestError(ValueError):
    """Raised when a diagnostic request is malformed or internally inconsistent."""


def _non_empty_string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise DiagnosticRequestError(f"{name} must be a non-empty string")
    return value


def _string_tuple(value: object, *, name: str, minimum: int = 0) -> tuple[str, ...]:
    if isinstance(value, str | bytes) or not isinstance(value, (tuple, list)):
        raise DiagnosticRequestError(f"{name} must be a list of strings")
    items = tuple(value)
    if len(items) < minimum:
        raise DiagnosticRequestError(f"{name} must contain at least {minimum} item(s)")
    for item in items:
        if not isinstance(item, str) or not item:
            raise DiagnosticRequestError(f"{name} must be a list of strings")
    if len(set(items)) != len(items):
        raise DiagnosticRequestError(f"{name} must not contain duplicates")
    return items


def _validate_output_location(value: str) -> str:
    if not value.strip():
        raise DiagnosticRequestError("output_location must be a non-empty string")
    candidate = Path(value)
    posix_absolute = value.startswith("/") or value.startswith("\\")
    drive_absolute = len(value) >= 2 and value[1] == ":" and value[0].isalpha()
    if candidate.is_absolute() or posix_absolute or drive_absolute:
        raise DiagnosticRequestError("output_location must be a repository-relative path")
    if ".." in candidate.parts:
        raise DiagnosticRequestError("output_location must not escape the repository")
    return value


@dataclass(frozen=True)
class CaptureSelection:
    """Which representation capture to diagnose, in domain terms only."""

    capture_id: str
    representation_identity: str
    axes: tuple[str, ...] = ("sample", "feature")

    def __post_init__(self) -> None:
        _non_empty_string(self.capture_id, name="capture_id")
        _non_empty_string(self.representation_identity, name="representation_identity")
        _string_tuple(self.axes, name="axes", minimum=1)


@dataclass(frozen=True)
class DiagnosticSelection:
    """Which representation-problem families to evaluate."""

    family_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _string_tuple(self.family_ids, name="family_ids", minimum=1)
        unknown = sorted(set(self.family_ids) - _DIAGNOSTIC_FAMILIES)
        if unknown:
            raise DiagnosticRequestError(f"unknown diagnostic families: {', '.join(unknown)}")


@dataclass(frozen=True)
class ControlSelection:
    """Declared controls bound to declared metrics."""

    control_ids: tuple[str, ...]
    metric_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _string_tuple(self.control_ids, name="control_ids", minimum=1)
        _string_tuple(self.metric_ids, name="metric_ids", minimum=1)


@dataclass(frozen=True)
class InterventionRequest:
    """One requested causal trial, identified by domain names only."""

    intervention_id: str
    target: str
    control_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _non_empty_string(self.intervention_id, name="intervention_id")
        _non_empty_string(self.target, name="target")
        _string_tuple(self.control_ids, name="control_ids")


@dataclass(frozen=True)
class ComparisonRequest:
    """One requested aligned run comparison."""

    comparison_id: str
    baseline_run: str
    candidate_run: str
    metric_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _non_empty_string(self.comparison_id, name="comparison_id")
        _non_empty_string(self.baseline_run, name="baseline_run")
        _non_empty_string(self.candidate_run, name="candidate_run")
        _string_tuple(self.metric_ids, name="metric_ids", minimum=1)
        if self.baseline_run == self.candidate_run:
            raise DiagnosticRequestError("baseline_run and candidate_run must differ")


@dataclass(frozen=True)
class OutputSelection:
    """Where the diagnostic output goes and what it should contain."""

    output_location: str
    artifact_name: str = "diagnostic-report"
    include_report: bool = True

    def __post_init__(self) -> None:
        _non_empty_string(self.output_location, name="output_location")
        _validate_output_location(self.output_location)
        _non_empty_string(self.artifact_name, name="artifact_name")
        if not isinstance(self.include_report, bool):
            raise DiagnosticRequestError("include_report must be boolean")


@dataclass(frozen=True)
class DiagnosticRequest:
    """Complete domain-level declaration of one diagnostic workflow run."""

    request_id: str
    manifest_id: str
    capture: CaptureSelection
    diagnostics: DiagnosticSelection
    controls: ControlSelection
    output: OutputSelection
    interventions: tuple[InterventionRequest, ...] = ()
    comparisons: tuple[ComparisonRequest, ...] = ()

    def __post_init__(self) -> None:
        _non_empty_string(self.request_id, name="request_id")
        _non_empty_string(self.manifest_id, name="manifest_id")
        if not isinstance(self.capture, CaptureSelection):
            raise DiagnosticRequestError("capture must be a CaptureSelection")
        if not isinstance(self.diagnostics, DiagnosticSelection):
            raise DiagnosticRequestError("diagnostics must be a DiagnosticSelection")
        if not isinstance(self.controls, ControlSelection):
            raise DiagnosticRequestError("controls must be a ControlSelection")
        if not isinstance(self.output, OutputSelection):
            raise DiagnosticRequestError("output must be an OutputSelection")
        if isinstance(self.interventions, list):
            object.__setattr__(self, "interventions", tuple(self.interventions))
        if isinstance(self.comparisons, list):
            object.__setattr__(self, "comparisons", tuple(self.comparisons))
        if not isinstance(self.interventions, tuple) or not isinstance(self.comparisons, tuple):
            raise DiagnosticRequestError("interventions and comparisons must be tuples")
        for intervention in self.interventions:
            if not isinstance(intervention, InterventionRequest):
                raise DiagnosticRequestError("interventions must be InterventionRequest items")
        for comparison in self.comparisons:
            if not isinstance(comparison, ComparisonRequest):
                raise DiagnosticRequestError("comparisons must be ComparisonRequest items")
        intervention_ids = [item.intervention_id for item in self.interventions]
        if len(set(intervention_ids)) != len(intervention_ids):
            raise DiagnosticRequestError("intervention identifiers must be unique")
        comparison_ids = [item.comparison_id for item in self.comparisons]
        if len(set(comparison_ids)) != len(comparison_ids):
            raise DiagnosticRequestError("comparison identifiers must be unique")
        known_controls = set(self.controls.control_ids)
        for intervention in self.interventions:
            unknown = sorted(set(intervention.control_ids) - known_controls)
            if unknown:
                raise DiagnosticRequestError(
                    f"intervention {intervention.intervention_id} references undeclared controls: {', '.join(unknown)}"
                )
        known_metrics = set(self.controls.metric_ids)
        for comparison in self.comparisons:
            unknown_metrics = sorted(set(comparison.metric_ids) - known_metrics)
            if unknown_metrics:
                raise DiagnosticRequestError(
                    f"comparison {comparison.comparison_id} references undeclared metrics: {', '.join(unknown_metrics)}"
                )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible mapping of this request."""

        return {
            "request_id": self.request_id,
            "manifest_id": self.manifest_id,
            "capture": {
                "capture_id": self.capture.capture_id,
                "representation_identity": self.capture.representation_identity,
                "axes": list(self.capture.axes),
            },
            "diagnostics": {"family_ids": list(self.diagnostics.family_ids)},
            "controls": {
                "control_ids": list(self.controls.control_ids),
                "metric_ids": list(self.controls.metric_ids),
            },
            "interventions": [
                {
                    "intervention_id": item.intervention_id,
                    "target": item.target,
                    "control_ids": list(item.control_ids),
                }
                for item in self.interventions
            ],
            "comparisons": [
                {
                    "comparison_id": item.comparison_id,
                    "baseline_run": item.baseline_run,
                    "candidate_run": item.candidate_run,
                    "metric_ids": list(item.metric_ids),
                }
                for item in self.comparisons
            ],
            "output": {
                "output_location": self.output.output_location,
                "artifact_name": self.output.artifact_name,
                "include_report": self.output.include_report,
            },
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> DiagnosticRequest:
        """Rebuild a request from its mapping form, failing closed on bad input."""

        if not isinstance(value, Mapping):
            raise DiagnosticRequestError("request must be a mapping")
        expected = {"request_id", "manifest_id", "capture", "diagnostics", "controls", "interventions", "comparisons", "output"}
        actual = set(value)
        if actual != expected:
            raise DiagnosticRequestError("request fields are invalid")
        capture = value["capture"]
        diagnostics = value["diagnostics"]
        controls = value["controls"]
        output = value["output"]
        if not isinstance(capture, Mapping) or not isinstance(diagnostics, Mapping):
            raise DiagnosticRequestError("capture and diagnostics must be mappings")
        if not isinstance(controls, Mapping) or not isinstance(output, Mapping):
            raise DiagnosticRequestError("controls and output must be mappings")
        interventions = value["interventions"]
        comparisons = value["comparisons"]
        if isinstance(interventions, str) or not isinstance(interventions, (tuple, list)):
            raise DiagnosticRequestError("interventions must be a list")
        if isinstance(comparisons, str) or not isinstance(comparisons, (tuple, list)):
            raise DiagnosticRequestError("comparisons must be a list")
        try:
            return cls(
                request_id=_non_empty_string(value["request_id"], name="request_id"),
                manifest_id=_non_empty_string(value["manifest_id"], name="manifest_id"),
                capture=CaptureSelection(
                    capture_id=_non_empty_string(capture.get("capture_id"), name="capture_id"),
                    representation_identity=_non_empty_string(
                        capture.get("representation_identity"), name="representation_identity"
                    ),
                    axes=_string_tuple(capture.get("axes"), name="axes", minimum=1),
                ),
                diagnostics=DiagnosticSelection(
                    family_ids=_string_tuple(diagnostics.get("family_ids"), name="family_ids", minimum=1),
                ),
                controls=ControlSelection(
                    control_ids=_string_tuple(controls.get("control_ids"), name="control_ids", minimum=1),
                    metric_ids=_string_tuple(controls.get("metric_ids"), name="metric_ids", minimum=1),
                ),
                interventions=tuple(
                    InterventionRequest(
                        intervention_id=_non_empty_string(item.get("intervention_id"), name="intervention_id"),
                        target=_non_empty_string(item.get("target"), name="target"),
                        control_ids=_string_tuple(item.get("control_ids", ()), name="control_ids"),
                    )
                    for item in (dict(item) if isinstance(item, Mapping) else {} for item in interventions)
                ),
                comparisons=tuple(
                    ComparisonRequest(
                        comparison_id=_non_empty_string(item.get("comparison_id"), name="comparison_id"),
                        baseline_run=_non_empty_string(item.get("baseline_run"), name="baseline_run"),
                        candidate_run=_non_empty_string(item.get("candidate_run"), name="candidate_run"),
                        metric_ids=_string_tuple(item.get("metric_ids"), name="metric_ids", minimum=1),
                    )
                    for item in (dict(item) if isinstance(item, Mapping) else {} for item in comparisons)
                ),
                output=OutputSelection(
                    output_location=_non_empty_string(output.get("output_location"), name="output_location"),
                    artifact_name=_non_empty_string(output.get("artifact_name", "diagnostic-report"), name="artifact_name"),
                    include_report=output.get("include_report", True)
                    if isinstance(output.get("include_report", True), bool)
                    else (_ for _ in ()).throw(DiagnosticRequestError("include_report must be boolean")),
                ),
            )
        except (AttributeError, TypeError) as exc:
            raise DiagnosticRequestError(f"request fields are invalid: {exc}") from exc


@dataclass(frozen=True)
class DiagnosticResult:
    """Resumable outcome carrier for one diagnostic request. No stage execution."""

    request_id: str
    manifest_id: str
    status: ResultStatus
    completed_stages: tuple[str, ...] = ()
    artifact_refs: tuple[str, ...] = ()
    report_id: str | None = None
    message: str = ""
    stage_results: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _non_empty_string(self.request_id, name="request_id")
        _non_empty_string(self.manifest_id, name="manifest_id")
        if self.status not in _RESULT_STATUSES:
            raise DiagnosticRequestError(f"unsupported result status: {self.status!r}")
        _string_tuple(self.completed_stages, name="completed_stages")
        unknown_stages = sorted(set(self.completed_stages) - set(_WORKFLOW_STAGES))
        if unknown_stages:
            raise DiagnosticRequestError(f"unknown workflow stages: {', '.join(unknown_stages)}")
        _string_tuple(self.artifact_refs, name="artifact_refs")
        if self.report_id is not None:
            _non_empty_string(self.report_id, name="report_id")
        if not isinstance(self.message, str):
            raise DiagnosticRequestError("message must be a string")
        if not isinstance(self.stage_results, Mapping):
            raise DiagnosticRequestError("stage_results must be a mapping")
        object.__setattr__(self, "stage_results", dict(self.stage_results))

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible mapping of this result."""

        return {
            "request_id": self.request_id,
            "manifest_id": self.manifest_id,
            "status": self.status,
            "completed_stages": list(self.completed_stages),
            "artifact_refs": list(self.artifact_refs),
            "report_id": self.report_id,
            "message": self.message,
            "stage_results": dict(self.stage_results),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> DiagnosticResult:
        """Rebuild a result from its mapping form, failing closed on bad input."""

        if not isinstance(value, Mapping):
            raise DiagnosticRequestError("result must be a mapping")
        expected = {
            "request_id",
            "manifest_id",
            "status",
            "completed_stages",
            "artifact_refs",
            "report_id",
            "message",
            "stage_results",
        }
        if set(value) != expected:
            raise DiagnosticRequestError("result fields are invalid")
        report_id = value["report_id"]
        if report_id is not None and (not isinstance(report_id, str) or not report_id):
            raise DiagnosticRequestError("report_id must be a non-empty string or null")
        stage_results = value["stage_results"]
        if not isinstance(stage_results, Mapping):
            raise DiagnosticRequestError("stage_results must be a mapping")
        return cls(
            request_id=_non_empty_string(value["request_id"], name="request_id"),
            manifest_id=_non_empty_string(value["manifest_id"], name="manifest_id"),
            status=value["status"]  # type: ignore[arg-type]
            if value["status"] in _RESULT_STATUSES
            else (_ for _ in ()).throw(DiagnosticRequestError(f"unsupported result status: {value['status']!r}")),
            completed_stages=_string_tuple(value["completed_stages"], name="completed_stages"),
            artifact_refs=_string_tuple(value["artifact_refs"], name="artifact_refs"),
            report_id=report_id,  # type: ignore[arg-type]
            message=value["message"] if isinstance(value["message"], str) else (_ for _ in ()).throw(
                DiagnosticRequestError("message must be a string")
            ),
            stage_results=dict(stage_results),
        )


__all__ = [
    "CaptureSelection",
    "ComparisonRequest",
    "ControlSelection",
    "DiagnosticRequest",
    "DiagnosticRequestError",
    "DiagnosticResult",
    "DiagnosticSelection",
    "InterventionRequest",
    "OutputSelection",
    "ResultStatus",
]

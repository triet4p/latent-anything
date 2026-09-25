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


def _require_capture(value: object) -> CaptureSelection:
    if not isinstance(value, CaptureSelection):
        raise DiagnosticRequestError("capture must be a CaptureSelection")
    return value


def _require_diagnostics(value: object) -> DiagnosticSelection:
    if not isinstance(value, DiagnosticSelection):
        raise DiagnosticRequestError("diagnostics must be a DiagnosticSelection")
    return value


def _require_controls(value: object) -> ControlSelection:
    if not isinstance(value, ControlSelection):
        raise DiagnosticRequestError("controls must be a ControlSelection")
    return value


def _require_include_report(value: object) -> bool:
    if not isinstance(value, bool):
        raise DiagnosticRequestError("include_report must be boolean")
    return value


def _require_result_status(value: object) -> ResultStatus:
    if value == "pending":
        return "pending"
    if value == "running":
        return "running"
    if value == "completed":
        return "completed"
    if value == "failed":
        return "failed"
    if value == "interrupted":
        return "interrupted"
    raise DiagnosticRequestError(f"unsupported result status: {value!r}")


def _require_output(value: object) -> OutputSelection:
    if not isinstance(value, OutputSelection):
        raise DiagnosticRequestError("output must be an OutputSelection")
    return value


def _require_interventions(value: object) -> tuple[InterventionRequest, ...]:
    if isinstance(value, (tuple, list)):
        items = tuple(value)
        for item in items:
            if not isinstance(item, InterventionRequest):
                raise DiagnosticRequestError("interventions must be InterventionRequest items")
        return items
    raise DiagnosticRequestError("interventions and comparisons must be tuples")


def _require_comparisons(value: object) -> tuple[ComparisonRequest, ...]:
    if isinstance(value, (tuple, list)):
        items = tuple(value)
        for item in items:
            if not isinstance(item, ComparisonRequest):
                raise DiagnosticRequestError("comparisons must be ComparisonRequest items")
        return items
    raise DiagnosticRequestError("interventions and comparisons must be tuples")


def _require_optional_string(value: object, *, name: str) -> str | None:
    if value is None:
        return None
    return _non_empty_string(value, name=name)


def _require_mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise DiagnosticRequestError(f"{name} must be a mapping")
    return value


def _require_text(value: object, *, name: str) -> str:
    if not isinstance(value, str):
        raise DiagnosticRequestError(f"{name} must be a string")
    return value


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
    """One aligned comparison with manifest-bound metrics evaluated on both sides.

    Comparison metrics may include task utility not selected for detection;
    the compare executor validates metric identities and requires its
    representation metric to remain detection-selected.
    """

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
        object.__setattr__(self, "include_report", _require_include_report(self.include_report))


@dataclass(frozen=True)
class DiagnosticRequest:
    """Complete domain-level diagnostic request.

    ``controls.metric_ids`` selects detector metrics. Comparison metric
    declarations are validated by the compare executor against the manifest
    and its representation/task roles.
    """

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
        capture = _require_capture(self.capture)
        diagnostics = _require_diagnostics(self.diagnostics)
        controls = _require_controls(self.controls)
        _require_output(self.output)
        interventions = _require_interventions(self.interventions)
        comparisons = _require_comparisons(self.comparisons)
        intervention_ids = [item.intervention_id for item in interventions]
        if len(set(intervention_ids)) != len(intervention_ids):
            raise DiagnosticRequestError("intervention identifiers must be unique")
        comparison_ids = [item.comparison_id for item in comparisons]
        if len(set(comparison_ids)) != len(comparison_ids):
            raise DiagnosticRequestError("comparison identifiers must be unique")
        known_controls = set(controls.control_ids)
        for intervention in interventions:
            unknown = sorted(set(intervention.control_ids) - known_controls)
            if unknown:
                raise DiagnosticRequestError(
                    f"intervention {intervention.intervention_id} references undeclared controls: {', '.join(unknown)}"
                )
        object.__setattr__(self, "capture", capture)
        object.__setattr__(self, "diagnostics", diagnostics)
        object.__setattr__(self, "controls", controls)
        object.__setattr__(self, "interventions", interventions)
        object.__setattr__(self, "comparisons", comparisons)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible mapping of this request."""

        capture = _require_capture(self.capture)
        diagnostics = _require_diagnostics(self.diagnostics)
        controls = _require_controls(self.controls)
        output = _require_output(self.output)
        interventions = _require_interventions(self.interventions)
        comparisons = _require_comparisons(self.comparisons)
        return {
            "request_id": self.request_id,
            "manifest_id": self.manifest_id,
            "capture": {
                "capture_id": capture.capture_id,
                "representation_identity": capture.representation_identity,
                "axes": list(capture.axes),
            },
            "diagnostics": {"family_ids": list(diagnostics.family_ids)},
            "controls": {
                "control_ids": list(controls.control_ids),
                "metric_ids": list(controls.metric_ids),
            },
            "interventions": [
                {
                    "intervention_id": item.intervention_id,
                    "target": item.target,
                    "control_ids": list(item.control_ids),
                }
                for item in interventions
            ],
            "comparisons": [
                {
                    "comparison_id": item.comparison_id,
                    "baseline_run": item.baseline_run,
                    "candidate_run": item.candidate_run,
                    "metric_ids": list(item.metric_ids),
                }
                for item in comparisons
            ],
            "output": {
                "output_location": output.output_location,
                "artifact_name": output.artifact_name,
                "include_report": output.include_report,
            },
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> DiagnosticRequest:
        """Rebuild a request from its mapping form, failing closed on bad input."""

        _require_mapping(value, name="request")
        expected = {
            "request_id",
            "manifest_id",
            "capture",
            "diagnostics",
            "controls",
            "interventions",
            "comparisons",
            "output",
        }
        actual = set(value)
        if actual != expected:
            raise DiagnosticRequestError("request fields are invalid")
        capture = _require_mapping(value["capture"], name="capture")
        diagnostics = _require_mapping(value["diagnostics"], name="diagnostics")
        controls = _require_mapping(value["controls"], name="controls")
        output = _require_mapping(value["output"], name="output")
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
                    artifact_name=_non_empty_string(
                        output.get("artifact_name", "diagnostic-report"), name="artifact_name"
                    ),
                    include_report=_require_include_report(output.get("include_report", True)),
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
        _require_result_status(self.status)
        _string_tuple(self.completed_stages, name="completed_stages")
        unknown_stages = sorted(set(self.completed_stages) - set(_WORKFLOW_STAGES))
        if unknown_stages:
            raise DiagnosticRequestError(f"unknown workflow stages: {', '.join(unknown_stages)}")
        _string_tuple(self.artifact_refs, name="artifact_refs")
        if self.report_id is not None:
            _non_empty_string(self.report_id, name="report_id")
        message = _require_text(self.message, name="message")
        stages = _require_mapping(self.stage_results, name="stage_results")
        object.__setattr__(self, "message", message)
        object.__setattr__(self, "stage_results", dict(stages))

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

        _require_mapping(value, name="result")
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
        report_id = _require_optional_string(value["report_id"], name="report_id")
        stage_results = _require_mapping(value["stage_results"], name="stage_results")
        status = _require_result_status(value["status"])
        return cls(
            request_id=_non_empty_string(value["request_id"], name="request_id"),
            manifest_id=_non_empty_string(value["manifest_id"], name="manifest_id"),
            status=status,
            completed_stages=_string_tuple(value["completed_stages"], name="completed_stages"),
            artifact_refs=_string_tuple(value["artifact_refs"], name="artifact_refs"),
            report_id=report_id,
            message=_require_text(value["message"], name="message"),
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

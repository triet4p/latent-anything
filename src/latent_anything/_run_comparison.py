"""Aligned checkpoint/run comparison executor for the workflow ``compare``
stage (Sprint 80.20).

One architecture-neutral private executor compares exactly two declared
sides (baseline and candidate run/checkpoint identities) under a strict
alignment contract, without a single model- or architecture-specific field.
The caller supplies one measurement callback; the executor owns identity
binding, per-side measurement with uncertainty, delta computation, and the
bounded classification.

Alignment contract (every check runs before any measurement callback
fires): each side declares immutable ``run_id``/``checkpoint_id`` identities
plus an ``alignment`` mapping over the exact field set ``ALIGNMENT_FIELDS``
using the80.14 vocabulary (``dataset_slice_id``,
``dataset_configuration``, ``preprocessing_identity``,
``layer_module_identity``, ``model_identity``, ``representation_identity``,
``seeds``, ``diagnostic_config``, ``manifest_identity``,
``taxonomy_identity``, ``schema_identity``, ``axes``,
``representation_metric``, ``task_metric``). Both sides must declare
identical alignment values; the shared values must then match the bound
prior evidence and the frozen manifest exactly: dataset configuration vs
manifest split identity, dataset slice vs a prior localize selection,
layer/module vs a localize selection or the captured representation,
model and representation identities vs the bound capture provenance,
seeds vs ``manifest_seed_identity``, diagnostic config vs
``detect_config_identity`` of the prior detect payload, manifest/schema
identity vs the manifest, taxonomy identity vs the prior localize family,
axes vs the bound capture axes, and the metric fields vs the spec's own
representation/task metrics. Metrics must be manifest-declared and request-declared
by the comparison; the representation metric additionally must be selected for
detection, equal the prior localize metric, and be carried by a non-omitted prior
explain row. The task metric is evaluated on the aligned comparison sides and
does not have to be a detector metric. Request-declared comparisons, sides, and
metric sets must match the declared specs exactly;
prior capture/detect/localize/explain/intervene records must be present and
structurally intact. Missing or duplicate sides, mismatched alignment,
invented metric/config identities, non-finite tolerances, and malformed
declarations fail closed before callbacks with ``ComparisonError`` or
``StageContractError``.

Measurement contract: for each side and each role (representation, task)
the callback receives a ``ComparisonApplication`` and must return a
``ComparisonMeasurement`` echoing the side's ``run_id``, the declared
``metric_id``, the shared captured-inputs digest, and a finite value; a
wrong side, wrong metric, mismatched inputs digest, non-finite value, or a
crashing callback fails the stage closed.

Classification (bounded, never inferring task change from representation
change or vice versa): per metric the signed delta is ``candidate -
baseline`` with an absolute delta and a conservative delta interval
``[candidate.lower - baseline.upper, candidate.upper - baseline.lower]``;
a metric is *changed* only when its absolute delta exceeds its own declared
tolerance, *unresolved* when that exceedance coincides with an interval
that strictly spans zero. Then:
1. any prior stage outcome ``unsupported`` -> ``unsupported`` (upstream
   evidence blocks interpretation; recorded, never promoted);
2. any metric unresolved -> ``inconclusive``;
3. otherwise the pair of independent change flags -> ``both`` /
   ``representation_only`` / ``task_only`` / ``neither``.

Every record carries both side identities, the shared alignment, per-metric
baseline/candidate measurements with uncertainty, signed/absolute deltas,
delta intervals, declared tolerances, metric definitions from the frozen
manifest, the classification, and bound provenance. Deterministic replay
reproduces the payload exactly. Method algorithms stay in the caller's
callback; no algorithm enters ``DiagnosticWorkflow`` itself.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from math import isfinite
from typing import Any, Literal

import numpy as np

from latent_anything._diagnostic_workflow import StageContractError, StageInvocation, StageOutput
from latent_anything._portable_contract import PortableNodeError, canonical_json
from latent_anything._statistical_controls import derive_stream_seed, run_bootstrap
from latent_anything.diagnostics import CaptureSelection, ComparisonRequest, ControlSelection, DiagnosticRequest

ALIGNMENT_FIELDS: tuple[str, ...] = (
    "axes",
    "dataset_configuration",
    "dataset_slice_id",
    "diagnostic_config",
    "layer_module_identity",
    "manifest_identity",
    "model_identity",
    "preprocessing_identity",
    "representation_identity",
    "representation_metric",
    "schema_identity",
    "seeds",
    "taxonomy_identity",
    "task_metric",
)
"""Exact alignment field set both sides must declare identically."""

COMPARISON_CLASSIFICATIONS: tuple[str, ...] = (
    "representation_only",
    "task_only",
    "both",
    "neither",
    "inconclusive",
    "unsupported",
)
"""Bounded comparison classifications. Anything else is rejected."""

COMPARISON_VERSION = "aligned-comparison-v1"
"""Stage executor version bound into the workflow config digest."""

_DIGEST_LENGTH = 64


class ComparisonError(ValueError):
    """Raised when a comparison declaration or measurement is fail-closed invalid."""


def _non_empty(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ComparisonError(f"{name} must be a non-empty string")
    return value


def _finite(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(float(value)):
        raise ComparisonError(f"{name} must be a finite number")
    return float(value)


def _require_prior(value: object) -> tuple[StageOutput, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (tuple, list)):
        raise ComparisonError("invocation prior outputs must be StageOutput items")
    items = tuple(value)
    for item in items:
        if not isinstance(item, StageOutput):
            raise ComparisonError("invocation prior outputs must be StageOutput items")
    return items


def _require_capture_payload(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise StageContractError("the prior capture payload must carry a bound capture under the 'capture' key")
    return value


def _require_detect_config(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise StageContractError("prior detect payload must declare a config object")
    return value


def _require_mapping(value: object, *, name: str, error: type[ValueError] = ComparisonError) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise error(f"{name} must be a mapping")
    return value


def _require_request(value: object) -> DiagnosticRequest:
    if not isinstance(value, DiagnosticRequest):
        raise StageContractError("compare executor requires a DiagnosticRequest")
    return value


def _require_manifest(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise StageContractError("compare executor requires a manifest mapping")
    return value


def _require_invocation(value: object) -> StageInvocation:
    if not isinstance(value, StageInvocation):
        raise StageContractError("compare executor requires a StageInvocation")
    return value


def _require_seed_rows(value: object, *, role: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not value:
        raise StageContractError(f"compare executor requires manifest seeds.{role}")
    return value


def _require_seed_int(value: object, *, role: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise StageContractError(f"manifest seeds.{role}[0] must be a non-negative integer")
    return int(value)


def _require_controls(value: object) -> ControlSelection:
    if not isinstance(value, ControlSelection):
        raise StageContractError("compare executor requires declared controls")
    return value


def _require_capture_selection(value: object) -> CaptureSelection:
    if not isinstance(value, CaptureSelection):
        raise StageContractError("compare executor requires a capture selection")
    return value


def _require_stage_str(value: object, *, name: str) -> str:
    if not isinstance(value, str):
        raise StageContractError(f"{name} must be a string")
    return value


def _require_non_empty(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise StageContractError(f"{name} must be a non-empty string")
    return value


def _require_metric_list(value: object) -> tuple[object, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        return ()
    return tuple(value)


def _require_trials(value: object) -> tuple[object, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not value:
        raise StageContractError("prior intervene payload must declare recorded trials")
    return tuple(value)


def _require_conclusions(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not value:
        raise StageContractError("prior intervene payload must declare recorded conclusions")
    return value


def _require_sequence(value: object, *, name: str) -> tuple[object, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise StageContractError(f"{name} must be a sequence")
    return tuple(value)


def _require_spec_list(value: object) -> tuple[ComparisonSpec, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ComparisonError("comparisons must be a sequence of ComparisonSpec items")
    items = tuple(value)
    if not items:
        raise ComparisonError("comparisons must declare at least one comparison")
    for position, item in enumerate(items):
        if not isinstance(item, ComparisonSpec):
            raise ComparisonError(f"comparisons[{position}] must be a ComparisonSpec")
    return items


def _require_record_list(value: object) -> tuple[Mapping[str, object], ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ComparisonError("records must be a sequence of comparison records")
    items = tuple(value)
    for position, item in enumerate(items):
        if not isinstance(item, Mapping):
            raise ComparisonError(f"records[{position}] must be an object")
    return items


def _require_evidence_row(value: object, hypothesis_id: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise StageContractError(f"explain hypothesis {hypothesis_id!r} has no family evidence")
    return value


def _require_evidence_outcome(value: object, hypothesis_id: str) -> str:
    if not isinstance(value, str) or not value:
        raise StageContractError(f"explain evidence for {hypothesis_id!r} must declare an outcome")
    return value


def _require_hypothesis_rows(value: object) -> tuple[object, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not value:
        raise StageContractError("prior explain payload must declare at least one hypothesis")
    return tuple(value)


def _require_axis_name(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ComparisonError("capture payload axes must be non-empty strings")
    return value


def _require_checkpoint_id(value: object) -> str:
    if not isinstance(value, str):
        raise ComparisonError("checkpoint_id must be a string (empty for run-level identity)")
    return value


def _require_alignment(value: object) -> Mapping[str, str]:
    if not isinstance(value, Mapping):
        raise ComparisonError("alignment must be a mapping")
    actual = set(value)
    expected = set(ALIGNMENT_FIELDS)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        details = []
        if missing:
            details.append(f"missing {', '.join(missing)}")
        if extra:
            details.append(f"unexpected {', '.join(extra)}")
        raise ComparisonError(f"alignment fields are invalid ({'; '.join(details)})")
    frozen: dict[str, str] = {}
    for key in ALIGNMENT_FIELDS:
        frozen[key] = _non_empty(value[key], name=f"alignment[{key!r}]")
    return dict(frozen)


def _require_side(value: object) -> RunSide:
    if not isinstance(value, RunSide):
        raise ComparisonError("baseline and candidate must be RunSide declarations")
    return value


def _require_app_checkpoint(value: object) -> str:
    if not isinstance(value, str):
        raise ComparisonError("application checkpoint_id must be a string")
    return value


def _require_app_seed(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ComparisonError("application stream_seed must be a non-negative integer")
    return int(value)


def _require_app_rng(value: object) -> np.random.Generator | None:
    if value is not None and not isinstance(value, np.random.Generator):
        raise ComparisonError("application rng must be a numpy Generator or None")
    return value


def _require_axes(value: object) -> tuple[str, ...]:
    if isinstance(value, str | bytes) or not isinstance(value, Sequence) or not value:
        raise ComparisonError("capture payload must declare a non-empty axes list")
    items = tuple(value)
    for item in items:
        if not isinstance(item, str) or not item:
            raise ComparisonError("capture payload axes must be non-empty strings")
    return items


def _require_generator(value: object) -> np.random.Generator:
    if not isinstance(value, np.random.Generator):
        raise ComparisonError("control stream must be a numpy Generator")
    return value


def _require_repetitions(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 2:
        raise StageContractError("manifest uncertainty.repetitions must be at least two")
    return int(value)


def _require_comparisons(value: object) -> tuple[ComparisonRequest, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (tuple, list)):
        raise StageContractError("requested comparisons must be ComparisonRequest items")
    items = tuple(value)
    for item in items:
        if not isinstance(item, ComparisonRequest):
            raise StageContractError("requested comparisons must be ComparisonRequest items")
    return items


def _require(condition: bool, message: str, *, error: type[ValueError] = ComparisonError) -> None:
    if not condition:
        raise error(message)


def _digest(value: object, *, name: str) -> str:
    text = _non_empty(value, name=name)
    if len(text) != _DIGEST_LENGTH or any(ch not in "0123456789abcdef" for ch in text):
        raise ComparisonError(f"{name} must be a lowercase SHA-256 digest")
    return text


# ---------------------------------------------------------------------------
# Alignment identity helpers (callers compute the same values they declare)
# ---------------------------------------------------------------------------


def manifest_seed_identity(manifest: object) -> str:
    """Return the canonical seed identity for one frozen manifest."""
    manifest = _require_manifest(manifest)
    seeds = _require_mapping(manifest.get("seeds"), name="manifest must declare seeds")
    parts: list[str] = []
    for role in ("training", "evaluation", "controls"):
        rows = _require_seed_rows(seeds.get(role), role=role)
        values: list[str] = []
        for raw in rows:
            values.append(str(_require_seed_int(raw, role=role)))
        parts.append(f"{role}={','.join(values)}")
    return ";".join(parts)


def detect_config_identity(detect_payload: object) -> str:
    """Return the canonical SHA-256 identity of a prior detect payload config."""
    detect_payload = _require_mapping(detect_payload, name="detect payload must be a mapping")
    config = _require_detect_config(detect_payload.get("config"))
    try:
        encoded = canonical_json(dict(config))
    except PortableNodeError as exc:
        raise ComparisonError(f"detect config is not canonical JSON: {exc}") from exc
    return sha256(encoded.encode("utf-8")).hexdigest()


def capture_axes_identity(capture_payload: object) -> str:
    """Return the canonical axes identity of a bound capture payload."""
    capture_payload = _require_mapping(capture_payload, name="capture payload must be a mapping")
    axes = _require_axes(capture_payload.get("axes"))
    names: list[str] = []
    for raw in axes:
        names.append(_require_axis_name(raw))
    return ",".join(names)


# ---------------------------------------------------------------------------
# Declared comparison carriers (validated before any callback runs)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunSide:
    """One declared comparison side: immutable run/checkpoint identity plus alignment."""

    run_id: str
    checkpoint_id: object
    alignment: object

    def __post_init__(self) -> None:
        _non_empty(self.run_id, name="run_id")
        _non_empty(self.run_id, name="run_id")
        object.__setattr__(self, "checkpoint_id", _require_checkpoint_id(self.checkpoint_id))
        object.__setattr__(self, "alignment", _require_alignment(self.alignment))

    def to_dict(self) -> dict[str, object]:
        return {
            "alignment": dict(_require_alignment(self.alignment)),
            "checkpoint_id": self.checkpoint_id,
            "run_id": self.run_id,
        }


@dataclass(frozen=True)
class ComparisonSpec:
    """One declared aligned comparison of exactly two sides."""

    comparison_id: str
    baseline: object
    candidate: object
    representation_metric_id: str
    task_metric_id: str
    representation_tolerance: float
    task_tolerance: float

    def __post_init__(self) -> None:
        _non_empty(self.comparison_id, name="comparison_id")
        baseline = _require_side(self.baseline)
        candidate = _require_side(self.candidate)
        object.__setattr__(self, "baseline", baseline)
        object.__setattr__(self, "candidate", candidate)
        _non_empty(self.representation_metric_id, name="representation_metric_id")
        _non_empty(self.task_metric_id, name="task_metric_id")
        if self.representation_metric_id == self.task_metric_id:
            raise ComparisonError("representation and task metrics must differ to distinguish their changes")
        if baseline.run_id == candidate.run_id:
            raise ComparisonError("baseline and candidate sides must declare different run identities")
        object.__setattr__(
            self, "representation_tolerance", _finite(self.representation_tolerance, name="representation_tolerance")
        )
        object.__setattr__(self, "task_tolerance", _finite(self.task_tolerance, name="task_tolerance"))
        if self.representation_tolerance < 0.0 or self.task_tolerance < 0.0:
            raise ComparisonError("declared tolerances must be non-negative")
        baseline_alignment = _require_alignment(baseline.alignment)
        candidate_alignment = _require_alignment(candidate.alignment)
        if baseline_alignment != candidate_alignment:
            differing = sorted(key for key in ALIGNMENT_FIELDS if baseline_alignment[key] != candidate_alignment[key])
            raise ComparisonError(
                f"baseline and candidate alignment must be identical; differing fields: {', '.join(differing)}"
            )


# ---------------------------------------------------------------------------
# Measurement contract (what the caller's callback must return)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ComparisonApplication:
    """One measurement request naming the exact side, role, and metric."""

    comparison_id: str
    side: Literal["baseline", "candidate"]
    run_id: str
    checkpoint_id: object
    metric_id: str
    metric_role: Literal["representation", "task"]
    inputs_digest: str
    stream_seed: object
    rng: object

    def __post_init__(self) -> None:
        _non_empty(self.comparison_id, name="application comparison_id")
        if self.side not in ("baseline", "candidate"):
            raise ComparisonError("application side must be 'baseline' or 'candidate'")
        _non_empty(self.run_id, name="application run_id")
        object.__setattr__(self, "checkpoint_id", _require_app_checkpoint(self.checkpoint_id))
        _non_empty(self.metric_id, name="application metric_id")
        if self.metric_role not in ("representation", "task"):
            raise ComparisonError("application metric_role must be 'representation' or 'task'")
        _digest(self.inputs_digest, name="application inputs_digest")
        object.__setattr__(self, "stream_seed", _require_app_seed(self.stream_seed))
        object.__setattr__(self, "rng", _require_app_rng(self.rng))


@dataclass(frozen=True)
class ComparisonMeasurement:
    """One side measurement echoing the side identity, metric, and inputs."""

    run_id: str
    metric_id: str
    value: float
    inputs_digest: str

    def __post_init__(self) -> None:
        _non_empty(self.run_id, name="measurement run_id")
        _non_empty(self.metric_id, name="measurement metric_id")
        _finite(self.value, name="measurement value")
        _digest(self.inputs_digest, name="measurement inputs_digest")


ComparisonMeasureFn = Callable[[ComparisonApplication], ComparisonMeasurement]
"""One caller-supplied measurement callback over the shared captured inputs."""


# ---------------------------------------------------------------------------
# Stage binding: every prior identity checked before any callback runs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _MetricBinding:
    metric_id: str
    definition: Mapping[str, object]
    tolerance: float


@dataclass(frozen=True)
class _BoundComparison:
    spec: ComparisonSpec
    request_entry: ComparisonRequest
    representation: _MetricBinding
    task: _MetricBinding


@dataclass(frozen=True)
class _CompareContext:
    """Frozen bound identities and predeclared inputs for one compare run."""

    manifest_id: str
    request_id: str
    capture_identity: str
    representation_identity: str
    localize_family_id: str
    detect_config_identity: str
    evaluation_seed: int
    control_seed: int
    repetitions: int
    confidence_level: float
    blocked_by: tuple[str, ...]
    workflow_identity: str
    request_digest: str
    manifest_digest: str
    config_digest: str
    comparisons: tuple[_BoundComparison, ...] = ()


def _prior_payloads(invocation: StageInvocation) -> dict[str, Mapping[str, object]]:
    prior = _require_prior(invocation.prior)
    return {item.stage: dict(_require_mapping(item.payload, name="prior payload")) for item in prior}


def _manifest_seeds(manifest: Mapping[str, object]) -> tuple[int, int]:
    seeds = _require_mapping(manifest.get("seeds"), name="compare executor requires manifest seeds")
    values: list[int] = []
    for role in ("evaluation", "controls"):
        rows = _require_seed_rows(seeds.get(role), role=role)
        values.append(_require_seed_int(rows[0], role=role))
    return values[0], values[1]


def _bind_capture(invocation: StageInvocation, request: DiagnosticRequest) -> tuple[str, str, str, str]:
    """Bind the prior capture record: identity, representation, model, axes."""
    prior = _prior_payloads(invocation)
    _require("capture" in prior, "compare executor requires a prior capture payload", error=StageContractError)
    raw = _require_capture_payload(prior["capture"].get("capture"))
    try:
        capture_identity = _digest(raw.get("capture_identity"), name="capture_identity")
        axes = capture_axes_identity(raw)
    except ComparisonError as exc:
        raise StageContractError(str(exc)) from exc
    capture_selection = _require_capture_selection(request.capture)
    for key, expected in (
        ("capture_id", capture_selection.capture_id),
        ("representation_identity", capture_selection.representation_identity),
        ("manifest_id", request.manifest_id),
        ("request_id", request.request_id),
    ):
        value = _require_stage_str(raw.get(key), name=f"prior capture {key!r}")
        _require(
            value == expected,
            f"prior capture {key!r} is {value!r}, expected {expected!r}",
            error=StageContractError,
        )
    model_id = _require_non_empty(raw.get("model_id"), name="prior capture payload must declare model_id")
    return capture_identity, capture_selection.representation_identity, axes, model_id


def _bind_localize(
    invocation: StageInvocation, request: DiagnosticRequest, manifest: Mapping[str, object]
) -> tuple[str, str, set[str]]:
    prior = _prior_payloads(invocation)
    _require("localize" in prior, "compare executor requires a prior localize payload", error=StageContractError)
    localize = prior["localize"]
    manifest_id = _require_non_empty(manifest.get("manifest_id"), name="prior localize manifest_id")
    localize_capture = _require_capture_selection(request.capture)
    for key, expected in (
        ("manifest_id", manifest_id),
        ("representation_identity", localize_capture.representation_identity),
    ):
        value = _require_stage_str(localize.get(key), name=f"prior localize {key!r}")
        _require(
            value == expected,
            f"prior localize {key!r} is {value!r}, expected {expected!r}",
            error=StageContractError,
        )
    family_id = _require_non_empty(localize.get("family_id"), name="prior localize payload must declare family_id")
    metric_id = _require_non_empty(localize.get("metric_id"), name="prior localize payload must declare metric_id")
    _require_stage_str(localize.get("verdict"), name="prior localize payload must declare verdict")
    selections: set[str] = set()
    selections: set[str] = set()
    for key in ("layer_order", "affected_layers", "declared_slice_ids", "affected_slices"):
        raw = localize.get(key)
        if isinstance(raw, Sequence) and not isinstance(raw, str | bytes):
            selections.update(item for item in raw if isinstance(item, str) and item)
    earliest = localize.get("earliest_layer")
    if isinstance(earliest, str) and earliest:
        selections.add(earliest)
    rows = localize.get("report_localization")
    if isinstance(rows, Sequence) and not isinstance(rows, str | bytes):
        for row in rows:
            if isinstance(row, Mapping):
                selection = row.get("selection")
                if isinstance(selection, str) and selection:
                    selections.add(selection)
    _require(
        bool(selections),
        "prior localize payload must declare at least one selection identity",
        error=StageContractError,
    )
    return family_id, metric_id, selections


def _detect_metric_ids(detect_payload: Mapping[str, object]) -> set[str]:
    found: set[str] = set()
    config = detect_payload.get("config")
    if isinstance(config, Mapping):
        for key in ("metric_id", "metric_ids"):
            raw = config.get(key)
            if isinstance(raw, str):
                found.add(raw)
            elif isinstance(raw, Sequence) and not isinstance(raw, str | bytes):
                found.update(item for item in raw if isinstance(item, str) and item)
    families = detect_payload.get("families")
    if isinstance(families, Sequence) and not isinstance(families, str | bytes):
        for item in families:
            if not isinstance(item, Mapping):
                continue
            observed = item.get("observed_metrics")
            if isinstance(observed, Mapping):
                found.update(str(key) for key in observed)
            passing = item.get("threshold_pass")
            if isinstance(passing, Mapping):
                found.update(str(key) for key in passing)
    return found


def _explain_representative_row(
    explain: Mapping[str, object], representation_metric_id: str
) -> tuple[str, Mapping[str, object]]:
    rows = _require_hypothesis_rows(explain.get("hypotheses"))
    evidence = _require_mapping(
        explain.get("family_evidence"),
        name="prior explain payload must declare structured family_evidence",
    )
    for raw in rows:
        hypothesis = _require_mapping(raw, name="explain hypotheses must be objects")
        metrics = _require_metric_list(hypothesis.get("metric_ids"))
        if representation_metric_id not in {str(item) for item in metrics}:
            continue
        hypothesis_id = _require_non_empty(
            hypothesis.get("hypothesis_id"), name="explain hypotheses must declare hypothesis_id"
        )
        row = _require_evidence_row(evidence.get(hypothesis_id), hypothesis_id)
        outcome = _require_evidence_outcome(row.get("outcome"), hypothesis_id)
        _require(
            outcome != "omitted",
            f"representation metric {representation_metric_id!r} is carried only by omitted "
            f"explanation {hypothesis_id!r}",
            error=StageContractError,
        )
        return hypothesis_id, hypothesis
    raise StageContractError(
        f"no prior explain hypothesis carries the representation metric {representation_metric_id!r}"
    )


def _bind_intervene(invocation: StageInvocation) -> tuple[str, ...]:
    prior = _prior_payloads(invocation)
    _require("intervene" in prior, "compare executor requires a prior intervene payload", error=StageContractError)
    intervene = prior["intervene"]
    _require_trials(intervene.get("trials"))
    _require_conclusions(intervene.get("conclusions"))
    blocked: list[str] = []
    for item in _require_prior(invocation.prior):
        if item.outcome == "unsupported":
            blocked.append(item.stage)
    return tuple(blocked)


def _metric_binding(
    metric_id: str,
    tolerance: float,
    manifest: Mapping[str, object],
) -> _MetricBinding:
    raw_metrics = _require_sequence(manifest.get("metrics"), name="compare executor requires manifest metrics")
    for raw in raw_metrics:
        metric = _require_mapping(raw, name="manifest metrics must be objects")
        if metric.get("id") != metric_id:
            continue
        definition = {key: metric.get(key) for key in ("direction", "estimator", "taxonomy_family_id", "unit")}
        _require(
            all(isinstance(value, str) and value for value in definition.values()),
            f"manifest metric {metric_id!r} must declare direction, estimator, taxonomy_family_id, and unit",
            error=StageContractError,
        )
        return _MetricBinding(metric_id=metric_id, definition=dict(definition), tolerance=tolerance)
    raise StageContractError(f"metric {metric_id!r} is not manifest-declared")


def _alignment_cross_checks(
    spec: ComparisonSpec,
    *,
    manifest: Mapping[str, object],
    representation_identity: str,
    model_identity: str,
    capture_axes: str,
    localize_family: str,
    localize_metric: str,
    localize_selections: set[str],
    detect_identity: str,
) -> None:
    alignment = _require_alignment(_require_side(spec.baseline).alignment)
    checks: tuple[tuple[str, str, str], ...] = (
        ("manifest_identity", alignment["manifest_identity"], str(manifest.get("manifest_id"))),
        ("schema_identity", alignment["schema_identity"], str(manifest.get("schema_version"))),
        ("taxonomy_identity", alignment["taxonomy_identity"], localize_family),
        ("representation_identity", alignment["representation_identity"], representation_identity),
        ("model_identity", alignment["model_identity"], model_identity),
        ("axes", alignment["axes"], capture_axes),
        ("seeds", alignment["seeds"], manifest_seed_identity(manifest)),
        ("diagnostic_config", alignment["diagnostic_config"], detect_identity),
        ("representation_metric", alignment["representation_metric"], spec.representation_metric_id),
        ("task_metric", alignment["task_metric"], spec.task_metric_id),
    )
    for field, declared, expected in checks:
        _require(
            declared == expected,
            f"alignment field {field!r} is {declared!r}, expected {expected!r}",
            error=StageContractError,
        )
    dataset = manifest.get("dataset")
    split_identity = dataset.get("split_identity") if isinstance(dataset, Mapping) else None
    _require(
        alignment["dataset_configuration"] == split_identity,
        f"alignment field 'dataset_configuration' is {alignment['dataset_configuration']!r}, "
        f"expected manifest split identity {split_identity!r}",
        error=StageContractError,
    )
    _require(
        alignment["dataset_slice_id"] in localize_selections,
        f"alignment field 'dataset_slice_id' {alignment['dataset_slice_id']!r} is not a prior localize selection",
        error=StageContractError,
    )
    layer = alignment["layer_module_identity"]
    _require(
        layer in localize_selections or layer == representation_identity,
        f"alignment field 'layer_module_identity' {layer!r} is neither a prior localize "
        "selection nor the captured representation identity",
        error=StageContractError,
    )
    _require(
        spec.representation_metric_id == localize_metric,
        f"representation metric {spec.representation_metric_id!r} does not match the prior "
        f"localize metric {localize_metric!r}",
        error=StageContractError,
    )


def _bind_comparisons(
    specs: tuple[ComparisonSpec, ...], invocation: StageInvocation
) -> tuple[_CompareContext, tuple[_BoundComparison, ...]]:
    request = _require_request(invocation.request)
    manifest = _require_manifest(invocation.manifest)
    manifest_id = _require_non_empty(manifest.get("manifest_id"), name="compare executor requires a manifest_id")
    _require(
        request.manifest_id == manifest_id,
        f"compare executor manifest mismatch: request {request.manifest_id!r} != manifest {manifest_id!r}",
        error=StageContractError,
    )
    expected_prior = ("capture", "detect", "localize", "explain", "intervene")
    stages = tuple(item.stage for item in _require_prior(invocation.prior))
    _require(
        stages == expected_prior,
        f"compare executor prior stages must be exactly {list(expected_prior)!r}, got {stages!r}",
        error=StageContractError,
    )

    capture_identity, representation_identity, capture_axes, model_identity = _bind_capture(invocation, request)
    localize_family, localize_metric, localize_selections = _bind_localize(invocation, request, manifest)

    prior = _prior_payloads(invocation)
    _require("detect" in prior, "compare executor requires a prior detect payload", error=StageContractError)
    detect_payload = prior["detect"]
    detect_identity = detect_config_identity(detect_payload)
    _require_detect_config(detect_payload.get("config"))
    detect_metrics = _detect_metric_ids(detect_payload)

    _require("explain" in prior, "compare executor requires a prior explain payload", error=StageContractError)
    blocked_by = _bind_intervene(invocation)

    request_comparisons = _require_comparisons(request.comparisons)
    request_index: dict[str, ComparisonRequest] = {entry.comparison_id: entry for entry in request_comparisons}
    _require(
        {spec.comparison_id for spec in specs} == set(request_index),
        "declared comparisons and requested comparisons must match exactly: "
        f"specs {sorted(spec.comparison_id for spec in specs)!r} != "
        f"request {sorted(request_index)!r}",
        error=StageContractError,
    )
    evaluation_seed, control_seed = _manifest_seeds(manifest)
    uncertainty = _require_mapping(manifest.get("uncertainty"), name="compare executor requires manifest uncertainty")
    repetitions = _require_repetitions(uncertainty.get("repetitions"))
    confidence = uncertainty.get("confidence_level")
    try:
        confidence_level = _finite(confidence, name="uncertainty.confidence_level")
    except ComparisonError as exc:
        raise StageContractError(str(exc)) from exc
    _require(
        0.0 < confidence_level < 1.0,
        "manifest confidence_level must be between zero and one",
        error=StageContractError,
    )

    bound: list[_BoundComparison] = []
    for spec in specs:
        entry = request_index[spec.comparison_id]
        spec_baseline = _require_side(spec.baseline)
        spec_candidate = _require_side(spec.candidate)
        _require(
            entry.baseline_run == spec_baseline.run_id,
            f"comparison {spec.comparison_id!r} baseline run {spec_baseline.run_id!r} != "
            f"requested {entry.baseline_run!r}",
            error=StageContractError,
        )
        _require(
            entry.candidate_run == spec_candidate.run_id,
            f"comparison {spec.comparison_id!r} candidate run {spec_candidate.run_id!r} != "
            f"requested {entry.candidate_run!r}",
            error=StageContractError,
        )
        expected_metrics = {spec.representation_metric_id, spec.task_metric_id}
        _require(
            set(entry.metric_ids) == expected_metrics,
            f"comparison {spec.comparison_id!r} metric set {sorted(set(entry.metric_ids))!r} != "
            f"declared {sorted(expected_metrics)!r}",
            error=StageContractError,
        )
        detected_metrics = set(_require_controls(request.controls).metric_ids)
        _require(
            spec.representation_metric_id in detected_metrics,
            f"comparison {spec.comparison_id!r} representation metric "
            f"{spec.representation_metric_id!r} is not request-declared for detection",
            error=StageContractError,
        )
        _require(
            spec.representation_metric_id in detect_metrics,
            f"comparison {spec.comparison_id!r} representation metric "
            f"{spec.representation_metric_id!r} has no prior detect metric evidence",
            error=StageContractError,
        )
        # The two sides must declare identical alignment before cross-checking.
        baseline_side = _require_side(spec.baseline)
        candidate_side = _require_side(spec.candidate)
        baseline_alignment = _require_alignment(baseline_side.alignment)
        candidate_alignment = _require_alignment(candidate_side.alignment)
        differing = sorted(key for key in ALIGNMENT_FIELDS if baseline_alignment[key] != candidate_alignment[key])
        _require(
            not differing,
            f"comparison {spec.comparison_id!r} side alignment mismatches on: {', '.join(differing)}",
            error=StageContractError,
        )
        _alignment_cross_checks(
            spec,
            manifest=manifest,
            representation_identity=representation_identity,
            model_identity=model_identity,
            capture_axes=capture_axes,
            localize_family=localize_family,
            localize_metric=localize_metric,
            localize_selections=localize_selections,
            detect_identity=detect_identity,
        )
        _explain_representative_row(prior["explain"], spec.representation_metric_id)
        bound.append(
            _BoundComparison(
                spec=spec,
                request_entry=entry,
                representation=_metric_binding(spec.representation_metric_id, spec.representation_tolerance, manifest),
                task=_metric_binding(spec.task_metric_id, spec.task_tolerance, manifest),
            )
        )

    context = _CompareContext(
        manifest_id=manifest_id,
        request_id=request.request_id,
        capture_identity=capture_identity,
        representation_identity=representation_identity,
        localize_family_id=localize_family,
        detect_config_identity=detect_identity,
        evaluation_seed=evaluation_seed,
        control_seed=control_seed,
        repetitions=int(repetitions),
        confidence_level=confidence_level,
        blocked_by=blocked_by,
        workflow_identity=invocation.workflow_identity,
        request_digest=invocation.request_digest,
        manifest_digest=invocation.manifest_digest,
        config_digest=invocation.config_digest,
    )
    return context, tuple(bound)


# ---------------------------------------------------------------------------
# Measurement, deltas, and the bounded classification
# ---------------------------------------------------------------------------


def _call_measure(measure: ComparisonMeasureFn, application: ComparisonApplication) -> object:
    return measure(application)


def _checked_measure(
    measure: ComparisonMeasureFn,
    application: ComparisonApplication,
    *,
    comparison_id: str,
) -> ComparisonMeasurement:
    result = _call_measure(measure, application)
    if not isinstance(result, ComparisonMeasurement):
        raise ComparisonError(
            f"measure callback for {comparison_id!r} must return a ComparisonMeasurement, got {type(result).__name__}"
        )
    if result.run_id != application.run_id:
        raise ComparisonError(
            f"measurement for {comparison_id!r} returned side {result.run_id!r}, expected "
            f"{application.run_id!r} ({application.side})"
        )
    if result.metric_id != application.metric_id:
        raise ComparisonError(
            f"measurement domain mismatch for {comparison_id!r}: got metric "
            f"{result.metric_id!r}, declared {application.metric_id!r}"
        )
    if result.inputs_digest != application.inputs_digest:
        raise ComparisonError(
            f"measurement inputs mismatch for {comparison_id!r}: got capture digest "
            f"{result.inputs_digest!r}, bound {application.inputs_digest!r}"
        )
    return result


def _application(
    bound: _BoundComparison,
    context: _CompareContext,
    *,
    side: Literal["baseline", "candidate"],
    run_side: RunSide,
    metric_role: Literal["representation", "task"],
    metric_id: str,
    stream_seed: int,
    rng: np.random.Generator | None,
) -> ComparisonApplication:
    return ComparisonApplication(
        comparison_id=bound.spec.comparison_id,
        side=side,
        run_id=run_side.run_id,
        checkpoint_id=run_side.checkpoint_id,
        metric_id=metric_id,
        metric_role=metric_role,
        inputs_digest=context.capture_identity,
        stream_seed=stream_seed,
        rng=rng,
    )


def _measurement_record(measurement: ComparisonMeasurement, application: ComparisonApplication) -> dict[str, object]:
    return {
        "checkpoint_id": application.checkpoint_id,
        "inputs_digest": measurement.inputs_digest,
        "metric_id": measurement.metric_id,
        "metric_role": application.metric_role,
        "run_id": measurement.run_id,
        "stream_seed": _require_app_seed(application.stream_seed),
        "value": float(measurement.value),
    }


def _metric_block(
    binding: _MetricBinding,
    *,
    baseline_point: float,
    baseline_interval: Mapping[str, object],
    baseline_application: ComparisonApplication,
    baseline_measurement: ComparisonMeasurement,
    candidate_point: float,
    candidate_interval: Mapping[str, object],
    candidate_application: ComparisonApplication,
    candidate_measurement: ComparisonMeasurement,
) -> dict[str, object]:
    signed_delta = float(candidate_point - baseline_point)
    absolute_delta = abs(signed_delta)
    baseline_lower = _finite(baseline_interval.get("lower"), name="baseline lower bound")
    baseline_upper = _finite(baseline_interval.get("upper"), name="baseline upper bound")
    candidate_lower = _finite(candidate_interval.get("lower"), name="candidate lower bound")
    candidate_upper = _finite(candidate_interval.get("upper"), name="candidate upper bound")
    delta_lower = float(candidate_lower - baseline_upper)
    delta_upper = float(candidate_upper - baseline_lower)
    changed = absolute_delta > binding.tolerance
    unresolved = changed and delta_lower < 0.0 < delta_upper
    return {
        "absolute_delta": absolute_delta,
        "baseline": {
            "interval": dict(baseline_interval),
            "measurement": _measurement_record(baseline_measurement, baseline_application),
        },
        "candidate": {
            "interval": dict(candidate_interval),
            "measurement": _measurement_record(candidate_measurement, candidate_application),
        },
        "changed": changed,
        "delta_interval": [delta_lower, delta_upper],
        "definition": dict(binding.definition),
        "metric_id": binding.metric_id,
        "resolved": not unresolved,
        "signed_delta": signed_delta,
        "tolerance": float(binding.tolerance),
        "unresolved": unresolved,
    }


def _classify(
    context: _CompareContext,
    representation: Mapping[str, object],
    task: Mapping[str, object],
) -> tuple[str, str]:
    if context.blocked_by:
        return (
            "unsupported",
            "upstream stage(s) "
            f"{', '.join(context.blocked_by)} recorded an unsupported outcome; the "
            "comparison cannot be interpreted",
        )
    unresolved = [name for name, block in (("representation", representation), ("task", task)) if block["unresolved"]]
    if unresolved:
        return (
            "inconclusive",
            f"{', '.join(unresolved)} change exceeds tolerance but its delta interval spans "
            "zero; the change is not resolved",
        )
    rep_changed = bool(representation["changed"])
    task_changed = bool(task["changed"])
    if rep_changed and task_changed:
        return (
            "both",
            "representation and task metrics each changed beyond their declared tolerances "
            "with resolved, non-overlapping zero; each change is reported independently",
        )
    if rep_changed:
        return (
            "representation_only",
            "only the representation metric changed beyond its declared tolerance; no "
            "task-metric change is claimed or inferred",
        )
    if task_changed:
        return (
            "task_only",
            "only the task metric changed beyond its declared tolerance; no "
            "representation change is claimed or inferred",
        )
    return (
        "neither",
        "neither metric changed beyond its declared tolerance; no change is claimed or inferred",
    )


def _execute_comparison(
    bound: _BoundComparison, measure: ComparisonMeasureFn, context: _CompareContext
) -> dict[str, object]:
    spec = bound.spec
    blocks: dict[str, dict[str, object]] = {}
    metrics: dict[str, _MetricBinding] = {
        "representation": bound.representation,
        "task": bound.task,
    }
    roles: tuple[Literal["representation", "task"], ...] = ("representation", "task")
    sides: tuple[tuple[Literal["baseline", "candidate"], RunSide], ...] = (
        ("baseline", _require_side(spec.baseline)),
        ("candidate", _require_side(spec.candidate)),
    )
    for role in roles:
        binding = metrics[role]
        points: dict[str, tuple[ComparisonApplication, ComparisonMeasurement]] = {}
        intervals: dict[str, dict[str, object]] = {}
        for side_label, run_side in sides:
            point_seed = derive_stream_seed(
                context.evaluation_seed,
                f"{spec.comparison_id}:{side_label}:{role}:point",
                role="evaluation",
            )
            point_application = _application(
                bound,
                context,
                side=side_label,
                run_side=run_side,
                metric_role=role,
                metric_id=binding.metric_id,
                stream_seed=point_seed,
                rng=None,
            )
            point_measurement = _checked_measure(measure, point_application, comparison_id=spec.comparison_id)
            points[side_label] = (point_application, point_measurement)

            uncertainty_id = f"compare:{spec.comparison_id}:{side_label}:{role}"

            def _draw(
                rng: object,
                *,
                label: Literal["baseline", "candidate"] = side_label,
                declared_side: RunSide = run_side,
                metric: str = binding.metric_id,
                metric_role: Literal["representation", "task"] = role,
                resample_id: str = uncertainty_id,
            ) -> float:
                rng = _require_generator(rng)
                draw_seed = derive_stream_seed(context.evaluation_seed, resample_id, role="evaluation")
                application = _application(
                    bound,
                    context,
                    side=label,
                    run_side=declared_side,
                    metric_role=metric_role,
                    metric_id=metric,
                    stream_seed=draw_seed,
                    rng=rng,
                )
                return float(_checked_measure(measure, application, comparison_id=spec.comparison_id).value)

            outcome, interval = run_bootstrap(
                control_id=uncertainty_id,
                base_seed=context.evaluation_seed,
                seed_role="evaluation",
                repetitions=context.repetitions,
                confidence_level=context.confidence_level,
                required=False,
                kind="bootstrap",
                metric_ids=(binding.metric_id,),
                expected_behavior="seeded resampling of one aligned comparison side",
                draw=_draw,
            )
            if outcome.status == "failed" or "lower" not in interval or "upper" not in interval:
                raise ComparisonError(
                    f"uncertainty failed for {spec.comparison_id!r} on {side_label}/{role}: "
                    f"{outcome.reason or 'interval unavailable'}"
                )
            intervals[side_label] = dict(interval)
        baseline_app, baseline_measurement = points["baseline"]
        candidate_app, candidate_measurement = points["candidate"]
        baseline_interval = intervals["baseline"]
        candidate_interval = intervals["candidate"]
        blocks[role] = _metric_block(
            binding,
            baseline_point=float(baseline_measurement.value),
            baseline_interval=baseline_interval,
            baseline_application=baseline_app,
            baseline_measurement=baseline_measurement,
            candidate_point=float(candidate_measurement.value),
            candidate_interval=candidate_interval,
            candidate_application=candidate_app,
            candidate_measurement=candidate_measurement,
        )

    representation = blocks["representation"]
    task = blocks["task"]
    classification, reason = _classify(context, representation, task)

    provenance: dict[str, object] = {
        "capture_identity": context.capture_identity,
        "config_digest": context.config_digest,
        "detect_config_identity": context.detect_config_identity,
        "localize_family_id": context.localize_family_id,
        "manifest_digest": context.manifest_digest,
        "manifest_id": context.manifest_id,
        "representation_identity": context.representation_identity,
        "request_digest": context.request_digest,
        "request_id": context.request_id,
        "workflow_identity": context.workflow_identity,
        "upstream_blocked_by": list(context.blocked_by),
    }

    record_baseline = _require_side(spec.baseline)
    record_candidate = _require_side(spec.candidate)
    return {
        "alignment": dict(_require_alignment(record_baseline.alignment)),
        "baseline": record_baseline.to_dict(),
        "candidate": record_candidate.to_dict(),
        "classification": classification,
        "comparison_id": spec.comparison_id,
        "provenance": provenance,
        "reason": reason,
        "representation": representation,
        "task": task,
    }


def _compare_output(records: Sequence[Mapping[str, object]], context: _CompareContext) -> StageOutput:
    classifications = {str(record["comparison_id"]): str(record["classification"]) for record in records}
    payload: dict[str, object] = {
        "classifications": dict(classifications),
        "comparison_version": COMPARISON_VERSION,
        "comparisons": list(records),
        "config": {
            "confidence_level": float(context.confidence_level),
            "config_digest": context.config_digest,
            "control_seed": int(context.control_seed),
            "evaluation_seed": int(context.evaluation_seed),
            "manifest_digest": context.manifest_digest,
            "manifest_id": context.manifest_id,
            "repetitions": int(context.repetitions),
            "request_digest": context.request_digest,
            "request_id": context.request_id,
            "workflow_identity": context.workflow_identity,
        },
    }
    aggregate = (
        "unsupported"
        if classifications and all(value == "unsupported" for value in classifications.values())
        else "completed"
    )
    return StageOutput(stage="compare", outcome=aggregate, payload=payload, artifact_refs=())


# ---------------------------------------------------------------------------
# Report-shaped output (bounded comparison rows, Phase-IV boundary)
# ---------------------------------------------------------------------------


def comparison_report_items(
    records: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Render comparison records as shape-compatible report comparison rows.

    Phase-IV boundary (not 80.21/80.22): rows pass ``validate_report_shape``
    but are NOT independently consumable by ``validate_diagnostic_report``.
    ``evidence_refs`` are provisional (the ``comparison-<id>-record`` name
    registers no report row or digest artifact yet). Change classifications
    map to the ``observed`` evidence status (a comparison is an observation,
    never a causal claim), ``inconclusive`` stays inconclusive, and blocked
    comparisons stay unsupported so the frozen validator fails closed on any
    promotion attempt.
    """
    records = _require_record_list(records)
    status_by_classification = {
        "representation_only": "observed",
        "task_only": "observed",
        "both": "observed",
        "neither": "observed",
        "inconclusive": "inconclusive",
        "unsupported": "unsupported",
    }
    items: list[dict[str, object]] = []
    seen: set[str] = set()
    for record in _require_record_list(records):
        comparison_id = _non_empty(record.get("comparison_id"), name="record comparison_id")
        if comparison_id in seen:
            raise ComparisonError("comparison identifiers must be unique in report rows")
        seen.add(comparison_id)
        classification = _non_empty(record.get("classification"), name=f"comparison {comparison_id!r} classification")
        if classification not in status_by_classification:
            raise ComparisonError(
                f"comparison {comparison_id!r} classification must be one of {list(status_by_classification)!r}"
            )
        baseline = _require_mapping(record.get("baseline"), name=f"comparison {comparison_id!r} baseline")
        candidate = _require_mapping(record.get("candidate"), name=f"comparison {comparison_id!r} candidate")
        alignment = _require_mapping(record.get("alignment"), name=f"comparison {comparison_id!r} alignment")
        baseline_run = _non_empty(baseline.get("run_id"), name="baseline run_id")
        candidate_run = _non_empty(candidate.get("run_id"), name="candidate run_id")
        representation = _require_mapping(
            record.get("representation"),
            name=f"comparison {comparison_id!r} representation",
        )
        task = _require_mapping(record.get("task"), name=f"comparison {comparison_id!r} task")
        representation_metric = _non_empty(representation.get("metric_id"), name="representation metric_id")
        task_metric = _non_empty(task.get("metric_id"), name="task metric_id")
        items.append(
            {
                "alignment": dict(alignment),
                "baseline": baseline_run,
                "candidate": candidate_run,
                "evidence_refs": [f"comparison-{comparison_id}-record"],
                "id": comparison_id,
                "metric_ids": [representation_metric, task_metric],
                "status": status_by_classification[classification],
            }
        )
    return items


# ---------------------------------------------------------------------------
# Stage executor factory (the single private execution seam)
# ---------------------------------------------------------------------------


def make_compare_executor(
    comparisons: object,
    measure: ComparisonMeasureFn,
    *,
    version: str = COMPARISON_VERSION,
) -> Any:
    """Build a supplied ``compare``-stage executor for declared comparisons.

    The coordinator invokes the returned callable with a ``StageInvocation``;
    the callable first verifies the invocation targets ``compare`` with the
    exact prior capture/detect/localize/explain/intervene records, the
    request-declared comparisons and metric sets, both sides' identical
    alignment against the bound prior evidence and frozen manifest, and the
    immutable run/checkpoint identities — raising ``StageContractError``
    before any measurement callback runs — then measures both sides on both
    metrics with per-side bootstrap uncertainty, computes signed/absolute
    deltas against the declared tolerances, and returns a ``completed``
    (or honest ``unsupported``) ``StageOutput`` whose payload carries the
    bounded classification per comparison. Measurement logic lives only in
    the caller's ``measure`` callback; no algorithm enters
    ``DiagnosticWorkflow`` itself.
    """
    _non_empty(version, name="version")
    frozen = _require_spec_list(comparisons)
    identities = [spec.comparison_id for spec in frozen]
    if len(set(identities)) != len(identities):
        raise ComparisonError("comparison identifiers must be unique")
    if not callable(measure):
        raise ComparisonError("measure must be callable")
    declared: tuple[ComparisonSpec, ...] = frozen

    def _execute(invocation: object) -> StageOutput:
        invocation = _require_invocation(invocation)
        if invocation.stage != "compare":
            raise StageContractError(f"compare executor received stage {invocation.stage!r}")
        context, bound = _bind_comparisons(declared, invocation)
        records = [_execute_comparison(item, measure, context) for item in bound]
        return _compare_output(records, context)

    _execute.comparison_version = version  # type: ignore[attr-defined]
    return _execute


__all__ = [
    "ALIGNMENT_FIELDS",
    "COMPARISON_CLASSIFICATIONS",
    "COMPARISON_VERSION",
    "ComparisonApplication",
    "ComparisonError",
    "ComparisonMeasureFn",
    "ComparisonMeasurement",
    "ComparisonSpec",
    "RunSide",
    "capture_axes_identity",
    "comparison_report_items",
    "detect_config_identity",
    "make_compare_executor",
    "manifest_seed_identity",
]

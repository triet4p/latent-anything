"""Architecture-neutral intervention-trial executors for the workflow
``intervene`` stage (Sprints 80.18 and 80.19).

One private execution seam covers every declared causal trial family without
a single model-, layer-, or architecture-specific field. The caller supplies
one measurement callback; the executor owns identity binding, control
scheduling, seed/provenance recording, uncertainty, and the bounded
conclusion decision.

80.18 — single-strength trials (``patch`` activation patching, ``ablate``,
``remove`` concept/direction removal):

Binding contract (every check runs before any measurement callback fires):
each trial is bound to the exact prior ``capture`` payload identity
(``BoundCapture`` provenance: capture id, representation identity, manifest,
request, and the deterministic ``capture_identity`` digest naming the shared
captured inputs), the exact prior ``localize`` payload identity (family,
metric, manifest, representation), and the exact prior ``explain`` payload
identity (hypothesis row plus a non-omitted ``family_evidence`` outcome,
with that hypothesis carrying the trial's downstream metric). The trial
metric must be a request-declared metric and a manifest causal-expectation
target metric with a positive predeclared threshold tolerance. Requested
intervention ids, targets, and control declarations must match the declared
trials exactly.

Control contract: every80.18 trial declares exactly one control of each
required class — ``zero_strength`` (identity / zero-strength rerun),
``random``, ``shuffled``, and ``off_target`` — all required, all evaluated
by the central ``_statistical_controls`` executor under identity-derived
streams. Every control measures the same captured inputs and the same
downstream metric as the baseline and intervened measurements; the executor
verifies that each returned measurement echoes the bound
``capture_identity`` digest and the declared metric id, and rejects
non-finite values.

Predeclared control rule: the zero-strength control must reproduce the
baseline within the manifest threshold tolerance; random, shuffled, and
off-target controls must move the metric strictly less than
``max(|intervened - baseline|, tolerance)``.

80.18 conclusion decision table (bounded supported / falsified /
inconclusive / unsupported, never promoted past a failed control):
1. failed zero-strength control -> ``unsupported`` (the measurement pipeline
   is untrustworthy; causal support is blocked);
2. failed random, shuffled, or off-target control -> ``falsified`` (the
   effect is not target-specific, matching the manifest falsification rule);
3. ``|intervened - baseline|`` within tolerance -> ``falsified`` (the
   declared intervention at the declared strength is inert beyond noise);
4. material effect opposite the declared direction -> ``falsified``;
5. baseline inside the intervened bootstrap interval -> ``inconclusive``
   (uncertainty cannot resolve the effect);
6. otherwise -> ``supported``.

80.19 — steering dose-response trials (``SteeringTrialSpec``): a predeclared
strength series that includes the zero/identity dose and at least two
nonzero doses, an immutable unit direction with a SHA-256 digest and full
direction provenance (method, carrier, source) bound before callbacks, and
three required control classes — ``zero_strength``, ``random_direction``
(an rng-drawn direction applied to the same captured inputs, same metric,
same scope, at the strongest declared dose), and ``off_target`` (the
declared direction applied to a different prior-localized selection). The
steering target must itself be a prior localize selection. Control rules:
the zero-strength control must reproduce the baseline within tolerance; the
random-direction control fails only when its effect has the declared sign
AND reaches ``max(|endpoint effect|, tolerance)`` (a random direction that
moves the metric differently does not reproduce the declared response and
is recorded, never hidden); the off-target control fails on any movement
(sign-agnostic) reaching that bound, enforcing selectivity.

80.19 response classification: ``monotonic_increase`` /
``monotonic_decrease`` (consecutive dose steps beyond tolerance share one
sign), ``non_monotonic`` (material steps of both signs), ``inert`` (no dose
moves the metric beyond tolerance), or ``unsupported`` (the identity control
broke the pipeline). Conclusion decision table:
1. failed zero-strength control -> ``unsupported`` (classification
   ``unsupported``; causal support blocked);
2. failed random-direction or off-target control -> ``falsified`` (direction
   specificity or selectivity fails);
3. ``inert`` -> ``falsified`` (no task-metric change beyond tolerance);
4. ``non_monotonic`` -> ``falsified`` (the predeclared monotonic dose
   expectation fails);
5. monotonic response opposite the declared direction -> ``falsified``;
6. endpoint bootstrap interval spanning the baseline -> ``inconclusive``;
7. otherwise -> ``supported``. A directional correlation never promotes to
   causality without passing every control, the dose expectation, and the
   uncertainty gate.

Measurement-contract violations (wrong metric, wrong inputs digest,
non-finite value, callback crash) fail the stage closed; missing or
mismatched identities, metrics, provenance, dose plans, controls, and
declared non-finite values raise before any callback runs. Method
algorithms stay in the caller-supplied callback; no algorithm enters
``DiagnosticWorkflow`` itself.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from hashlib import sha256
from math import isfinite
from typing import Any, Literal

import numpy as np

from latent_anything._diagnostic_workflow import StageContractError, StageInvocation, StageOutput
from latent_anything._statistical_controls import (
    derive_stream_seed,
    execute_plan,
    run_bootstrap,
    ControlPlan as _ControlPlan,
    ControlSpec as _ControlSpec,
)
from latent_anything.diagnostics import DiagnosticRequest, InterventionRequest

INTERVENTION_KINDS: tuple[str, ...] = ("patch", "ablate", "remove")
"""The three80.18 single-strength trial kinds. Anything else rejects there."""

STEERING_KIND: str = "steer"
"""The80.19 steering dose-response trial kind."""

INTERVENTION_CONTROL_CLASSES: tuple[str, ...] = (
    "zero_strength",
    "random",
    "shuffled",
    "off_target",
)
"""Every80.18 trial requires exactly these four control classes, all required."""

STEERING_CONTROL_CLASSES: tuple[str, ...] = (
    "zero_strength",
    "random_direction",
    "off_target",
)
"""Every80.19 steering trial requires exactly these three classes, all required."""

_KNOWN_CONTROL_CLASSES: frozenset[str] = frozenset(
    (*INTERVENTION_CONTROL_CLASSES, *STEERING_CONTROL_CLASSES)
)

INTERVENTION_VERSION = "intervention-trials-v1"
"""Stage executor version bound into the workflow config digest."""

_CONTROL_CLASS_KINDS: dict[str, str] = {
    "zero_strength": "null",
    "random": "randomized",
    "random_direction": "randomized",
    "shuffled": "shuffled",
    "off_target": "negative",
}
"""Binding from intervention control class to central control kind."""

_CONCLUSION_OUTCOMES: tuple[str, ...] = ("supported", "falsified", "inconclusive", "unsupported")
_EXPECTED_EFFECTS: tuple[str, ...] = ("increase", "decrease")
_ROLES: tuple[str, ...] = ("baseline", "intervened", "control")
_STEERING_CLASSIFICATIONS: tuple[str, ...] = (
    "monotonic_increase",
    "monotonic_decrease",
    "non_monotonic",
    "inert",
    "unsupported",
)
_DIGEST_LENGTH = 64
_REQUIRED_PROVENANCE_KEYS: tuple[str, ...] = ("method", "carrier")
_REQUIRED_DIRECTION_PROVENANCE_KEYS: tuple[str, ...] = ("method", "carrier", "source")


class InterventionError(ValueError):
    """Raised when a declared trial or a returned measurement is fail-closed invalid."""


def _non_empty_string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InterventionError(f"{name} must be a non-empty string")
    return value


def _finite(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(float(value)):
        raise InterventionError(f"{name} must be a finite number")
    return float(value)


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise InterventionError(f"{name} must be an object")
    for key in value:
        if not isinstance(key, str):
            raise InterventionError(f"{name} must use string keys")
    return value


def _digest_string(value: object, *, name: str) -> str:
    text = _non_empty_string(value, name=name)
    if len(text) != _DIGEST_LENGTH or any(ch not in "0123456789abcdef" for ch in text):
        raise InterventionError(f"{name} must be a lowercase SHA-256 digest")
    return text


def _require(condition: bool, message: str, *, error: type[ValueError] = InterventionError) -> None:
    if not condition:
        raise error(message)


def _provenance_map(value: object, required: tuple[str, ...], *, name: str) -> dict[str, str]:
    provenance = _mapping(value, name=name)
    for key in required:
        _non_empty_string(provenance.get(key), name=f"{name}[{key!r}]")
    for key, item in provenance.items():
        if not isinstance(item, str):
            raise InterventionError(f"{name}[{key!r}] must be a string")
    return dict(provenance)


# ---------------------------------------------------------------------------
# Declared trial carriers (caller-declared before any callback runs)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrialControl:
    """One declared control for one trial: identity, class, and expectation."""

    control_id: str
    control_class: str
    expected_behavior: str
    target: str = ""

    def __post_init__(self) -> None:
        _non_empty_string(self.control_id, name="control_id")
        if self.control_class not in _KNOWN_CONTROL_CLASSES:
            raise InterventionError(
                f"unsupported intervention control class: {self.control_class!r}; "
                f"declared classes are {sorted(_KNOWN_CONTROL_CLASSES)!r}"
            )
        _non_empty_string(self.expected_behavior, name="expected_behavior")
        if not isinstance(self.target, str):
            raise InterventionError("control target must be a string")
        if self.control_class == "off_target":
            _non_empty_string(self.target, name="off_target control target")
        elif self.target:
            raise InterventionError(
                f"only off_target controls may declare a target override, got {self.target!r}"
            )


def _validate_controls(
    controls: object, expected_classes: tuple[str, ...], *, target: str
) -> tuple[TrialControl, ...]:
    if isinstance(controls, str | bytes) or not isinstance(controls, Sequence):
        raise InterventionError("controls must be a sequence of TrialControl items")
    resolved = tuple(controls)
    if not resolved:
        raise InterventionError("a trial must declare controls")
    for position, control in enumerate(resolved):
        if not isinstance(control, TrialControl):
            raise InterventionError(f"controls[{position}] must be a TrialControl")
    classes = [control.control_class for control in resolved]
    if len(classes) != len(expected_classes) or sorted(classes) != sorted(expected_classes):
        raise InterventionError(
            "a trial must declare exactly one control of each required class "
            f"{list(expected_classes)!r}, got {sorted(classes)!r}"
        )
    identities = [control.control_id for control in resolved]
    if len(set(identities)) != len(identities):
        raise InterventionError("trial control identifiers must be unique")
    off_target = next(c for c in resolved if c.control_class == "off_target")
    if off_target.target == target:
        raise InterventionError("the off_target control must name a different target than the trial")
    return resolved


@dataclass(frozen=True)
class TrialSpec:
    """One declared patch/ablate/remove causal trial, fully validated up front."""

    intervention_id: str
    kind: str
    target: str
    metric_id: str
    hypothesis_id: str
    strength: float
    strength_semantics: str
    expected_effect: str
    controls: tuple[TrialControl, ...]
    provenance: Mapping[str, str]

    def __post_init__(self) -> None:
        _non_empty_string(self.intervention_id, name="intervention_id")
        if self.kind not in INTERVENTION_KINDS:
            raise InterventionError(
                f"unsupported intervention kind: {self.kind!r}; "
                f"expected one of {list(INTERVENTION_KINDS)!r}"
            )
        _non_empty_string(self.target, name="target")
        _non_empty_string(self.metric_id, name="metric_id")
        _non_empty_string(self.hypothesis_id, name="hypothesis_id")
        object.__setattr__(self, "strength", _finite(self.strength, name="strength"))
        _non_empty_string(self.strength_semantics, name="strength_semantics")
        if self.expected_effect not in _EXPECTED_EFFECTS:
            raise InterventionError(
                f"expected_effect must be one of {list(_EXPECTED_EFFECTS)!r}, "
                f"got {self.expected_effect!r}"
            )
        object.__setattr__(
            self, "controls", _validate_controls(self.controls, INTERVENTION_CONTROL_CLASSES, target=self.target)
        )
        object.__setattr__(self, "provenance", _provenance_map(self.provenance, _REQUIRED_PROVENANCE_KEYS, name="provenance"))


@dataclass(frozen=True)
class SteeringTrialSpec:
    """One declared steering dose-response trial with immutable direction provenance.

    ``strengths`` is the predeclared dose series: it must contain the
    zero/identity dose and at least two nonzero doses so the response can be
    characterized. ``direction`` must be a finite unit vector (the
    ``SteeringVector.direction`` contract); it is frozen read-only and
    fingerprinted into ``direction_digest`` at construction, and the
    provenance mapping must name the direction's method, carrier, and
    source before any callback runs.
    """

    intervention_id: str
    target: str
    metric_id: str
    hypothesis_id: str
    strengths: tuple[float, ...]
    strength_semantics: str
    expected_effect: str
    direction: np.ndarray
    provenance: Mapping[str, str]
    controls: tuple[TrialControl, ...]
    direction_digest: str = field(init=False)

    def __post_init__(self) -> None:
        _non_empty_string(self.intervention_id, name="intervention_id")
        _non_empty_string(self.target, name="target")
        _non_empty_string(self.metric_id, name="metric_id")
        _non_empty_string(self.hypothesis_id, name="hypothesis_id")
        if isinstance(self.strengths, str | bytes) or not isinstance(self.strengths, Sequence):
            raise InterventionError("strengths must be a sequence of finite doses")
        doses = tuple(
            _finite(raw, name=f"strengths[{position}]")
            for position, raw in enumerate(self.strengths)
        )
        if not doses:
            raise InterventionError("strengths must declare at least the zero/identity dose")
        if len(set(doses)) != len(doses):
            raise InterventionError("strength doses must be unique")
        if 0.0 not in doses:
            raise InterventionError("strengths must include the zero/identity dose")
        if sum(1 for dose in doses if dose != 0.0) < 2:
            raise InterventionError(
                "strengths must declare at least two nonzero doses to characterize the response"
            )
        object.__setattr__(self, "strengths", doses)
        _non_empty_string(self.strength_semantics, name="strength_semantics")
        if self.expected_effect not in _EXPECTED_EFFECTS:
            raise InterventionError(
                f"expected_effect must be one of {list(_EXPECTED_EFFECTS)!r}, "
                f"got {self.expected_effect!r}"
            )
        try:
            direction = np.asarray(self.direction, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise InterventionError(f"direction must be numeric: {exc}") from exc
        if direction.ndim != 1 or direction.size < 1:
            raise InterventionError("direction must be a 1-D array")
        if not bool(np.all(np.isfinite(direction))):
            raise InterventionError("direction must be finite")
        norm = float(np.linalg.norm(direction))
        if not isfinite(norm) or abs(norm - 1.0) > 1e-6:
            raise InterventionError(
                "direction must be a unit vector (the SteeringVector.direction contract); "
                f"got norm {norm}"
            )
        frozen = np.array(direction, dtype=np.float64, copy=True)
        frozen.setflags(write=False)
        object.__setattr__(self, "direction", frozen)
        fingerprint = sha256()
        fingerprint.update(b"steering-direction-v1")
        fingerprint.update(str(frozen.shape).encode("utf-8"))
        fingerprint.update(frozen.astype("<f8", copy=False).tobytes())
        object.__setattr__(self, "direction_digest", fingerprint.hexdigest())
        object.__setattr__(
            self,
            "provenance",
            _provenance_map(self.provenance, _REQUIRED_DIRECTION_PROVENANCE_KEYS, name="provenance"),
        )
        object.__setattr__(
            self, "controls", _validate_controls(self.controls, STEERING_CONTROL_CLASSES, target=self.target)
        )

    @property
    def kind(self) -> str:
        """The recorded trial kind for steering dose-response records."""
        return STEERING_KIND


# ---------------------------------------------------------------------------
# Measurement contract (what the caller's callback must return)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Measurement:
    """One downstream metric measurement echoing the shared captured inputs."""

    metric_id: str
    value: float
    inputs_digest: str

    def __post_init__(self) -> None:
        _non_empty_string(self.metric_id, name="measurement metric_id")
        _digest_string(self.inputs_digest, name="measurement inputs_digest")
        object.__setattr__(self, "value", _finite(self.value, name="measurement value"))


@dataclass(frozen=True)
class TrialApplication:
    """One measurement request: the exact intervention (or baseline) to apply."""

    intervention_id: str
    kind: str
    target: str
    strength: float
    strength_semantics: str
    role: Literal["baseline", "intervened", "control"]
    control_class: str | None
    control_id: str | None
    metric_id: str
    inputs_digest: str
    stream_seed: int
    rng: np.random.Generator | None

    def __post_init__(self) -> None:
        _non_empty_string(self.intervention_id, name="application intervention_id")
        if self.kind not in (*INTERVENTION_KINDS, STEERING_KIND):
            raise InterventionError(
                f"application kind must be one of {[*INTERVENTION_KINDS, STEERING_KIND]!r}"
            )
        _non_empty_string(self.target, name="application target")
        object.__setattr__(self, "strength", _finite(self.strength, name="application strength"))
        _non_empty_string(self.strength_semantics, name="application strength_semantics")
        if self.role not in _ROLES:
            raise InterventionError(f"application role must be one of {list(_ROLES)!r}")
        _non_empty_string(self.metric_id, name="application metric_id")
        _digest_string(self.inputs_digest, name="application inputs_digest")
        if isinstance(self.stream_seed, bool) or not isinstance(self.stream_seed, int) or self.stream_seed < 0:
            raise InterventionError("application stream_seed must be a non-negative integer")
        if self.role == "control":
            if self.control_class not in _KNOWN_CONTROL_CLASSES:
                raise InterventionError(
                    "control applications must declare a required control class"
                )
            _non_empty_string(self.control_id, name="application control_id")
        elif self.control_class is not None or self.control_id is not None:
            raise InterventionError(
                "baseline/intervened applications must not declare control identity"
            )
        if self.rng is not None and not isinstance(self.rng, np.random.Generator):
            raise InterventionError("application rng must be a numpy Generator or None")


MeasureFn = Callable[[TrialApplication], Measurement]
"""One caller-supplied measurement callback over the shared captured inputs."""


# ---------------------------------------------------------------------------
# Stage binding: every prior identity checked before any callback runs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ResolvedTrial:
    """One80.18 trial plus its manifest/request-derived decision inputs."""

    spec: TrialSpec
    request_entry: InterventionRequest
    metric_direction: str
    threshold: Mapping[str, object]
    zero_tolerance: float


@dataclass(frozen=True)
class _ResolvedSteering:
    """ One80.19 steering trial plus its decision inputs."""

    spec: SteeringTrialSpec
    request_entry: InterventionRequest
    metric_direction: str
    threshold: Mapping[str, object]
    zero_tolerance: float


@dataclass(frozen=True)
class _DeclarationTables:
    """Shared request/manifest/localize/explain lookup tables for one run."""

    request_index: Mapping[str, InterventionRequest]
    declared_metrics: frozenset[str]
    causal_targets: frozenset[str]
    metric_directions: Mapping[str, str]
    threshold_rows: Mapping[str, Mapping[str, object]]
    localize_family_id: str
    localize_metric_id: str
    localize_selections: frozenset[str]
    explain_rows: Mapping[str, Mapping[str, object]]
    explain_evidence: Mapping[str, object]
    manifest_id: str
    representation_identity: str


@dataclass(frozen=True)
class _BoundDeclaration:
    """Result of binding one declared trial identity set."""

    metric_direction: str
    threshold: Mapping[str, object]
    zero_tolerance: float


@dataclass(frozen=True)
class _StageContext:
    """Frozen bound identities and predeclared inputs for one intervene run."""

    manifest_id: str
    request_id: str
    capture_identity: str
    representation_identity: str
    localize_family_id: str
    localize_metric_id: str
    localize_verdict: str
    evaluation_seed: int
    control_seed: int
    repetitions: int
    confidence_level: float
    causal_expectation: Mapping[str, object]
    workflow_identity: str
    request_digest: str
    manifest_digest: str
    config_digest: str
    trials: tuple[_ResolvedTrial, ...] = ()


def _prior_payloads(invocation: StageInvocation) -> dict[str, Mapping[str, object]]:
    return {item.stage: dict(item.payload) for item in invocation.prior}


def _seed_from_manifest(manifest: Mapping[str, object], role: str) -> int:
    seeds = manifest.get("seeds")
    _require(isinstance(seeds, Mapping), "intervene executor requires manifest seeds", error=StageContractError)
    rows = seeds.get(role)  # pyright: ignore[reportUnknownMemberType]
    _require(
        isinstance(rows, Sequence) and not isinstance(rows, str | bytes) and len(rows) > 0,
        f"intervene executor requires manifest seeds.{role}",
        error=StageContractError,
    )
    first = rows[0]  # pyright: ignore[reportUnknownVariableType]
    _require(
        isinstance(first, int) and not isinstance(first, bool) and first >= 0,
        f"manifest seeds.{role}[0] must be a non-negative integer",
        error=StageContractError,
    )
    return int(first)


def _bind_capture(invocation: StageInvocation, request: DiagnosticRequest) -> str:
    prior = _prior_payloads(invocation)
    _require("capture" in prior, "intervene executor requires a prior capture payload", error=StageContractError)
    raw_capture = prior["capture"].get("capture")
    _require(
        isinstance(raw_capture, Mapping),
        "the prior capture payload must carry a bound capture under the 'capture' key",
        error=StageContractError,
    )
    try:
        capture_identity = _digest_string(raw_capture.get("capture_identity"), name="capture_identity")
    except InterventionError as exc:
        raise StageContractError(str(exc)) from exc
    fields = (
        ("capture_id", request.capture.capture_id),
        ("representation_identity", request.capture.representation_identity),
        ("manifest_id", request.manifest_id),
        ("request_id", request.request_id),
    )
    for key, expected in fields:
        value = raw_capture.get(key)
        _require(
            isinstance(value, str) and value == expected,
            f"prior capture {key!r} is {value!r}, expected {expected!r}",
            error=StageContractError,
        )
    return capture_identity


def _bind_localize(
    invocation: StageInvocation, request: DiagnosticRequest, manifest: Mapping[str, object]
) -> tuple[str, str, str]:
    prior = _prior_payloads(invocation)
    _require("localize" in prior, "intervene executor requires a prior localize payload", error=StageContractError)
    localize = prior["localize"]
    manifest_id = manifest.get("manifest_id")
    for key, expected in (
        ("manifest_id", manifest_id),
        ("representation_identity", request.capture.representation_identity),
    ):
        value = localize.get(key)
        _require(
            isinstance(value, str) and isinstance(expected, str) and value == expected,
            f"prior localize {key!r} is {value!r}, expected {expected!r}",
            error=StageContractError,
        )
    family_id = localize.get("family_id")
    metric_id = localize.get("metric_id")
    verdict = localize.get("verdict")
    _require(
        isinstance(family_id, str) and bool(family_id.strip()),
        "prior localize payload must declare family_id",
        error=StageContractError,
    )
    _require(
        isinstance(metric_id, str) and bool(metric_id.strip()),
        "prior localize payload must declare metric_id",
        error=StageContractError,
    )
    _require(isinstance(verdict, str), "prior localize payload must declare verdict", error=StageContractError)
    return family_id, metric_id, verdict


def _localize_selections(localize: Mapping[str, object]) -> frozenset[str]:
    """Extract structured selection identities a steering target must bind to."""
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
    axes = localize.get("axes")
    if isinstance(axes, Sequence) and not isinstance(axes, str | bytes):
        for entry in axes:
            if not isinstance(entry, Mapping):
                continue
            for key in ("affected", "earliest"):
                value = entry.get(key)
                if isinstance(value, str) and value:
                    selections.add(value)
                elif isinstance(value, Sequence) and not isinstance(value, str | bytes):
                    selections.update(item for item in value if isinstance(item, str) and item)
            report_rows = entry.get("report_rows")
            if isinstance(report_rows, Sequence) and not isinstance(report_rows, str | bytes):
                for row in report_rows:
                    if isinstance(row, Mapping):
                        selection = row.get("selection")
                        if isinstance(selection, str) and selection:
                            selections.add(selection)
    return frozenset(selections)


def _explain_index(
    invocation: StageInvocation,
) -> tuple[dict[str, Mapping[str, object]], Mapping[str, object]]:
    prior = _prior_payloads(invocation)
    _require("explain" in prior, "intervene executor requires a prior explain payload", error=StageContractError)
    explain = prior["explain"]
    rows = explain.get("hypotheses")
    _require(
        isinstance(rows, Sequence) and not isinstance(rows, str | bytes) and len(rows) > 0,
        "prior explain payload must declare at least one hypothesis",
        error=StageContractError,
    )
    index: dict[str, Mapping[str, object]] = {}
    for position, raw in enumerate(rows):  # pyright: ignore[reportUnknownVariableType]
        _require(
            isinstance(raw, Mapping),
            f"explain hypotheses[{position}] must be an object",
            error=StageContractError,
        )
        hypothesis_id = raw.get("hypothesis_id")
        _require(
            isinstance(hypothesis_id, str) and bool(hypothesis_id.strip()),
            f"explain hypotheses[{position}] must declare hypothesis_id",
            error=StageContractError,
        )
        _require(
            hypothesis_id not in index,
            "explain hypothesis identifiers must be unique",
            error=StageContractError,
        )
        index[hypothesis_id] = dict(raw)
    evidence = explain.get("family_evidence")
    _require(
        isinstance(evidence, Mapping),
        "prior explain payload must declare structured family_evidence",
        error=StageContractError,
    )
    return index, evidence


def _string_entries(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,) if value else ()
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return tuple(item for item in value if isinstance(item, str) and item)
    return ()


def _causal_and_metric_tables(
    manifest: Mapping[str, object],
) -> tuple[frozenset[str], dict[str, str], dict[str, Mapping[str, object]]]:
    causal = manifest.get("causal_expectation")
    _require(
        isinstance(causal, Mapping),
        "intervene executor requires a causal expectation",
        error=StageContractError,
    )
    _require(
        causal.get("applicable") is True,
        "manifest causal expectation is not applicable; intervention trials cannot run",
        error=StageContractError,
    )
    targets = causal.get("target_metric_ids")
    _require(
        isinstance(targets, Sequence) and not isinstance(targets, str | bytes),
        "manifest causal expectation must declare target_metric_ids",
        error=StageContractError,
    )
    causal_targets = frozenset(str(item) for item in targets)

    metric_directions: dict[str, str] = {}
    raw_metrics = manifest.get("metrics")
    _require(
        isinstance(raw_metrics, Sequence) and not isinstance(raw_metrics, str | bytes) and len(raw_metrics) > 0,
        "intervene executor requires a non-empty manifest metrics list",
        error=StageContractError,
    )
    for position, raw in enumerate(raw_metrics):  # pyright: ignore[reportUnknownVariableType]
        _require(
            isinstance(raw, Mapping),
            f"manifest metrics[{position}] must be an object",
            error=StageContractError,
        )
        metric_id = raw.get("id")
        direction = raw.get("direction")
        _require(
            isinstance(metric_id, str) and isinstance(direction, str),
            f"manifest metrics[{position}] must declare id and direction",
            error=StageContractError,
        )
        metric_directions[metric_id] = direction

    threshold_rows: dict[str, Mapping[str, object]] = {}
    raw_thresholds = manifest.get("thresholds")
    _require(
        isinstance(raw_thresholds, Sequence)
        and not isinstance(raw_thresholds, str | bytes)
        and len(raw_thresholds) > 0,
        "intervene executor requires a non-empty manifest thresholds list",
        error=StageContractError,
    )
    for position, raw in enumerate(raw_thresholds):  # pyright: ignore[reportUnknownVariableType]
        _require(
            isinstance(raw, Mapping),
            f"manifest thresholds[{position}] must be an object",
            error=StageContractError,
        )
        metric_id = raw.get("metric_id")
        _require(
            isinstance(metric_id, str),
            f"manifest thresholds[{position}] must declare metric_id",
            error=StageContractError,
        )
        _require(
            metric_id not in threshold_rows,
            "manifest thresholds must be unique per metric",
            error=StageContractError,
        )
        threshold_rows[metric_id] = dict(raw)
    return causal_targets, metric_directions, threshold_rows


def _bind_stage(invocation: StageInvocation) -> tuple[_StageContext, _DeclarationTables]:
    """Verify every shared prior identity and predeclared manifest input.

    Runs in full before any measurement callback is invoked for either trial
    family; the first gap raises ``StageContractError`` so the stage fails
    closed without touching model code.
    """
    request = invocation.request
    manifest = invocation.manifest
    manifest_id = manifest.get("manifest_id")
    _require(
        isinstance(manifest_id, str) and bool(manifest_id),
        "intervene executor requires a manifest_id",
        error=StageContractError,
    )
    _require(
        request.manifest_id == manifest_id,
        f"intervene executor manifest mismatch: request {request.manifest_id!r} != "
        f"manifest {manifest_id!r}",
        error=StageContractError,
    )
    expected_prior = ("capture", "detect", "localize", "explain")
    stages = tuple(item.stage for item in invocation.prior)
    _require(
        stages == expected_prior,
        f"intervene executor prior stages must be exactly {list(expected_prior)!r}, got {stages!r}",
        error=StageContractError,
    )
    capture_identity = _bind_capture(invocation, request)
    localize_family_id, localize_metric_id, localize_verdict = _bind_localize(
        invocation, request, manifest
    )
    explain_rows, explain_evidence = _explain_index(invocation)
    uncertainty = manifest.get("uncertainty")
    _require(
        isinstance(uncertainty, Mapping),
        "intervene executor requires manifest uncertainty",
        error=StageContractError,
    )
    repetitions = uncertainty.get("repetitions")
    _require(
        isinstance(repetitions, int) and not isinstance(repetitions, bool) and repetitions >= 2,
        "manifest uncertainty.repetitions must be at least two",
        error=StageContractError,
    )
    try:
        confidence = _finite(uncertainty.get("confidence_level"), name="uncertainty.confidence_level")
    except InterventionError as exc:
        raise StageContractError(str(exc)) from exc
    _require(
        0.0 < confidence < 1.0,
        "manifest confidence_level must be between zero and one",
        error=StageContractError,
    )
    causal = manifest.get("causal_expectation")
    assert isinstance(causal, Mapping)  # validated in _causal_and_metric_tables
    causal_targets, metric_directions, threshold_rows = _causal_and_metric_tables(manifest)
    causal_record: dict[str, object] = {
        "applicable": True,
        "expectation": str(causal.get("expectation")),
        "falsification_rule": str(causal.get("falsification_rule")),
        "target_metric_ids": sorted(causal_targets),
    }
    prior_localize = _prior_payloads(invocation)["localize"]
    context = _StageContext(
        manifest_id=manifest_id,
        request_id=request.request_id,
        capture_identity=capture_identity,
        representation_identity=request.capture.representation_identity,
        localize_family_id=localize_family_id,
        localize_metric_id=localize_metric_id,
        localize_verdict=localize_verdict,
        evaluation_seed=_seed_from_manifest(manifest, "evaluation"),
        control_seed=_seed_from_manifest(manifest, "controls"),
        repetitions=int(repetitions),
        confidence_level=confidence,
        causal_expectation=dict(causal_record),
        workflow_identity=invocation.workflow_identity,
        request_digest=invocation.request_digest,
        manifest_digest=invocation.manifest_digest,
        config_digest=invocation.config_digest,
    )
    tables = _DeclarationTables(
        request_index={entry.intervention_id: entry for entry in request.interventions},
        declared_metrics=frozenset(request.controls.metric_ids),
        causal_targets=causal_targets,
        metric_directions=dict(metric_directions),
        threshold_rows=dict(threshold_rows),
        localize_family_id=localize_family_id,
        localize_metric_id=localize_metric_id,
        localize_selections=_localize_selections(prior_localize),
        explain_rows=dict(explain_rows),
        explain_evidence=dict(explain_evidence),
        manifest_id=manifest_id,
        representation_identity=request.capture.representation_identity,
    )
    return context, tables


def _bind_declaration(
    *,
    intervention_id: str,
    target: str,
    metric_id: str,
    hypothesis_id: str,
    control_ids: set[str],
    tables: _DeclarationTables,
) -> _BoundDeclaration:
    """Bind one declared trial identity set against request/manifest/prior evidence."""
    entry = tables.request_index.get(intervention_id)
    _require(
        entry is not None,
        f"trial {intervention_id!r} has no matching requested intervention",
        error=StageContractError,
    )
    assert entry is not None
    _require(
        entry.target == target,
        f"trial {intervention_id!r} target {target!r} != requested {entry.target!r}",
        error=StageContractError,
    )
    _require(
        set(entry.control_ids) == control_ids,
        f"trial {intervention_id!r} controls {sorted(control_ids)!r} != "
        f"requested {sorted(set(entry.control_ids))!r}",
        error=StageContractError,
    )
    _require(
        metric_id in tables.declared_metrics,
        f"trial {intervention_id!r} metric {metric_id!r} is not a request-declared metric",
        error=StageContractError,
    )
    _require(
        metric_id in tables.causal_targets,
        f"trial {intervention_id!r} metric {metric_id!r} is not a "
        "manifest causal-expectation target metric",
        error=StageContractError,
    )
    _require(
        metric_id == tables.localize_metric_id,
        f"trial {intervention_id!r} metric {metric_id!r} does not match the "
        f"prior localize metric {tables.localize_metric_id!r}",
        error=StageContractError,
    )
    _require(
        metric_id in tables.metric_directions,
        f"trial {intervention_id!r} metric {metric_id!r} is not manifest-declared",
        error=StageContractError,
    )
    row = tables.explain_rows.get(hypothesis_id)
    _require(
        row is not None,
        f"trial {intervention_id!r} hypothesis {hypothesis_id!r} has no prior explain row",
        error=StageContractError,
    )
    row_metrics = row.get("metric_ids")
    _require(
        isinstance(row_metrics, Sequence) and not isinstance(row_metrics, str | bytes),
        f"explain hypothesis {hypothesis_id!r} must declare metric_ids",
        error=StageContractError,
    )
    _require(
        metric_id in {str(item) for item in row_metrics},
        f"trial {intervention_id!r} metric {metric_id!r} is not carried by "
        f"explain hypothesis {hypothesis_id!r}",
        error=StageContractError,
    )
    _require(
        row.get("family_id") == tables.localize_family_id,
        f"explain hypothesis {hypothesis_id!r} family does not match "
        f"the prior localize family {tables.localize_family_id!r}",
        error=StageContractError,
    )
    row_representation = row.get("representation_id")
    _require(
        isinstance(row_representation, str)
        and (row_representation == "" or row_representation == tables.representation_identity),
        f"explain hypothesis {hypothesis_id!r} representation contradicts the capture",
        error=StageContractError,
    )
    row_manifest = row.get("manifest_id")
    _require(
        isinstance(row_manifest, str)
        and (row_manifest == "" or row_manifest == tables.manifest_id),
        f"explain hypothesis {hypothesis_id!r} manifest contradicts the run manifest",
        error=StageContractError,
    )
    evidence_row = tables.explain_evidence.get(hypothesis_id)
    _require(
        isinstance(evidence_row, Mapping),
        f"trial {intervention_id!r} hypothesis {hypothesis_id!r} has no explain evidence",
        error=StageContractError,
    )
    outcome = evidence_row.get("outcome")
    _require(
        isinstance(outcome, str) and bool(outcome),
        f"explain evidence for {hypothesis_id!r} must declare an outcome",
        error=StageContractError,
    )
    _require(
        outcome != "omitted",
        f"trial {intervention_id!r} binds hypothesis {hypothesis_id!r} whose "
        "explanation was never evaluated (omitted)",
        error=StageContractError,
    )
    threshold = tables.threshold_rows.get(metric_id)
    _require(
        threshold is not None,
        f"trial {intervention_id!r} metric {metric_id!r} has no predeclared threshold",
        error=StageContractError,
    )
    try:
        zero_tolerance = _finite(threshold.get("tolerance"), name="threshold tolerance")
    except InterventionError as exc:
        raise StageContractError(str(exc)) from exc
    _require(
        zero_tolerance > 0.0,
        f"trial {intervention_id!r} requires a positive predeclared threshold tolerance "
        "for identity/zero-strength gating",
        error=StageContractError,
    )
    return _BoundDeclaration(
        metric_direction=tables.metric_directions[metric_id],
        threshold=dict(threshold),
        zero_tolerance=zero_tolerance,
    )


def _bind_request_set(declared_ids: set[str], tables: _DeclarationTables) -> None:
    _require(
        declared_ids == set(tables.request_index),
        "declared trials and requested interventions must match exactly: "
        f"trials {sorted(declared_ids)!r} != request {sorted(tables.request_index)!r}",
        error=StageContractError,
    )


def _bind_trials(
    trials: tuple[TrialSpec, ...], tables: _DeclarationTables
) -> tuple[_ResolvedTrial, ...]:
    _bind_request_set({spec.intervention_id for spec in trials}, tables)
    resolved: list[_ResolvedTrial] = []
    for spec in trials:
        bound = _bind_declaration(
            intervention_id=spec.intervention_id,
            target=spec.target,
            metric_id=spec.metric_id,
            hypothesis_id=spec.hypothesis_id,
            control_ids={control.control_id for control in spec.controls},
            tables=tables,
        )
        entry = tables.request_index[spec.intervention_id]
        resolved.append(
            _ResolvedTrial(
                spec=spec,
                request_entry=entry,
                metric_direction=bound.metric_direction,
                threshold=bound.threshold,
                zero_tolerance=bound.zero_tolerance,
            )
        )
    return tuple(resolved)


def _bind_context(trials: tuple[TrialSpec, ...], invocation: StageInvocation) -> _StageContext:
    """Full pre-callback binding for one80.18 intervene run."""
    context, tables = _bind_stage(invocation)
    return replace(context, trials=_bind_trials(trials, tables))


def _bind_steering(
    specs: tuple[SteeringTrialSpec, ...], invocation: StageInvocation
) -> tuple[_StageContext, tuple[_ResolvedSteering, ...]]:
    """Full pre-callback binding for one80.19 steering intervene run."""
    context, tables = _bind_stage(invocation)
    _bind_request_set({spec.intervention_id for spec in specs}, tables)
    resolved: list[_ResolvedSteering] = []
    for spec in specs:
        bound = _bind_declaration(
            intervention_id=spec.intervention_id,
            target=spec.target,
            metric_id=spec.metric_id,
            hypothesis_id=spec.hypothesis_id,
            control_ids={control.control_id for control in spec.controls},
            tables=tables,
        )
        _require(
            spec.target in tables.localize_selections,
            f"steering trial {spec.intervention_id!r} target {spec.target!r} is not a "
            "prior localize selection",
            error=StageContractError,
        )
        entry = tables.request_index[spec.intervention_id]
        resolved.append(
            _ResolvedSteering(
                spec=spec,
                request_entry=entry,
                metric_direction=bound.metric_direction,
                threshold=bound.threshold,
                zero_tolerance=bound.zero_tolerance,
            )
        )
    return context, tuple(resolved)


# ---------------------------------------------------------------------------
# Trial execution: measurements, controls, uncertainty, bounded conclusion
# ---------------------------------------------------------------------------


def _checked_measure(measure: MeasureFn, spec: Any, application: TrialApplication) -> Measurement:
    result = measure(application)
    if not isinstance(result, Measurement):
        raise InterventionError(
            f"measure callback for {spec.intervention_id!r} must return a Measurement, "
            f"got {type(result).__name__}"
        )
    if result.metric_id != spec.metric_id:
        raise InterventionError(
            f"measurement domain mismatch for {spec.intervention_id!r}: got metric "
            f"{result.metric_id!r}, declared {spec.metric_id!r}"
        )
    if result.inputs_digest != application.inputs_digest:
        raise InterventionError(
            f"measurement inputs mismatch for {spec.intervention_id!r}: got capture digest "
            f"{result.inputs_digest!r}, bound {application.inputs_digest!r}"
        )
    return result


def _application(
    spec: Any,
    context: _StageContext,
    *,
    role: Literal["baseline", "intervened", "control"],
    strength: float,
    target: str,
    control_class: str | None,
    control_id: str | None,
    stream_seed: int,
    rng: np.random.Generator | None,
) -> TrialApplication:
    return TrialApplication(
        intervention_id=spec.intervention_id,
        kind=spec.kind,
        target=target,
        strength=strength,
        strength_semantics=spec.strength_semantics,
        role=role,
        control_class=control_class,
        control_id=control_id,
        metric_id=spec.metric_id,
        inputs_digest=context.capture_identity,
        stream_seed=stream_seed,
        rng=rng,
    )


def _measurement_record(measurement: Measurement, application: TrialApplication) -> dict[str, object]:
    return {
        "inputs_digest": measurement.inputs_digest,
        "metric_id": measurement.metric_id,
        "role": application.role,
        "stream_seed": int(application.stream_seed),
        "value": float(measurement.value),
    }


def _run_central_controls(
    controls: Sequence[TrialControl],
    *,
    plan_id: str,
    metric_id: str,
    thresholds: Mapping[str, float],
    evaluation_seed: int,
    control_seed: int,
    repetitions: int,
    confidence_level: float,
    statistics: Mapping[str, Callable[[np.random.Generator], Mapping[str, float]]],
) -> tuple[dict[str, str], dict[str, dict[str, object]]]:
    """Run every declared control through the central executor, verified."""
    specs = tuple(
        _ControlSpec(
            control_id=control.control_id,
            kind=_CONTROL_CLASS_KINDS[control.control_class],
            required=True,
            metric_ids=(metric_id,),
            expected_behavior=control.expected_behavior,
            comparator="below_threshold",
            threshold_value=float(thresholds[control.control_id]),
        )
        for control in controls
    )
    plan = _ControlPlan(
        plan_id=plan_id,
        repetitions=repetitions,
        confidence_level=confidence_level,
        evaluation_seed=evaluation_seed,
        control_seed=control_seed,
        controls=specs,
    )
    outcomes = execute_plan(plan, {}, statistics=statistics)
    status: dict[str, str] = {}
    records: dict[str, dict[str, object]] = {}
    for control in sorted(controls, key=lambda item: item.control_id):
        control_id = control.control_id
        outcome = outcomes[control_id]
        expected_seed = derive_stream_seed(control_seed, control_id, role="control")
        if expected_seed != outcome.stream_seed:
            raise InterventionError(
                f"control {control_id!r} stream seed mismatch: executor used "
                f"{outcome.stream_seed}, bound {expected_seed}"
            )
        status[control_id] = outcome.status
        records[control_id] = outcome.to_dict()
    return status, records


def _decide_conclusion(
    spec: TrialSpec,
    effect: float,
    zero_tolerance: float,
    control_status: Mapping[str, str],
    interval_lower: float,
    interval_upper: float,
    baseline_value: float,
) -> tuple[str, str]:
    """Apply the predeclared80.18 decision table; never promote past a failed control."""
    zero_id = next(c.control_id for c in spec.controls if c.control_class == "zero_strength")
    if control_status[zero_id] == "failed":
        return (
            "unsupported",
            f"identity/zero-strength control {zero_id} failed; the measurement pipeline is "
            "untrustworthy and causal support is blocked",
        )
    specificity = [
        control.control_id
        for control in spec.controls
        if control.control_class in ("random", "shuffled", "off_target")
        and control_status[control.control_id] == "failed"
    ]
    if specificity:
        return (
            "falsified",
            f"control(s) {', '.join(sorted(specificity))} reproduced the intervention effect; "
            "target specificity is falsified",
        )
    if abs(effect) <= zero_tolerance:
        return (
            "falsified",
            f"declared {spec.kind} intervention is inert beyond noise: |effect| {abs(effect)} "
            f"does not exceed the predeclared tolerance {zero_tolerance}",
        )
    expected_sign = 1.0 if spec.expected_effect == "increase" else -1.0
    if (effect > 0.0) != (expected_sign > 0.0):
        return (
            "falsified",
            f"intervention moved {spec.metric_id} by {effect} opposite the declared "
            f"{spec.expected_effect} direction",
        )
    if interval_lower <= baseline_value <= interval_upper:
        return (
            "inconclusive",
            f"intervened uncertainty interval [{interval_lower}, {interval_upper}] spans the "
            f"baseline {baseline_value}; the effect is not resolved",
        )
    return (
        "supported",
        f"material {spec.expected_effect} effect {effect} beyond tolerance {zero_tolerance} with "
        "identity, random, shuffled, and off-target controls passing and uncertainty excluding "
        "the baseline",
    )


def _execute_trial(
    resolved: _ResolvedTrial, measure: MeasureFn, context: _StageContext
) -> dict[str, object]:
    spec = resolved.spec

    baseline_seed = derive_stream_seed(
        context.evaluation_seed, f"{spec.intervention_id}#baseline", role="evaluation"
    )
    baseline_application = _application(
        spec,
        context,
        role="baseline",
        strength=0.0,
        target=spec.target,
        control_class=None,
        control_id=None,
        stream_seed=baseline_seed,
        rng=None,
    )
    baseline = _checked_measure(measure, spec, baseline_application)

    intervened_seed = derive_stream_seed(
        context.evaluation_seed, f"{spec.intervention_id}#intervened", role="evaluation"
    )
    intervened_application = _application(
        spec,
        context,
        role="intervened",
        strength=spec.strength,
        target=spec.target,
        control_class=None,
        control_id=None,
        stream_seed=intervened_seed,
        rng=None,
    )
    intervened = _checked_measure(measure, spec, intervened_application)
    effect = float(intervened.value - baseline.value)

    # Predeclared control rule: the zero-strength control must reproduce the
    # baseline within the manifest threshold tolerance; random, shuffled, and
    # off-target controls must stay strictly below max(|effect|, tolerance).
    specificity_threshold = max(abs(effect), resolved.zero_tolerance)
    control_thresholds: dict[str, float] = {}
    control_seeds: dict[str, int] = {}
    statistics: dict[str, Callable[[np.random.Generator], Mapping[str, float]]] = {}
    measurement_errors: dict[str, Exception] = {}
    control_measurements: dict[str, dict[str, object]] = {}

    for control in spec.controls:
        threshold = (
            resolved.zero_tolerance
            if control.control_class == "zero_strength"
            else specificity_threshold
        )
        control_thresholds[control.control_id] = float(threshold)
        stream_seed = derive_stream_seed(context.control_seed, control.control_id, role="control")
        control_seeds[control.control_id] = stream_seed

        def _statistic(
            rng: np.random.Generator,
            *,
            control_id: str = control.control_id,
            control_class: str = control.control_class,
            control_target: str = control.target,
            bound_seed: int = stream_seed,
        ) -> Mapping[str, float]:
            try:
                application = _application(
                    spec,
                    context,
                    role="control",
                    strength=0.0 if control_class == "zero_strength" else spec.strength,
                    target=control_target or spec.target,
                    control_class=control_class,
                    control_id=control_id,
                    stream_seed=bound_seed,
                    rng=rng,
                )
                measurement = _checked_measure(measure, spec, application)
            except Exception as exc:  # noqa: BLE001 - classified before any conclusion is drawn
                measurement_errors[control_id] = exc
                return {spec.metric_id: 0.0}
            record = _measurement_record(measurement, application)
            record["control_class"] = control_class
            record["strength"] = float(application.strength)
            record["target"] = application.target
            control_measurements[control_id] = record
            return {spec.metric_id: abs(float(measurement.value) - float(baseline.value))}

        statistics[control.control_id] = _statistic

    control_status, control_outcomes = _run_central_controls(
        spec.controls,
        plan_id=f"intervene:{spec.intervention_id}",
        metric_id=spec.metric_id,
        thresholds=control_thresholds,
        evaluation_seed=context.evaluation_seed,
        control_seed=context.control_seed,
        repetitions=context.repetitions,
        confidence_level=context.confidence_level,
        statistics=statistics,
    )
    if measurement_errors:
        control_id, exc = sorted(measurement_errors.items())[0]
        raise InterventionError(
            f"control {control_id!r} measurement failed for {spec.intervention_id!r}: {exc}"
        ) from exc

    uncertainty_id = f"intervene:{spec.intervention_id}:intervened"
    uncertainty_seed = derive_stream_seed(
        context.evaluation_seed, uncertainty_id, role="evaluation"
    )

    def _uncertainty_draw(rng: np.random.Generator) -> float:
        application = _application(
            spec,
            context,
            role="intervened",
            strength=spec.strength,
            target=spec.target,
            control_class=None,
            control_id=None,
            stream_seed=uncertainty_seed,
            rng=rng,
        )
        return float(_checked_measure(measure, spec, application).value)

    uncertainty_outcome, interval = run_bootstrap(
        control_id=uncertainty_id,
        base_seed=context.evaluation_seed,
        seed_role="evaluation",
        repetitions=context.repetitions,
        confidence_level=context.confidence_level,
        required=False,
        kind="bootstrap",
        metric_ids=(spec.metric_id,),
        expected_behavior="seeded resampling of the intervened downstream metric",
        draw=_uncertainty_draw,
    )
    if uncertainty_outcome.status == "failed" or "lower" not in interval or "upper" not in interval:
        raise InterventionError(
            f"intervened uncertainty failed for {spec.intervention_id!r}: "
            f"{uncertainty_outcome.reason or 'interval unavailable'}"
        )
    interval_lower = _finite(interval.get("lower"), name="uncertainty lower bound")
    interval_upper = _finite(interval.get("upper"), name="uncertainty upper bound")

    conclusion, reason = _decide_conclusion(
        spec,
        effect,
        resolved.zero_tolerance,
        control_status,
        interval_lower,
        interval_upper,
        float(baseline.value),
    )

    provenance: dict[str, object] = dict(spec.provenance)
    provenance.update(
        {
            "capture_identity": context.capture_identity,
            "hypothesis_id": spec.hypothesis_id,
            "localize_family_id": context.localize_family_id,
            "localize_metric_id": context.localize_metric_id,
            "localize_verdict": context.localize_verdict,
            "manifest_id": context.manifest_id,
            "metric_direction": resolved.metric_direction,
            "representation_identity": context.representation_identity,
            "request_id": context.request_id,
            "seeds": {
                "baseline_stream": int(baseline_seed),
                "controls": int(context.control_seed),
                "evaluation": int(context.evaluation_seed),
                "intervened_stream": int(intervened_seed),
                "uncertainty_stream": int(uncertainty_seed),
            },
        }
    )

    return {
        "conclusion": conclusion,
        "control_outcomes": control_outcomes,
        "control_thresholds": dict(control_thresholds),
        "effect": float(effect),
        "expected_effect": spec.expected_effect,
        "hypothesis_id": spec.hypothesis_id,
        "inputs_digest": context.capture_identity,
        "intervention_id": spec.intervention_id,
        "kind": spec.kind,
        "measurements": {
            "baseline": _measurement_record(baseline, baseline_application),
            "controls": dict(control_measurements),
            "intervened": _measurement_record(intervened, intervened_application),
        },
        "metric_direction": resolved.metric_direction,
        "metric_id": spec.metric_id,
        "provenance": provenance,
        "reason": reason,
        "strength": float(spec.strength),
        "strength_semantics": spec.strength_semantics,
        "target": spec.target,
        "threshold": dict(resolved.threshold),
        "uncertainty": dict(interval),
        "zero_tolerance": float(resolved.zero_tolerance),
    }


# ---------------------------------------------------------------------------
#80.19 steering dose-response execution
# ---------------------------------------------------------------------------


def _classify_response(
    strengths: tuple[float, ...],
    values: Mapping[float, float],
    baseline: float,
    zero_tolerance: float,
) -> str:
    """Classify the declared dose series from its consecutive material steps."""
    ordered = sorted(strengths)
    material_change = any(
        abs(values[strength] - baseline) > zero_tolerance
        for strength in ordered
        if strength != 0.0
    )
    if not material_change:
        return "inert"
    signs: set[int] = set()
    for left, right in zip(ordered, ordered[1:], strict=False):
        step = values[right] - values[left]
        if abs(step) > zero_tolerance:
            signs.add(1 if step > 0.0 else -1)
    if len(signs) >= 2:
        return "non_monotonic"
    if len(signs) == 1:
        return "monotonic_increase" if next(iter(signs)) > 0 else "monotonic_decrease"
    endpoint = max(ordered, key=abs)
    endpoint_change = values[endpoint] - baseline
    return "monotonic_increase" if endpoint_change > 0.0 else "monotonic_decrease"


def _decide_steering(
    spec: SteeringTrialSpec,
    classification: str,
    control_status: Mapping[str, str],
    zero_tolerance: float,
    endpoint_effect: float,
    interval_lower: float,
    interval_upper: float,
    baseline_value: float,
) -> tuple[str, str]:
    """Apply the predeclared80.19 decision table; causality never outranks controls."""
    zero_id = next(c.control_id for c in spec.controls if c.control_class == "zero_strength")
    if control_status[zero_id] == "failed":
        return (
            "unsupported",
            f"identity/zero-strength control {zero_id} failed; the measurement pipeline is "
            "untrustworthy and causal support is blocked",
        )
    random_control = next(c for c in spec.controls if c.control_class == "random_direction")
    if control_status[random_control.control_id] == "failed":
        return (
            "falsified",
            f"random-direction control {random_control.control_id} reproduced a material "
            f"{spec.expected_effect} response at the endpoint dose; direction specificity is "
            "falsified",
        )
    off_target = next(c for c in spec.controls if c.control_class == "off_target")
    if control_status[off_target.control_id] == "failed":
        return (
            "falsified",
            f"off-target control {off_target.control_id} moved {spec.metric_id} as much as the "
            "declared on-target dose response; selectivity fails",
        )
    if classification == "unsupported":
        return "unsupported", "the response could not be classified under a failed identity control"
    if classification == "inert":
        return (
            "falsified",
            f"steering dose series is inert beyond tolerance {zero_tolerance}: no declared "
            f"strength moved {spec.metric_id}",
        )
    if classification == "non_monotonic":
        return (
            "falsified",
            f"response is non-monotonic across the declared strengths; the predeclared "
            f"monotonic {spec.expected_effect} dose expectation fails",
        )
    expected_classification = (
        "monotonic_increase" if spec.expected_effect == "increase" else "monotonic_decrease"
    )
    if classification != expected_classification:
        return (
            "falsified",
            f"monotonic {classification} response contradicts the declared "
            f"{spec.expected_effect} dose expectation",
        )
    if interval_lower <= baseline_value <= interval_upper:
        return (
            "inconclusive",
            f"endpoint uncertainty interval [{interval_lower}, {interval_upper}] spans the "
            f"baseline {baseline_value}; the task-metric change is not resolved",
        )
    return (
        "supported",
        f"monotonic {classification} response with endpoint task-metric change "
        f"{endpoint_effect} beyond tolerance {zero_tolerance}; zero-identity, random-direction, "
        "and off-target controls pass, uncertainty excludes the baseline, and the directional "
        "dose response is target-specific",
    )


def _execute_steering_trial(
    resolved: _ResolvedSteering, measure: MeasureFn, context: _StageContext
) -> dict[str, object]:
    spec = resolved.spec
    nonzero = sorted(strength for strength in spec.strengths if strength != 0.0)
    endpoint_strength = max(nonzero, key=abs)

    baseline_seed = derive_stream_seed(
        context.evaluation_seed, f"{spec.intervention_id}#baseline", role="evaluation"
    )
    baseline_application = _application(
        spec,
        context,
        role="baseline",
        strength=0.0,
        target=spec.target,
        control_class=None,
        control_id=None,
        stream_seed=baseline_seed,
        rng=None,
    )
    baseline = _checked_measure(measure, spec, baseline_application)
    values: dict[float, float] = {0.0: float(baseline.value)}
    dose_streams: dict[str, int] = {}
    dose_records: list[dict[str, object]] = []
    endpoint_record: dict[str, object] | None = None

    for strength in sorted(spec.strengths):
        if strength == 0.0:
            continue
        stream_seed = derive_stream_seed(
            context.evaluation_seed, f"{spec.intervention_id}#dose:{strength!r}", role="evaluation"
        )
        dose_streams[repr(strength)] = int(stream_seed)
        application = _application(
            spec,
            context,
            role="intervened",
            strength=strength,
            target=spec.target,
            control_class=None,
            control_id=None,
            stream_seed=stream_seed,
            rng=None,
        )
        measurement = _checked_measure(measure, spec, application)
        uncertainty_id = f"steer:{spec.intervention_id}:dose:{strength!r}"
        uncertainty_seed = derive_stream_seed(
            context.evaluation_seed, uncertainty_id, role="evaluation"
        )

        def _dose_draw(
            rng: np.random.Generator,
            *,
            dose: float = strength,
            bound_seed: int = uncertainty_seed,
        ) -> float:
            draw_application = _application(
                spec,
                context,
                role="intervened",
                strength=dose,
                target=spec.target,
                control_class=None,
                control_id=None,
                stream_seed=bound_seed,
                rng=rng,
            )
            return float(_checked_measure(measure, spec, draw_application).value)

        dose_outcome, interval = run_bootstrap(
            control_id=uncertainty_id,
            base_seed=context.evaluation_seed,
            seed_role="evaluation",
            repetitions=context.repetitions,
            confidence_level=context.confidence_level,
            required=False,
            kind="bootstrap",
            metric_ids=(spec.metric_id,),
            expected_behavior="seeded resampling of the steered downstream metric",
            draw=_dose_draw,
        )
        if dose_outcome.status == "failed" or "lower" not in interval or "upper" not in interval:
            raise InterventionError(
                f"dose uncertainty failed for {spec.intervention_id!r} at strength "
                f"{strength}: {dose_outcome.reason or 'interval unavailable'}"
            )
        values[strength] = float(measurement.value)
        record = {
            "effect": float(measurement.value - baseline.value),
            "measurement": _measurement_record(measurement, application),
            "strength": float(strength),
            "uncertainty": dict(interval),
        }
        dose_records.append(record)
        if strength == endpoint_strength:
            endpoint_record = record

    assert endpoint_record is not None  # endpoint_strength came from the declared doses
    endpoint_effect = values[endpoint_strength] - baseline.value

    # Predeclared steering control rules: zero-strength must reproduce the
    # baseline within tolerance; the random-direction control is evaluated on
    # its declared-direction-aligned effect (an rng-drawn direction that moves
    # the metric differently never reproduces the response); the off-target
    # control is sign-agnostic selectivity.
    specificity_threshold = max(abs(endpoint_effect), resolved.zero_tolerance)
    control_thresholds: dict[str, float] = {}
    statistics: dict[str, Callable[[np.random.Generator], Mapping[str, float]]] = {}
    measurement_errors: dict[str, Exception] = {}
    control_measurements: dict[str, dict[str, object]] = {}
    expected_positive = spec.expected_effect == "increase"

    for control in spec.controls:
        threshold = (
            resolved.zero_tolerance
            if control.control_class == "zero_strength"
            else specificity_threshold
        )
        control_thresholds[control.control_id] = float(threshold)
        control_strength = 0.0 if control.control_class == "zero_strength" else endpoint_strength

        def _statistic(
            rng: np.random.Generator,
            *,
            control_id: str = control.control_id,
            control_class: str = control.control_class,
            control_target: str = control.target,
            applied_strength: float = control_strength,
        ) -> Mapping[str, float]:
            bound_seed = derive_stream_seed(context.control_seed, control_id, role="control")
            try:
                application = _application(
                    spec,
                    context,
                    role="control",
                    strength=applied_strength,
                    target=control_target or spec.target,
                    control_class=control_class,
                    control_id=control_id,
                    stream_seed=bound_seed,
                    rng=rng,
                )
                measurement = _checked_measure(measure, spec, application)
            except Exception as exc:  # noqa: BLE001 - classified before any conclusion is drawn
                measurement_errors[control_id] = exc
                return {spec.metric_id: 0.0}
            record = _measurement_record(measurement, application)
            record["control_class"] = control_class
            record["strength"] = float(application.strength)
            record["target"] = application.target
            control_measurements[control_id] = record
            delta = float(measurement.value) - float(baseline.value)
            if control_class == "random_direction":
                aligned = (
                    abs(delta)
                    if delta != 0.0 and (delta > 0.0) == expected_positive
                    else 0.0
                )
                return {spec.metric_id: float(aligned)}
            return {spec.metric_id: abs(delta)}

        statistics[control.control_id] = _statistic

    control_status, control_outcomes = _run_central_controls(
        spec.controls,
        plan_id=f"steer:{spec.intervention_id}",
        metric_id=spec.metric_id,
        thresholds=control_thresholds,
        evaluation_seed=context.evaluation_seed,
        control_seed=context.control_seed,
        repetitions=context.repetitions,
        confidence_level=context.confidence_level,
        statistics=statistics,
    )
    if measurement_errors:
        control_id, exc = sorted(measurement_errors.items())[0]
        raise InterventionError(
            f"control {control_id!r} measurement failed for {spec.intervention_id!r}: {exc}"
        ) from exc

    classification = _classify_response(spec.strengths, values, float(baseline.value), resolved.zero_tolerance)
    endpoint_uncertainty = dict(endpoint_record["uncertainty"])  # pyright: ignore[reportUnknownMemberType]
    interval_lower = _finite(endpoint_uncertainty.get("lower"), name="endpoint lower bound")
    interval_upper = _finite(endpoint_uncertainty.get("upper"), name="endpoint upper bound")
    conclusion, reason = _decide_steering(
        spec,
        classification,
        control_status,
        resolved.zero_tolerance,
        endpoint_effect,
        interval_lower,
        interval_upper,
        float(baseline.value),
    )
    if conclusion == "unsupported":
        classification = "unsupported"

    provenance: dict[str, object] = dict(spec.provenance)
    provenance.update(
        {
            "capture_identity": context.capture_identity,
            "hypothesis_id": spec.hypothesis_id,
            "localize_family_id": context.localize_family_id,
            "localize_metric_id": context.localize_metric_id,
            "localize_verdict": context.localize_verdict,
            "manifest_id": context.manifest_id,
            "metric_direction": resolved.metric_direction,
            "representation_identity": context.representation_identity,
            "request_id": context.request_id,
            "seeds": {
                "baseline_stream": int(baseline_seed),
                "controls": int(context.control_seed),
                "dose_streams": dict(dose_streams),
                "evaluation": int(context.evaluation_seed),
            },
        }
    )

    return {
        "conclusion": conclusion,
        "control_outcomes": control_outcomes,
        "control_thresholds": dict(control_thresholds),
        "direction": {
            "digest": spec.direction_digest,
            "norm": float(np.linalg.norm(spec.direction)),
            "provenance": dict(spec.provenance),
        },
        "doses": dose_records,
        "effect": float(endpoint_effect),
        "endpoint_strength": float(endpoint_strength),
        "expected_effect": spec.expected_effect,
        "hypothesis_id": spec.hypothesis_id,
        "inputs_digest": context.capture_identity,
        "intervention_id": spec.intervention_id,
        "kind": STEERING_KIND,
        "measurements": {
            "baseline": _measurement_record(baseline, baseline_application),
            "controls": dict(control_measurements),
        },
        "metric_direction": resolved.metric_direction,
        "metric_id": spec.metric_id,
        "provenance": provenance,
        "reason": reason,
        "response_classification": classification,
        "strength_semantics": spec.strength_semantics,
        "strengths": [float(strength) for strength in spec.strengths],
        "task_metric_change": float(endpoint_effect),
        "target": spec.target,
        "threshold": dict(resolved.threshold),
        "uncertainty": endpoint_uncertainty,
        "zero_tolerance": float(resolved.zero_tolerance),
    }


# ---------------------------------------------------------------------------
# Report-shaped output (shape-compatible claim fragments, Phase-IV boundary)
# ---------------------------------------------------------------------------


def intervention_report_items(trials: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    """Render trial records as shape-compatible causal claim fragments.

    Phase-IV boundary (not 80.21/80.22): items pass ``validate_report_shape``
    but are NOT independently consumable by ``validate_diagnostic_report``.
    ``evidence_refs`` are provisional (the ``intervention-<id>-record`` name
    registers no report row or digest artifact yet) and ``control_refs`` name
    the trial-declared intervention controls. Every item carries
    ``kind="causal_result"`` with ``causal=True``; supported and falsified
    conclusions allow the claim, inconclusive and unsupported conclusions
    never do, and every blocked record names its blocking reason explicitly
    so the frozen validator fails closed on any overclaim.
    """
    if isinstance(trials, str | bytes) or not isinstance(trials, Sequence):
        raise InterventionError("trials must be a sequence of trial records")
    items: list[dict[str, object]] = []
    seen: set[str] = set()
    allowed_kinds = (*INTERVENTION_KINDS, STEERING_KIND)
    for position, raw in enumerate(trials):
        record = _mapping(raw, name=f"trials[{position}]")
        intervention_id = _non_empty_string(record.get("intervention_id"), name="trial intervention_id")
        if intervention_id in seen:
            raise InterventionError("intervention identifiers must be unique in report items")
        seen.add(intervention_id)
        kind = _non_empty_string(record.get("kind"), name=f"trial {intervention_id!r} kind")
        if kind not in allowed_kinds:
            raise InterventionError(f"trial {intervention_id!r} kind is unsupported")
        target = _non_empty_string(record.get("target"), name=f"trial {intervention_id!r} target")
        conclusion = _non_empty_string(
            record.get("conclusion"), name=f"trial {intervention_id!r} conclusion"
        )
        if conclusion not in _CONCLUSION_OUTCOMES:
            raise InterventionError(
                f"trial {intervention_id!r} conclusion must be one of {list(_CONCLUSION_OUTCOMES)!r}"
            )
        reason = _non_empty_string(record.get("reason"), name=f"trial {intervention_id!r} reason")
        control_outcomes = _mapping(
            record.get("control_outcomes"), name=f"trial {intervention_id!r} control_outcomes"
        )
        control_refs = sorted(control_outcomes)
        if not control_refs:
            raise InterventionError(f"trial {intervention_id!r} must record control outcomes")
        claim_allowed = conclusion in ("supported", "falsified")
        items.append(
            {
                "causal": True,
                "claim": f"{kind} intervention {intervention_id} on target {target}: {reason}",
                "claim_allowed": claim_allowed,
                "control_refs": control_refs,
                "evidence_refs": [f"intervention-{intervention_id}-record"],
                "id": f"intervention-{intervention_id}",
                "kind": "causal_result",
                "missing_evidence": [] if claim_allowed else [reason],
                "status": conclusion,
            }
        )
    return items


# ---------------------------------------------------------------------------
# Stage executor factories (the single private execution seam per family)
# ---------------------------------------------------------------------------


def _intervene_output(records: Sequence[Mapping[str, object]], context: _StageContext) -> StageOutput:
    conclusions = {
        str(record["intervention_id"]): str(record["conclusion"]) for record in records
    }
    payload: dict[str, object] = {
        "config": {
            "causal_expectation": dict(context.causal_expectation),
            "config_digest": context.config_digest,
            "confidence_level": float(context.confidence_level),
            "control_seed": int(context.control_seed),
            "evaluation_seed": int(context.evaluation_seed),
            "manifest_digest": context.manifest_digest,
            "manifest_id": context.manifest_id,
            "repetitions": int(context.repetitions),
            "request_digest": context.request_digest,
            "request_id": context.request_id,
            "workflow_identity": context.workflow_identity,
        },
        "conclusions": dict(conclusions),
        "intervention_version": INTERVENTION_VERSION,
        "trials": list(records),
    }
    aggregate = (
        "unsupported"
        if conclusions and all(value == "unsupported" for value in conclusions.values())
        else "completed"
    )
    return StageOutput(stage="intervene", outcome=aggregate, payload=payload, artifact_refs=())


def _validate_factory_inputs(trials: object, measure: object, version: object) -> tuple[Any, ...]:
    _non_empty_string(version, name="version")
    if isinstance(trials, str | bytes) or not isinstance(trials, Sequence):
        raise InterventionError("trials must be a sequence of trial spec items")
    frozen = tuple(trials)  # pyright: ignore[reportUnknownVariableType]
    if not frozen:
        raise InterventionError("trials must declare at least one trial")
    if not callable(measure):
        raise InterventionError("measure must be callable")
    return frozen


def make_intervene_executor(
    trials: Sequence[TrialSpec],
    measure: MeasureFn,
    *,
    version: str = INTERVENTION_VERSION,
) -> Any:
    """Build a supplied ``intervene``-stage executor for declared80.18 trials.

    The coordinator invokes the returned callable with a ``StageInvocation``;
    the callable first verifies the invocation targets ``intervene`` with the
    exact prior capture/localize/explain identities, request-declared
    intervention and control declarations, manifest causal target metrics,
    and provenance bindings — raising ``StageContractError`` before any
    measurement callback runs — then measures baseline, intervened, and the
    four required controls over the same captured inputs and downstream
    metric, records seeds/provenance/uncertainty/control outcomes, and
    returns a ``completed`` (or honest ``unsupported``) ``StageOutput`` whose
    payload carries the bounded per-trial conclusions. Method-specific
    application logic lives only in the caller's ``measure`` callback; no
    algorithm enters ``DiagnosticWorkflow`` itself.
    """
    frozen = _validate_factory_inputs(trials, measure, version)
    for position, spec in enumerate(frozen):
        if not isinstance(spec, TrialSpec):
            raise InterventionError(f"trials[{position}] must be a TrialSpec")
    identities = [spec.intervention_id for spec in frozen]
    if len(set(identities)) != len(identities):
        raise InterventionError("trial intervention identifiers must be unique")
    resolved_specs: tuple[TrialSpec, ...] = tuple(frozen)

    def _execute(invocation: StageInvocation) -> StageOutput:
        if invocation.stage != "intervene":
            raise StageContractError(f"intervene executor received stage {invocation.stage!r}")
        context = _bind_context(resolved_specs, invocation)
        records = [_execute_trial(resolved, measure, context) for resolved in context.trials]
        return _intervene_output(records, context)

    _execute.intervention_version = version  # type: ignore[attr-defined]
    return _execute


def make_steering_intervene_executor(
    trials: Sequence[SteeringTrialSpec],
    measure: MeasureFn,
    *,
    version: str = INTERVENTION_VERSION,
) -> Any:
    """Build a supplied ``intervene``-stage executor for steering dose trials.

    The coordinator invokes the returned callable with a ``StageInvocation``;
    the callable first verifies the invocation targets ``intervene`` with the
    exact prior capture/localize/explain/request/manifest identities, the
    steering target as a prior localize selection, the immutable direction
    provenance and digest, the predeclared dose plan, and the required
    zero-identity/random-direction/off-target control declarations — raising
    ``StageContractError`` before any measurement callback runs — then
    measures the zero/identity dose, every nonzero dose with a per-dose
    bootstrap interval, and the three required controls over the same
    captured inputs and downstream metric, classifies the response
    (monotonic/non-monotonic/inert/unsupported), and returns a
    ``completed`` (or honest ``unsupported``) ``StageOutput`` whose payload
    carries direction provenance, every strength and measurement, task-metric
    change, random-direction and off-target outcomes, and the bounded
    conclusion. Dose application logic lives only in the caller's ``measure``
    callback; no algorithm enters ``DiagnosticWorkflow`` itself.
    """
    frozen = _validate_factory_inputs(trials, measure, version)
    for position, spec in enumerate(frozen):
        if not isinstance(spec, SteeringTrialSpec):
            raise InterventionError(f"trials[{position}] must be a SteeringTrialSpec")
    identities = [spec.intervention_id for spec in frozen]
    if len(set(identities)) != len(identities):
        raise InterventionError("trial intervention identifiers must be unique")
    resolved_specs: tuple[SteeringTrialSpec, ...] = tuple(frozen)

    def _execute(invocation: StageInvocation) -> StageOutput:
        if invocation.stage != "intervene":
            raise StageContractError(f"intervene executor received stage {invocation.stage!r}")
        context, resolved = _bind_steering(resolved_specs, invocation)
        records = [_execute_steering_trial(item, measure, context) for item in resolved]
        return _intervene_output(records, context)

    _execute.intervention_version = version  # type: ignore[attr-defined]
    return _execute


__all__ = [
    "INTERVENTION_CONTROL_CLASSES",
    "INTERVENTION_KINDS",
    "INTERVENTION_VERSION",
    "STEERING_CONTROL_CLASSES",
    "STEERING_KIND",
    "InterventionError",
    "MeasureFn",
    "Measurement",
    "SteeringTrialSpec",
    "TrialApplication",
    "TrialControl",
    "TrialSpec",
    "intervention_report_items",
    "make_intervene_executor",
    "make_steering_intervene_executor",
]

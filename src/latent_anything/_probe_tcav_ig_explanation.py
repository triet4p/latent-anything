"""Probe, TCAV, and Integrated-Gradients explanation evidence (Sprint 80.16).

Private explain-stage adapter behind the frozen taxonomy/workflow/report
contracts. It binds one predeclared immutable hypothesis per method run,
executes only declared methods through caller-supplied callbacks, and emits
one canonical non-causal evidence record per executed method with five gated
dimensions: fidelity, stability, selectivity, leakage, and uncertainty.

Reused primitives (no new estimators):

- a local bounded probe fit (training-only scaler + logreg): training-only
  ``StandardScaler`` plus ``LogisticRegression`` (C=1.0, lbfgs, balanced).
- ``tcav.learn_mean_diff_direction`` / ``learn_linear_separator_direction``
  own CAV fitting; ``_tcav_statistics.assemble_tcav_result`` is not reused
  for scoring because its internal bootstrap/random-concept loops own local
  RNG streams. This adapter replays the same statistics (directional-derivative
  fractions, bootstrap CAV stability, permutation random-concept baselines)
  with caller-fitted gradients under central executor-owned streams.
- ``integrated_gradients.IntegratedGradients`` owns the activation-space
  path integral; completeness error is read from its typed result.
- ``_probe_split.stratified_split`` is reused only inside the existing CAV
  learners; hypothesis splits are caller-declared and validated disjoint.
- ``_statistical_controls`` (``ControlPlan``/``execute_plan``/
  ``run_bootstrap``/``failed_required``) owns every repetition count, RNG
  stream, percentile interval, and required/optional control verdict.

Single-evaluation seam: :func:`evaluate_explanations` fits every probe,
CAV, and gradient exactly once per executor call inside one
:class:`_ExplainContext`, then shares that context between decisions and
payload assembly. Uncertainty resamples already-fitted predictions,
scores, and attributions without refitting. The ``explain`` stage executor
(:func:`make_explain_executor`) carries no algorithm; it verifies identity
and returns the shared payload.

Fail-closed: undeclared hypotheses or methods never invoke method code; the
callback stays uncalled and the record reads ``omitted`` (undeclared) or
``unsupported`` (declared but inapplicable, e.g. disconnected target).
Missing or failed fidelity, stability, selectivity, leakage, uncertainty,
or any required control blocks promotion (``inconclusive`` or
``unsupported``, ``claim_allowed=False``). Probe train accuracy is never
promoted. Records are ``kind="explanation"`` with ``causal=False`` always.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, SupportsFloat, cast

import numpy as np

from latent_anything._diagnostic_workflow import StageInvocation, StageOutput
from latent_anything._portable_contract import PortableNodeError, canonical_json
from latent_anything._statistical_controls import ControlPlan as _ControlPlan
from latent_anything._statistical_controls import ControlSpec as _ControlSpec
from latent_anything._statistical_controls import execute_plan as _execute_plan
from latent_anything._statistical_controls import failed_required as _failed_required
from latent_anything._statistical_controls import run_bootstrap as _central_bootstrap
from latent_anything._statistical_controls import (
    run_permutation_control as _central_control,
)
from latent_anything.probes import LinearProbeConfig, LinearProbeResult

EXPLAINER_VERSION = "probe-tcav-ig-explainer-v1"
"""Version string bound into explain-stage payloads."""

DECLARED_PROBE_CAPACITY = "linear-logreg-C1.0-standardized"
"""The only probe capacity this adapter may promote."""

SUPPORTED_METHODS: tuple[str, ...] = ("probe", "tcav", "integrated_gradients")
"""Explanation methods this adapter may execute. Anything else is rejected."""

EXTENDED_METHODS: tuple[str, ...] = ("sae_sparse", "lens", "geometry", "density", "clustering")
"""Feature explanation methods owned by the 80.17 adapter.

Listed here so the shared hypothesis/evidence/input carriers stay single-schema:
both :func:`evaluate_explanations` and :func:`make_explain_executor` reject
these methods with :class:`ExplanationError` before any callback fires (they
belong to the feature explainer). ``SUPPORTED_METHODS`` itself is frozen.
"""

_ALL_METHODS: tuple[str, ...] = SUPPORTED_METHODS + EXTENDED_METHODS
"""Every method name the shared carriers accept. Private; never re-exported."""

EvidenceOutcome = Literal["supported", "inconclusive", "unsupported", "omitted"]
"""Honest per-method outcomes. Omitted methods never executed any code."""


class ExplanationError(ValueError):
    """Raised when explanation input, hypothesis, or evidence is fail-closed invalid."""


def _require_generator(value: object) -> np.random.Generator:
    if not isinstance(value, np.random.Generator):
        raise ExplanationError("control stream must be a numpy Generator")
    return value


def _require_float(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (str, bytes, bytearray, SupportsFloat)):
        raise ExplanationError(f"{name} must be a finite number")
    try:
        level = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ExplanationError(f"{name} must be a finite number") from exc
    if not np.isfinite(level):
        raise ExplanationError(f"{name} must be a finite number")
    return level


def _require_count(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ExplanationError(f"{name} must be an integer")
    return int(value)


def _require_confidence_level(value: object) -> float:
    level = _require_float(value, name="confidence_level")
    if not 0.0 < level < 1.0:
        raise ExplanationError("confidence_level must be between zero and one")
    return level


def _require_observed_floats(value: object, *, name: str) -> Mapping[str, float]:
    if not isinstance(value, Mapping):
        raise ExplanationError(f"{name} must be a mapping")
    narrowed: dict[str, float] = {}
    for key, entry in value.items():
        if not isinstance(key, str):
            raise ExplanationError(f"{name} must map strings to numbers")
        narrowed[key] = _require_float(entry, name=f"{name}[{key!r}]")
    return narrowed


def _require_central_outcomes(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ExplanationError("context central_outcomes must be a mapping")
    return dict(value)


def _require_control_outcomes(value: object) -> dict[str, dict[str, str]]:
    if not isinstance(value, Mapping):
        raise ExplanationError("context control_outcomes must be a mapping")
    normalized: dict[str, dict[str, str]] = {}
    for key, entry in value.items():
        if not isinstance(key, str) or not isinstance(entry, Mapping):
            raise ExplanationError("context control_outcomes must map strings to string mappings")
        normalized[key] = {str(item): str(val) for item, val in entry.items()}
    return normalized


def _optional_str_items(value: object) -> tuple[str, ...]:
    """Return the non-empty string entries of an optional payload list."""
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        return ()
    return tuple(item for item in value if isinstance(item, str) and item)


def _non_empty_string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ExplanationError(f"{name} must be a non-empty string")
    return value


def _require_hypotheses(value: object) -> tuple[ExplanationHypothesis, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ExplanationError("hypotheses must declare at least one hypothesis")
    checked: list[ExplanationHypothesis] = []
    for item in value:
        if not isinstance(item, ExplanationHypothesis):
            raise ExplanationError("hypotheses must hold ExplanationHypothesis items")
        checked.append(item)
    if not checked:
        raise ExplanationError("hypotheses must declare at least one hypothesis")
    return tuple(checked)


def _require_inputs(value: object) -> MethodInputs:
    if not isinstance(value, MethodInputs):
        raise ExplanationError("inputs must be a MethodInputs")
    return value


def _require_evidence_items(value: object) -> tuple[ExplanationEvidence, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ExplanationError("evidence must hold ExplanationEvidence items")
    checked: list[ExplanationEvidence] = []
    for item in value:
        if not isinstance(item, ExplanationEvidence):
            raise ExplanationError("evidence must hold ExplanationEvidence items")
        checked.append(item)
    return tuple(checked)


def _require_payload_evidence(value: object) -> tuple[Mapping[str, object], ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ExplanationError("explain payload evidence must be a list")
    checked: list[Mapping[str, object]] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise ExplanationError("explain payload evidence must hold objects")
        checked.append(item)
    return tuple(checked)


def _raw(record: object, field: str) -> object:
    return object.__getattribute__(record, field)


def _finite_number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(float(value)):
        raise ExplanationError(f"{name} must be a finite number")
    return float(value)


def _non_negative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ExplanationError(f"{name} must be a non-negative integer")
    return value


def _string_tuple(value: object, *, name: str, minimum: int = 1) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ExplanationError(f"{name} must be a list of strings")
    items = tuple(value)
    if len(items) < minimum:
        raise ExplanationError(f"{name} must contain at least {minimum} item(s)")
    for position, item in enumerate(items):
        if not isinstance(item, str) or not item.strip():
            raise ExplanationError(f"{name}[{position}] must be a non-empty string")
    if len(set(items)) != len(items):
        raise ExplanationError(f"{name} must not contain duplicates")
    return items


METHOD_CONTROL_ROLES: Mapping[str, tuple[str, ...]] = {
    "probe": ("capacity", "randomized", "negative"),
    "tcav": ("fidelity", "stability", "selectivity"),
    "integrated_gradients": ("fidelity", "stability", "selectivity"),
    "sae_sparse": ("fidelity", "stability", "selectivity"),
    "lens": ("fidelity", "stability", "selectivity"),
    "geometry": ("fidelity", "stability", "selectivity"),
    "density": ("fidelity", "stability", "selectivity"),
    "clustering": ("fidelity", "stability", "selectivity"),
}
"""One private method-specific role binding for declared control IDs.

Every hypothesis must declare exactly ``len(roles)`` control IDs in method
order, each ``"<role>:<name>"`` prefixed (e.g. ``"capacity:control-capacity"``).
The role prefix selects which executed control the declared ID binds to; the
suffix is the exact central ``ControlSpec``/callback identity and the exact
report ``control_refs`` entry. No adapter-invented identity may execute or
report unless the declaration bound it.
"""


def _declared_control_ids(hypothesis: ExplanationHypothesis) -> tuple[str, ...]:
    declared = _string_tuple(tuple(hypothesis.control_ids), name="control_ids", minimum=1)
    return declared


def _bound_control_ids(hypothesis: ExplanationHypothesis) -> dict[str, str]:
    """Validate ``control_ids`` against the method roles; return role->ID."""
    roles = METHOD_CONTROL_ROLES.get(hypothesis.method)
    if roles is None:
        raise ExplanationError(f"method {hypothesis.method!r} has no control-role binding")
    declared = _declared_control_ids(hypothesis)
    if len(declared) != len(roles):
        raise ExplanationError(
            f"hypothesis {hypothesis.hypothesis_id!r} must declare exactly "
            f"{len(roles)} role-bound controls ({', '.join(roles)}), got {len(declared)}"
        )
    bound: dict[str, str] = {}
    for role, declaration in zip(roles, declared, strict=True):
        if ":" not in declaration:
            raise ExplanationError(
                f"hypothesis {hypothesis.hypothesis_id!r} control {declaration!r} "
                f"must be '<role>:<id>' with role {role!r}"
            )
        prefix, _, identity = declaration.partition(":")
        if prefix != role or not identity.strip():
            raise ExplanationError(
                f"hypothesis {hypothesis.hypothesis_id!r} control {declaration!r} "
                f"must carry role prefix {role!r} and a non-empty identity"
            )
        if identity in bound.values():
            raise ExplanationError(
                f"hypothesis {hypothesis.hypothesis_id!r} declares duplicate control identity {identity!r}"
            )
        bound[role] = identity
    return bound


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    flat_a = np.asarray(a, dtype=np.float64).ravel()
    flat_b = np.asarray(b, dtype=np.float64).ravel()
    denominator = float(np.linalg.norm(flat_a) * np.linalg.norm(flat_b))
    if denominator == 0.0:
        return 0.0
    return float(np.dot(flat_a, flat_b) / denominator)


def _sign_aware_cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Return the sign/permutation-aware cosine between two directions."""
    return abs(_cosine(a, b))


# ---------------------------------------------------------------------------
# Hypothesis declaration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExplanationHypothesis:
    """One immutable predeclared hypothesis binding a symptom to one method run.

    Every field is caller-declared before any method code executes. The
    ``hypothesis_id`` links one diagnosed symptom/family to one target
    output, representation slice, method, expected direction, dataset/split
    identities, seeds, controls, thresholds, and baseline/concept
    definitions as applicable. ``localization_bindings`` is the explicit
    declared localization contract: a non-empty tuple of ``(axis, identity)``
    records with axes drawn from ``layer``/``slice``/``checkpoint``/``token``/
    ``time``/``feature``. Only declared bindings are upstream-localized;
    ``layer_id``/``slice_id`` remain method context and prove nothing by their
    mere non-emptiness. Optional fields use ``""``/``()`` when the method does
    not need them; required per-method fields are enforced by
    :func:`evaluate_explanations`, not here, so declaration stays uniform.
    """

    hypothesis_id: str
    symptom_id: str
    family_id: str
    target_id: str
    representation_id: str
    layer_id: str
    slice_id: str
    method: str
    expected_direction: str
    dataset_id: str
    train_split_identity: str
    eval_split_identity: str
    seeds: tuple[int, ...]
    control_ids: tuple[str, ...]
    metric_ids: tuple[str, ...]
    thresholds: tuple[tuple[str, str, float], ...]
    manifest_id: str = ""
    baseline_policy: str = ""
    concept_id: str = ""
    negative_concept_ids: tuple[str, ...] = ()
    localization_bindings: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        _non_empty_string(self.hypothesis_id, name="hypothesis_id")
        _non_empty_string(self.symptom_id, name="symptom_id")
        _non_empty_string(self.family_id, name="family_id")
        _non_empty_string(self.target_id, name="target_id")
        _non_empty_string(self.representation_id, name="representation_id")
        if self.method not in _ALL_METHODS:
            raise ExplanationError(f"unsupported explanation method: {self.method!r}")
        _non_empty_string(self.layer_id, name="layer_id")
        _non_empty_string(self.slice_id, name="slice_id")
        _non_empty_string(self.expected_direction, name="expected_direction")
        _non_empty_string(self.dataset_id, name="dataset_id")
        _non_empty_string(self.train_split_identity, name="train_split_identity")
        _non_empty_string(self.eval_split_identity, name="eval_split_identity")
        if self.train_split_identity == self.eval_split_identity:
            raise ExplanationError("train/eval split identities must be distinct")
        if not self.seeds:
            raise ExplanationError("seeds must declare at least one seed")
        for position, seed in enumerate(self.seeds):
            _non_negative_int(seed, name=f"seeds[{position}]")
        object.__setattr__(self, "seeds", tuple(int(item) for item in self.seeds))
        object.__setattr__(self, "control_ids", _string_tuple(tuple(self.control_ids), name="control_ids", minimum=1))
        raw_thresholds: object = _raw(self, "thresholds")
        if isinstance(raw_thresholds, (str, bytes)) or not isinstance(raw_thresholds, Sequence):
            raise ExplanationError("thresholds must declare at least one (metric, comparator, value) rule")
        items = tuple(raw_thresholds)
        if not items:
            raise ExplanationError("thresholds must declare at least one (metric, comparator, value) rule")
        seen: set[str] = set()
        normalized_thresholds: list[tuple[str, str, float]] = []
        for position, item in enumerate(items):
            if not isinstance(item, (tuple, list)) or len(item) != 3:
                raise ExplanationError(f"thresholds[{position}] must be a (metric_id, comparator, value) triple")
            parts = tuple(item)
            metric_id, comparator, target = parts[0], parts[1], parts[2]
            _non_empty_string(metric_id, name=f"thresholds[{position}].metric_id")
            if comparator not in (">=", "<="):
                raise ExplanationError(f"thresholds[{position}].comparator must be '>=' or '<='")
            _finite_number(target, name=f"thresholds[{position}].value")
            key = str(metric_id)
            if key in seen:
                raise ExplanationError(f"thresholds declare duplicate metric: {metric_id!r}")
            seen.add(key)
            normalized_thresholds.append((str(metric_id), str(comparator), float(target)))
        object.__setattr__(self, "thresholds", tuple(normalized_thresholds))
        raw_baseline: object = _raw(self, "baseline_policy")
        if not isinstance(raw_baseline, str):
            raise ExplanationError("baseline_policy must be a string")
        _non_empty_string(self.concept_id, name="concept_id") if self.method == "tcav" else None
        if self.method == "tcav" and not self.concept_id.strip():
            raise ExplanationError("tcav hypotheses must declare a positive concept_id")
        if self.method == "integrated_gradients" and not self.baseline_policy.strip():
            raise ExplanationError("integrated_gradients hypotheses must declare a baseline_policy")
        object.__setattr__(self, "negative_concept_ids", tuple(str(item) for item in self.negative_concept_ids))
        raw_bindings: object = _raw(self, "localization_bindings")
        if isinstance(raw_bindings, (str, bytes)) or not isinstance(raw_bindings, Sequence):
            raise ExplanationError("localization_bindings must be a list of (axis, identity) pairs")
        bindings = tuple(raw_bindings)
        allowed_axes = ("layer", "slice", "checkpoint", "token", "time", "feature")
        seen_axes: set[str] = set()
        seen_pairs: set[tuple[str, str]] = set()
        normalized: list[tuple[str, str]] = []
        for position, entry in enumerate(bindings):
            if not isinstance(entry, (tuple, list)) or len(entry) != 2:
                raise ExplanationError(f"localization_bindings[{position}] must be an (axis, identity) pair")
            axis, identity = entry[0], entry[1]
            if axis not in allowed_axes:
                raise ExplanationError(
                    f"localization_bindings[{position}].axis must be one of {list(allowed_axes)}, got {axis!r}"
                )
            if not isinstance(identity, str) or not identity.strip():
                raise ExplanationError(f"localization_bindings[{position}].identity must be a non-empty string")
            if axis in seen_axes:
                raise ExplanationError(f"localization_bindings declares duplicate axis {axis!r}")
            if (str(axis), str(identity)) in seen_pairs:
                raise ExplanationError("localization_bindings must not contain duplicate pairs")
            seen_axes.add(str(axis))
            seen_pairs.add((str(axis), str(identity)))
            normalized.append((str(axis), str(identity)))
        object.__setattr__(self, "localization_bindings", tuple(normalized))

    def threshold_for(self, metric_id: str) -> tuple[str, float] | None:
        """Return the ``(comparator, value)`` pair for one metric, if declared."""
        for declared_metric, comparator, target in self.thresholds:
            if declared_metric == metric_id:
                return comparator, float(target)
        return None

    def to_dict(self) -> dict[str, object]:
        return {
            "baseline_policy": self.baseline_policy,
            "concept_id": self.concept_id,
            "control_ids": list(self.control_ids),
            "dataset_id": self.dataset_id,
            "eval_split_identity": self.eval_split_identity,
            "expected_direction": self.expected_direction,
            "family_id": self.family_id,
            "hypothesis_id": self.hypothesis_id,
            "layer_id": self.layer_id,
            "manifest_id": self.manifest_id,
            "metric_ids": list(self.metric_ids),
            "method": self.method,
            "negative_concept_ids": list(self.negative_concept_ids),
            "representation_id": self.representation_id,
            "thresholds": [
                {"comparator": comparator, "metric_id": metric_id, "value": float(target)}
                for metric_id, comparator, target in self.thresholds
            ],
            "target_id": self.target_id,
            "train_split_identity": self.train_split_identity,
            "localization_bindings": [
                {"axis": axis, "identity": identity} for axis, identity in self.localization_bindings
            ],
        }


_KNOWN_FAMILY_IDS: tuple[str, ...] = (
    "collapse_rank_loss",
    "anisotropy_inactive_dimensions",
    "redundancy_superposition",
    "separability_probe_leakage",
    "density_ood_distribution_drift",
    "sparse_feature_instability",
    "sequence_trajectory_drift",
)
"""Frozen taxonomy family identities. The explain gate links a hypothesis to a
detect family only through these structured identities, never through status
words or other payload strings."""


def _string_entries(value: object) -> tuple[str, ...]:
    """Return the non-empty string entries of a string or string list."""
    if isinstance(value, str):
        return (value,) if value else ()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(item for item in value if isinstance(item, str) and item)
    return ()


def _detect_family_ids(detect_payload: Mapping[str, object]) -> set[str]:
    """Extract structured detect family identities from a detect payload.

    Family IDs come only from explicit ``family_id``/``family_ids`` fields,
    explicit string entries of a known family-ID list, or identity-keyed
    ``family_evidence`` keys. Status words, control identifiers, measurement
    labels, and other nested strings never bind a family.
    """
    found: set[str] = set()
    for key in ("family_id", "family_ids"):
        for entry in _string_entries(detect_payload.get(key)):
            if entry in _KNOWN_FAMILY_IDS:
                found.add(entry)
    families = detect_payload.get("families")
    if isinstance(families, Sequence) and not isinstance(families, (str, bytes)):
        for item in families:
            if isinstance(item, str):
                if item in _KNOWN_FAMILY_IDS:
                    found.add(item)
            elif isinstance(item, Mapping):
                for entry in _string_entries(item.get("family_id")):
                    if entry in _KNOWN_FAMILY_IDS:
                        found.add(entry)
                for entry in _string_entries(item.get("family_ids")):
                    if entry in _KNOWN_FAMILY_IDS:
                        found.add(entry)
    evidence = detect_payload.get("family_evidence")
    if isinstance(evidence, Mapping):
        for key in evidence:
            if isinstance(key, str) and key in _KNOWN_FAMILY_IDS:
                found.add(key)
    for key in ("config", "measurements"):
        nested = detect_payload.get(key)
        if isinstance(nested, Mapping):
            for entry in _string_entries(nested.get("family_id")):
                if entry in _KNOWN_FAMILY_IDS:
                    found.add(entry)
            for entry in _string_entries(nested.get("family_ids")):
                if entry in _KNOWN_FAMILY_IDS:
                    found.add(entry)
    return found


def _detect_metric_ids(detect_payload: Mapping[str, object]) -> set[str]:
    """Extract structured detect metric identities from a detect payload.

    Metric IDs come only from explicit ``metric_id``/``metric_ids`` fields
    and the detector's declared ``config`` metric identities. Status words,
    control kinds/IDs, arbitrary measurement labels, and other nested strings
    never bind a metric.
    """
    found: set[str] = set()
    for key in ("metric_id", "metric_ids"):
        found.update(_string_entries(detect_payload.get(key)))
    families = detect_payload.get("families")
    if isinstance(families, Sequence) and not isinstance(families, (str, bytes)):
        for item in families:
            if isinstance(item, Mapping):
                found.update(_string_entries(item.get("metric_id")))
                found.update(_string_entries(item.get("metric_ids")))
                observed = item.get("observed_metrics")
                if isinstance(observed, Mapping):
                    for key in observed:
                        if isinstance(key, str) and key:
                            found.add(key)
                passing = item.get("threshold_pass")
                if isinstance(passing, Mapping):
                    for key in passing:
                        if isinstance(key, str) and key:
                            found.add(key)
    config = detect_payload.get("config")
    if isinstance(config, Mapping):
        found.update(_string_entries(config.get("metric_id")))
        found.update(_string_entries(config.get("metric_ids")))
        thresholds = config.get("thresholds")
        if isinstance(thresholds, Sequence) and not isinstance(thresholds, (str, bytes)):
            for item in thresholds:
                if isinstance(item, Mapping):
                    found.update(_string_entries(item.get("metric_id")))
                    found.update(_string_entries(item.get("metric_ids")))
    return found


def _localize_declared_ids(localize_payload: Mapping[str, object]) -> tuple[set[str], set[str]]:
    """Return the exact declared localize ``(families, metrics)`` identities."""
    families = set(_string_entries(localize_payload.get("family_id")))
    families.update(_string_entries(localize_payload.get("family_ids")))
    metrics = set(_string_entries(localize_payload.get("metric_id")))
    metrics.update(_string_entries(localize_payload.get("metric_ids")))
    return families, metrics


def _localize_axis_map(localize_payload: Mapping[str, object]) -> dict[str, dict[str, set[str]]]:
    """Build a structured ``axis -> {supported, status}`` map from a localize payload.

    Layer/slice axes read the layer/slice payload's declared/affected/earliest
    layer and declared/affected slice identities. Checkpoint/token/time/feature
    axes read the axial payload's ``axes`` entries plus ``report_localization``
    and earliest+affected selections: each entry contributes its axis, status,
    and every localized/supported selection identity it carries.
    Applicability is never decided by key presence: an axis counts as covered
    only when its entry (or payload-level verdict for layer/slice) marks a
    localized/supported selection.
    """
    axis_map: dict[str, dict[str, set[str]]] = {}

    def _record(axis: str, identity: str, *, supported: bool) -> None:
        entry = axis_map.setdefault(axis, {"supported": set(), "status": set()})
        if identity:
            entry["supported" if supported else "status"].add(identity)

    verdict = localize_payload.get("verdict")
    verdict_supported = verdict in ("localized",)
    order_items = _optional_str_items(localize_payload.get("layer_order", []))
    affected_items = _optional_str_items(localize_payload.get("affected_layers", []))
    affected_set = set(affected_items)
    layer_identities = list(order_items) + list(affected_items)
    for identity in layer_identities:
        _record(
            "layer",
            identity,
            supported=bool(verdict_supported and identity in affected_set),
        )
    earliest_layer = localize_payload.get("earliest_layer")
    if isinstance(earliest_layer, str) and earliest_layer:
        _record("layer", earliest_layer, supported=bool(verdict_supported))
    slice_items = _optional_str_items(localize_payload.get("declared_slice_ids", []))
    slice_hits = _optional_str_items(localize_payload.get("affected_slices", []))
    slice_hit_set = set(slice_hits)
    slice_identities = list(slice_items) + list(slice_hits)
    for identity in slice_identities:
        _record(
            "slice",
            identity,
            supported=bool(verdict_supported and identity in slice_hit_set),
        )
    axes = localize_payload.get("axes")
    if isinstance(axes, Sequence) and not isinstance(axes, (str, bytes)):
        for entry in axes:
            if not isinstance(entry, Mapping):
                continue
            axis = entry.get("axis")
            if not isinstance(axis, str) or not axis:
                continue
            status = entry.get("status")
            supported = status in ("localized", "supported")
            for key in ("affected", "earliest"):
                value = entry.get(key)
                if isinstance(value, str) and value:
                    _record(axis, value, supported=supported)
                elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                    for identity in value:
                        if isinstance(identity, str) and identity:
                            _record(axis, identity, supported=supported)
            rows_raw = entry.get("report_rows", [])
            rows_items = (
                tuple(rows_raw) if isinstance(rows_raw, Sequence) and not isinstance(rows_raw, (str, bytes)) else ()
            )
            for row in rows_items:
                if isinstance(row, Mapping):
                    selection = row.get("selection")
                    if isinstance(selection, str) and selection:
                        _record(axis, selection, supported=supported)
    axial_supported: dict[str, bool] = {}
    if isinstance(axes, Sequence) and not isinstance(axes, (str, bytes)):
        for entry in axes:
            if isinstance(entry, Mapping) and isinstance(entry.get("axis"), str):
                axial_supported[str(entry.get("axis"))] = axial_supported.get(str(entry.get("axis")), False) or (
                    entry.get("status") in ("localized", "supported")
                )
    report_raw = localize_payload.get("report_localization", [])
    report_items = (
        tuple(report_raw) if isinstance(report_raw, Sequence) and not isinstance(report_raw, (str, bytes)) else ()
    )
    for row in report_items:
        if isinstance(row, Mapping):
            axis = row.get("axis")
            selection = row.get("selection")
            status = row.get("status")
            if isinstance(axis, str) and isinstance(selection, str) and axis and selection:
                gate = axial_supported.get(axis, status in ("supported", "localized"))
                _record(axis, selection, supported=bool(gate and status in ("supported", "localized")))
    return axis_map


def _check_explain_identities(hypotheses: Sequence[ExplanationHypothesis], invocation: Any) -> None:
    """Bind declared hypothesis identity against the workflow invocation.

    Fails closed with ``StageContractError`` when the invocation's manifest
    identity, request capture representation, or prior detect/localize
    evidence contradicts a declared hypothesis. Verified identities: run
    manifest, capture representation, manifest dataset split, upstream
    diagnostic linkage (hypothesis ``family_id`` against real detect family
    fields, declared metrics against structured detect metric fields), and —
    for every entry of the explicit ``localization_bindings`` contract — the
    declared ``(axis, identity)`` against the structured prior localization
    result. Each binding requires the prior localize payload to actually
    cover that axis with a localized/supported selection carrying the exact
    identity; a missing axis, a not_applicable/negative/unsupported axis, or
    a mismatched identity raises before any callback. ``layer_id``/
    ``slice_id``/``symptom_id``/``target_id`` are hypothesis-local method
    context validated by method inputs, never upstream evidence by their mere
    presence. Prior stage order is verified through ``StageOutput.stage``.
    Missing required bindable upstream fields fail closed rather than
    completing.
    """
    from latent_anything._diagnostic_workflow import StageContractError as _ContractError

    request = invocation.request
    manifest = invocation.manifest
    if not isinstance(manifest, Mapping):
        raise _ContractError("explain executor requires a manifest mapping")
    manifest_id = manifest.get("manifest_id")
    if not isinstance(manifest_id, str) or not manifest_id:
        raise _ContractError("explain executor requires a manifest_id")
    if request.manifest_id != manifest_id:
        raise _ContractError(
            f"explain executor manifest mismatch: request {request.manifest_id!r} != manifest {manifest_id!r}"
        )
    capture_rep = getattr(getattr(request, "capture", None), "representation_identity", "")
    prior = tuple(getattr(invocation, "prior", ()) or ())
    stages = tuple(str(getattr(item, "stage", "")) for item in prior)
    expected_prior = ("capture", "detect", "localize")
    if stages and stages != expected_prior[: len(stages)]:
        raise _ContractError(f"explain executor prior stages are out of order: {stages!r}")
    prior_payloads: dict[str, Mapping[str, object]] = {}
    for item in prior:
        stage = str(getattr(item, "stage", ""))
        payload = getattr(item, "payload", {})
        prior_payloads[stage] = dict(payload) if isinstance(payload, Mapping) else {}
    detect_payload = prior_payloads.get("detect", {})
    localize_payload = prior_payloads.get("localize", {})
    detect_families = _detect_family_ids(detect_payload)
    detect_metrics = _detect_metric_ids(detect_payload)
    localize_families, localize_metrics = _localize_declared_ids(localize_payload)
    axis_map = _localize_axis_map(localize_payload) if "localize" in prior_payloads else {}
    for hypothesis in hypotheses:
        if hypothesis.manifest_id and hypothesis.manifest_id != request.manifest_id:
            raise _ContractError(
                f"explain hypothesis {hypothesis.hypothesis_id!r} declares manifest "
                f"{hypothesis.manifest_id!r} != run manifest {request.manifest_id!r}"
            )
        if capture_rep and hypothesis.representation_id and hypothesis.representation_id != capture_rep:
            raise _ContractError(
                f"explain hypothesis {hypothesis.hypothesis_id!r} declares representation "
                f"{hypothesis.representation_id!r} != capture {capture_rep!r}"
            )
        dataset_split = ""
        try:
            dataset = manifest.get("dataset")
            if isinstance(dataset, Mapping):
                candidate = dataset.get("split_identity")
                if isinstance(candidate, str):
                    dataset_split = candidate
        except AttributeError:
            dataset_split = ""
        if dataset_split and hypothesis.dataset_id and hypothesis.dataset_id not in dataset_split:
            declared_tokens = set(hypothesis.dataset_id.split(":"))
            split_tokens = set(dataset_split.split("-"))
            if not (declared_tokens & split_tokens) and hypothesis.dataset_id != dataset_split:
                raise _ContractError(
                    f"explain hypothesis {hypothesis.hypothesis_id!r} declares dataset "
                    f"{hypothesis.dataset_id!r} outside manifest split {dataset_split!r}"
                )
        if "detect" in prior_payloads:
            if not detect_families:
                raise _ContractError(
                    f"explain hypothesis {hypothesis.hypothesis_id!r} has no bindable "
                    "detect family evidence in the prior detect payload"
                )
            if hypothesis.family_id not in detect_families:
                raise _ContractError(
                    f"explain hypothesis {hypothesis.hypothesis_id!r} declares family "
                    f"{hypothesis.family_id!r} with no matching prior detect family evidence"
                )
            for metric_id in hypothesis.metric_ids:
                if metric_id not in detect_metrics:
                    raise _ContractError(
                        f"explain hypothesis {hypothesis.hypothesis_id!r} declares metric "
                        f"{metric_id!r} with no matching prior detect metric evidence"
                    )
        if "localize" in prior_payloads:
            if localize_families and hypothesis.family_id not in localize_families:
                raise _ContractError(
                    f"explain hypothesis {hypothesis.hypothesis_id!r} declares family "
                    f"{hypothesis.family_id!r} with no matching prior localize family evidence"
                )
            if localize_metrics and not set(hypothesis.metric_ids) & localize_metrics:
                raise _ContractError(
                    f"explain hypothesis {hypothesis.hypothesis_id!r} declares metrics "
                    f"{hypothesis.metric_ids!r} with no matching prior localize metric evidence"
                )
            if not hypothesis.localization_bindings:
                raise _ContractError(
                    f"explain hypothesis {hypothesis.hypothesis_id!r} declares no localization "
                    "bindings but the run carries a prior localize payload"
                )
            for axis, identity in hypothesis.localization_bindings:
                entry = axis_map.get(axis)
                if entry is None or identity not in entry.get("supported", set()):
                    raise _ContractError(
                        f"explain hypothesis {hypothesis.hypothesis_id!r} declares localization "
                        f"({axis!r}, {identity!r}) with no localized/supported prior "
                        f"{axis} selection carrying that identity"
                    )


# ---------------------------------------------------------------------------
# Method input carriers (caller-fitted data + one callback per method)
# ---------------------------------------------------------------------------

ProbeFn = Callable[[], Mapping[str, object]]
"""Zero-argument probe fit returning already-shaped headline arrays."""

TCAVFn = Callable[[], Mapping[str, object]]
"""Zero-argument TCAV fit returning gradients plus concept batches."""

IGFn = Callable[[], Mapping[str, object]]
"""Zero-argument IG evaluation returning attributions plus completeness."""


@dataclass(frozen=True)
class MethodInputs:
    """Caller-supplied data and exactly one estimator callback per method."""

    probe: Mapping[str, object] | None = None
    tcav: Mapping[str, object] | None = None
    integrated_gradients: Mapping[str, object] | None = None
    sae_sparse: Mapping[str, object] | None = None
    lens: Mapping[str, object] | None = None
    geometry: Mapping[str, object] | None = None
    density: Mapping[str, object] | None = None
    clustering: Mapping[str, object] | None = None

    def callback_for(self, method: str) -> Mapping[str, object] | None:
        """Return the declared input bundle for one method, if any."""
        if method == "probe":
            return self.probe
        if method == "tcav":
            return self.tcav
        if method == "integrated_gradients":
            return self.integrated_gradients
        if method == "sae_sparse":
            return self.sae_sparse
        if method == "lens":
            return self.lens
        if method == "geometry":
            return self.geometry
        if method == "density":
            return self.density
        if method == "clustering":
            return self.clustering
        raise ExplanationError(f"unsupported explanation method: {method!r}")


# ---------------------------------------------------------------------------
# Canonical evidence record
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExplanationEvidence:
    """One canonical non-causal evidence record for one hypothesis/method."""

    hypothesis_id: str
    method: str
    outcome: EvidenceOutcome
    claim_allowed: bool
    observed_effect: Mapping[str, float]
    fidelity: Mapping[str, object]
    stability: Mapping[str, object]
    selectivity: Mapping[str, object]
    leakage: Mapping[str, object]
    uncertainty: Mapping[str, object]
    control_outcomes: Mapping[str, str]
    central_outcomes: Mapping[str, object]
    missing_evidence: tuple[str, ...]
    limitations: tuple[str, ...]
    reason: str
    provenance: Mapping[str, object]

    def __post_init__(self) -> None:
        _non_empty_string(self.hypothesis_id, name="hypothesis_id")
        for name in ("fidelity", "stability", "selectivity", "leakage", "uncertainty"):
            item = _raw(self, name)
            if not isinstance(item, Mapping):
                raise ExplanationError(f"{name} must be a mapping")
            object.__setattr__(self, name, dict(item))
        control_records: object = _raw(self, "control_outcomes")
        if not isinstance(control_records, Mapping):
            raise ExplanationError("control_outcomes must be a mapping")
        object.__setattr__(self, "control_outcomes", dict(control_records))
        central_records: object = _raw(self, "central_outcomes")
        if not isinstance(central_records, Mapping):
            raise ExplanationError("central_outcomes must be a mapping")
        object.__setattr__(self, "central_outcomes", dict(central_records))
        raw_missing: object = _raw(self, "missing_evidence")
        if isinstance(raw_missing, (str, bytes)) or not isinstance(raw_missing, Sequence):
            raise ExplanationError("missing_evidence must be a list of strings")
        object.__setattr__(self, "missing_evidence", tuple(raw_missing))
        raw_limits: object = _raw(self, "limitations")
        if isinstance(raw_limits, (str, bytes)) or not isinstance(raw_limits, Sequence):
            raise ExplanationError("limitations must be a list of strings")
        object.__setattr__(self, "limitations", tuple(raw_limits))
        _non_empty_string(self.reason, name="reason")
        raw_provenance: object = _raw(self, "provenance")
        if not isinstance(raw_provenance, Mapping):
            raise ExplanationError("provenance must be a mapping")
        object.__setattr__(self, "provenance", dict(raw_provenance))
        if self.outcome == "supported" and (not self.claim_allowed or self.missing_evidence):
            raise ExplanationError("supported outcomes must allow the claim and admit no missing evidence")
        if self.outcome in ("unsupported", "omitted") and self.claim_allowed:
            raise ExplanationError(f"{self.outcome} outcomes must not allow the claim")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible mapping of this evidence record."""
        return {
            "causal": False,
            "central_outcomes": dict(self.central_outcomes),
            "claim_allowed": self.claim_allowed,
            "control_outcomes": dict(self.control_outcomes),
            "fidelity": dict(self.fidelity),
            "hypothesis_id": self.hypothesis_id,
            "kind": "explanation",
            "leakage": dict(self.leakage),
            "limitations": list(self.limitations),
            "method": self.method,
            "missing_evidence": list(self.missing_evidence),
            "observed_effect": dict(self.observed_effect),
            "outcome": self.outcome,
            "provenance": dict(self.provenance),
            "reason": self.reason,
            "selectivity": dict(self.selectivity),
            "stability": dict(self.stability),
            "uncertainty": dict(self.uncertainty),
        }


def _passes(comparator: str, observed: float, target: float) -> bool:
    if comparator == ">=":
        return bool(observed >= target)
    return bool(observed <= target)


def _finite_array(values: object, *, name: str) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ExplanationError(f"{name} must be numeric: {exc}") from exc
    if array.size == 0:
        raise ExplanationError(f"{name} must not be empty")
    if not np.isfinite(array).all():
        raise ExplanationError(f"{name} contains non-finite values")
    return array


def _check_disjoint_identities(
    train_ids: Sequence[object], eval_ids: Sequence[object], *, hypothesis_id: str
) -> tuple[str, ...]:
    train = tuple(str(item) for item in train_ids)
    eval_items = tuple(str(item) for item in eval_ids)
    if not train or not eval_items:
        raise ExplanationError(f"hypothesis {hypothesis_id!r} sample identities must not be empty")
    if any(not item.strip() for item in (*train, *eval_items)):
        raise ExplanationError(f"hypothesis {hypothesis_id!r} sample identities must be non-empty strings")
    if len(set(train)) != len(train) or len(set(eval_items)) != len(eval_items):
        raise ExplanationError(f"hypothesis {hypothesis_id!r} sample identities must be unique within each split")
    overlap = sorted(set(train) & set(eval_items))
    if overlap:
        raise ExplanationError(f"hypothesis {hypothesis_id!r} train/eval split leaks: {', '.join(overlap[:4])}")
    return train


@dataclass(frozen=True)
class _ExplainContext:
    """One executor call evaluated exactly once and shared by decisions."""

    hypotheses: object
    evidence: object
    central_outcomes: object
    control_outcomes: object

    def __post_init__(self) -> None:
        checked_hypotheses = list(_require_hypotheses(self.hypotheses))
        object.__setattr__(self, "hypotheses", tuple(checked_hypotheses))
        checked_evidence = list(_require_evidence_items(self.evidence))
        object.__setattr__(self, "evidence", tuple(checked_evidence))
        raw_central: object = self.central_outcomes
        if not isinstance(raw_central, Mapping):
            raise ExplanationError("context central_outcomes must be a mapping")
        object.__setattr__(self, "central_outcomes", dict(raw_central))
        raw_controls: object = self.control_outcomes
        if not isinstance(raw_controls, Mapping):
            raise ExplanationError("context control_outcomes must be a mapping")
        normalized: dict[str, dict[str, str]] = {}
        for key, value in raw_controls.items():
            if not isinstance(key, str) or not isinstance(value, Mapping):
                raise ExplanationError("context control_outcomes must map strings to string mappings")
            normalized[key] = {str(item): str(entry) for item, entry in value.items()}
        object.__setattr__(self, "control_outcomes", normalized)


# ---------------------------------------------------------------------------
# Probe evaluation (one fit, leakage-safe)
# ---------------------------------------------------------------------------


def _fit_probe(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    random_state: int,
) -> LinearProbeResult:
    """Fit one bounded linear probe (training-only scaler + C=1.0/lbfgs/balanced logreg)."""
    from sklearn.linear_model import LogisticRegression  # type: ignore[reportMissingTypeStubs]
    from sklearn.preprocessing import StandardScaler  # type: ignore[reportMissingTypeStubs]

    scaler = StandardScaler()
    train_scaled: np.ndarray = np.asarray(scaler.fit_transform(np.asarray(train_x, dtype=np.float64)))
    test_scaled: np.ndarray = np.asarray(scaler.transform(np.asarray(test_x, dtype=np.float64)))
    classifier = LogisticRegression(
        C=1.0, solver="lbfgs", max_iter=1000, random_state=int(random_state), class_weight="balanced"
    )
    classifier.fit(train_scaled, np.asarray(train_y))
    predictions: np.ndarray = np.asarray(classifier.predict(test_scaled))
    probabilities: np.ndarray = np.asarray(classifier.predict_proba(test_scaled))
    coefficients: np.ndarray = np.asarray(classifier.coef_)
    n_classes = len(np.unique(np.asarray(train_y)))
    return LinearProbeResult(
        accuracy=float(np.mean(predictions == np.asarray(test_y))),
        val_accuracy=0.0,
        classes=np.unique(np.asarray(train_y)),
        predictions=predictions,
        probabilities=probabilities,
        coefficients=coefficients[0] if n_classes == 2 and coefficients.shape[0] == 1 else coefficients,
        intercept=np.asarray(classifier.intercept_),
        n_iter=int(classifier.n_iter_[0]) if hasattr(classifier, "n_iter_") else 0,
        train_indices=np.full(len(np.asarray(train_x)) + len(np.asarray(test_x)), False),
        val_indices=np.full(len(np.asarray(train_x)) + len(np.asarray(test_x)), False),
        test_indices=np.full(len(np.asarray(train_x)) + len(np.asarray(test_x)), False),
        feature_means=np.asarray(scaler.mean_) if hasattr(scaler, "mean_") else None,
        feature_stds=np.asarray(scaler.scale_) if hasattr(scaler, "scale_") else None,
        config=LinearProbeConfig(random_state=int(random_state)),
        provenance={"method": "diagnostic-probe"},
    )


def _evaluate_probe(
    hypothesis: ExplanationHypothesis,
    bundle: Mapping[str, object],
    *,
    repetitions: int,
    confidence_level: float,
    evaluation_seed: int,
    control_seed: int,
    training_seed: int,
) -> ExplanationEvidence:
    """Fit one leakage-safe probe and gate all five evidence dimensions."""
    missing: list[str] = []
    provenance: dict[str, object] = {
        "explainer_version": EXPLAINER_VERSION,
        "method": "probe",
        "hypothesis_id": hypothesis.hypothesis_id,
        "representation_id": hypothesis.representation_id,
        "dataset_id": hypothesis.dataset_id,
        "capacity": DECLARED_PROBE_CAPACITY,
    }

    def _blocked(reason: str, extra_missing: Sequence[str] = ()) -> ExplanationEvidence:
        return ExplanationEvidence(
            hypothesis_id=hypothesis.hypothesis_id,
            method="probe",
            outcome="inconclusive",
            claim_allowed=False,
            observed_effect={},
            fidelity={"status": "missing"},
            stability={"status": "missing"},
            selectivity={"status": "missing"},
            leakage={"status": "missing"},
            uncertainty={"status": "missing"},
            control_outcomes={},
            central_outcomes={},
            missing_evidence=tuple([*missing, *extra_missing]),
            limitations=("probe accuracy alone is not an explanation; causal claims require 80.18+ trials",),
            reason=reason,
            provenance=provenance,
        )

    try:
        bound_controls = _bound_control_ids(hypothesis)
    except ExplanationError as exc:
        missing.append("missing-evidence:control-declaration")
        return _blocked(str(exc), ("missing-evidence:control-declaration",))
    try:
        train_x = _finite_array(bundle.get("train_matrix"), name="probe train_matrix")
        eval_x = _finite_array(bundle.get("eval_matrix"), name="probe eval_matrix")
        train_y = _finite_array(bundle.get("train_labels"), name="probe train_labels").ravel()
        eval_y = _finite_array(bundle.get("eval_labels"), name="probe eval_labels").ravel()
    except ExplanationError as exc:
        missing.append("missing-evidence:probe-matrices")
        return _blocked(f"probe input is incomplete: {exc}", ("missing-evidence:probe-matrices",))
    train_ids = bundle.get("train_ids")
    eval_ids = bundle.get("eval_ids")
    capacity = bundle.get("capacity")
    negative_accuracy = bundle.get("negative_accuracy")
    if train_ids is None or eval_ids is None:
        return _blocked("probe sample identities are missing", ("missing-evidence:leakage-identities",))
    if not isinstance(capacity, str) or not capacity:
        return _blocked("probe capacity declaration is missing", ("missing-evidence:capacity",))
    if train_x.ndim != 2 or eval_x.ndim != 2:
        return _blocked("probe matrices must be 2D", ("missing-evidence:probe-matrices",))
    if train_x.shape[1] != eval_x.shape[1]:
        return _blocked("probe train/eval feature dimensions disagree", ("missing-evidence:probe-matrices",))
    if train_x.shape[0] != train_y.shape[0] or eval_x.shape[0] != eval_y.shape[0]:
        return _blocked("probe labels must cover every row", ("missing-evidence:probe-matrices",))
    if len(np.unique(train_y)) < 2 or len(np.unique(eval_y)) < 2:
        return _blocked("probe splits must contain at least two classes", ("missing-evidence:probe-matrices",))
    try:
        _check_disjoint_identities(
            tuple(train_ids) if isinstance(train_ids, Sequence) and not isinstance(train_ids, (str, bytes)) else [],
            tuple(eval_ids) if isinstance(eval_ids, Sequence) and not isinstance(eval_ids, (str, bytes)) else [],
            hypothesis_id=hypothesis.hypothesis_id,
        )
    except ExplanationError as exc:
        return _blocked(str(exc), ("failed-leakage:train-eval-overlap",))
    if hypothesis.train_split_identity == hypothesis.eval_split_identity:
        return _blocked("probe split identities are identical", ("failed-leakage:split-identity",))
    declared_capacity = str(capacity)
    provenance["declared_capacity"] = declared_capacity
    capacity_passed = declared_capacity == DECLARED_PROBE_CAPACITY
    n_classes = int(len(np.unique(train_y)))
    n_params = int(train_x.shape[1] * (1 if n_classes == 2 else n_classes) + (1 if n_classes == 2 else n_classes))
    if n_params > int(train_x.shape[0]):
        capacity_passed = False

    try:
        headline = _fit_probe(train_x, train_y, eval_x, eval_y, int(training_seed))
    except ValueError as exc:
        return _blocked(f"probe fit failed: {exc}", ("missing-evidence:fidelity",))
    heldout = float(headline.accuracy)
    headline_coef = np.asarray(headline.coefficients, dtype=np.float64).ravel()
    headline_pred = np.asarray(headline.predictions).ravel()
    truth = np.asarray(eval_y).ravel()
    n_eval = int(truth.shape[0])
    if headline_pred.shape[0] != n_eval:
        return _blocked("probe predictions do not cover the eval split", ("missing-evidence:fidelity",))

    # Central randomized control: the actual label permutation plus the probe
    # refit runs inside the executor-owned stream; gap binds from the outcome.
    # Declared identities extract here: the callback below runs inside the
    # central stream under the declared randomized ID.
    capacity_id = bound_controls["capacity"]
    randomized_id = bound_controls["randomized"]
    negative_id = bound_controls["negative"]
    randomized_holder: dict[str, object] = {}

    def _randomized_statistic(rng: object) -> Mapping[str, float]:
        rng = _require_generator(rng)
        shuffled = np.asarray(rng.permutation(np.asarray(train_y).ravel()))

        refit = _fit_probe(train_x, shuffled, eval_x, eval_y, int(training_seed))
        randomized = float(refit.accuracy)
        randomized_holder["predictions"] = np.asarray(refit.predictions).ravel()
        randomized_holder["coefficients"] = np.asarray(refit.coefficients, dtype=np.float64).ravel()
        return {
            "heldout_accuracy": float(heldout),
            "randomized_accuracy": float(randomized),
            "leakage_gap": float(float(heldout) - float(randomized)),
        }

    shuffle_outcome = _central_control(
        control_id=randomized_id,
        base_seed=int(control_seed),
        seed_role="control",
        required=False,
        kind="randomized",
        metric_ids=("heldout_accuracy", "leakage_gap"),
        expected_behavior="one seeded permutation of the training labels",
        statistic=_randomized_statistic,
    )
    if shuffle_outcome.status == "failed":
        return _blocked(
            f"probe randomized control failed: {shuffle_outcome.reason}",
            (f"failed-control:{randomized_id}",),
        )
    probe_observed = _require_observed_floats(shuffle_outcome.observed, name="probe randomized control observed")
    randomized_accuracy = probe_observed["randomized_accuracy"]
    gap = probe_observed["leakage_gap"]
    randomized_pred = np.asarray(randomized_holder["predictions"], dtype=np.float64).ravel()

    # Stability: compare the headline fit with either declared estimator-seed
    # refits or caller-supplied, identity-bound training-set perturbations.
    coef_cosines: list[float] = []
    seed_accuracies: list[float] = []
    stability_raw = bundle.get("stability_replicates")
    stability_replicates: list[dict[str, object]] = []
    if stability_raw is None:
        for position, seed in enumerate(hypothesis.seeds):
            if position == 0:
                coef_cosines.append(1.0)
                seed_accuracies.append(heldout)
                continue
            refit = _fit_probe(train_x, train_y, eval_x, eval_y, int(seed))
            coef_cosines.append(
                _sign_aware_cosine(headline_coef, np.asarray(refit.coefficients, dtype=np.float64).ravel())
            )
            seed_accuracies.append(float(refit.accuracy))
    else:
        if isinstance(stability_raw, (str, bytes)) or not isinstance(stability_raw, Sequence) or len(stability_raw) < 2:
            return _blocked(
                "probe stability requires at least two training-set replicates",
                ("failed-stability:stability-replicates",),
            )
        if (
            isinstance(train_ids, (str, bytes))
            or not isinstance(train_ids, Sequence)
            or isinstance(eval_ids, (str, bytes))
            or not isinstance(eval_ids, Sequence)
        ):
            return _blocked(
                "probe stability requires explicit sample identities",
                ("failed-stability:stability-replicates",),
            )
        train_identity_set = {str(item) for item in train_ids}
        eval_identity_set = {str(item) for item in eval_ids}
        replicate_identity_sets: set[frozenset[str]] = set()
        for position, raw_replicate in enumerate(stability_raw):
            if not isinstance(raw_replicate, Mapping):
                return _blocked(
                    f"probe stability replicate {position} must be an object",
                    ("failed-stability:stability-replicates",),
                )
            try:
                replicate_x = _finite_array(raw_replicate.get("train_matrix"), name="stability train_matrix")
                replicate_y = _finite_array(raw_replicate.get("train_labels"), name="stability train_labels").ravel()
                raw_ids = raw_replicate.get("train_ids")
                if isinstance(raw_ids, (str, bytes)) or not isinstance(raw_ids, Sequence):
                    raise ExplanationError("stability train_ids must be a sequence")
                replicate_ids = tuple(str(item) for item in raw_ids)
                fit_seed = _require_count(raw_replicate.get("fit_seed"), name="stability fit_seed")
                _non_negative_int(fit_seed, name="stability fit_seed")
                if replicate_x.ndim != 2 or replicate_x.shape[1] != train_x.shape[1]:
                    raise ExplanationError("stability train features must match the probe feature dimension")
                if replicate_x.shape[0] != replicate_y.shape[0] or replicate_y.shape[0] != len(replicate_ids):
                    raise ExplanationError("stability rows, labels, and identities must align")
                if any(not item.strip() for item in replicate_ids) or len(set(replicate_ids)) != len(replicate_ids):
                    raise ExplanationError("stability train identities must be non-empty and unique")
                identity_set = frozenset(replicate_ids)
                if not identity_set < train_identity_set or identity_set & eval_identity_set:
                    raise ExplanationError(
                        "stability train identities must be a proper subset of the headline train split"
                    )
                if identity_set in replicate_identity_sets:
                    raise ExplanationError("stability replicates must use distinct training identities")
                replicate_identity_sets.add(identity_set)
                if not np.array_equal(np.unique(replicate_y), np.unique(train_y)):
                    raise ExplanationError("stability training labels must retain every headline class")
                if n_params > replicate_x.shape[0]:
                    raise ExplanationError("stability training subset does not satisfy probe capacity")
            except ExplanationError as exc:
                return _blocked(
                    f"probe stability replicate {position} is invalid: {exc}",
                    ("failed-stability:stability-replicates",),
                )
            try:
                refit = _fit_probe(replicate_x, replicate_y, eval_x, eval_y, fit_seed)
            except ValueError as exc:
                return _blocked(
                    f"probe stability replicate {position} fit failed: {exc}",
                    ("failed-stability:stability-replicates",),
                )
            cosine = _sign_aware_cosine(headline_coef, np.asarray(refit.coefficients, dtype=np.float64).ravel())
            coef_cosines.append(cosine)
            stability_replicates.append(
                {
                    "coefficient_cosine": float(cosine),
                    "fit_seed": int(fit_seed),
                    "train_rows": int(replicate_x.shape[0]),
                }
            )
    coef_stability = float(min(coef_cosines)) if coef_cosines else 0.0

    def _heldout_draw(rng: object) -> float:
        rng = _require_generator(rng)
        positions = rng.integers(0, n_eval, size=n_eval)

        return float(np.mean(headline_pred[positions] == truth[positions]))

    def _gap_draw(rng: object) -> float:
        rng = _require_generator(rng)
        positions = rng.integers(0, n_eval, size=n_eval)

        held = float(np.mean(headline_pred[positions] == truth[positions]))
        drawn = float(np.mean(randomized_pred[positions] == truth[positions]))
        return float(held - drawn)

    _, held_interval = _central_bootstrap(
        control_id="bootstrap:probe-heldout-accuracy",
        base_seed=int(evaluation_seed),
        seed_role="evaluation",
        repetitions=int(repetitions),
        confidence_level=float(confidence_level),
        required=False,
        kind="bootstrap",
        metric_ids=("heldout_accuracy",),
        expected_behavior="seeded resampling of fitted probe predictions",
        draw=_heldout_draw,
    )
    _, gap_interval = _central_bootstrap(
        control_id="bootstrap:probe-leakage-gap",
        base_seed=int(evaluation_seed),
        seed_role="evaluation",
        repetitions=int(repetitions),
        confidence_level=float(confidence_level),
        required=False,
        kind="bootstrap",
        metric_ids=("leakage_gap",),
        expected_behavior="seeded resampling of fitted probe predictions",
        draw=_gap_draw,
    )

    fidelity_threshold = hypothesis.threshold_for("heldout_accuracy")
    stability_threshold = hypothesis.threshold_for("coef_stability")
    selectivity_threshold = hypothesis.threshold_for("leakage_gap")
    fidelity_met = (
        True if fidelity_threshold is None else _passes(fidelity_threshold[0], heldout, fidelity_threshold[1])
    )
    stability_met = (
        True if stability_threshold is None else _passes(stability_threshold[0], coef_stability, stability_threshold[1])
    )
    selectivity_met = (
        True if selectivity_threshold is None else _passes(selectivity_threshold[0], gap, selectivity_threshold[1])
    )

    # Required-control gating through the central executor under the exact
    # declared identities (extracted above). A declared required control not
    # executed/passed blocks support; nothing adapter-invented may gate.
    specs: list[_ControlSpec] = []
    supplied: dict[str, Mapping[str, float] | None] = {}
    capacity_value = 1.0 if capacity_passed else 0.0
    specs.append(
        _ControlSpec(
            control_id=capacity_id,
            kind="capacity",
            required=True,
            metric_ids=("capacity",),
            expected_behavior="declared probe capacity bound",
            comparator="meets_threshold",
            threshold_value=1.0,
        )
    )
    supplied[capacity_id] = {"capacity": capacity_value}
    gap_gate = selectivity_threshold if selectivity_threshold is not None else (">=", 0.15)
    gap_comparator = "meets_threshold" if gap_gate[0] == ">=" else "below_threshold"
    specs.append(
        _ControlSpec(
            control_id=randomized_id,
            kind="randomized",
            required=True,
            metric_ids=("leakage_gap",),
            expected_behavior="label-randomization gap under predeclared threshold",
            comparator=gap_comparator,  # type: ignore[arg-type]
            threshold_value=float(gap_gate[1]),
        )
    )
    supplied[randomized_id] = {"leakage_gap": float(gap)}
    negative_gate = fidelity_threshold if fidelity_threshold is not None else (">=", 0.7)
    neg_comparator = "below_threshold" if negative_gate[0] == ">=" else "meets_threshold"
    if negative_accuracy is None:
        specs.append(
            _ControlSpec(
                control_id=negative_id,
                kind="counterexample",
                required=True,
                metric_ids=("heldout_accuracy",),
                expected_behavior="non-separable negative must not meet headline accuracy",
                comparator=neg_comparator,  # type: ignore[arg-type]
                threshold_value=float(negative_gate[1]),
            )
        )
        supplied[negative_id] = None
    else:
        try:
            negative_value = _require_float(negative_accuracy, name="probe negative control")
        except ExplanationError:
            return _blocked("probe negative control is non-numeric", (f"failed-control:{negative_id}",))
        if not np.isfinite(negative_value):
            return _blocked("probe negative control is non-finite", (f"failed-control:{negative_id}",))
        specs.append(
            _ControlSpec(
                control_id=negative_id,
                kind="counterexample",
                required=True,
                metric_ids=("heldout_accuracy",),
                expected_behavior="non-separable negative must not meet headline accuracy",
                comparator=neg_comparator,  # type: ignore[arg-type]
                threshold_value=float(negative_gate[1]),
            )
        )
        supplied[negative_id] = {"heldout_accuracy": float(negative_value)}
    plan = _ControlPlan(
        plan_id=f"{hypothesis.hypothesis_id}:probe",
        repetitions=int(repetitions),
        confidence_level=float(confidence_level),
        evaluation_seed=int(evaluation_seed),
        control_seed=int(control_seed),
        controls=tuple(specs),
    )
    outcomes = _execute_plan(plan, supplied)
    central_records = {control_id: outcome.to_dict() for control_id, outcome in outcomes.items()}
    blocked = list(_failed_required(outcomes))
    control_status = {
        control_id: ("passed" if outcome.status in ("passed", "recorded") else "failed")
        for control_id, outcome in outcomes.items()
    }

    observed = {
        "heldout_accuracy": float(heldout),
        "randomized_accuracy": float(randomized_accuracy),
        "leakage_gap": float(gap),
        "coef_stability": float(coef_stability),
    }
    if negative_accuracy is not None:
        observed["negative_accuracy"] = float(cast(float, negative_accuracy))
    fidelity = {
        "status": "passed" if fidelity_met else "failed",
        "metric": "heldout_accuracy",
        "observed": float(heldout),
        "threshold": None if fidelity_threshold is None else [fidelity_threshold[0], float(fidelity_threshold[1])],
        "train_accuracy_note": "train accuracy is recorded nowhere and never promoted",
    }
    stability: dict[str, object] = {
        "status": "passed" if stability_met else "failed",
        "metric": "coef_stability",
        "observed": float(coef_stability),
        "threshold": None if stability_threshold is None else [stability_threshold[0], float(stability_threshold[1])],
        "seed_accuracies": [float(item) for item in seed_accuracies],
        "sign_aware": True,
    }
    if stability_raw is not None:
        stability["basis"] = "identity-bound-training-set-replicates"
        stability["replicates"] = stability_replicates
    selectivity = {
        "status": "passed" if selectivity_met else "failed",
        "metric": "leakage_gap",
        "observed": float(gap),
        "threshold": None
        if selectivity_threshold is None
        else [selectivity_threshold[0], float(selectivity_threshold[1])],
        "randomized_accuracy": float(randomized_accuracy),
    }
    leakage = {
        "status": "passed" if capacity_passed and not blocked else "failed",
        "train_split_identity": hypothesis.train_split_identity,
        "eval_split_identity": hypothesis.eval_split_identity,
        "capacity_declared": declared_capacity,
        "capacity_passed": bool(capacity_passed),
        "disjoint_sample_identities": True,
    }
    uncertainty = {
        "status": "passed",
        "heldout_accuracy": dict(held_interval),
        "leakage_gap": dict(gap_interval),
        "evaluation_seed": int(evaluation_seed),
        "repetitions": int(repetitions),
    }
    gaps: list[str] = []
    if not fidelity_met:
        gaps.append("failed-fidelity:heldout_accuracy")
    if not stability_met:
        gaps.append("failed-stability:coef_stability")
    if not selectivity_met:
        gaps.append("failed-selectivity:leakage_gap")
    if not capacity_passed:
        gaps.append("failed-leakage:capacity")
    for control_id in blocked:
        gaps.append(f"failed-control:{control_id}")
    if gaps:
        return ExplanationEvidence(
            hypothesis_id=hypothesis.hypothesis_id,
            method="probe",
            outcome="inconclusive",
            claim_allowed=False,
            observed_effect=dict(observed),
            fidelity=dict(fidelity),
            stability=dict(stability),
            selectivity=dict(selectivity),
            leakage=dict(leakage),
            uncertainty=dict(uncertainty),
            control_outcomes=dict(control_status),
            central_outcomes=dict(central_records),
            missing_evidence=tuple(gaps),
            limitations=("probe accuracy alone is not an explanation; causal claims require 80.18+ trials",),
            reason=f"probe evidence blocked: {', '.join(gaps)}",
            provenance=dict(provenance),
        )
    return ExplanationEvidence(
        hypothesis_id=hypothesis.hypothesis_id,
        method="probe",
        outcome="supported",
        claim_allowed=True,
        observed_effect=dict(observed),
        fidelity=dict(fidelity),
        stability=dict(stability),
        selectivity=dict(selectivity),
        leakage=dict(leakage),
        uncertainty=dict(uncertainty),
        control_outcomes=dict(control_status),
        central_outcomes=dict(central_records),
        missing_evidence=(),
        limitations=("probe accuracy alone is not an explanation; causal claims require 80.18+ trials",),
        reason="leakage-safe probe meets fidelity, stability, selectivity, leakage, and uncertainty gates",
        provenance=dict(provenance),
    )


# ---------------------------------------------------------------------------
# TCAV evaluation (one CAV fit family, caller-fitted gradients)
# ---------------------------------------------------------------------------


def _evaluate_tcav(
    hypothesis: ExplanationHypothesis,
    bundle: Mapping[str, object],
    *,
    repetitions: int,
    confidence_level: float,
    evaluation_seed: int,
    control_seed: int,
) -> ExplanationEvidence:
    """Score one declared concept against caller-fitted gradients."""
    provenance: dict[str, object] = {
        "explainer_version": EXPLAINER_VERSION,
        "method": "tcav",
        "hypothesis_id": hypothesis.hypothesis_id,
        "representation_id": hypothesis.representation_id,
        "dataset_id": hypothesis.dataset_id,
        "concept_id": hypothesis.concept_id,
    }

    def _blocked(reason: str, gaps: Sequence[str]) -> ExplanationEvidence:
        return ExplanationEvidence(
            hypothesis_id=hypothesis.hypothesis_id,
            method="tcav",
            outcome="unsupported" if any(item.startswith("unsupported") for item in gaps) else "inconclusive",
            claim_allowed=False,
            observed_effect={},
            fidelity={"status": "missing"},
            stability={"status": "missing"},
            selectivity={"status": "missing"},
            leakage={"status": "missing"},
            uncertainty={"status": "missing"},
            control_outcomes={},
            central_outcomes={},
            missing_evidence=tuple(gaps),
            limitations=("a CAV direction is not a semantic label; causal claims require 80.18+ trials",),
            reason=reason,
            provenance=provenance,
        )

    try:
        bound_controls = _bound_control_ids(hypothesis)
    except ExplanationError as exc:
        return _blocked(str(exc), ("missing-evidence:control-declaration",))
    try:
        grad_matrix = _finite_array(bundle.get("gradients"), name="tcav gradients")
        concept_x = _finite_array(bundle.get("concept_matrix"), name="tcav concept_matrix")
        reference_x = _finite_array(bundle.get("reference_matrix"), name="tcav reference_matrix")
    except ExplanationError as exc:
        return _blocked(f"tcav input is incomplete: {exc}", ("missing-evidence:tcav-matrices",))
    if grad_matrix.ndim != 2 or concept_x.ndim != 2 or reference_x.ndim != 2:
        return _blocked("tcav matrices must be 2D", ("missing-evidence:tcav-matrices",))
    dim = int(grad_matrix.shape[1])
    if concept_x.shape[1] != dim or reference_x.shape[1] != dim:
        return _blocked(
            "tcav concept/reference dimensions disagree with gradients", ("missing-evidence:tcav-matrices",)
        )
    if concept_x.shape[0] < 2 or reference_x.shape[0] < 2:
        return _blocked(
            "tcav concept/reference sets need at least two examples each", ("missing-evidence:tcav-matrices",)
        )
    direction_method = bundle.get("direction_method")
    method_name = str(direction_method) if isinstance(direction_method, str) and direction_method else "mean_diff"
    if method_name not in ("mean_diff", "linear_separator"):
        return _blocked(f"tcav direction_method {method_name!r} is not supported", ("unsupported:direction-method",))
    target_identity = bundle.get("target_identity")
    if not isinstance(target_identity, str) or not target_identity.strip():
        return _blocked("tcav target identity is missing", ("missing-evidence:target-identity",))
    provenance["target_identity"] = str(target_identity)
    provenance["direction_method"] = method_name

    from latent_anything.tcav import ConceptDataset
    from latent_anything.tcav import learn_linear_separator_direction as _learn_linear
    from latent_anything.tcav import learn_mean_diff_direction as _learn_mean

    dataset = ConceptDataset(
        concept_examples=np.asarray(concept_x, dtype=np.float64),
        reference_examples=np.asarray(reference_x, dtype=np.float64),
        concept_name=hypothesis.concept_id,
        source=hypothesis.dataset_id,
        representation_space=hypothesis.representation_id,
        model_version=hypothesis.layer_id,
    )
    primary_seed = int(hypothesis.seeds[0])
    try:
        if method_name == "mean_diff":
            primary = _learn_mean(dataset, n_bootstrap=50, bootstrap_seed=primary_seed)
        else:
            primary = _learn_linear(dataset, n_bootstrap=50, bootstrap_seed=primary_seed, split_seed=primary_seed)
    except ValueError as exc:
        return _blocked(f"tcav concept fit failed: {exc}", ("missing-evidence:fidelity",))
    sensitivities = np.asarray(grad_matrix, dtype=np.float64) @ np.asarray(primary.direction, dtype=np.float64).ravel()
    n_examples = int(sensitivities.shape[0])
    headline = float(np.mean(sensitivities > 0))
    fidelity_threshold = hypothesis.threshold_for("concept_separability")
    fidelity_met = (
        True
        if fidelity_threshold is None
        else _passes(fidelity_threshold[0], float(primary.separability_accuracy), float(fidelity_threshold[1]))
    )
    if not np.isfinite(primary.separability_accuracy):
        return _blocked("tcav concept fidelity is non-finite", ("missing-evidence:fidelity",))

    # Stability: CAV direction across declared seeds (sign-aware) plus the
    # TCAV fraction spread. A disconnected CAV (near-zero norm handled
    # inside the learner) that flips sign across seeds fails here.
    cav_cosines: list[float] = [1.0]
    seed_scores: list[float] = [headline]
    for seed in hypothesis.seeds[1:]:
        if method_name == "mean_diff":
            candidate = _learn_mean(dataset, n_bootstrap=50, bootstrap_seed=int(seed))
        else:
            candidate = _learn_linear(dataset, n_bootstrap=50, bootstrap_seed=int(seed), split_seed=int(seed))
        cav_cosines.append(_sign_aware_cosine(primary.direction, candidate.direction))
        candidate_sens = np.asarray(grad_matrix, dtype=np.float64) @ np.asarray(candidate.direction).ravel()
        seed_scores.append(float(np.mean(candidate_sens > 0)))
    cav_stability = float(min(cav_cosines)) if cav_cosines else 0.0
    stability_threshold = hypothesis.threshold_for("cav_stability")
    stability_met = (
        True
        if stability_threshold is None
        else _passes(stability_threshold[0], cav_stability, float(stability_threshold[1]))
    )

    # Selectivity: headline fraction minus the averaged permutation
    # random-concept baseline, executed inside the central stream; plus the
    # declared negative concepts scored without refitting.
    pooled = np.concatenate(
        [np.asarray(concept_x, dtype=np.float64), np.asarray(reference_x, dtype=np.float64)], axis=0
    )
    n_concept = int(concept_x.shape[0])

    def _random_selectivity(rng: np.random.Generator) -> Mapping[str, float]:
        # Average ten seeded permutations inside the single central callback:
        # one executor-owned stream drives all draws, and the mean random
        # score is stable enough to gate selectivity against.
        grads_local = np.asarray(grad_matrix, dtype=np.float64)
        random_scores: list[float] = []
        for _ in range(10):
            perm = rng.permutation(pooled.shape[0])
            group_a = pooled[perm[:n_concept]]
            group_b = pooled[perm[n_concept:]]
            direction = group_a.mean(axis=0) - group_b.mean(axis=0)
            norm = float(np.linalg.norm(direction))
            if norm < 1e-15:
                random_scores.append(0.5)
                continue
            unit = direction / norm
            random_scores.append(float(np.mean((grads_local @ unit) > 0)))
        return {"tcav_score": float(headline), "random_score": float(sum(random_scores) / len(random_scores))}

    random_outcome = _central_control(
        control_id=bound_controls["selectivity"],
        base_seed=int(control_seed),
        seed_role="control",
        required=False,
        kind="randomized",
        metric_ids=("tcav_score", "random_score"),
        expected_behavior="ten seeded permutations of the pooled concept/reference sets",
        statistic=_random_selectivity,
    )
    if random_outcome.status == "failed":
        return _blocked(
            f"tcav random-concept control failed: {random_outcome.reason}",
            (f"failed-control:{bound_controls['selectivity']}",),
        )
    tcav_observed = _require_observed_floats(random_outcome.observed, name="tcav random-concept control observed")
    random_score = tcav_observed["random_score"]
    selectivity_margin = float(headline - random_score)
    selectivity_threshold = hypothesis.threshold_for("selectivity_margin")
    selectivity_met = (
        True
        if selectivity_threshold is None
        else _passes(selectivity_threshold[0], selectivity_margin, float(selectivity_threshold[1]))
    )
    negative_scores: dict[str, float] = {}
    negative_bundle = bundle.get("negative_concepts")
    if isinstance(negative_bundle, Mapping):
        for negative_id in hypothesis.negative_concept_ids:
            entry = negative_bundle.get(negative_id)
            if not isinstance(entry, Mapping):
                continue
            try:
                neg_concept = _finite_array(entry.get("concept_matrix"), name=f"tcav negative {negative_id}")
                neg_reference = _finite_array(entry.get("reference_matrix"), name=f"tcav negative {negative_id}")
            except ExplanationError:
                continue
            if neg_concept.ndim != 2 or neg_concept.shape[1] != dim:
                continue
            neg_dataset = ConceptDataset(
                concept_examples=np.asarray(neg_concept, dtype=np.float64),
                reference_examples=np.asarray(neg_reference, dtype=np.float64),
                concept_name=str(negative_id),
                source=hypothesis.dataset_id,
                representation_space=hypothesis.representation_id,
                model_version=hypothesis.layer_id,
            )
            neg_direction = _learn_mean(neg_dataset, n_bootstrap=10, bootstrap_seed=primary_seed)
            negative_scores[str(negative_id)] = float(
                np.mean((np.asarray(grad_matrix, dtype=np.float64) @ np.asarray(neg_direction.direction).ravel()) > 0)
            )
    negative_margin = min((float(headline - score) for score in negative_scores.values()), default=float("nan"))

    # Sample-level uncertainty: resample directional-derivative signs without
    # refitting the CAV, under the central repetition schedule.
    signs = (np.asarray(sensitivities).ravel() > 0).astype(np.float64)

    def _tcav_draw(rng: object) -> float:
        rng = _require_generator(rng)
        positions = rng.integers(0, n_examples, size=n_examples)

        return float(np.mean(signs[positions]))

    _, tcav_interval = _central_bootstrap(
        control_id="bootstrap:tcav-score",
        base_seed=int(evaluation_seed),
        seed_role="evaluation",
        repetitions=int(repetitions),
        confidence_level=float(confidence_level),
        required=False,
        kind="bootstrap",
        metric_ids=("tcav_score",),
        expected_behavior="seeded resampling of fitted directional-derivative signs",
        draw=_tcav_draw,
    )

    # Leakage: concept/reference/train-eval disjointness plus target binding.
    leakage_passed = True
    leakage_notes: list[str] = []
    concept_ids = bundle.get("concept_ids")
    reference_ids = bundle.get("reference_ids")
    if concept_ids is None or reference_ids is None:
        leakage_passed = False
        leakage_notes.append("concept/reference identities are missing")
    else:
        try:
            _check_disjoint_identities(
                tuple(concept_ids)
                if isinstance(concept_ids, Sequence) and not isinstance(concept_ids, (str, bytes))
                else [],
                tuple(reference_ids)
                if isinstance(reference_ids, Sequence) and not isinstance(reference_ids, (str, bytes))
                else [],
                hypothesis_id=hypothesis.hypothesis_id,
            )
        except ExplanationError as exc:
            leakage_passed = False
            leakage_notes.append(str(exc))
    if hypothesis.train_split_identity == hypothesis.eval_split_identity:
        leakage_passed = False
        leakage_notes.append("train/eval split identities are identical")

    specs: list[_ControlSpec] = []
    supplied: dict[str, Mapping[str, float] | None] = {}
    fidelity_id = bound_controls["fidelity"]
    stability_id = bound_controls["stability"]
    selectivity_id = bound_controls["selectivity"]
    separability_gate = fidelity_threshold if fidelity_threshold is not None else (">=", 0.7)
    neg_gate: Literal["below_threshold", "meets_threshold"] = (
        "below_threshold" if separability_gate[0] == ">=" else "meets_threshold"
    )
    assert neg_gate in ("below_threshold", "meets_threshold")
    fidelity_comparator = "meets_threshold" if separability_gate[0] == ">=" else "below_threshold"
    specs.append(
        _ControlSpec(
            control_id=fidelity_id,
            kind="counterexample",
            required=True,
            metric_ids=("concept_separability",),
            expected_behavior="concept classifier fidelity under predeclared threshold",
            comparator=fidelity_comparator,  # type: ignore[arg-type]
            threshold_value=float(separability_gate[1]),
        )
    )
    supplied[fidelity_id] = {"concept_separability": float(primary.separability_accuracy)}
    stability_gate = stability_threshold if stability_threshold is not None else (">=", 0.9)
    stability_comparator = "meets_threshold" if stability_gate[0] == ">=" else "below_threshold"
    specs.append(
        _ControlSpec(
            control_id=stability_id,
            kind="cross_seed",
            required=True,
            metric_ids=("cav_stability",),
            expected_behavior="CAV stability across declared seeds",
            comparator=stability_comparator,  # type: ignore[arg-type]
            threshold_value=float(stability_gate[1]),
        )
    )
    supplied[stability_id] = {"cav_stability": float(cav_stability)}
    margin_gate = selectivity_threshold if selectivity_threshold is not None else (">=", 0.4)
    margin_comparator = "meets_threshold" if margin_gate[0] == ">=" else "below_threshold"
    specs.append(
        _ControlSpec(
            control_id=selectivity_id,
            kind="randomized",
            required=True,
            metric_ids=("selectivity_margin",),
            expected_behavior="TCAV selectivity versus the random-concept baseline",
            comparator=margin_comparator,  # type: ignore[arg-type]
            threshold_value=float(margin_gate[1]),
        )
    )
    supplied[selectivity_id] = {"selectivity_margin": float(selectivity_margin)}
    plan = _ControlPlan(
        plan_id=f"{hypothesis.hypothesis_id}:tcav",
        repetitions=int(repetitions),
        confidence_level=float(confidence_level),
        evaluation_seed=int(evaluation_seed),
        control_seed=int(control_seed),
        controls=tuple(specs),
    )
    outcomes = _execute_plan(plan, supplied)
    central_records = {control_id: outcome.to_dict() for control_id, outcome in outcomes.items()}
    blocked = list(_failed_required(outcomes))
    control_status = {
        control_id: ("passed" if outcome.status in ("passed", "recorded") else "failed")
        for control_id, outcome in outcomes.items()
    }

    observed = {
        "tcav_score": float(headline),
        "random_score": float(random_score),
        "selectivity_margin": float(selectivity_margin),
        "concept_separability": float(primary.separability_accuracy),
        "cav_stability": float(cav_stability),
    }
    fidelity = {
        "status": "passed" if fidelity_met else "failed",
        "metric": "concept_separability",
        "observed": float(primary.separability_accuracy),
        "threshold": None if fidelity_threshold is None else [fidelity_threshold[0], float(fidelity_threshold[1])],
    }
    stability = {
        "status": "passed" if stability_met else "failed",
        "metric": "cav_stability",
        "observed": float(cav_stability),
        "threshold": None if stability_threshold is None else [stability_threshold[0], float(stability_threshold[1])],
        "seed_scores": [float(item) for item in seed_scores],
        "sign_aware": True,
    }
    selectivity = {
        "status": "passed" if selectivity_met else "failed",
        "metric": "selectivity_margin",
        "observed": float(selectivity_margin),
        "threshold": None
        if selectivity_threshold is None
        else [selectivity_threshold[0], float(selectivity_threshold[1])],
        "random_score": float(random_score),
        "negative_scores": {key: float(value) for key, value in negative_scores.items()},
        "negative_margin": float(negative_margin),
    }
    leakage = {
        "status": "passed" if leakage_passed and not blocked else "failed",
        "train_split_identity": hypothesis.train_split_identity,
        "eval_split_identity": hypothesis.eval_split_identity,
        "target_identity": str(target_identity),
        "notes": list(leakage_notes),
    }
    uncertainty = {
        "status": "passed",
        "tcav_score": dict(tcav_interval),
        "evaluation_seed": int(evaluation_seed),
        "repetitions": int(repetitions),
    }
    gaps: list[str] = []
    if not fidelity_met:
        gaps.append("failed-fidelity:concept_separability")
    if not stability_met:
        gaps.append("failed-stability:cav_stability")
    if not selectivity_met:
        gaps.append("failed-selectivity:selectivity_margin")
    if not leakage_passed:
        gaps.append("failed-leakage:concept-identities")
    for control_id in blocked:
        gaps.append(f"failed-control:{control_id}")
    if gaps:
        return ExplanationEvidence(
            hypothesis_id=hypothesis.hypothesis_id,
            method="tcav",
            outcome="inconclusive",
            claim_allowed=False,
            observed_effect=dict(observed),
            fidelity=dict(fidelity),
            stability=dict(stability),
            selectivity=dict(selectivity),
            leakage=dict(leakage),
            uncertainty=dict(uncertainty),
            control_outcomes=dict(control_status),
            central_outcomes=dict(central_records),
            missing_evidence=tuple(gaps),
            limitations=("a CAV direction is not a semantic label; causal claims require 80.18+ trials",),
            reason=f"tcav evidence blocked: {', '.join(gaps)}",
            provenance=dict(provenance),
        )
    return ExplanationEvidence(
        hypothesis_id=hypothesis.hypothesis_id,
        method="tcav",
        outcome="supported",
        claim_allowed=True,
        observed_effect=dict(observed),
        fidelity=dict(fidelity),
        stability=dict(stability),
        selectivity=dict(selectivity),
        leakage=dict(leakage),
        uncertainty=dict(uncertainty),
        control_outcomes=dict(control_status),
        central_outcomes=dict(central_records),
        missing_evidence=(),
        limitations=("a CAV direction is not a semantic label; causal claims require 80.18+ trials",),
        reason="declared concept meets fidelity, CAV stability, selectivity, leakage, and uncertainty gates",
        provenance=dict(provenance),
    )


# ---------------------------------------------------------------------------
# Integrated-gradients evaluation (one path integral family, no refits)
# ---------------------------------------------------------------------------


def _evaluate_integrated_gradients(
    hypothesis: ExplanationHypothesis,
    bundle: Mapping[str, object],
    *,
    repetitions: int,
    confidence_level: float,
    evaluation_seed: int,
    control_seed: int,
) -> ExplanationEvidence:
    """Gate one declared IG target/baseline run on all five dimensions."""
    provenance: dict[str, object] = {
        "explainer_version": EXPLAINER_VERSION,
        "method": "integrated_gradients",
        "hypothesis_id": hypothesis.hypothesis_id,
        "representation_id": hypothesis.representation_id,
        "dataset_id": hypothesis.dataset_id,
        "baseline_policy": hypothesis.baseline_policy,
    }

    def _blocked(reason: str, gaps: Sequence[str], *, unsupported: bool = False) -> ExplanationEvidence:
        return ExplanationEvidence(
            hypothesis_id=hypothesis.hypothesis_id,
            method="integrated_gradients",
            outcome="unsupported" if unsupported else "inconclusive",
            claim_allowed=False,
            observed_effect={},
            fidelity={"status": "missing"},
            stability={"status": "missing"},
            selectivity={"status": "missing"},
            leakage={"status": "missing"},
            uncertainty={"status": "missing"},
            control_outcomes={},
            central_outcomes={},
            missing_evidence=tuple(gaps),
            limitations=("attribution is not causation; causal claims require 80.18+ trials",),
            reason=reason,
            provenance=provenance,
        )

    try:
        bound_controls = _bound_control_ids(hypothesis)
    except ExplanationError as exc:
        return _blocked(str(exc), ("missing-evidence:control-declaration",))
    if bundle.get("disconnected") is True:
        return _blocked(
            "ig target is disconnected or non-differentiable",
            ("unsupported:disconnected-target",),
            unsupported=True,
        )
    if bundle.get("target_differentiable") is False:
        return _blocked(
            "ig target is declared non-differentiable",
            ("unsupported:non-differentiable-target",),
            unsupported=True,
        )
    try:
        attributions = _finite_array(bundle.get("attributions"), name="ig attributions")
        baseline_attributions = bundle.get("baseline_attributions")
        off_target_attributions = bundle.get("off_target_attributions")
        random_attributions = bundle.get("random_attributions")
    except ExplanationError as exc:
        return _blocked(f"ig input is incomplete: {exc}", ("missing-evidence:ig-attributions",))
    completeness_error = bundle.get("completeness_error")
    completeness_delta = bundle.get("completeness_delta")
    if completeness_error is None or completeness_delta is None:
        return _blocked("ig completeness report is missing", ("missing-evidence:completeness",))
    try:
        completeness_value = _require_float(completeness_error, name="ig completeness_error")
        delta_value = _require_float(completeness_delta, name="ig completeness_delta")
    except ExplanationError:
        return _blocked("ig completeness report is non-numeric", ("missing-evidence:completeness",))
    if not np.isfinite(completeness_value) or not np.isfinite(delta_value):
        return _blocked("ig completeness report is non-finite", ("missing-evidence:completeness",))
    target_identity = bundle.get("target_identity")
    baseline_identity = bundle.get("baseline_identity")
    input_identity = bundle.get("input_identity")
    if not isinstance(target_identity, str) or not target_identity.strip():
        return _blocked("ig target identity is missing", ("missing-evidence:target-identity",))
    if not isinstance(baseline_identity, str) or not baseline_identity.strip():
        return _blocked("ig baseline identity is missing", ("missing-evidence:baseline-identity",))
    if not isinstance(input_identity, str) or not input_identity.strip():
        return _blocked("ig input identity is missing", ("missing-evidence:input-identity",))
    if str(baseline_identity) == str(input_identity):
        return _blocked(
            "ig baseline/input identity drift: baseline equals the input",
            ("failed-leakage:baseline-input-drift",),
        )
    declared_baseline = hypothesis.baseline_policy.strip().lower().replace("_", "-")
    observed_baseline = str(baseline_identity).strip().lower()
    if declared_baseline not in observed_baseline and observed_baseline not in declared_baseline:
        return _blocked(
            f"ig baseline {observed_baseline!r} does not match the declared policy {hypothesis.baseline_policy!r}",
            ("failed-leakage:baseline-policy",),
        )
    provenance["target_identity"] = str(target_identity)
    provenance["baseline_identity"] = str(baseline_identity)
    provenance["input_identity"] = str(input_identity)
    if str(target_identity) != hypothesis.target_id and hypothesis.target_id not in str(target_identity):
        return _blocked(
            "ig target does not match the declared hypothesis target",
            ("failed-leakage:target-identity",),
        )

    flat = np.asarray(attributions, dtype=np.float64).ravel()
    attribution_norm = float(np.linalg.norm(flat))
    fidelity_threshold = hypothesis.threshold_for("completeness_error")
    fidelity_bound = float(fidelity_threshold[1]) if fidelity_threshold is not None else 0.05
    fidelity_comparator = fidelity_threshold[0] if fidelity_threshold is not None else "<="
    if attribution_norm < 1e-15:
        return _blocked(
            "ig attributions vanish: disconnected or saturated path",
            ("unsupported:vanishing-attribution",),
            unsupported=True,
        )
    fidelity_met = _passes(fidelity_comparator, abs(completeness_value), fidelity_bound)
    relative_error = abs(completeness_value) / max(1e-12, abs(delta_value))

    # Stability: cosine across declared baseline/seed attributions; missing
    # variants fail closed rather than defaulting to self-similarity.
    variants: list[np.ndarray] = []
    if baseline_attributions is not None:
        if isinstance(baseline_attributions, Mapping):
            for key in sorted(baseline_attributions):
                try:
                    variants.append(_finite_array(baseline_attributions[key], name=f"ig variant {key}").ravel())
                except ExplanationError as exc:
                    return _blocked(str(exc), ("missing-evidence:stability-variants",))
        else:
            try:
                variants.append(_finite_array(baseline_attributions, name="ig baseline_attributions").ravel())
            except ExplanationError as exc:
                return _blocked(str(exc), ("missing-evidence:stability-variants",))
    seed_variants = bundle.get("seed_attributions")
    if isinstance(seed_variants, Mapping):
        for key in sorted(seed_variants):
            try:
                variants.append(_finite_array(seed_variants[key], name=f"ig seed variant {key}").ravel())
            except ExplanationError as exc:
                return _blocked(str(exc), ("missing-evidence:stability-variants",))
    elif seed_variants is not None:
        try:
            variants.append(_finite_array(seed_variants, name="ig seed_attributions").ravel())
        except ExplanationError as exc:
            return _blocked(str(exc), ("missing-evidence:stability-variants",))
    if not variants:
        return _blocked(
            "ig stability variants are missing: declare at least one baseline/seed alternative",
            ("missing-evidence:stability-variants",),
        )
    shape_mismatch = [int(item.shape[0]) for item in variants if item.shape[0] != flat.shape[0]]
    if shape_mismatch:
        return _blocked("ig stability variants disagree on attribution shape", ("missing-evidence:stability-variants",))
    stability_cosines = [_cosine(flat, item) for item in variants]
    attribution_stability = float(min(stability_cosines)) if stability_cosines else 0.0
    stability_threshold = hypothesis.threshold_for("attribution_stability")
    stability_met = (
        True
        if stability_threshold is None
        else _passes(stability_threshold[0], attribution_stability, float(stability_threshold[1]))
    )

    # Selectivity: on-target attributions must separate from the off-target
    # and random-baseline controls by the predeclared margin.
    if off_target_attributions is None or random_attributions is None:
        return _blocked(
            "ig selectivity controls are missing: declare off-target and random-baseline attributions",
            ("missing-evidence:selectivity-controls",),
        )
    try:
        off_target = _finite_array(off_target_attributions, name="ig off_target_attributions").ravel()
        random_baseline = _finite_array(random_attributions, name="ig random_attributions").ravel()
    except ExplanationError as exc:
        return _blocked(str(exc), ("missing-evidence:selectivity-controls",))
    if off_target.shape[0] != flat.shape[0] or random_baseline.shape[0] != flat.shape[0]:
        return _blocked(
            "ig selectivity controls disagree on attribution shape", ("missing-evidence:selectivity-controls",)
        )
    off_cosine = _cosine(flat, off_target)
    random_cosine = _cosine(flat, random_baseline)
    selectivity_margin = float(min(1.0 - off_cosine, 1.0 - random_cosine))
    selectivity_threshold = hypothesis.threshold_for("selectivity_margin")
    selectivity_met = (
        True
        if selectivity_threshold is None
        else _passes(selectivity_threshold[0], selectivity_margin, float(selectivity_threshold[1]))
    )

    # Sample-level uncertainty: resample attribution coordinates without
    # recomputing any gradient, under the central repetition schedule.
    coordinate_magnitudes = np.abs(flat)

    def _attribution_draw(rng: object) -> float:
        rng = _require_generator(rng)
        positions = rng.integers(0, flat.shape[0], size=flat.shape[0])

        return float(np.sum(coordinate_magnitudes[positions]) / flat.shape[0])

    _, attribution_interval = _central_bootstrap(
        control_id="bootstrap:ig-attribution-mass",
        base_seed=int(evaluation_seed),
        seed_role="evaluation",
        repetitions=int(repetitions),
        confidence_level=float(confidence_level),
        required=False,
        kind="bootstrap",
        metric_ids=("attribution_mass",),
        expected_behavior="seeded resampling of fitted attribution coordinates",
        draw=_attribution_draw,
    )

    specs: list[_ControlSpec] = []
    supplied: dict[str, Mapping[str, float] | None] = {}
    fidelity_id = bound_controls["fidelity"]
    stability_id = bound_controls["stability"]
    selectivity_id = bound_controls["selectivity"]
    fidelity_central = "below_threshold" if fidelity_comparator == "<=" else "meets_threshold"
    specs.append(
        _ControlSpec(
            control_id=fidelity_id,
            kind="seed",
            required=True,
            metric_ids=("completeness_error",),
            expected_behavior="path-integral completeness under the declared tolerance",
            comparator=fidelity_central,  # type: ignore[arg-type]
            threshold_value=float(fidelity_bound),
        )
    )
    supplied[fidelity_id] = {"completeness_error": abs(completeness_value)}
    stability_gate = stability_threshold if stability_threshold is not None else (">=", 0.9)
    stability_central = "meets_threshold" if stability_gate[0] == ">=" else "below_threshold"
    specs.append(
        _ControlSpec(
            control_id=stability_id,
            kind="seed",
            required=True,
            metric_ids=("attribution_stability",),
            expected_behavior="attribution stability across declared baselines/seeds",
            comparator=stability_central,  # type: ignore[arg-type]
            threshold_value=float(stability_gate[1]),
        )
    )
    supplied[stability_id] = {"attribution_stability": float(attribution_stability)}
    margin_gate = selectivity_threshold if selectivity_threshold is not None else (">=", 0.3)
    margin_central = "meets_threshold" if margin_gate[0] == ">=" else "below_threshold"
    specs.append(
        _ControlSpec(
            control_id=selectivity_id,
            kind="negative",
            required=True,
            metric_ids=("selectivity_margin",),
            expected_behavior="on-target attributions separate from off-target/random baselines",
            comparator=margin_central,  # type: ignore[arg-type]
            threshold_value=float(margin_gate[1]),
        )
    )
    supplied[selectivity_id] = {"selectivity_margin": float(selectivity_margin)}
    plan = _ControlPlan(
        plan_id=f"{hypothesis.hypothesis_id}:integrated-gradients",
        repetitions=int(repetitions),
        confidence_level=float(confidence_level),
        evaluation_seed=int(evaluation_seed),
        control_seed=int(control_seed),
        controls=tuple(specs),
    )
    outcomes = _execute_plan(plan, supplied)
    central_records = {control_id: outcome.to_dict() for control_id, outcome in outcomes.items()}
    blocked = list(_failed_required(outcomes))
    control_status = {
        control_id: ("passed" if outcome.status in ("passed", "recorded") else "failed")
        for control_id, outcome in outcomes.items()
    }

    observed = {
        "completeness_error": abs(completeness_value),
        "attribution_stability": float(attribution_stability),
        "selectivity_margin": float(selectivity_margin),
        "off_target_cosine": float(off_cosine),
        "random_cosine": float(random_cosine),
    }
    fidelity = {
        "status": "passed" if fidelity_met else "failed",
        "metric": "completeness_error",
        "observed": abs(completeness_value),
        "threshold": [fidelity_comparator, float(fidelity_bound)],
        "relative_error": float(relative_error),
        "completeness_delta": float(delta_value),
    }
    stability = {
        "status": "passed" if stability_met else "failed",
        "metric": "attribution_stability",
        "observed": float(attribution_stability),
        "threshold": None if stability_threshold is None else [stability_threshold[0], float(stability_threshold[1])],
        "variant_cosines": [float(item) for item in stability_cosines],
    }
    selectivity = {
        "status": "passed" if selectivity_met else "failed",
        "metric": "selectivity_margin",
        "observed": float(selectivity_margin),
        "threshold": None
        if selectivity_threshold is None
        else [selectivity_threshold[0], float(selectivity_threshold[1])],
        "off_target_cosine": float(off_cosine),
        "random_cosine": float(random_cosine),
    }
    leakage = {
        "status": "passed" if not blocked else "failed",
        "target_identity": str(target_identity),
        "baseline_identity": str(baseline_identity),
        "input_identity": str(input_identity),
        "baseline_policy": hypothesis.baseline_policy,
    }
    uncertainty = {
        "status": "passed",
        "attribution_mass": dict(attribution_interval),
        "evaluation_seed": int(evaluation_seed),
        "repetitions": int(repetitions),
    }
    gaps: list[str] = []
    if not fidelity_met:
        gaps.append("failed-fidelity:completeness_error")
    if not stability_met:
        gaps.append("failed-stability:attribution_stability")
    if not selectivity_met:
        gaps.append("failed-selectivity:selectivity_margin")
    for control_id in blocked:
        gaps.append(f"failed-control:{control_id}")
    if gaps:
        return ExplanationEvidence(
            hypothesis_id=hypothesis.hypothesis_id,
            method="integrated_gradients",
            outcome="inconclusive",
            claim_allowed=False,
            observed_effect=dict(observed),
            fidelity=dict(fidelity),
            stability=dict(stability),
            selectivity=dict(selectivity),
            leakage=dict(leakage),
            uncertainty=dict(uncertainty),
            control_outcomes=dict(control_status),
            central_outcomes=dict(central_records),
            missing_evidence=tuple(gaps),
            limitations=("attribution is not causation; causal claims require 80.18+ trials",),
            reason=f"ig evidence blocked: {', '.join(gaps)}",
            provenance=dict(provenance),
        )
    return ExplanationEvidence(
        hypothesis_id=hypothesis.hypothesis_id,
        method="integrated_gradients",
        outcome="supported",
        claim_allowed=True,
        observed_effect=dict(observed),
        fidelity=dict(fidelity),
        stability=dict(stability),
        selectivity=dict(selectivity),
        leakage=dict(leakage),
        uncertainty=dict(uncertainty),
        control_outcomes=dict(control_status),
        central_outcomes=dict(central_records),
        missing_evidence=(),
        limitations=("attribution is not causation; causal claims require 80.18+ trials",),
        reason=(
            "declared target/baseline run meets completeness, stability, selectivity, leakage, and uncertainty gates"
        ),
        provenance=dict(provenance),
    )


# ---------------------------------------------------------------------------
# Shared evaluation seam (undeclared methods never execute)
# ---------------------------------------------------------------------------


def _omitted(hypothesis_id: str, method: str, *, reason: str) -> ExplanationEvidence:
    return ExplanationEvidence(
        hypothesis_id=hypothesis_id,
        method=method,
        outcome="omitted",
        claim_allowed=False,
        observed_effect={},
        fidelity={"status": "omitted"},
        stability={"status": "omitted"},
        selectivity={"status": "omitted"},
        leakage={"status": "omitted"},
        uncertainty={"status": "omitted"},
        control_outcomes={},
        central_outcomes={},
        missing_evidence=("omitted:undeclared-hypothesis-or-method",),
        limitations=("no method code executed for this hypothesis/method pair",),
        reason=reason,
        provenance={"explainer_version": EXPLAINER_VERSION, "method": method, "hypothesis_id": hypothesis_id},
    )


def evaluate_explanations(
    hypotheses: object,
    inputs: object,
    *,
    repetitions: object = 20,
    confidence_level: object = 0.95,
    evaluation_seed: object = 0,
    control_seed: object = 1,
    training_seed: object = 0,
) -> tuple[tuple[ExplanationEvidence, ...], dict[str, object]]:
    """Evaluate every declared hypothesis exactly once and return evidence plus payload.

    This is the single-evaluation seam every caller must use: each method
    callback fires at most once per declared hypothesis, undeclared
    hypothesis/method pairs yield honest ``omitted`` records without touching
    any callback, and one shared context feeds decisions and payload
    assembly. Uncertainty resamples already-fitted state without refitting.
    """
    declared = list(_require_hypotheses(hypotheses))
    identities = [item.hypothesis_id for item in declared]
    if len(set(identities)) != len(identities):
        raise ExplanationError("hypothesis identifiers must be unique")
    resolved_inputs = _require_inputs(inputs)
    run_repetitions = _require_count(repetitions, name="repetitions")
    if run_repetitions < 2:
        raise ExplanationError("repetitions must be at least two")
    run_confidence = _require_confidence_level(confidence_level)
    run_evaluation_seed = _require_count(evaluation_seed, name="evaluation_seed")
    run_control_seed = _require_count(control_seed, name="control_seed")
    run_training_seed = _require_count(training_seed, name="training_seed")
    for name, seed in (
        ("evaluation_seed", run_evaluation_seed),
        ("control_seed", run_control_seed),
        ("training_seed", run_training_seed),
    ):
        _non_negative_int(seed, name=name)

    evidence: list[ExplanationEvidence] = []
    for hypothesis in declared:
        if hypothesis.method not in SUPPORTED_METHODS:
            raise ExplanationError(
                f"method {hypothesis.method!r} is not an 80.16 explanation method; "
                "feature methods belong to the 80.17 feature explainer"
            )
        bundle = resolved_inputs.callback_for(hypothesis.method)
        if bundle is None:
            evidence.append(
                _omitted(
                    hypothesis.hypothesis_id,
                    hypothesis.method,
                    reason=(
                        f"method {hypothesis.method!r} has no declared input "
                        f"for hypothesis {hypothesis.hypothesis_id!r}; callback uncalled"
                    ),
                )
            )
            continue
        resolved = bundle
        if hypothesis.method == "probe":
            evidence.append(
                _evaluate_probe(
                    hypothesis,
                    resolved,
                    repetitions=run_repetitions,
                    confidence_level=run_confidence,
                    evaluation_seed=run_evaluation_seed,
                    control_seed=run_control_seed,
                    training_seed=run_training_seed,
                )
            )
        elif hypothesis.method == "tcav":
            evidence.append(
                _evaluate_tcav(
                    hypothesis,
                    resolved,
                    repetitions=run_repetitions,
                    confidence_level=run_confidence,
                    evaluation_seed=run_evaluation_seed,
                    control_seed=run_control_seed,
                )
            )
        else:
            evidence.append(
                _evaluate_integrated_gradients(
                    hypothesis,
                    resolved,
                    repetitions=run_repetitions,
                    confidence_level=run_confidence,
                    evaluation_seed=run_evaluation_seed,
                    control_seed=run_control_seed,
                )
            )
    context = _ExplainContext(
        hypotheses=declared,
        evidence=tuple(evidence),
        central_outcomes={record.hypothesis_id: dict(record.central_outcomes) for record in evidence},
        control_outcomes={record.hypothesis_id: dict(record.control_outcomes) for record in evidence},
    )
    payload = explanation_payload(context)
    return tuple(evidence), payload


def explanation_payload(context: object) -> dict[str, object]:
    """Assemble the canonical machine-readable explain-stage payload."""
    if not isinstance(context, _ExplainContext):
        raise ExplanationError("context must be a _ExplainContext")
    typed = context
    hypotheses = _require_hypotheses(typed.hypotheses)
    records_evidence = _require_evidence_items(typed.evidence)
    records = [record.to_dict() for record in records_evidence]
    try:
        canonical_json(records)
    except PortableNodeError as exc:
        raise ExplanationError(f"explanation records are not canonical JSON: {exc}") from exc
    central = _require_central_outcomes(typed.central_outcomes)
    controls = _require_control_outcomes(typed.control_outcomes)
    return {
        "explainer_version": EXPLAINER_VERSION,
        "hypotheses": [hypothesis.to_dict() for hypothesis in hypotheses],
        "evidence": records,
        "family_evidence": {
            record.hypothesis_id: {
                "outcome": record.outcome,
                "claim_allowed": record.claim_allowed,
                "method": record.method,
            }
            for record in records_evidence
        },
        "controls": {key: dict(entry) if isinstance(entry, Mapping) else {} for key, entry in central.items()},
        "control_outcomes": {key: dict(entry) for key, entry in controls.items()},
    }


def explanation_report_items(evidence: object) -> list[dict[str, object]]:
    """Render evidence records as shape-compatible report claim fragments.

    Phase-IV boundary (not 80.21/80.22): items pass ``validate_report_shape``
    but are NOT independently consumable by ``validate_diagnostic_report``.
    Their ``evidence_refs`` are provisional (``explanation-<id>-record`` names
    no registered report row or digest artifact yet) and ``control_refs``
    name hypothesis-declared controls, not manifest-registered controls; a
    promoted explanation claim additionally has no metric-bearing
    statistical-evidence row linkage. Later report assembly (80.21/80.22)
    must register evidence/control refs and emit metric-bearing
    statistical-evidence rows before the frozen validator can consume them.
    Every item carries ``kind="explanation"`` with ``causal=False``; only
    ``supported`` records allow the claim, and every record names its
    missing evidence explicitly. Unsupported/omitted records map to the
    ``unsupported`` claim status so the report validator fails closed on
    any promotion attempt.
    """
    raw_records = _require_evidence_items(evidence)
    items: list[dict[str, object]] = []
    for record in raw_records:
        status = (
            "supported"
            if record.outcome == "supported"
            else ("inconclusive" if record.outcome == "inconclusive" else "unsupported")
        )
        items.append(
            {
                "id": f"explanation-{record.hypothesis_id}",
                "kind": "explanation",
                "status": status,
                "claim": (f"{record.method} explanation for hypothesis {record.hypothesis_id}: {record.reason}"),
                "evidence_refs": [f"explanation-{record.hypothesis_id}-record"],
                "control_refs": sorted(record.control_outcomes),
                "causal": False,
                "claim_allowed": bool(record.claim_allowed),
                "missing_evidence": list(record.missing_evidence),
            }
        )
    return items


def make_explain_executor(
    hypotheses: object,
    inputs: object,
    *,
    repetitions: object = 20,
    confidence_level: object = 0.95,
    evaluation_seed: object = 0,
    control_seed: object = 1,
    training_seed: object = 0,
    version: object = EXPLAINER_VERSION,
) -> Any:
    """Build a supplied ``explain``-stage executor bound to declared hypotheses.

    The coordinator invokes the returned callable with a ``StageInvocation``;
    the callable binds the expected manifest/run identity (run manifest,
    capture representation, manifest dataset split, plus upstream diagnostic
    linkage: hypothesis family against the structured detect payload family
    identities, declared metrics against the structured detect payload metric
    identities, and the exact declared localize ``family_id``/``metric_id``
    only when the prior localize payload declares them), then exact-matches
    every entry of the non-empty ``localization_bindings`` contract against
    the structured prior localization result, evaluates
    :func:`evaluate_explanations` once, and returns a ``completed``
    :class:`StageOutput` whose payload carries hypotheses, evidence records,
    control outcomes, and uncertainty. ``localization_bindings`` is the only
    upstream localization contract; ``layer_id``/``slice_id`` (like
    ``symptom_id``/``target_id``) are hypothesis-local method context
    validated by method inputs, never upstream-localized by their mere
    presence. Missing required upstream fields fail closed. No algorithm
    enters ``DiagnosticWorkflow`` itself.
    """
    _non_empty_string(version, name="version")
    raw_frozen = _require_hypotheses(hypotheses)
    frozen: list[ExplanationHypothesis] = []
    for item in raw_frozen:
        if item.method not in SUPPORTED_METHODS:
            raise ExplanationError(f"method {item.method!r} is not an 80.16 explanation method")
        frozen.append(item)
    resolved_inputs = _require_inputs(inputs)

    def _execute(invocation: StageInvocation) -> StageOutput:
        from latent_anything._diagnostic_workflow import StageContractError as _ContractError

        if invocation.stage != "explain":
            raise _ContractError(f"explain executor received stage {invocation.stage!r}")
        _check_explain_identities(frozen, invocation)
        _, payload = evaluate_explanations(
            tuple(frozen),
            resolved_inputs,
            repetitions=_require_count(repetitions, name="repetitions"),
            confidence_level=_require_confidence_level(confidence_level),
            evaluation_seed=_require_count(evaluation_seed, name="evaluation_seed"),
            control_seed=_require_count(control_seed, name="control_seed"),
            training_seed=_require_count(training_seed, name="training_seed"),
        )
        aggregate: Literal["completed", "unsupported"] = "completed"
        raw_evidence = _require_payload_evidence(payload.get("evidence"))
        rows = list(raw_evidence)
        if rows and all(str(item.get("outcome")) == "omitted" for item in rows):
            aggregate = "unsupported"
        return StageOutput(stage="explain", outcome=aggregate, payload=payload, artifact_refs=())

    _execute.explainer_version = version  # type: ignore[attr-defined]
    return _execute


__all__ = [
    "DECLARED_PROBE_CAPACITY",
    "EXPLAINER_VERSION",
    "SUPPORTED_METHODS",
    "EvidenceOutcome",
    "ExplanationError",
    "ExplanationEvidence",
    "ExplanationHypothesis",
    "MethodInputs",
    "evaluate_explanations",
    "explanation_payload",
    "explanation_report_items",
    "make_explain_executor",
]

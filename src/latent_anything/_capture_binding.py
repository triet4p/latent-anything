"""Architecture-neutral capture-axis binding (Sprint 80.7).

Turns a domain :class:`CaptureSelection` plus a frozen benchmark manifest into
deterministic capture identities and typed axis/provenance metadata, then wraps
existing :class:`CapturedActivation` records into the existing
:class:`LatentValue` and :class:`Trajectory` primitives.

One generic path serves both core cases (encoder bottleneck and transformer
hidden states): the feature dimension is always the last array axis, batch and
sequence roles are assigned from array rank alone, and layer/slice/checkpoint
remain selection (non-array) axes. There is no architecture-specific
branching in this module.

Scope: binding only. No workflow orchestration (80.8), no detector
algorithms, no benchmark execution, no localization algorithms, and no
changes to manifest thresholds.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from latent_anything._portable_contract import canonical_json
from latent_anything.capture import CapturedActivation
from latent_anything.diagnostics import CaptureSelection
from latent_anything.latent_space import LatentSpace
from latent_anything.latent_value import LatentValue
from latent_anything.trajectory import Trajectory


class CaptureBindingError(ValueError):
    """Raised when a capture selection cannot be bound fail-closed."""


SUPPORTED_AXES: frozenset[str] = frozenset(
    {"sample", "slice", "token", "time", "checkpoint", "layer", "module", "feature"}
)
"""Axis names the binder understands. Anything else is rejected, not guessed."""

_SEQUENCE_AXES: frozenset[str] = frozenset({"token", "time"})
_SELECTION_AXES: frozenset[str] = frozenset({"layer", "module", "checkpoint"})
_TRAJECTORY_AXES: frozenset[str] = frozenset({"token", "time"})


def _non_empty_string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CaptureBindingError(f"{name} must be a non-empty string")
    return value


def _manifest_section(manifest: Mapping[str, object], key: str) -> Mapping[str, object]:
    section = manifest.get(key)
    return _require_section(section, name=f"manifest {key}")


def _require_plan(value: object) -> BoundCapture:
    if not isinstance(value, BoundCapture):
        raise CaptureBindingError("plan must be a BoundCapture")
    return value


def _require_bound(value: object) -> BoundCapture:
    if not isinstance(value, BoundCapture):
        raise CaptureBindingError("bound must be a BoundCapture")
    return value


def _require_value(value: object) -> LatentValue:
    if not isinstance(value, LatentValue):
        raise CaptureBindingError("value must be a LatentValue")
    return value


def _require_captured(value: object) -> CapturedActivation:
    if not isinstance(value, CapturedActivation):
        raise CaptureBindingError("captured must be a CapturedActivation")
    return value


def _require_captures(value: object) -> tuple[CapturedActivation, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not value:
        raise CaptureBindingError("captures must be a non-empty sequence")
    items = tuple(value)
    for item in items:
        if not isinstance(item, CapturedActivation):
            raise CaptureBindingError("captures must be CapturedActivation items")
    return items


def _require_array(value: object) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise CaptureBindingError("captured values must be a NumPy array")
    return value


def _require_selection(value: object) -> CaptureSelection:
    if not isinstance(value, CaptureSelection):
        raise CaptureBindingError("selection must be a CaptureSelection")
    return value


def _require_section(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise CaptureBindingError(f"{name} must be an object")
    return cast(Mapping[str, object], value)


def _require_positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise CaptureBindingError(f"{name} must be a positive integer")
    return int(value)


def _require_optional_positive_int(value: object, *, name: str) -> int | None:
    if value is None:
        return None
    return _require_positive_int(value, name=name)


def _require_optional_non_negative_int(value: object, *, name: str) -> int | None:
    if value is None:
        return None
    return _require_non_negative_int(value, name=name)


def _require_non_negative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CaptureBindingError(f"{name} must be a non-negative integer")
    return int(value)


@dataclass(frozen=True)
class AxisBinding:
    """One bound axis: requested name plus deterministic selection provenance."""

    name: str
    selection: str
    identity: str
    size: object = None
    axis_index: object = None

    def __post_init__(self) -> None:
        _non_empty_string(self.name, name="axis name")
        _non_empty_string(self.selection, name=f"axis {self.name} selection")
        _non_empty_string(self.identity, name=f"axis {self.name} identity")
        if self.name not in SUPPORTED_AXES:
            raise CaptureBindingError(f"unsupported axis: {self.name!r}")
        size = _require_optional_positive_int(self.size, name=f"axis {self.name} size")
        axis_index = _require_optional_non_negative_int(self.axis_index, name=f"axis {self.name} index")
        object.__setattr__(self, "size", size)
        object.__setattr__(self, "axis_index", axis_index)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible mapping of this axis binding."""
        return {
            "name": self.name,
            "selection": self.selection,
            "identity": self.identity,
            "size": self.size,
            "axis_index": self.axis_index,
        }


@dataclass(frozen=True)
class BoundCapture:
    """Deterministic identity plus complete provenance for one bound capture."""

    capture_identity: str
    capture_id: str
    representation_identity: str
    manifest_id: str
    request_id: str
    model_id: str
    model_revision: str
    dataset_id: str
    dataset_revision: str
    dataset_split: str
    split_identity: str
    axes: tuple[AxisBinding, ...]
    absent_axes: tuple[str, ...]
    manifest_axes: tuple[AxisBinding, ...]
    location: str | None = None
    call_index: int | None = None
    order_index: int | None = None
    total: int | None = None
    dtype: str | None = None
    shape: tuple[int, ...] | None = None
    device: str | None = None

    def __post_init__(self) -> None:
        _non_empty_string(self.capture_identity, name="capture_identity")
        _non_empty_string(self.capture_id, name="capture_id")
        _non_empty_string(self.representation_identity, name="representation_identity")
        _non_empty_string(self.manifest_id, name="manifest_id")
        _non_empty_string(self.request_id, name="request_id")
        if not self.axes:
            raise CaptureBindingError("bound axes must not be empty")

    @property
    def is_plan(self) -> bool:
        """Whether this is a plan-level binding without observed array data."""
        return self.shape is None

    def provenance(self) -> dict[str, object]:
        """Return a JSON-compatible provenance mapping for primitive metadata."""
        capture_point: dict[str, object] | None = None
        if self.location is not None and self.call_index is not None:
            capture_point = {"location": self.location, "call_index": self.call_index}
        ordering: dict[str, object] | None = None
        if self.order_index is not None and self.total is not None:
            ordering = {"order_index": self.order_index, "total": self.total}
        return {
            "capture_identity": self.capture_identity,
            "capture_id": self.capture_id,
            "source_representation_identity": self.representation_identity,
            "representation_identity": self.representation_identity,
            "manifest_id": self.manifest_id,
            "request_id": self.request_id,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "model_version": self.model_revision,
            "revision": self.model_revision,
            "dataset_id": self.dataset_id,
            "dataset_revision": self.dataset_revision,
            "dataset_split": self.dataset_split,
            "split_identity": self.split_identity,
            "axes": [axis.name for axis in self.axes],
            "axis_details": [axis.to_dict() for axis in self.axes],
            "absent_axes": list(self.absent_axes),
            "manifest_axes": [axis.to_dict() for axis in self.manifest_axes],
            "capture_point": capture_point,
            "ordering": ordering,
            "dtype": self.dtype,
            "shape": list(self.shape) if self.shape is not None else None,
            "device": self.device,
        }


def _axis_table(representation: Mapping[str, object]) -> dict[str, dict[str, str]]:
    raw_axes = representation.get("axes")
    if isinstance(raw_axes, (str, bytes)) or not isinstance(raw_axes, Sequence) or not raw_axes:
        raise CaptureBindingError("manifest representation.axes must be a non-empty list")
    table: dict[str, dict[str, str]] = {}
    identities: set[str] = set()
    for index, raw_axis in enumerate(raw_axes):
        if not isinstance(raw_axis, Mapping):
            raise CaptureBindingError(f"manifest representation.axes[{index}] must be an object")
        axis = cast(Mapping[str, object], raw_axis)
        name = _non_empty_string(axis.get("name"), name=f"manifest representation.axes[{index}].name")
        selection = _non_empty_string(axis.get("selection"), name=f"manifest representation.axes[{index}].selection")
        identity = _non_empty_string(axis.get("identity"), name=f"manifest representation.axes[{index}].identity")
        if name in table:
            raise CaptureBindingError(f"manifest declares duplicate axis name: {name!r}")
        if identity in identities:
            raise CaptureBindingError(f"manifest declares duplicate axis identity: {identity!r}")
        identities.add(identity)
        table[name] = {"selection": selection, "identity": identity}
    return table


def _digest(payload: object) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def bind_selection(
    selection: object,
    *,
    manifest: Mapping[str, object],
    request_id: str,
) -> BoundCapture:
    """Bind one domain selection plus manifest source identity to a plan.

    Returns a plan-level :class:`BoundCapture` (no observed dtype/shape/device)
    with a deterministic ``capture_identity``. Array data is bound later via
    :func:`resolve_capture` / :func:`resolve_captures` so the same generic path
    serves the encoder bottleneck and transformer hidden-state selections.
    """
    selection = _require_selection(selection)
    _require_section(manifest, name="manifest")
    clean_request = _non_empty_string(request_id, name="request_id")
    manifest_id = _non_empty_string(manifest.get("manifest_id"), name="manifest.manifest_id")

    representation = _manifest_section(manifest, "representation")
    expected_identity = _non_empty_string(representation.get("identity"), name="manifest representation.identity")
    if selection.representation_identity != expected_identity:
        raise CaptureBindingError(
            "provenance mismatch: selection representation_identity "
            f"{selection.representation_identity!r} != manifest {expected_identity!r}"
        )
    table = _axis_table(representation)

    requested = tuple(selection.axes)
    if not requested:
        raise CaptureBindingError("requested axes must not be empty")
    if len(set(requested)) != len(requested):
        raise CaptureBindingError(f"requested axes must not contain duplicates: {list(requested)!r}")
    for name in requested:
        if name not in SUPPORTED_AXES:
            raise CaptureBindingError(f"unsupported axis: {name!r}")
    if "layer" in requested and "module" in requested:
        raise CaptureBindingError("ambiguous axes: 'layer' and 'module' name the same seam")

    model = _manifest_section(manifest, "model")
    dataset = _manifest_section(manifest, "dataset")
    model_id = _non_empty_string(model.get("id"), name="manifest model.id")
    model_revision = _non_empty_string(model.get("revision"), name="manifest model.revision")
    dataset_id = _non_empty_string(dataset.get("id"), name="manifest dataset.id")
    dataset_revision = _non_empty_string(dataset.get("revision"), name="manifest dataset.revision")
    dataset_split = _non_empty_string(dataset.get("split"), name="manifest dataset.split")
    split_identity = _non_empty_string(dataset.get("split_identity"), name="manifest dataset.split_identity")

    axes: list[AxisBinding] = []
    for name in requested:
        declared = table.get(name)
        if declared is not None:
            axes.append(AxisBinding(name=name, selection=declared["selection"], identity=declared["identity"]))
        elif name == "feature":
            axes.append(
                AxisBinding(
                    name=name,
                    selection="all-features",
                    identity=f"{expected_identity}:feature-axis",
                )
            )
        else:
            raise CaptureBindingError(
                f"provenance mismatch: requested axis {name!r} is not declared by manifest {manifest_id!r}"
            )

    covered = set(requested)
    if "layer" in covered or "module" in covered:
        covered |= {"layer", "module"}
    absent = tuple(sorted(SUPPORTED_AXES - covered))

    manifest_axes = tuple(
        AxisBinding(name=name, selection=entry["selection"], identity=entry["identity"])
        for name, entry in table.items()
    )
    identity_payload = {
        "capture_id": selection.capture_id,
        "representation_identity": expected_identity,
        "manifest_id": manifest_id,
        "model_id": model_id,
        "model_revision": model_revision,
        "dataset_id": dataset_id,
        "dataset_revision": dataset_revision,
        "split_identity": split_identity,
        "axes": sorted(
            ({"identity": axis.identity, "name": axis.name, "selection": axis.selection} for axis in axes),
            key=lambda row: cast(str, row["name"]),
        ),
    }
    return BoundCapture(
        capture_identity=_digest(identity_payload),
        capture_id=selection.capture_id,
        representation_identity=expected_identity,
        manifest_id=manifest_id,
        request_id=clean_request,
        model_id=model_id,
        model_revision=model_revision,
        dataset_id=dataset_id,
        dataset_revision=dataset_revision,
        dataset_split=dataset_split,
        split_identity=split_identity,
        axes=tuple(axes),
        absent_axes=absent,
        manifest_axes=manifest_axes,
    )


def _assign_indices(names: tuple[str, ...], ndim: int) -> dict[str, int | None]:
    """Assign array-axis indices from rank alone; selection axes stay ``None``."""
    if ndim < 1:
        raise CaptureBindingError(f"incompatible shape: rank {ndim} carries no feature axis")
    if ndim > 3:
        raise CaptureBindingError(f"incompatible shape: rank {ndim} exceeds the bound sample/sequence/feature layout")
    requested = set(names)
    has_batch = "sample" in requested or "slice" in requested
    sequences = [name for name in names if name in _SEQUENCE_AXES]
    if len(sequences) > 1:
        raise CaptureBindingError(f"ambiguous axes: {sequences!r} cannot share one sequence dimension")
    sequence = sequences[0] if sequences else None

    indices: dict[str, int | None] = dict.fromkeys(names)
    indices["feature"] = ndim - 1 if "feature" in requested else None
    for name in names:
        if name in _SELECTION_AXES or name == "checkpoint":
            indices[name] = None
    if ndim == 1:
        if has_batch or sequence is not None:
            raise CaptureBindingError("incompatible shape: rank 1 carries feature only, not sample/sequence axes")
        return indices
    if ndim == 2:
        roles = (1 if has_batch else 0) + (1 if sequence is not None else 0)
        if roles != 1:
            raise CaptureBindingError(
                "ambiguous axes: rank 2 carries exactly one of batch (sample/slice)"
                " or sequence (token/time) plus feature"
            )
        target = 0
        for name in names:
            if name == "feature" or name in _SELECTION_AXES or name == "checkpoint":
                continue
            indices[name] = target
        return indices
    # ndim == 3: batch (sample and/or slice share axis 0), one sequence axis, feature last.
    if not has_batch:
        raise CaptureBindingError("incompatible shape: rank 3 requires a batch axis (sample or slice) plus feature")
    if sequence is None:
        raise CaptureBindingError("incompatible shape: rank 3 requires one sequence axis (token or time) plus feature")
    for name in names:
        if name in {"sample", "slice"}:
            indices[name] = 0
        elif name in _SEQUENCE_AXES:
            indices[name] = 1
    return indices


def _concrete_identity(
    plan: BoundCapture, *, location: str, call_index: int, shape: tuple[int, ...], dtype: str
) -> str:
    return _digest(
        {
            "plan_identity": plan.capture_identity,
            "location": location,
            "call_index": call_index,
            "shape": list(shape),
            "dtype": dtype,
        }
    )


def resolve_capture(
    plan: object,
    captured: object,
    *,
    order_index: int = 0,
    total: int = 1,
) -> tuple[BoundCapture, LatentValue]:
    """Resolve one captured record against a plan into a bound ``LatentValue``.

    The tensor is passed straight into the existing ``LatentValue`` primitive
    (which owns its single copy); this binder performs no additional copy.
    """
    plan = _require_plan(plan)
    if not plan.is_plan:
        raise CaptureBindingError("resolve_capture requires a plan-level binding without observed shape")
    captured = _require_captured(captured)
    if order_index < 0 or total < 1 or order_index >= total:
        raise CaptureBindingError("ordering must satisfy 0 <= order_index < total with total >= 1")

    values = _require_array(captured.values)
    shape = tuple(int(size) for size in values.shape)
    if tuple(int(size) for size in captured.metadata.shape) != shape:
        raise CaptureBindingError("incompatible shape/axis metadata: capture metadata shape differs from values shape")

    version = captured.metadata.source_model_version
    if version and plan.model_revision not in version and version != plan.model_id:
        raise CaptureBindingError(
            "provenance mismatch: capture source_model_version matches neither model id nor revision"
        )
    location = _non_empty_string(captured.metadata.location, name="capture location")

    indices = _assign_indices(tuple(axis.name for axis in plan.axes), values.ndim)
    axes: list[AxisBinding] = []
    for axis in plan.axes:
        index = indices[axis.name]
        axes.append(
            AxisBinding(
                name=axis.name,
                selection=axis.selection,
                identity=axis.identity,
                size=shape[index] if index is not None else None,
                axis_index=index,
            )
        )

    concrete = BoundCapture(
        capture_identity=_concrete_identity(
            plan, location=location, call_index=captured.metadata.call_index, shape=shape, dtype=captured.metadata.dtype
        ),
        capture_id=plan.capture_id,
        representation_identity=plan.representation_identity,
        manifest_id=plan.manifest_id,
        request_id=plan.request_id,
        model_id=plan.model_id,
        model_revision=plan.model_revision,
        dataset_id=plan.dataset_id,
        dataset_revision=plan.dataset_revision,
        dataset_split=plan.dataset_split,
        split_identity=plan.split_identity,
        axes=tuple(axes),
        absent_axes=plan.absent_axes,
        manifest_axes=plan.manifest_axes,
        location=location,
        call_index=captured.metadata.call_index,
        order_index=order_index,
        total=total,
        dtype=captured.metadata.dtype,
        shape=shape,
        device=captured.metadata.device,
    )
    space = LatentSpace(
        dim=shape[-1],
        source_model=plan.model_id,
        metadata={
            "source_representation_identity": plan.representation_identity,
            "model_version": plan.model_revision,
            "role": "diagnostic-capture",
        },
    )
    value = LatentValue(values, space, concrete.provenance())
    return concrete, value


def resolve_captures(plan: object, captures: object) -> tuple[tuple[BoundCapture, ...], tuple[LatentValue, ...]]:
    """Resolve captures in forward execution order, preserving alignment."""
    plan = _require_plan(plan)
    captures = _require_captures(captures)
    points: set[tuple[str, int]] = set()
    shapes: set[tuple[int, ...]] = set()
    dtypes: set[str] = set()
    for item in captures:
        point = (item.metadata.location, item.metadata.call_index)
        if point in points:
            raise CaptureBindingError(f"duplicate capture point: {point!r}")
        points.add(point)
        shapes.add(tuple(int(size) for size in item.values.shape))
        dtypes.add(item.metadata.dtype)
    if len(shapes) != 1:
        raise CaptureBindingError(
            f"incompatible shape/axis metadata: layer captures disagree on shape {sorted(shapes)!r}"
        )
    if len(dtypes) != 1:
        raise CaptureBindingError(
            f"incompatible shape/axis metadata: layer captures disagree on dtype {sorted(dtypes)!r}"
        )
    bounds: list[BoundCapture] = []
    values: list[LatentValue] = []
    total = len(captures)
    for order_index, item in enumerate(captures):
        bound, value = resolve_capture(plan, item, order_index=order_index, total=total)
        bounds.append(bound)
        values.append(value)
    return tuple(bounds), tuple(values)


def bound_trajectory(bound: object, value: object, *, axis: str) -> Trajectory:
    """Project one resolved 2D value onto an explicit sequence axis.

    Only the declared ``token``/``time`` axes convert; every other axis name
    (including ``sample``) is an explicit non-applicability rejection, never an
    empty success. Rank-3 values must be sliced to one sample first via the
    existing ``LatentValue`` indexing rather than squeezed silently here.
    """
    bound = _require_bound(bound)
    value = _require_value(value)
    if bound.is_plan or bound.shape is None:
        raise CaptureBindingError("bound_trajectory requires a resolved capture with observed shape")
    if axis not in _TRAJECTORY_AXES:
        raise CaptureBindingError(f"axis {axis!r} is non-applicable for trajectory binding: requires 'token' or 'time'")
    if axis not in {entry.name for entry in bound.axes}:
        raise CaptureBindingError(f"axis {axis!r} is non-applicable: it is absent from this bound capture")
    if value.shape[-1] != bound.shape[-1]:
        raise CaptureBindingError(
            "incompatible shape/axis metadata: value feature width differs from the bound capture"
        )
    if len(value.shape) != 2:
        raise CaptureBindingError(
            "incompatible shape/axis metadata: trajectory binding requires one 2D (n_points, dim) value"
        )
    metadata: dict[str, Any] = dict(bound.provenance())
    metadata["trajectory_axis"] = axis
    return Trajectory(value.to_numpy(), metadata=metadata)


__all__ = [
    "AxisBinding",
    "BoundCapture",
    "CaptureBindingError",
    "SUPPORTED_AXES",
    "bind_selection",
    "bound_trajectory",
    "resolve_capture",
    "resolve_captures",
]

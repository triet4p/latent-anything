"""Internal recursive handlers for portable manifest nodes."""

from __future__ import annotations

import base64
from collections.abc import Mapping
from typing import Any, cast

import numpy as np

from latent_anything._portable_contract import (
    PortableLimits,
    PortableNodeError,
    canonical_json,
    checked_shape,
    expected_array_bytes,
)
from latent_anything.covariance import CovarianceState
from latent_anything.latent_space import LatentSpace
from latent_anything.latent_value import LatentValue
from latent_anything.trajectory import Trajectory


class PortableEncoder:
    def __init__(self, limits: PortableLimits) -> None:
        self.limits = limits
        self.arrays: list[tuple[str, tuple[int, ...], bytes]] = []
        self.total_array_bytes = 0
        self.nodes = 0
        self.active: set[int] = set()

    def _visit(self, depth: int) -> None:
        if depth > self.limits.max_depth:
            raise PortableNodeError("portable value exceeds maximum nesting depth")
        self.nodes += 1
        if self.nodes > self.limits.max_nodes:
            raise PortableNodeError("portable value exceeds maximum node count")

    def _enter(self, value: object) -> int:
        identity = id(value)
        if identity in self.active:
            raise PortableNodeError("cyclic portable values are not supported")
        self.active.add(identity)
        return identity

    def _leave(self, identity: int) -> None:
        self.active.remove(identity)

    def _array(self, value: np.ndarray) -> dict[str, object]:
        if value.dtype.hasobject:
            raise PortableNodeError("object-dtype NumPy arrays are not portable")
        contiguous = np.ascontiguousarray(value)
        payload = contiguous.tobytes(order="C")
        if len(payload) > self.limits.max_array_bytes:
            raise PortableNodeError("array exceeds maximum per-array byte limit")
        self.total_array_bytes += len(payload)
        if self.total_array_bytes > self.limits.max_total_array_bytes:
            raise PortableNodeError("arrays exceed maximum total byte limit")
        array_id = len(self.arrays)
        shape = checked_shape(value.shape, self.limits)
        self.arrays.append((value.dtype.str, shape, payload))
        return {"kind": "ndarray", "array_id": array_id}

    def value(self, value: object, depth: int = 0) -> object:
        self._visit(depth)
        if value is None or isinstance(value, str | bool | int):
            return value
        if isinstance(value, float):
            if not np.isfinite(value):
                raise PortableNodeError("non-finite metadata floats are not portable")
            return value
        if isinstance(value, np.ndarray):
            return self._array(value)
        if isinstance(value, np.generic):
            return self.value(value.item(), depth + 1)
        if isinstance(value, bytes):
            return {"kind": "bytes", "value": base64.b64encode(value).decode("ascii")}
        if isinstance(value, LatentValue):
            identity = self._enter(value)
            try:
                return {
                    "kind": "latent_value",
                    "data": self.value(value.to_numpy(), depth + 1),
                    "space": self.value(value.space, depth + 1),
                    "metadata": self.value(dict(value.metadata), depth + 1),
                }
            finally:
                self._leave(identity)
        if isinstance(value, Trajectory):
            identity = self._enter(value)
            try:
                return {
                    "kind": "trajectory",
                    "data": self.value(value.to_numpy(), depth + 1),
                    "metadata": self.value(dict(value.metadata), depth + 1),
                }
            finally:
                self._leave(identity)
        if isinstance(value, LatentSpace):
            identity = self._enter(value)
            try:
                space_state = cast(Any, value)
                payload: dict[str, object] = {
                    "kind": "latent_space",
                    "dim": value.dim,
                    "geometry": value.geometry,
                    "source_model": value.source_model,
                    "metadata": self.value(value.metadata, depth + 1),
                    "n_gaussians": value.n_gaussians,
                    "position_dim": space_state._position_dim,
                    "scale_dim": space_state._scale_dim,
                    "color_dim": space_state._color_dim,
                    "codebook_size": value.codebook_size,
                }
                if value.covariance is not None:
                    payload["covariance"] = self.value(value.covariance, depth + 1)
                return payload
            finally:
                self._leave(identity)
        if isinstance(value, CovarianceState):
            identity = self._enter(value)
            try:
                return {
                    "kind": "covariance_state",
                    "mean": self.value(value.mean, depth + 1),
                    "covariance": self.value(value.covariance, depth + 1),
                    "n_samples": value.n_samples,
                    "source_representation_identity": value.source_representation_identity,
                    "reg_coef": value.reg_coef,
                    "provenance": self.value(dict(value.provenance), depth + 1),
                }
            finally:
                self._leave(identity)
        if isinstance(value, Mapping):
            identity = self._enter(value)
            try:
                result: dict[str, object] = {}
                for key in sorted(value, key=lambda item: str(item)):
                    if not isinstance(key, str):
                        raise PortableNodeError("portable mapping keys must be strings")
                    result[key] = self.value(value[key], depth + 1)
                return {"kind": "mapping", "items": result}
            finally:
                self._leave(identity)
        if isinstance(value, tuple):
            identity = self._enter(value)
            try:
                return {"kind": "tuple", "items": [self.value(item, depth + 1) for item in value]}
            finally:
                self._leave(identity)
        if isinstance(value, list):
            identity = self._enter(value)
            try:
                return {"kind": "list", "items": [self.value(item, depth + 1) for item in value]}
            finally:
                self._leave(identity)
        if isinstance(value, frozenset):
            identity = self._enter(value)
            try:
                items = [self.value(item, depth + 1) for item in value]
                items.sort(key=canonical_json)
                return {"kind": "frozenset", "items": items}
            finally:
                self._leave(identity)
        raise PortableNodeError(f"unsupported portable value type: {type(value).__name__}")


class PortableDecoder:
    def __init__(self, arrays: list[tuple[str, tuple[int, ...], bytes]], limits: PortableLimits) -> None:
        self.arrays = arrays
        self.limits = limits
        self.nodes = 0
        self.total_array_bytes = 0

    def value(self, value: object, depth: int = 0) -> object:
        if depth > self.limits.max_depth:
            raise PortableNodeError("portable value exceeds maximum nesting depth")
        self.nodes += 1
        if self.nodes > self.limits.max_nodes:
            raise PortableNodeError("portable value exceeds maximum node count")
        if value is None or isinstance(value, str | bool | int | float):
            return value
        if not isinstance(value, dict):
            raise PortableNodeError("portable manifest node must be a JSON object")
        kind = value.get("kind")
        if kind == "ndarray":
            raw_id = value.get("array_id")
            if isinstance(raw_id, bool) or not isinstance(raw_id, int) or raw_id < 0 or raw_id >= len(self.arrays):
                raise PortableNodeError("portable array reference is invalid")
            dtype_text, shape, payload = self.arrays[raw_id]
            if len(payload) > self.limits.max_array_bytes:
                raise PortableNodeError("array exceeds maximum per-array byte limit")
            self.total_array_bytes += len(payload)
            if self.total_array_bytes > self.limits.max_total_array_bytes:
                raise PortableNodeError("arrays exceed maximum total byte limit")
            try:
                dtype = np.dtype(dtype_text)
            except (TypeError, ValueError) as exc:
                raise PortableNodeError("portable array dtype is invalid") from exc
            if dtype.hasobject:
                raise PortableNodeError("object-dtype NumPy arrays are not portable")
            expected = expected_array_bytes(shape, dtype)
            if expected > self.limits.max_total_array_bytes:
                raise PortableNodeError("array shape exceeds portable allocation guard")
            if expected != len(payload):
                raise PortableNodeError("portable array payload length does not match dtype and shape")
            return np.frombuffer(payload, dtype=dtype).reshape(shape).copy()
        if kind == "bytes":
            encoded = value.get("value")
            if not isinstance(encoded, str):
                raise PortableNodeError("portable bytes node is malformed")
            try:
                return base64.b64decode(encoded.encode("ascii"), validate=True)
            except (ValueError, UnicodeEncodeError) as exc:
                raise PortableNodeError("portable bytes node is not valid base64") from exc
        if kind == "mapping":
            items = value.get("items")
            if not isinstance(items, dict):
                raise PortableNodeError("portable mapping node is malformed")
            return {key: self.value(item, depth + 1) for key, item in items.items()}
        if kind in {"list", "tuple", "frozenset"}:
            items = value.get("items")
            if not isinstance(items, list):
                raise PortableNodeError("portable sequence node is malformed")
            decoded = [self.value(item, depth + 1) for item in items]
            if kind == "tuple":
                return tuple(decoded)
            if kind == "frozenset":
                return frozenset(decoded)
            return decoded
        if kind == "latent_space":
            metadata = self.value(value.get("metadata"), depth + 1)
            if not isinstance(metadata, dict):
                raise PortableNodeError("latent space metadata is malformed")
            kwargs = {
                "dim": value.get("dim"),
                "geometry": value.get("geometry"),
                "source_model": value.get("source_model"),
                "metadata": metadata,
                "n_gaussians": value.get("n_gaussians"),
                "position_dim": value.get("position_dim"),
                "scale_dim": value.get("scale_dim"),
                "color_dim": value.get("color_dim"),
                "codebook_size": value.get("codebook_size"),
            }
            covariance = value.get("covariance")
            if covariance is not None:
                decoded_covariance = self.value(covariance, depth + 1)
                if not isinstance(decoded_covariance, CovarianceState):
                    raise PortableNodeError("latent space covariance is malformed")
                kwargs["covariance"] = decoded_covariance
            try:
                return LatentSpace(**cast(Any, kwargs))
            except (TypeError, ValueError) as exc:
                raise PortableNodeError(f"latent space is invalid: {exc}") from exc
        if kind == "covariance_state":
            mean = self.value(value.get("mean"), depth + 1)
            covariance = self.value(value.get("covariance"), depth + 1)
            provenance = self.value(value.get("provenance"), depth + 1)
            if (
                not isinstance(mean, np.ndarray)
                or not isinstance(covariance, np.ndarray)
                or not isinstance(provenance, dict)
            ):
                raise PortableNodeError("covariance state is malformed")
            try:
                return CovarianceState(
                    mean,
                    covariance,
                    cast(int, value.get("n_samples")),
                    cast(str, value.get("source_representation_identity")),
                    cast(float, value.get("reg_coef")),
                    provenance,
                )
            except (TypeError, ValueError) as exc:
                raise PortableNodeError(f"covariance state is invalid: {exc}") from exc
        if kind == "latent_value":
            data = self.value(value.get("data"), depth + 1)
            space = self.value(value.get("space"), depth + 1)
            metadata = self.value(value.get("metadata"), depth + 1)
            if not isinstance(data, np.ndarray) or not isinstance(space, LatentSpace) or not isinstance(metadata, dict):
                raise PortableNodeError("latent value is malformed")
            try:
                return LatentValue(data, space, metadata)
            except (TypeError, ValueError) as exc:
                raise PortableNodeError(f"latent value is invalid: {exc}") from exc
        if kind == "trajectory":
            data = self.value(value.get("data"), depth + 1)
            metadata = self.value(value.get("metadata"), depth + 1)
            if not isinstance(data, np.ndarray) or not isinstance(metadata, dict):
                raise PortableNodeError("trajectory is malformed")
            try:
                return Trajectory(data, metadata=metadata)
            except (TypeError, ValueError) as exc:
                raise PortableNodeError(f"trajectory is invalid: {exc}") from exc
        raise PortableNodeError(f"unsupported portable manifest kind: {kind!r}")

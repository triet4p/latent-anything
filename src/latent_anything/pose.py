"""Matrix-backed SO(3)/SE(3) geometry for robot poses.

Rotations are represented by validated 3x3 matrices and poses by validated
4x4 homogeneous matrices. Euler angles are intentionally not part of the
public representation; callers must choose a quaternion order explicitly when
converting from quaternions.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, cast

import numpy as np
from pydantic import BaseModel, Field


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = vector
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def _immutable_array(array: np.ndarray) -> np.ndarray:
    """Return an array whose read-only flag cannot be re-enabled by callers."""

    immutable = np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)
    immutable.setflags(write=False)
    return immutable


def _freeze_metadata(value: Any) -> Any:
    if isinstance(value, Mapping):
        mapping = cast(Mapping[Any, Any], value)
        return MappingProxyType({str(key): _freeze_metadata(item) for key, item in mapping.items()})
    if isinstance(value, (list, tuple)):
        sequence = cast(list[Any] | tuple[Any, ...], value)
        return tuple(_freeze_metadata(item) for item in sequence)
    if isinstance(value, (set, frozenset)):
        items = cast(set[Any] | frozenset[Any], value)
        return frozenset(_freeze_metadata(item) for item in items)
    if isinstance(value, np.ndarray):
        return _immutable_array(np.asarray(value, dtype=np.float64))
    return value


def _thaw_metadata(value: Any) -> Any:
    if isinstance(value, Mapping):
        mapping = cast(Mapping[Any, Any], value)
        return {str(key): _thaw_metadata(item) for key, item in mapping.items()}
    if isinstance(value, (list, tuple)):
        sequence = cast(list[Any] | tuple[Any, ...], value)
        return [_thaw_metadata(item) for item in sequence]
    if isinstance(value, (set, frozenset)):
        items = cast(set[Any] | frozenset[Any], value)
        return [_thaw_metadata(item) for item in items]
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _validate_rotation(matrix: np.ndarray) -> np.ndarray:
    value = np.asarray(matrix, dtype=np.float64)
    if value.shape != (3, 3) or not np.isfinite(value).all():
        raise ValueError("rotation matrix must be finite with shape (3, 3)")
    if not np.allclose(value.T @ value, np.eye(3), atol=1e-8) or not np.isclose(np.linalg.det(value), 1.0, atol=1e-8):
        raise ValueError("rotation matrix must be orthonormal with determinant +1")
    return value.copy()


@dataclass(frozen=True)
class PoseMetadata:
    """Frame and unit contract attached to a pose."""

    parent_frame: str = "world"
    child_frame: str = "tool"
    position_unit: str = "m"
    angle_unit: str = "rad"

    def __post_init__(self) -> None:
        if not self.parent_frame or not self.child_frame:
            raise ValueError("parent_frame and child_frame must be non-empty")
        if self.position_unit != "m" or self.angle_unit != "rad":
            raise ValueError("pose boundaries require position_unit='m' and angle_unit='rad'")

    def to_dict(self) -> dict[str, str]:
        return {
            "parent_frame": self.parent_frame,
            "child_frame": self.child_frame,
            "position_unit": self.position_unit,
            "angle_unit": self.angle_unit,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PoseMetadata:
        return cls(**data)


class PoseConfig(BaseModel):
    """Serializable pose contract; no implicit Euler-angle convention."""

    parent_frame: str = Field(default="world", min_length=1)
    child_frame: str = Field(default="tool", min_length=1)
    position_unit: str = "m"
    angle_unit: str = "rad"

    def metadata(self) -> PoseMetadata:
        return PoseMetadata(**self.model_dump())


class SO3:
    """An immutable validated 3D rotation represented as a matrix."""

    def __init__(self, matrix: np.ndarray | None = None) -> None:
        self._matrix = _validate_rotation(np.eye(3) if matrix is None else matrix)

    @classmethod
    def identity(cls) -> SO3:
        return cls()

    @classmethod
    def from_quaternion(cls, quaternion: np.ndarray, *, order: str = "xyzw") -> SO3:
        q = np.asarray(quaternion, dtype=np.float64)
        if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-12:
            raise ValueError("quaternion must be a finite non-zero vector with shape (4,)")
        if order == "wxyz":
            w, x, y, z = q / np.linalg.norm(q)
        elif order == "xyzw":
            x, y, z, w = q / np.linalg.norm(q)
        else:
            raise ValueError("quaternion order must be explicitly 'xyzw' or 'wxyz'")
        return cls(
            np.array(
                [
                    [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                    [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                    [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
                ]
            )
        )

    @classmethod
    def exp(cls, rotation_vector: np.ndarray) -> SO3:
        v = np.asarray(rotation_vector, dtype=np.float64)
        if v.shape != (3,) or not np.isfinite(v).all():
            raise ValueError("rotation_vector must be finite with shape (3,)")
        theta = float(np.linalg.norm(v))
        k = _skew(v)
        if theta < 1e-10:
            return cls(np.eye(3) + k + 0.5 * k @ k)
        return cls(np.eye(3) + np.sin(theta) / theta * k + (1 - np.cos(theta)) / theta**2 * (k @ k))

    @property
    def matrix(self) -> np.ndarray:
        return self._matrix.copy()

    def compose(self, other: SO3) -> SO3:
        return SO3(self._matrix @ other._matrix)

    def inverse(self) -> SO3:
        return SO3(self._matrix.T)

    def log(self) -> np.ndarray:
        cosine = float(np.clip((np.trace(self._matrix) - 1) / 2, -1, 1))
        theta = float(np.arccos(cosine))
        if theta < 1e-10:
            return np.array(
                [
                    (self._matrix[2, 1] - self._matrix[1, 2]) / 2,
                    (self._matrix[0, 2] - self._matrix[2, 0]) / 2,
                    (self._matrix[1, 0] - self._matrix[0, 1]) / 2,
                ]
            )
        if np.pi - theta < 1e-6:
            symmetric = 0.5 * (self._matrix + np.eye(3))
            eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
            axis = np.asarray(eigenvectors[:, int(np.argmax(eigenvalues))], dtype=np.float64)
            skew_vector = 0.5 * np.array(
                [
                    self._matrix[2, 1] - self._matrix[1, 2],
                    self._matrix[0, 2] - self._matrix[2, 0],
                    self._matrix[1, 0] - self._matrix[0, 1],
                ]
            )
            if float(np.dot(axis, skew_vector)) < 0.0:
                axis = -axis
            if float(np.linalg.norm(skew_vector)) < 1e-8:
                pivot = int(np.argmax(np.abs(axis)))
                if axis[pivot] < 0.0:
                    axis = -axis
            return theta * axis / np.linalg.norm(axis)
        return (
            theta
            / (2 * np.sin(theta))
            * np.array(
                [
                    self._matrix[2, 1] - self._matrix[1, 2],
                    self._matrix[0, 2] - self._matrix[2, 0],
                    self._matrix[1, 0] - self._matrix[0, 1],
                ]
            )
        )

    def distance(self, other: SO3) -> float:
        return float(np.linalg.norm(self.inverse().compose(other).log()))

    def interpolate(self, other: SO3, t: float) -> SO3:
        if not 0 <= t <= 1:
            raise ValueError("t must be in [0, 1]")
        return self.compose(SO3.exp(t * self.inverse().compose(other).log()))


class SE3:
    """An immutable rigid transform mapping ``child_frame`` into ``parent_frame``."""

    def __init__(
        self,
        rotation: SO3 | np.ndarray | None = None,
        translation: np.ndarray | None = None,
        *,
        metadata: PoseMetadata | None = None,
    ) -> None:
        self.rotation = rotation if isinstance(rotation, SO3) else SO3(rotation)
        self._translation = np.zeros(3) if translation is None else np.asarray(translation, dtype=np.float64).copy()
        if self._translation.shape != (3,) or not np.isfinite(self._translation).all():
            raise ValueError("translation must be finite with shape (3,) and use metres")
        self._metadata = metadata or PoseMetadata()

    @property
    def translation(self) -> np.ndarray:
        """Return a read-only defensive copy of the translation."""

        return _immutable_array(self._translation)

    @property
    def metadata(self) -> PoseMetadata:
        """Return the immutable frame and unit metadata."""

        return self._metadata

    @classmethod
    def identity(cls, *, metadata: PoseMetadata | None = None) -> SE3:
        return cls(metadata=metadata)

    @classmethod
    def from_matrix(cls, matrix: np.ndarray, *, metadata: PoseMetadata | None = None) -> SE3:
        value = np.asarray(matrix, dtype=np.float64)
        if value.shape != (4, 4) or not np.allclose(value[3], [0, 0, 0, 1], atol=1e-8):
            raise ValueError("homogeneous pose matrix must have shape (4, 4) and final row [0, 0, 0, 1]")
        return cls(SO3(value[:3, :3]), value[:3, 3], metadata=metadata)

    @classmethod
    def exp(cls, twist: np.ndarray, *, metadata: PoseMetadata | None = None) -> SE3:
        value = np.asarray(twist, dtype=np.float64)
        if value.shape != (6,):
            raise ValueError("SE(3) twist must have shape (6,), ordered [translation, rotation]")
        rho, omega = value[:3], value[3:]
        theta = float(np.linalg.norm(omega))
        skew = _skew(omega)
        v_matrix = (
            np.eye(3) + (1 - np.cos(theta)) / theta**2 * skew + (theta - np.sin(theta)) / theta**3 * (skew @ skew)
            if theta >= 1e-10
            else np.eye(3) + 0.5 * skew + (skew @ skew) / 6
        )
        return cls(SO3.exp(omega), v_matrix @ rho, metadata=metadata)

    @property
    def matrix(self) -> np.ndarray:
        result = np.eye(4)
        result[:3, :3] = self.rotation.matrix
        result[:3, 3] = self.translation
        return result

    def compose(self, other: SE3) -> SE3:
        if self.metadata.child_frame != other.metadata.parent_frame:
            raise ValueError(
                f"frame mismatch: {self.metadata.child_frame!r} cannot compose with parent "
                f"{other.metadata.parent_frame!r}"
            )
        return SE3.from_matrix(
            self.matrix @ other.matrix, metadata=PoseMetadata(self.metadata.parent_frame, other.metadata.child_frame)
        )

    def inverse(self) -> SE3:
        return SE3(
            self.rotation.inverse(),
            -(self.rotation.inverse().matrix @ self.translation),
            metadata=PoseMetadata(self.metadata.child_frame, self.metadata.parent_frame),
        )

    def log(self) -> np.ndarray:
        omega = self.rotation.log()
        theta = float(np.linalg.norm(omega))
        skew = _skew(omega)
        if theta < 1e-10:
            v_inverse = np.eye(3) - 0.5 * skew + (skew @ skew) / 12
        else:
            v_inverse = (
                np.eye(3)
                - 0.5 * skew
                + (1 / theta**2 - (1 + np.cos(theta)) / (2 * theta * np.sin(theta))) * (skew @ skew)
            )
        return np.concatenate([v_inverse @ self.translation, omega])

    def distance(self, other: SE3, *, translation_weight: float = 1.0, rotation_weight: float = 1.0) -> float:
        if (
            self.metadata.parent_frame != other.metadata.parent_frame
            or self.metadata.child_frame != other.metadata.child_frame
        ):
            raise ValueError("pose distance requires matching parent and child frames")
        delta = self.inverse().compose(SE3(other.rotation, other.translation, metadata=self.metadata))
        twist = delta.log()
        return float(
            np.sqrt(translation_weight * np.dot(twist[:3], twist[:3]) + rotation_weight * np.dot(twist[3:], twist[3:]))
        )

    def interpolate(self, other: SE3, t: float) -> SE3:
        if self.metadata != other.metadata:
            raise ValueError("pose interpolation requires matching frame and unit metadata")
        if not 0 <= t <= 1:
            raise ValueError("t must be in [0, 1]")
        return SE3(
            self.rotation.interpolate(other.rotation, t),
            (1 - t) * self.translation + t * other.translation,
            metadata=self.metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        return {"matrix": self.matrix.tolist(), "metadata": self.metadata.to_dict()}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SE3:
        return cls.from_matrix(
            np.asarray(data["matrix"], dtype=np.float64), metadata=PoseMetadata.from_dict(data["metadata"])
        )


class PoseTrajectory:
    """Immutable pose sequence with slicing and LeRobot-compatible metadata."""

    def __init__(self, poses: Sequence[SE3], *, metadata: Mapping[str, Any] | None = None) -> None:
        if not poses:
            raise ValueError("PoseTrajectory requires at least one pose")
        self._poses = tuple(poses)
        first = self._poses[0].metadata
        if any(p.metadata != first for p in self._poses):
            raise ValueError("all poses in a trajectory must share frame and unit metadata")
        trajectory_metadata = dict(metadata or {})
        trajectory_metadata.setdefault("observation.frame_id", first.child_frame)
        trajectory_metadata.setdefault("observation.parent_frame", first.parent_frame)
        trajectory_metadata.setdefault("observation.position_unit", first.position_unit)
        trajectory_metadata.setdefault("observation.angle_unit", first.angle_unit)
        self._metadata = _freeze_metadata(trajectory_metadata)

    def __len__(self) -> int:
        return len(self._poses)

    def __getitem__(self, key: int | slice) -> SE3 | PoseTrajectory:
        if isinstance(key, int):
            return self._poses[key]
        return PoseTrajectory(self._poses[key], metadata=self.metadata)

    @property
    def poses(self) -> tuple[SE3, ...]:
        return self._poses

    @property
    def metadata(self) -> Mapping[str, Any]:
        """Return immutable trajectory metadata."""

        return self._metadata

    def to_numpy(self) -> np.ndarray:
        return np.stack([pose.matrix for pose in self._poses])

    def group_distance(self) -> np.ndarray:
        return np.array([0.0 if i == 0 else self._poses[i - 1].distance(self._poses[i]) for i in range(len(self))])

    def lerobot_metadata(self) -> dict[str, Any]:
        return {
            "features": {"observation.state": {"dtype": "float64", "shape": [4, 4], "names": ["pose_matrix"]}},
            **_thaw_metadata(self.metadata),
        }

    def to_dict(self) -> dict[str, Any]:
        return {"poses": [pose.to_dict() for pose in self._poses], "metadata": _thaw_metadata(self.metadata)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PoseTrajectory:
        return cls([SE3.from_dict(item) for item in data["poses"]], metadata=data.get("metadata"))

"""Typed, provenance-rich views over LeRobot v3 dataset objects.

The classes in this module intentionally sit outside LeRobot's storage model.
They read metadata through the canonical ``meta`` object and delegate sample
loading to ``LeRobotDataset`` or ``StreamingLeRobotDataset``.  In particular,
the bridge never opens Parquet or video files itself and never converts a
processor-ready sample merely because it crossed this boundary.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from itertools import islice
from typing import cast

import numpy as np

from latent_anything.integrations.lerobot import LeRobotAPI, load_lerobot_api

_INDEX_KEYS = ("index", "episode_index", "frame_index", "timestamp", "task_index")


def _scalar(value: object) -> object:
    """Unwrap a scalar tensor/NumPy value without changing structured values."""

    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except (TypeError, ValueError, RuntimeError):
            return value
    return value


def _int_value(value: object, default: int | None = None) -> int | None:
    value = _scalar(value)
    if value is None:
        return default
    if isinstance(value, (int, float, str, np.integer, np.floating)):
        try:
            return int(value)
        except (TypeError, ValueError):
            return default
    return default


def _float_value(value: object, default: float | None = None) -> float | None:
    value = _scalar(value)
    if value is None:
        return default
    if isinstance(value, (int, float, str, np.integer, np.floating)):
        try:
            return float(value)
        except (TypeError, ValueError):
            return default
    return default


def _json_value(value: object) -> object:
    """Convert metadata values to immutable, JSON-friendly Python values."""

    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return tuple(_json_value(item) for item in value.tolist())
    if isinstance(value, (list, tuple)):
        return tuple(_json_value(item) for item in value)
    value = _scalar(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _row_value(row: object, key: str, default: object = None) -> object:
    if isinstance(row, Mapping):
        return row.get(key, default)
    getter = getattr(row, "get", None)
    if callable(getter):
        return getter(key, default)
    return default


def _mapping_value(value: object) -> Mapping[str, object]:
    converted = _json_value(value)
    return converted if isinstance(converted, Mapping) else {}


def _row_at(rows: object, index: int) -> Mapping[str, object]:
    """Read one metadata row from HF Dataset, pandas, or a fixture sequence."""

    iloc = getattr(rows, "iloc", None)
    if iloc is not None:
        row = iloc[index]
        to_dict = getattr(row, "to_dict", None)
        if callable(to_dict):
            converted = to_dict()
            if isinstance(converted, Mapping):
                return {str(key): value for key, value in converted.items()}
    row = rows[index]  # type: ignore[index]
    if isinstance(row, Mapping):
        return dict(row)
    to_dict = getattr(row, "to_dict", None)
    if callable(to_dict):
        converted = to_dict()
        if isinstance(converted, Mapping):
            return {str(key): value for key, value in converted.items()}
    raise TypeError("LeRobot episode metadata must expose mapping-like rows")


def _first_row_value(row: Mapping[str, object], suffix: str) -> object:
    direct = row.get(suffix)
    if direct is not None:
        return direct
    for key, value in row.items():
        if str(key).endswith(suffix):
            return value
    return None


def _task_labels(tasks: object) -> dict[int, str]:
    """Extract the canonical task-index-to-text mapping from ``meta.tasks``."""

    if tasks is None:
        return {}
    labels: dict[int, str] = {}
    index = getattr(tasks, "index", None)
    if index is not None:
        for position, label in enumerate(index):
            row = None
            iloc = getattr(tasks, "iloc", None)
            if iloc is not None:
                row = iloc[position]
            task_index = _int_value(_row_value(row, "task_index"), position)
            if task_index is not None:
                labels[task_index] = str(label)
        return labels
    if isinstance(tasks, Mapping):
        for key, value in tasks.items():
            task_index = _int_value(key)
            label = _row_value(value, "task", value)
            if task_index is not None:
                labels[task_index] = str(_scalar(label))
        return labels
    for position, value in enumerate(tasks):  # type: ignore[reportUnknownVariableType]
        labels[position] = str(_scalar(value))
    return labels


def _metadata_source(dataset: object) -> object:
    meta = getattr(dataset, "meta", None)
    return meta if meta is not None else dataset


def _feature_role(key: str, feature: Mapping[str, object]) -> str:
    dtype = str(feature.get("dtype", ""))
    if dtype in {"image", "video"}:
        return "camera"
    if key == "action" or key.startswith("action."):
        return "action"
    if key == "observation.state" or key.startswith("observation.state."):
        return "state"
    return "auxiliary"


@dataclass(frozen=True)
class LeRobotFeatureDescriptor:
    """Bridge-owned description of one LeRobot feature."""

    key: str
    dtype: str
    shape: tuple[int, ...]
    role: str
    names: object | None = None
    normalization: Mapping[str, object] = field(default_factory=dict)
    metadata: Mapping[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class LeRobotEpisodeSlice:
    """Half-open frame range and task/timing metadata for one episode."""

    episode_index: int
    frame_start: int
    frame_stop: int
    length: int
    task_indices: tuple[int, ...] = ()
    task_labels: tuple[str, ...] = ()
    timestamp_start: float | None = None
    timestamp_stop: float | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.episode_index < 0:
            raise ValueError("episode_index must be non-negative")
        if self.frame_start < 0 or self.frame_stop < self.frame_start:
            raise ValueError("episode frame range must be a valid half-open interval")
        if self.frame_stop - self.frame_start != self.length:
            raise ValueError("episode length must match frame_stop - frame_start")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class LeRobotDatasetDescriptor:
    """Schema and provenance snapshot derived from a canonical LeRobot object."""

    repo_id: str
    revision: str
    codebase_version: str
    fps: int
    total_frames: int
    total_episodes: int
    features: Mapping[str, LeRobotFeatureDescriptor]
    cameras: tuple[str, ...]
    state_features: tuple[str, ...]
    action_features: tuple[str, ...]
    index_features: tuple[str, ...]
    task_labels: Mapping[int, str]
    episodes: tuple[LeRobotEpisodeSlice, ...]
    stats: Mapping[str, Mapping[str, object]]
    provenance: Mapping[str, object]

    def episode(self, episode_index: int) -> LeRobotEpisodeSlice:
        for episode in self.episodes:
            if episode.episode_index == episode_index:
                return episode
        raise IndexError(f"episode_index {episode_index} is not present in the dataset descriptor")

    def to_dict(self) -> dict[str, object]:
        return {
            "repo_id": self.repo_id,
            "revision": self.revision,
            "codebase_version": self.codebase_version,
            "fps": self.fps,
            "total_frames": self.total_frames,
            "total_episodes": self.total_episodes,
            "features": {key: value.to_dict() for key, value in self.features.items()},
            "cameras": list(self.cameras),
            "state_features": list(self.state_features),
            "action_features": list(self.action_features),
            "index_features": list(self.index_features),
            "task_labels": {str(key): value for key, value in self.task_labels.items()},
            "episodes": [episode.to_dict() for episode in self.episodes],
            "stats": {key: dict(value) for key, value in self.stats.items()},
            "provenance": dict(self.provenance),
        }


@dataclass(frozen=True)
class LeRobotSampleProvenance:
    """Identity and alignment metadata attached to one raw processor sample."""

    repo_id: str
    revision: str
    episode_index: int | None
    frame_index: int | None
    timestamp: float | None
    task_index: int | None
    task: str | None
    source: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class LeRobotSample:
    """One sample whose values remain in LeRobot's processor-ready format.

    ``values`` intentionally contains the original mapping returned by
    LeRobot. Tensor values are not detached, copied, or converted here.
    """

    values: Mapping[str, object]
    provenance: LeRobotSampleProvenance

    def __getitem__(self, key: str) -> object:
        return self.values[key]


@dataclass(frozen=True)
class LeRobotCapturedLatent:
    """NumPy boundary for a latent captured from a processor/policy path."""

    values: np.ndarray
    provenance: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        values = np.array(self.values, copy=True)
        values.setflags(write=False)
        object.__setattr__(self, "values", values)

    def to_numpy(self) -> np.ndarray:
        return self.values.copy()


def describe_lerobot_dataset(dataset: object) -> LeRobotDatasetDescriptor:
    """Describe a loaded v3 dataset without reading any sample or video data."""

    meta = _metadata_source(dataset)
    raw_features = getattr(meta, "features", getattr(dataset, "features", {}))
    feature_descriptors: dict[str, LeRobotFeatureDescriptor] = {}
    for key, raw_feature in raw_features.items():
        feature = raw_feature if isinstance(raw_feature, Mapping) else {}
        raw_stats = getattr(meta, "stats", {}) or {}
        normalization = raw_stats.get(key, {}) if isinstance(raw_stats, Mapping) else {}
        shape_value = feature.get("shape", ())
        shape = tuple(int(item) for item in shape_value) if shape_value is not None else ()
        feature_descriptors[str(key)] = LeRobotFeatureDescriptor(
            key=str(key),
            dtype=str(feature.get("dtype", "unknown")),
            shape=shape,
            role=_feature_role(str(key), feature),
            names=_json_value(feature.get("names")),
            normalization=_mapping_value(normalization),
            metadata=_mapping_value(feature.get("info", {})),
        )

    task_labels = _task_labels(getattr(meta, "tasks", None))
    task_indices_by_label = {label: index for index, label in task_labels.items()}
    total_episodes = int(getattr(meta, "total_episodes", getattr(dataset, "num_episodes", 0)))
    episodes: list[LeRobotEpisodeSlice] = []
    raw_episodes = getattr(meta, "episodes", None)
    if raw_episodes is not None:
        for position in range(len(raw_episodes)):  # type: ignore[arg-type]
            row = _row_at(raw_episodes, position)
            episode_index = _int_value(_row_value(row, "episode_index"), position)
            length = _int_value(_row_value(row, "length"), 0) or 0
            frame_start = _int_value(_row_value(row, "dataset_from_index"), 0) or 0
            frame_stop = _int_value(_row_value(row, "dataset_to_index"), frame_start + length) or frame_start + length
            raw_tasks = _row_value(row, "tasks", ())
            if isinstance(raw_tasks, (str, int)):
                raw_tasks = (raw_tasks,)
            task_indices: list[int] = []
            task_text: list[str] = []
            task_values: Sequence[object] = (
                (raw_tasks,)
                if isinstance(raw_tasks, (str, int))
                else raw_tasks
                if isinstance(raw_tasks, Sequence)
                else ()
            )
            for raw_task in task_values:
                task_index = _int_value(raw_task)
                if task_index is not None:
                    task_indices.append(task_index)
                    if task_index in task_labels:
                        task_text.append(task_labels[task_index])
                else:
                    task_label = str(_scalar(raw_task))
                    task_text.append(task_label)
                    if task_label in task_indices_by_label:
                        task_indices.append(task_indices_by_label[task_label])
            if not task_indices:
                task_index = _int_value(_row_value(row, "task_index"))
                if task_index is not None:
                    task_indices.append(task_index)
                    task_text.append(task_labels.get(task_index, str(task_index)))
            metadata = {
                str(key): _json_value(value)
                for key, value in row.items()
                if key not in {"episode_index", "length", "dataset_from_index", "dataset_to_index", "tasks"}
            }
            episodes.append(
                LeRobotEpisodeSlice(
                    episode_index=episode_index if episode_index is not None else position,
                    frame_start=frame_start,
                    frame_stop=frame_stop,
                    length=length,
                    task_indices=tuple(task_indices),
                    task_labels=tuple(task_text),
                    timestamp_start=_float_value(_first_row_value(row, "from_timestamp")),
                    timestamp_stop=_float_value(_first_row_value(row, "to_timestamp")),
                    metadata=metadata,
                )
            )

    raw_stats = getattr(meta, "stats", {}) or {}
    stats = (
        {str(key): _mapping_value(value) for key, value in raw_stats.items()} if isinstance(raw_stats, Mapping) else {}
    )
    repo_id = str(getattr(dataset, "repo_id", getattr(meta, "repo_id", "")))
    revision = str(getattr(dataset, "revision", getattr(meta, "revision", "")))
    info = getattr(meta, "info", None)
    if isinstance(info, Mapping):
        codebase_version = str(info.get("codebase_version", "v3.0"))
    else:
        codebase_version = str(getattr(info, "codebase_version", getattr(meta, "_version", "v3.0")))
    fps = int(getattr(meta, "fps", getattr(dataset, "fps", 0)))
    total_frames = int(getattr(meta, "total_frames", getattr(dataset, "num_frames", 0)))
    provenance = {
        "source": "lerobot_dataset_v3",
        "repo_id": repo_id,
        "revision": revision,
        "codebase_version": codebase_version,
        "fps": fps,
    }
    return LeRobotDatasetDescriptor(
        repo_id=repo_id,
        revision=revision,
        codebase_version=codebase_version,
        fps=fps,
        total_frames=total_frames,
        total_episodes=total_episodes,
        features=feature_descriptors,
        cameras=tuple(key for key, feature in feature_descriptors.items() if feature.role == "camera"),
        state_features=tuple(key for key, feature in feature_descriptors.items() if feature.role == "state"),
        action_features=tuple(key for key, feature in feature_descriptors.items() if feature.role == "action"),
        index_features=_INDEX_KEYS,
        task_labels=task_labels,
        episodes=tuple(episodes),
        stats=stats,
        provenance=provenance,
    )


def _sample_from_raw(
    raw: Mapping[str, object],
    descriptor: LeRobotDatasetDescriptor,
    *,
    source: str,
) -> LeRobotSample:
    episode_index = _int_value(raw.get("episode_index"))
    frame_index = _int_value(raw.get("frame_index", raw.get("index")))
    task_index = _int_value(raw.get("task_index"))
    raw_task = _scalar(raw.get("task"))
    task = (
        str(raw_task)
        if raw_task is not None
        else descriptor.task_labels.get(task_index)
        if task_index is not None
        else None
    )
    return LeRobotSample(
        values=raw,
        provenance=LeRobotSampleProvenance(
            repo_id=descriptor.repo_id,
            revision=descriptor.revision,
            episode_index=episode_index,
            frame_index=frame_index,
            timestamp=_float_value(raw.get("timestamp")),
            task_index=task_index,
            task=task,
            source=source,
        ),
    )


class LeRobotDatasetReader:
    """Lazy episode reader delegating every frame read to ``LeRobotDataset``."""

    def __init__(self, dataset: object, descriptor: LeRobotDatasetDescriptor | None = None) -> None:
        self.dataset = dataset
        self.descriptor = descriptor if descriptor is not None else describe_lerobot_dataset(dataset)

    def _relative_index(self, absolute_index: int, episode_index: int) -> int:
        mapping = getattr(self.dataset, "absolute_to_relative_idx", None)
        if isinstance(mapping, Mapping):
            if absolute_index not in mapping:
                raise IndexError(f"absolute frame {absolute_index} is not selected by the loaded dataset")
            return int(mapping[absolute_index])
        selected = getattr(self.dataset, "episodes", None)
        if selected is None:
            return absolute_index
        selected_indices = [int(value) for value in selected]
        if episode_index not in selected_indices:
            raise IndexError(f"episode {episode_index} is not selected by the loaded dataset")
        offset = 0
        for selected_episode in selected_indices:
            if selected_episode == episode_index:
                return offset + absolute_index - self.descriptor.episode(selected_episode).frame_start
            offset += self.descriptor.episode(selected_episode).length
        raise IndexError(f"episode {episode_index} is not selected by the loaded dataset")

    def sample_at(self, absolute_index: int, *, source: str = "lerobot_dataset") -> LeRobotSample:
        episode = next(
            (item for item in self.descriptor.episodes if item.frame_start <= absolute_index < item.frame_stop),
            None,
        )
        dataset_index = (
            self._relative_index(absolute_index, episode.episode_index) if episode is not None else absolute_index
        )
        raw = self.dataset[dataset_index]  # type: ignore[index]
        if not isinstance(raw, Mapping):
            raise TypeError("LeRobotDataset samples must be mapping-like")
        return _sample_from_raw(raw, self.descriptor, source=source)

    def iter_episode(
        self,
        episode_index: int,
        *,
        start_frame: int = 0,
        stop_frame: int | None = None,
    ) -> Iterator[LeRobotSample]:
        episode = self.descriptor.episode(episode_index)
        if start_frame < 0 or start_frame > episode.length:
            raise ValueError("start_frame is outside the episode")
        end = episode.length if stop_frame is None else stop_frame
        if end < start_frame or end > episode.length:
            raise ValueError("stop_frame is outside the episode")
        for relative_frame in range(start_frame, end):
            absolute_index = episode.frame_start + relative_frame
            dataset_index = self._relative_index(absolute_index, episode_index)
            raw = self.dataset[dataset_index]  # type: ignore[index]
            if not isinstance(raw, Mapping):
                raise TypeError("LeRobotDataset samples must be mapping-like")
            yield _sample_from_raw(raw, self.descriptor, source="lerobot_dataset")

    def iter_samples(self, *, max_samples: int | None = None) -> Iterator[LeRobotSample]:
        """Iterate selected samples lazily, optionally bounded by a sample count."""

        source: Iterator[LeRobotSample] = (
            sample for episode in self.descriptor.episodes for sample in self.iter_episode(episode.episode_index)
        )
        yield from source if max_samples is None else islice(source, max_samples)


def read_lerobot_episode(
    dataset: object,
    episode_index: int,
    *,
    start_frame: int = 0,
    stop_frame: int | None = None,
    descriptor: LeRobotDatasetDescriptor | None = None,
) -> Iterator[LeRobotSample]:
    """Return a lazy iterator for one episode and frame range."""

    return LeRobotDatasetReader(dataset, descriptor).iter_episode(
        episode_index, start_frame=start_frame, stop_frame=stop_frame
    )


class LeRobotStreamingReader:
    """Bounded bridge buffer over an upstream ``StreamingLeRobotDataset``."""

    def __init__(
        self,
        dataset: object,
        *,
        buffer_size: int = 1024,
        descriptor: LeRobotDatasetDescriptor | None = None,
    ) -> None:
        if buffer_size < 1:
            raise ValueError("buffer_size must be positive")
        self.dataset = dataset
        self.buffer_size = buffer_size
        self.descriptor = descriptor if descriptor is not None else describe_lerobot_dataset(dataset)
        self._buffer: deque[LeRobotSample] = deque(maxlen=buffer_size)

    @property
    def buffered_samples(self) -> tuple[LeRobotSample, ...]:
        """Return the bounded recent-sample window without exposing mutable state."""

        return tuple(self._buffer)

    def iter_samples(self, *, max_samples: int | None = None) -> Iterator[LeRobotSample]:
        if max_samples is not None and max_samples < 0:
            raise ValueError("max_samples must be non-negative or None")
        self._buffer.clear()
        upstream = iter(cast(Iterator[object], self.dataset))
        count = 0
        while max_samples is None or count < max_samples:
            try:
                raw = next(upstream)
            except StopIteration:
                return
            if not isinstance(raw, Mapping):
                raise TypeError("StreamingLeRobotDataset samples must be mapping-like")
            sample = _sample_from_raw(raw, self.descriptor, source="streaming_lerobot_dataset")
            self._buffer.append(sample)
            count += 1
            yield sample


def stream_lerobot_samples(
    dataset: object,
    *,
    max_samples: int | None = None,
    buffer_size: int = 1024,
    descriptor: LeRobotDatasetDescriptor | None = None,
) -> Iterator[LeRobotSample]:
    """Stream processor-ready samples with a bounded bridge-owned window."""

    return LeRobotStreamingReader(dataset, buffer_size=buffer_size, descriptor=descriptor).iter_samples(
        max_samples=max_samples
    )


def load_streaming_lerobot_dataset(
    repo_id: str,
    *,
    buffer_size: int = 1024,
    api: LeRobotAPI | None = None,
    **kwargs: object,
) -> LeRobotStreamingReader:
    """Construct the upstream streaming dataset and wrap it without re-decoding data."""

    if buffer_size < 1:
        raise ValueError("buffer_size must be positive")
    upstream_api = api if api is not None else load_lerobot_api()
    factory = cast(Callable[..., object], upstream_api.streaming_dataset_type)
    dataset = factory(repo_id, buffer_size=buffer_size, **kwargs)
    return LeRobotStreamingReader(dataset, buffer_size=buffer_size)


def captured_latent_to_numpy(value: object) -> np.ndarray:
    """Detach one captured latent result at the framework's NumPy boundary.

    This is intentionally not used by dataset readers. It accepts a NumPy
    array or a PyTorch-like ``detach().cpu().numpy()`` result and returns an
    owned, read-only NumPy array.
    """

    if isinstance(value, np.ndarray):
        result = np.array(value, copy=True)
    else:
        detached = getattr(value, "detach", None)
        current = detached() if callable(detached) else value
        cpu = getattr(current, "cpu", None)
        current = cpu() if callable(cpu) else current
        numpy = getattr(current, "numpy", None)
        current = numpy() if callable(numpy) else current
        result = np.array(current, copy=True)
    result.setflags(write=False)
    return result


def captured_latent(
    value: object,
    *,
    provenance: Mapping[str, object] | None = None,
) -> LeRobotCapturedLatent:
    """Create a typed captured-latent result while preserving provenance."""

    return LeRobotCapturedLatent(captured_latent_to_numpy(value), dict(provenance or {}))


__all__ = [
    "LeRobotCapturedLatent",
    "LeRobotDatasetDescriptor",
    "LeRobotDatasetReader",
    "LeRobotEpisodeSlice",
    "LeRobotFeatureDescriptor",
    "LeRobotSample",
    "LeRobotSampleProvenance",
    "LeRobotStreamingReader",
    "captured_latent",
    "captured_latent_to_numpy",
    "describe_lerobot_dataset",
    "load_streaming_lerobot_dataset",
    "read_lerobot_episode",
    "stream_lerobot_samples",
]

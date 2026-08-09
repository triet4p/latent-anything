"""Offline tests for the LeRobot v3 dataset bridge."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from latent_anything.integrations.lerobot import (
    LeRobotDatasetReader,
    LeRobotStreamingReader,
    captured_latent,
    describe_lerobot_dataset,
    read_lerobot_episode,
)


@dataclass
class SyntheticMeta:
    features: dict[str, dict[str, object]]
    stats: dict[str, dict[str, object]]
    tasks: dict[int, str]
    episodes: list[dict[str, object]]
    repo_id: str = "synthetic/lerobot-v3"
    revision: str = "fixture-rev"
    fps: int = 10
    total_frames: int = 5
    total_episodes: int = 2
    info: object = None


class SyntheticDataset:
    def __init__(self) -> None:
        self.meta = SyntheticMeta(
            features={
                "observation.images.front": {"dtype": "video", "shape": [3, 4, 5]},
                "observation.state": {"dtype": "float32", "shape": [2], "names": ["x", "y"]},
                "action": {"dtype": "float32", "shape": [1]},
            },
            stats={
                "observation.state": {
                    "mean": np.array([1.0, 2.0]),
                    "std": np.array([0.5, 0.25]),
                }
            },
            tasks={0: "pick", 1: "place"},
            episodes=[
                {
                    "episode_index": 0,
                    "length": 3,
                    "dataset_from_index": 0,
                    "dataset_to_index": 3,
                    "tasks": [0],
                    "from_timestamp": 0.0,
                    "to_timestamp": 0.2,
                },
                {
                    "episode_index": 1,
                    "length": 2,
                    "dataset_from_index": 3,
                    "dataset_to_index": 5,
                    "tasks": [1],
                    "from_timestamp": 0.0,
                    "to_timestamp": 0.1,
                },
            ],
        )
        self.meta.info = SimpleNamespace(codebase_version="v3.0")
        self.repo_id = self.meta.repo_id
        self.revision = self.meta.revision
        self.fps = self.meta.fps
        self.num_frames = self.meta.total_frames
        self.num_episodes = self.meta.total_episodes
        self.episodes: list[int] | None = None
        self.read_indices: list[int] = []
        self.samples = [
            {
                "episode_index": torch.tensor(0),
                "frame_index": torch.tensor(0),
                "timestamp": torch.tensor(0.0),
                "task_index": torch.tensor(0),
                "observation.state": torch.tensor([1.0, 2.0]),
            },
            {
                "episode_index": torch.tensor(0),
                "frame_index": torch.tensor(1),
                "timestamp": torch.tensor(0.1),
                "task_index": torch.tensor(0),
                "observation.state": torch.tensor([1.1, 2.1]),
            },
            {
                "episode_index": torch.tensor(0),
                "frame_index": torch.tensor(2),
                "timestamp": torch.tensor(0.2),
                "task_index": torch.tensor(0),
                "observation.state": torch.tensor([1.2, 2.2]),
            },
            {
                "episode_index": torch.tensor(1),
                "frame_index": torch.tensor(0),
                "timestamp": torch.tensor(0.0),
                "task_index": torch.tensor(1),
                "observation.state": torch.tensor([3.0, 4.0]),
            },
            {
                "episode_index": torch.tensor(1),
                "frame_index": torch.tensor(1),
                "timestamp": torch.tensor(0.1),
                "task_index": torch.tensor(1),
                "observation.state": torch.tensor([3.1, 4.1]),
            },
        ]

    def __getitem__(self, index: int) -> Mapping[str, object]:
        self.read_indices.append(index)
        return self.samples[index]


class SyntheticStreamingDataset(SyntheticDataset):
    def __iter__(self):
        yield from self.samples


def test_descriptor_maps_v3_schema_normalization_tasks_and_episode_ranges() -> None:
    descriptor = describe_lerobot_dataset(SyntheticDataset())

    assert descriptor.repo_id == "synthetic/lerobot-v3"
    assert descriptor.revision == "fixture-rev"
    assert descriptor.cameras == ("observation.images.front",)
    assert descriptor.state_features == ("observation.state",)
    assert descriptor.action_features == ("action",)
    assert descriptor.features["observation.state"].normalization["mean"] == (1.0, 2.0)
    assert descriptor.task_labels == {0: "pick", 1: "place"}
    assert descriptor.episodes[1].frame_start == 3
    assert descriptor.episodes[1].frame_stop == 5
    assert descriptor.episodes[1].task_labels == ("place",)


def test_episode_reader_is_lazy_and_preserves_processor_ready_tensor_objects() -> None:
    dataset = SyntheticDataset()
    samples = read_lerobot_episode(dataset, 1, start_frame=0, stop_frame=1)

    assert dataset.read_indices == []
    sample = next(samples)
    assert dataset.read_indices == [3]
    assert sample.values["observation.state"] is dataset.samples[3]["observation.state"]
    assert sample.provenance.episode_index == 1
    assert sample.provenance.frame_index == 0
    assert sample.provenance.timestamp == 0.0
    assert sample.provenance.task == "place"
    assert list(samples) == []


def test_dataset_reader_maps_selected_episodes_to_relative_dataset_rows() -> None:
    dataset = SyntheticDataset()
    dataset.episodes = [1]
    reader = LeRobotDatasetReader(dataset)

    samples = list(reader.iter_episode(1))

    assert dataset.read_indices == [0, 1]
    assert [sample.provenance.frame_index for sample in samples] == [0, 1]

    dataset.read_indices.clear()
    selected_samples = list(reader.iter_samples())

    assert dataset.read_indices == [0, 1]
    assert len(selected_samples) == 2


def test_streaming_reader_has_bounded_recent_window_and_raw_values() -> None:
    dataset = SyntheticStreamingDataset()
    reader = LeRobotStreamingReader(dataset, buffer_size=2)

    samples = list(reader.iter_samples(max_samples=4))

    assert len(samples) == 4
    assert len(reader.buffered_samples) == 2
    assert reader.buffered_samples[0].values["observation.state"] is dataset.samples[2]["observation.state"]
    assert reader.buffered_samples[-1].provenance.episode_index == 1


def test_captured_latent_conversion_is_explicit_and_read_only() -> None:
    tensor = torch.tensor([[1.0, 2.0]])
    result = captured_latent(tensor, provenance={"layer": "encoder"})

    assert result.values.dtype == np.float32
    assert np.array_equal(result.to_numpy(), np.array([[1.0, 2.0]], dtype=np.float32))
    assert result.provenance["layer"] == "encoder"
    assert not result.values.flags.writeable
    assert isinstance(tensor, torch.Tensor)


def test_timestamp_episode_and_task_alignment_is_preserved_across_boundaries() -> None:
    dataset = SyntheticStreamingDataset()
    samples = list(LeRobotStreamingReader(dataset).iter_samples())

    assert [(s.provenance.episode_index, s.provenance.frame_index) for s in samples] == [
        (0, 0),
        (0, 1),
        (0, 2),
        (1, 0),
        (1, 1),
    ]
    assert [s.provenance.timestamp for s in samples] == pytest.approx([0.0, 0.1, 0.2, 0.0, 0.1])
    assert [s.provenance.task for s in samples] == ["pick", "pick", "pick", "place", "place"]

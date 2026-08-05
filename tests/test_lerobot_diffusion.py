"""Offline Diffusion capture tests plus an opt-in pinned checkpoint smoke."""

from __future__ import annotations

import sys
from collections.abc import Mapping
from types import ModuleType, SimpleNamespace
from typing import Literal, cast

import numpy as np
import pytest
import torch
from torch import Tensor, nn

from latent_anything.integrations.lerobot import LeRobotAPI, LeRobotPolicyContext
from latent_anything.integrations.lerobot_diffusion import (
    DEFAULT_DIFFUSION_CHECKPOINT,
    DIFFUSION_CONDITIONING_LOCATION,
    DIFFUSION_DENOISING_LOCATION,
    DiffusionCheckpointSpec,
    DiffusionEpisodeTrace,
    DiffusionPolicyAdapter,
    analyze_diffusion_traces,
    load_diffusion_policy,
)


class TinyDenoiser(nn.Module):
    def forward(self, sample: Tensor, timestep: Tensor, *, global_cond: Tensor) -> Tensor:
        del timestep
        return sample + global_cond[:, : sample.shape[-1]].unsqueeze(1)


class TinyDiffusion(nn.Module):
    conditioning_dim = 3

    def __init__(self) -> None:
        super().__init__()
        self.unet = TinyDenoiser()


class TinyDiffusionPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(action_feature=SimpleNamespace(shape=(2,)), horizon=4)
        self.diffusion = TinyDiffusion()
        self._action_queue: list[Tensor] = []
        self.reset_calls = 0

    def reset(self) -> None:
        self._action_queue.clear()
        self.reset_calls += 1

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], *, noise: Tensor | None = None) -> Tensor:
        if not self._action_queue:
            base = noise if noise is not None else torch.zeros((1, 4, 2))
            actions: list[Tensor] = []
            for timestep in (9, 4, 0):
                base = self.diffusion.unet(base, torch.tensor([timestep]), global_cond=batch["state"])
                actions.append(base[:, 0])
            self._action_queue.extend(actions)
        return self._action_queue.pop(0)


class AddOnePreprocessor:
    def __call__(self, sample: Mapping[str, object]) -> dict[str, Tensor]:
        value = sample["state"]
        if not isinstance(value, Tensor):
            raise TypeError("fixture state must be a Tensor")
        return {"state": value + 1.0}


class ScalePostprocessor:
    def __call__(self, action: Tensor) -> dict[str, Tensor]:
        return {"action": action * 2.0}


def make_fixture_adapter(*, checkpoint: DiffusionCheckpointSpec | None = None) -> DiffusionPolicyAdapter:
    context = LeRobotPolicyContext(
        policy_name="diffusion",
        policy=TinyDiffusionPolicy(),
        preprocessor=AddOnePreprocessor(),
        postprocessor=ScalePostprocessor(),
        dataset=SimpleNamespace(repo_id="fixture/dataset", revision="fixture-revision"),
    )
    return DiffusionPolicyAdapter(context, checkpoint=checkpoint or DEFAULT_DIFFUSION_CHECKPOINT)


def test_diffusion_capture_keeps_conditioning_and_timestep_axes_explicit() -> None:
    adapter = make_fixture_adapter()
    selected = adapter.select_action(
        {"state": torch.tensor([[1.0, 2.0, 3.0]])},
        noise=np.zeros((1, 4, 2)),
    )

    assert len(selected.representations) == 4
    condition = [item for item in selected.representations if item.kind == "observation_conditioning"]
    denoising = [item for item in selected.representations if item.kind == "denoising_action"]
    assert len(condition) == 1
    assert len(denoising) == 3
    assert [item.diffusion_timestep for item in denoising] == [9, 4, 0]
    assert all(item.episode_step == 0 for item in selected.representations)
    assert condition[0].capture_metadata.location == DIFFUSION_CONDITIONING_LOCATION
    assert denoising[0].capture_metadata.location == DIFFUSION_DENOISING_LOCATION
    np.testing.assert_array_equal(condition[0].latent.values, np.array([2.0, 3.0, 4.0]))


def test_diffusion_action_matches_direct_preprocess_select_postprocess_with_fixed_noise() -> None:
    adapter = make_fixture_adapter()
    sample = {"state": torch.tensor([[1.0, 2.0, 3.0]])}
    noise = np.full((1, 4, 2), 0.25)
    preprocessor = cast(AddOnePreprocessor, adapter.context.preprocessor)
    policy = cast(TinyDiffusionPolicy, adapter.context.policy)
    postprocessor = cast(ScalePostprocessor, adapter.context.postprocessor)
    direct_action = postprocessor(policy.select_action(preprocessor(sample), noise=torch.as_tensor(noise)))

    policy.reset()
    selected = adapter.select_action(sample, noise=noise)

    np.testing.assert_array_equal(selected.action_array, direct_action["action"].numpy())


def test_diffusion_capture_episode_preserves_action_queue_misses() -> None:
    adapter = make_fixture_adapter()
    trace = adapter.capture_episode(
        [{"state": torch.tensor([[float(step), 0.0, 1.0]])} for step in range(7)],
        episode_id="fixture-success",
        outcome="success",
    )

    assert len(trace.selections) == 3
    assert [selection.representations[0].episode_step for selection in trace.selections] == [0, 3, 6]
    assert all(selection.representations[0].kind == "observation_conditioning" for selection in trace.selections)


def test_load_diffusion_policy_uses_supported_upstream_factories(monkeypatch: pytest.MonkeyPatch) -> None:
    config_calls: list[tuple[str, str]] = []
    policy_calls: list[tuple[object, object]] = []
    processor_calls: list[tuple[object, str, str]] = []

    class FakeDiffusionConfig:
        @classmethod
        def from_pretrained(cls, repo_id: str, *, revision: str) -> SimpleNamespace:
            config_calls.append((repo_id, revision))
            return SimpleNamespace(pretrained_path=None, pretrained_revision=None, device=None)

    module = ModuleType("lerobot.policies.diffusion.configuration_diffusion")
    module.DiffusionConfig = FakeDiffusionConfig  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, module.__name__, module)

    policy = TinyDiffusionPolicy()

    def make_policy(config: object, *, ds_meta: object) -> object:
        policy_calls.append((config, ds_meta))
        return policy

    def make_processors(
        config: object,
        *,
        pretrained_path: str,
        pretrained_revision: str,
        **kwargs: object,
    ) -> tuple[AddOnePreprocessor, ScalePostprocessor]:
        del kwargs
        processor_calls.append((config, pretrained_path, pretrained_revision))
        return AddOnePreprocessor(), ScalePostprocessor()

    api = LeRobotAPI(
        make_policy=make_policy,
        make_pre_post_processors=make_processors,
        dataset_type=object,
        streaming_dataset_type=object,
        policy_processor_pipeline_type=object,
        make_env=lambda **kwargs: kwargs,
        evaluation_main=lambda **kwargs: kwargs,
        register_third_party_plugins=lambda: None,
    )
    spec = DiffusionCheckpointSpec(
        policy_repo_id="fixture/diffusion",
        policy_revision="policy-revision",
        dataset_repo_id="fixture/dataset",
        dataset_revision="dataset-revision",
    )
    adapter = load_diffusion_policy(spec, api=api, dataset_meta=SimpleNamespace(stats={}), device="cpu")

    assert config_calls == [("fixture/diffusion", "policy-revision")]
    assert len(policy_calls) == 1
    assert processor_calls == [(policy_calls[0][0], "fixture/diffusion", "policy-revision")]
    assert adapter.context.policy is policy
    assert adapter.context.metadata["dataset_revision"] == "dataset-revision"


def test_diffusion_analysis_separates_episode_time_and_diffusion_timestep_metrics() -> None:
    traces: list[DiffusionEpisodeTrace] = []
    for outcome, offset in (("success", 2.0), ("failure", -2.0)):
        for episode_index in range(4):
            adapter = make_fixture_adapter()
            trace = adapter.capture_episode(
                [{"state": torch.tensor([[offset + episode_index * 0.1 + step, 0.0, 1.0]])} for step in range(7)],
                episode_id=f"{outcome}-{episode_index}",
                outcome=cast(Literal["success", "failure"], outcome),
            )
            traces.append(trace)

    result = analyze_diffusion_traces(traces, random_state=0)

    assert result.projected_conditioning.shape == (24, 2)
    assert result.conditioning_probe.accuracy >= result.conditioning_controls.majority_class
    assert 0.0 <= result.conditioning_density_auroc <= 1.0
    assert set(result.episode_time_lengths) == {trace.episode_id for trace in traces}
    assert all(set(metrics) == {"9", "4", "0"} for metrics in result.timestep_time_lengths.values())
    assert result.metadata["axes"] == ["episode_time", "action_chunk_position", "diffusion_timestep"]
    assert result.metadata["causal_intervention"] is False


@pytest.mark.network
@pytest.mark.large_download
def test_pinned_public_diffusion_checkpoint_pair_loads_through_lerobot_factories() -> None:
    pytest.importorskip("lerobot")
    from latent_anything.integrations.lerobot import check_lerobot_compatibility

    report = check_lerobot_compatibility()
    if not report.supported:
        pytest.skip(report.diagnostic)
    adapter = load_diffusion_policy(DEFAULT_DIFFUSION_CHECKPOINT, device="cpu")
    assert adapter.checkpoint.policy_revision == DEFAULT_DIFFUSION_CHECKPOINT.policy_revision
    assert adapter.checkpoint.dataset_revision == DEFAULT_DIFFUSION_CHECKPOINT.dataset_revision
    assert adapter.checkpoint.environment_task == "AlohaInsertion-v0"

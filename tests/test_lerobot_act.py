"""Offline ACT capture tests plus an opt-in pinned checkpoint smoke test."""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Mapping
from types import ModuleType, SimpleNamespace
from typing import Literal, cast

import numpy as np
import pytest
import torch
from torch import nn

from latent_anything.integrations.lerobot import LeRobotAPI, LeRobotPolicyContext
from latent_anything.integrations.lerobot_act import (
    ACT_CAPTURE_LOCATION,
    DEFAULT_ACT_CHECKPOINT,
    ACTCheckpointSpec,
    ACTEpisodeTrace,
    ACTPolicyAdapter,
    analyze_act_traces,
    load_act_policy,
)


def test_act_module_import_stays_lazy_in_base_install() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import latent_anything.integrations.lerobot_act; "
            "assert not any(name == 'lerobot' or name.startswith('lerobot.') for name in sys.modules)",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


class TinyDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(3, 3, bias=False)
        with torch.no_grad():
            self.projection.weight.copy_(torch.eye(3))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        token = self.projection(state)
        return torch.stack((token, token + 0.25), dim=0)


class TinyACTNetwork(nn.Module):
    decoder: TinyDecoder

    def __init__(self) -> None:
        super().__init__()
        self.decoder = TinyDecoder()


class TinyACTPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(dim_model=3)
        self.model = TinyACTNetwork()
        self.reset_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1

    @torch.no_grad()
    def select_action(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        decoder_output = self.model.decoder(batch["state"])
        return {"action": decoder_output[0, 0]}


class AddOnePreprocessor:
    def __call__(self, sample: Mapping[str, object]) -> dict[str, torch.Tensor]:
        value = sample["state"]
        if not isinstance(value, torch.Tensor):
            raise TypeError("fixture state must be a Tensor")
        return {"state": value + 1.0}


class ScalePostprocessor:
    def __call__(self, action: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {"action": action["action"] * 2.0}


def make_fixture_adapter(*, checkpoint: ACTCheckpointSpec | None = None) -> ACTPolicyAdapter:
    policy = TinyACTPolicy()
    context = LeRobotPolicyContext(
        policy_name="act",
        policy=policy,
        preprocessor=AddOnePreprocessor(),
        postprocessor=ScalePostprocessor(),
        dataset=SimpleNamespace(repo_id="fixture/dataset", revision="fixture-dataset-rev"),
    )
    return ACTPolicyAdapter(context, checkpoint=checkpoint or ACTCheckpointSpec(), representation_dim=3)


def test_act_adapter_matches_direct_preprocess_select_postprocess_action() -> None:
    adapter = make_fixture_adapter()
    sample = {"state": torch.tensor([[1.0, 2.0, 3.0]])}

    preprocessor = cast(AddOnePreprocessor, adapter.context.preprocessor)
    policy = cast(TinyACTPolicy, adapter.context.policy)
    postprocessor = cast(ScalePostprocessor, adapter.context.postprocessor)
    prepared = preprocessor(sample)
    direct_raw = policy.select_action(prepared)
    direct_action = postprocessor(direct_raw)

    adapter.reset()
    selected = adapter.select_action(sample)

    assert isinstance(direct_action, dict)
    np.testing.assert_array_equal(selected.action_array, direct_action["action"].numpy())
    assert selected.representation is not None
    assert selected.representation.capture_metadata.location == ACT_CAPTURE_LOCATION
    assert selected.representation.capture_metadata.shape == (2, 1, 3)
    np.testing.assert_array_equal(selected.representation.latent.values, np.array([2.0, 3.0, 4.0]))
    assert selected.representation.latent.provenance["query_index"] == 0


def test_act_adapter_preserves_queue_misses_as_uncaptured_action() -> None:
    adapter = make_fixture_adapter()
    policy = cast(TinyACTPolicy, adapter.context.policy)
    original_select = policy.select_action
    calls = 0

    def select_once(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        nonlocal calls
        calls += 1
        if calls == 1:
            return original_select(batch)
        return {"action": torch.zeros(3)}

    policy.select_action = select_once
    first = adapter.select_action({"state": torch.zeros((1, 3))})
    second = adapter.select_action({"state": torch.ones((1, 3))})

    assert first.representation is not None
    assert second.representation is None
    np.testing.assert_array_equal(second.action_array, np.zeros(3))


def test_load_act_policy_uses_supported_upstream_factories(monkeypatch: pytest.MonkeyPatch) -> None:
    config_calls: list[tuple[str, str]] = []
    policy_calls: list[tuple[object, object]] = []
    processor_calls: list[tuple[object, str, str]] = []

    class FakeACTConfig:
        @classmethod
        def from_pretrained(cls, repo_id: str, *, revision: str) -> SimpleNamespace:
            config_calls.append((repo_id, revision))
            return SimpleNamespace(
                pretrained_path=None,
                pretrained_revision=None,
                device=None,
                dim_model=3,
            )

    module = ModuleType("lerobot.policies.act.configuration_act")
    module.ACTConfig = FakeACTConfig  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, module.__name__, module)

    policy = TinyACTPolicy()

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
    spec = ACTCheckpointSpec(
        policy_repo_id="fixture/act",
        policy_revision="policy-rev",
        dataset_repo_id="fixture/dataset",
        dataset_revision="dataset-rev",
    )
    adapter = load_act_policy(spec, api=api, dataset_meta=SimpleNamespace(stats={}), device="cpu")

    assert config_calls == [("fixture/act", "policy-rev")]
    assert len(policy_calls) == 1
    assert processor_calls == [(policy_calls[0][0], "fixture/act", "policy-rev")]
    assert adapter.context.policy is policy
    assert adapter.context.metadata["dataset_revision"] == "dataset-rev"


def test_act_trace_analysis_reports_projection_probe_trajectory_and_controls() -> None:
    adapter = make_fixture_adapter()
    traces: list[ACTEpisodeTrace] = []
    for outcome, offset in (("success", 2.0), ("failure", -2.0)):
        outcome_value = cast(Literal["success", "failure"], outcome)
        for episode_index in range(4):
            samples = [{"state": torch.tensor([[offset + episode_index * 0.1 + step, 0.0, 0.0]])} for step in range(4)]
            traces.append(
                adapter.capture_episode(
                    samples,
                    episode_id=f"{outcome}-{episode_index}",
                    outcome=outcome_value,
                )
            )

    result = analyze_act_traces(traces, random_state=0)

    assert result.projected.shape == (32, 2)
    assert result.probe.accuracy >= result.controls.majority_class
    assert result.controls.shuffled_label <= 1.0
    expected_ids = {f"{outcome}-{index}" for outcome in ("success", "failure") for index in range(4)}
    assert set(result.trajectory_lengths) == expected_ids
    assert result.metadata["causal_intervention"] is False


@pytest.mark.network
@pytest.mark.large_download
def test_pinned_public_act_checkpoint_pair_loads_through_lerobot_factories() -> None:
    pytest.importorskip("lerobot")
    adapter = load_act_policy(DEFAULT_ACT_CHECKPOINT, device="cpu")

    assert adapter.checkpoint.policy_revision == ACTCheckpointSpec().policy_revision
    assert adapter.checkpoint.dataset_revision == ACTCheckpointSpec().dataset_revision
    assert adapter.metadata.capture_location == "model.decoder"
    assert adapter.metadata.representation_role == "first_action_decoder_query"

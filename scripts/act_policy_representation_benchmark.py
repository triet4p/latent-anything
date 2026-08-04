"""Deterministic observational ACT representation benchmark using a tiny policy fixture.

The fixture exercises the same adapter lifecycle as a real checkpoint while
keeping the default evidence run offline. The marked integration test covers
the pinned public LeRobot checkpoint separately.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Literal, cast

import numpy as np
import torch
from torch import nn

from latent_anything.integrations.lerobot import LeRobotPolicyContext
from latent_anything.integrations.lerobot_act import (
    DEFAULT_ACT_CHECKPOINT,
    ACTEpisodeTrace,
    ACTPolicyAdapter,
    analyze_act_traces,
)


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

    def reset(self) -> None:
        pass

    @torch.no_grad()
    def select_action(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        decoder_output = self.model.decoder(batch["state"])
        return {"action": decoder_output[0, 0]}


class AddOnePreprocessor:
    def __call__(self, sample: dict[str, object]) -> dict[str, torch.Tensor]:
        state = sample["state"]
        if not isinstance(state, torch.Tensor):
            raise TypeError("fixture state must be a Tensor")
        return {"state": state + 1.0}


class IdentityPostprocessor:
    def __call__(self, action: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return action


def build_fixture_adapter() -> ACTPolicyAdapter:
    context = LeRobotPolicyContext(
        policy_name="act",
        policy=TinyACTPolicy(),
        preprocessor=AddOnePreprocessor(),
        postprocessor=IdentityPostprocessor(),
        dataset=SimpleNamespace(repo_id="fixture/act-dataset", revision="fixture-rev"),
    )
    return ACTPolicyAdapter(context, checkpoint=DEFAULT_ACT_CHECKPOINT, representation_dim=3)


def build_traces(adapter: ACTPolicyAdapter) -> list[ACTEpisodeTrace]:
    traces: list[ACTEpisodeTrace] = []
    for outcome, offset in (("success", 2.0), ("failure", -2.0)):
        for episode_index in range(6):
            samples = [{"state": torch.tensor([[offset + episode_index * 0.1 + step, 0.0, 0.0]])} for step in range(5)]
            outcome_value = cast(Literal["success", "failure"], outcome)
            traces.append(
                adapter.capture_episode(
                    samples,
                    episode_id=f"{outcome}-{episode_index}",
                    outcome=outcome_value,
                )
            )
    return traces


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/act_policy_representation_benchmark.json"),
    )
    args = parser.parse_args()

    np.random.seed(0)
    adapter = build_fixture_adapter()
    traces = build_traces(adapter)
    result = analyze_act_traces(traces, random_state=0)
    payload = {
        "claim_scope": "offline fixture evidence for observational ACT representation analysis",
        "causal_intervention": False,
        "checkpoint_pair": DEFAULT_ACT_CHECKPOINT.to_dict(),
        "capture_metadata": adapter.metadata.to_dict(),
        "episodes": {
            "total": len(traces),
            "success": sum(1 for trace in traces if trace.outcome == "success"),
            "failure": sum(1 for trace in traces if trace.outcome == "failure"),
        },
        "acceptance": {
            "both_outcomes_present": True,
            "probe_beats_majority_control": result.probe.accuracy >= result.controls.majority_class,
            "projection_components": 2,
            "controls": ["majority_class", "shuffled_label", "raw_input"],
        },
        "analysis": result.to_dict(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload["acceptance"], sort_keys=True))


if __name__ == "__main__":
    main()

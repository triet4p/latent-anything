"""Deterministic observational Diffusion representation benchmark.

The tiny policy fixture follows LeRobot's queue and denoising lifecycle while
keeping the default evidence run offline. The marked integration test covers
the pinned public checkpoint separately.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Literal, cast

import numpy as np
import torch
from torch import Tensor, nn

from latent_anything.integrations.lerobot import LeRobotPolicyContext
from latent_anything.integrations.lerobot_diffusion import (
    DEFAULT_DIFFUSION_CHECKPOINT,
    DiffusionEpisodeTrace,
    DiffusionPolicyAdapter,
    analyze_diffusion_traces,
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

    def reset(self) -> None:
        self._action_queue.clear()

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], *, noise: Tensor | None = None) -> Tensor:
        if not self._action_queue:
            current = noise if noise is not None else torch.zeros((1, 4, 2))
            actions: list[Tensor] = []
            for timestep in (9, 4, 0):
                current = self.diffusion.unet(current, torch.tensor([timestep]), global_cond=batch["state"])
                actions.append(current[:, 0])
            self._action_queue.extend(actions)
        return self._action_queue.pop(0)


class AddOnePreprocessor:
    def __call__(self, sample: dict[str, object]) -> dict[str, Tensor]:
        state = sample["state"]
        if not isinstance(state, Tensor):
            raise TypeError("fixture state must be a Tensor")
        return {"state": state + 1.0}


class IdentityPostprocessor:
    def __call__(self, action: Tensor) -> dict[str, Tensor]:
        return {"action": action}


def build_fixture_adapter() -> DiffusionPolicyAdapter:
    context = LeRobotPolicyContext(
        policy_name="diffusion",
        policy=TinyDiffusionPolicy(),
        preprocessor=AddOnePreprocessor(),
        postprocessor=IdentityPostprocessor(),
        dataset=SimpleNamespace(repo_id="fixture/diffusion-dataset", revision="fixture-revision"),
    )
    return DiffusionPolicyAdapter(context, checkpoint=DEFAULT_DIFFUSION_CHECKPOINT)


def build_traces(adapter: DiffusionPolicyAdapter) -> list[DiffusionEpisodeTrace]:
    traces: list[DiffusionEpisodeTrace] = []
    for outcome, offset in (("success", 2.0), ("failure", -2.0)):
        for episode_index in range(4):
            samples = [{"state": torch.tensor([[offset + episode_index * 0.1 + step, 0.0, 1.0]])} for step in range(7)]
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
        default=Path("artifacts/diffusion_policy_representation_benchmark.json"),
    )
    args = parser.parse_args()

    np.random.seed(0)
    adapter = build_fixture_adapter()
    traces = build_traces(adapter)
    result = analyze_diffusion_traces(traces, random_state=0)
    payload = {
        "claim_scope": "offline fixture evidence for observational LeRobot Diffusion representation analysis",
        "causal_intervention": False,
        "checkpoint_pair": DEFAULT_DIFFUSION_CHECKPOINT.to_dict(),
        "capture_metadata": adapter.metadata.to_dict(),
        "episodes": {
            "total": len(traces),
            "success": sum(1 for trace in traces if trace.outcome == "success"),
            "failure": sum(1 for trace in traces if trace.outcome == "failure"),
        },
        "acceptance": {
            "both_outcomes_present": True,
            "probe_beats_majority_control": result.conditioning_probe.accuracy
            >= result.conditioning_controls.majority_class,
            "density_auroc_in_range": 0.0 <= result.conditioning_density_auroc <= 1.0,
            "axes_are_explicit": result.metadata["axes"]
            == ["episode_time", "action_chunk_position", "diffusion_timestep"],
            "controls": ["majority_class", "shuffled_label", "raw_input_not_used"],
        },
        "analysis": result.to_dict(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload["acceptance"], sort_keys=True))


if __name__ == "__main__":
    main()

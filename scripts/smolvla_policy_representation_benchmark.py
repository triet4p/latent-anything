"""Deterministic offline SmolVLA capture/intervention benchmark using a tiny fixture.

The fixture mirrors LeRobot's official SmolVLA ``select_action`` seams (SigLIP
vision encoder, language embedding table, state projection, action-expert norm,
flow-matching denoising, action queue). The marked public checkpoint lane runs
separately on a CUDA device.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import torch
from torch import Tensor, nn

from latent_anything.integrations.lerobot import LeRobotPolicyContext
from latent_anything.integrations.lerobot_smolvla import (
    DEFAULT_SMOLVLA_CHECKPOINT,
    SmolVLAIntervention,
    SmolVLAPolicyAdapter,
    measure_smolvla_intervention,
)

HIDDEN = 8
EXPERT = 6
PATCHES = 4
LANG_LEN = 4
CHUNK = 4
NUM_STEPS = 3
MAX_STATE = 4
MAX_ACTION = 4
ACTION_DIM = 2
VOCAB = 16


class TinyVisionModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(3 * 8 * 8, HIDDEN)

    def forward(self, *, pixel_values: Tensor, patch_attention_mask: Tensor | None = None) -> SimpleNamespace:
        del patch_attention_mask
        flat = pixel_values.reshape(pixel_values.shape[0], -1)
        pooled = self.projection(flat)
        tokens = pooled[:, :PATCHES, None].repeat(1, 1, HIDDEN) + pooled[:, None, :HIDDEN]
        return SimpleNamespace(last_hidden_state=tokens)


class TinyTextModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(VOCAB, HIDDEN)


class TinyVLMContainer(nn.Module):
    vision_model: TinyVisionModel
    text_model: TinyTextModel

    def __init__(self) -> None:
        super().__init__()
        self.vision_model = TinyVisionModel()
        self.text_model = TinyTextModel()


class TinyVLM(nn.Module):
    model: TinyVLMContainer

    def __init__(self) -> None:
        super().__init__()
        self.model = TinyVLMContainer()


class TinyLMExpert(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([nn.Identity()])
        self.norm = nn.LayerNorm(EXPERT)


class TinyVLMWithExpert(nn.Module):
    expert_hidden_size = EXPERT

    def __init__(self) -> None:
        super().__init__()
        self.vlm = TinyVLM()
        self.lm_expert = TinyLMExpert()
        self.config = SimpleNamespace(text_config=SimpleNamespace(hidden_size=HIDDEN))
        self.processor = SimpleNamespace(tokenizer=SimpleNamespace(fake_image_token_id=0, global_image_token_id=1))

    def embed_image(self, image: Tensor) -> Tensor:
        return self.vlm.model.vision_model(pixel_values=image).last_hidden_state

    def embed_language_tokens(self, tokens: Tensor) -> Tensor:
        return self.vlm.model.text_model.embed_tokens(tokens)

    def forward(
        self,
        inputs_embeds: list[Tensor | None],
        **kwargs: object,
    ) -> tuple[list[Tensor | None], None]:
        del kwargs
        suffix = inputs_embeds[1]
        if suffix is None:
            return [inputs_embeds[0], None], None
        return [None, self.lm_expert.norm(suffix)], None


class TinyVLAFlowMatching(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.vlm_with_expert = TinyVLMWithExpert()
        self.state_proj = nn.Linear(MAX_STATE, HIDDEN)
        self.action_in_proj = nn.Linear(MAX_ACTION, EXPERT)
        self.action_out_proj = nn.Linear(EXPERT, MAX_ACTION)
        self.context_proj = nn.Linear(2 * HIDDEN + HIDDEN + HIDDEN, MAX_ACTION)

    @torch.no_grad()
    def sample_actions(
        self,
        images: list[Tensor],
        img_masks: list[Tensor],
        lang_tokens: Tensor,
        lang_masks: Tensor,
        state: Tensor,
        noise: Tensor | None = None,
    ) -> Tensor:
        del img_masks, lang_masks
        prefix_parts: list[Tensor] = [self.vlm_with_expert.embed_image(img) for img in images]
        lang_emb = self.vlm_with_expert.embed_language_tokens(lang_tokens)
        state_emb = self.state_proj(state)[:, None, :]
        vision_mean = torch.cat([part.mean(dim=1) for part in prefix_parts], dim=1)
        context = torch.cat([vision_mean, lang_emb.mean(dim=1), state_emb[:, 0, :]], dim=1)
        context_bias = self.context_proj(context)[:, None, :]
        x_t = noise.to(torch.float32) if noise is not None else torch.zeros(1, CHUNK, MAX_ACTION)
        for _ in range(NUM_STEPS):
            suffix = self.action_in_proj(x_t)
            outputs, _ = self.vlm_with_expert.forward(inputs_embeds=[None, suffix])
            velocity = self.action_out_proj(cast(Tensor, outputs[1])) + context_bias
            x_t = x_t - 0.1 * velocity
        return x_t


class TinySmolVLAPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.config = SimpleNamespace(
            action_feature=SimpleNamespace(shape=(ACTION_DIM,)),
            max_action_dim=MAX_ACTION,
            max_state_dim=MAX_STATE,
            chunk_size=CHUNK,
            num_steps=NUM_STEPS,
            n_action_steps=CHUNK,
            image_features={
                "observation.images.camera1": SimpleNamespace(),
                "observation.images.camera2": SimpleNamespace(),
                "observation.images.camera3": SimpleNamespace(),
            },
        )
        self.model = TinyVLAFlowMatching()
        self._action_queue: list[Tensor] = []

    def reset(self) -> None:
        self._action_queue.clear()

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], *, noise: Tensor | None = None) -> Tensor:
        if not self._action_queue:
            images, img_masks = self.prepare_images(batch)
            state = self.prepare_state(batch)
            lang_tokens = batch["observation.language.tokens"]
            lang_masks = batch["observation.language.attention_mask"]
            actions = self.model.sample_actions(images, img_masks, lang_tokens, lang_masks, state, noise=noise)
            actions = actions[:, :, :ACTION_DIM]
            self._action_queue.extend(actions.transpose(0, 1)[: self.config.n_action_steps])
        return self._action_queue.pop(0)

    def prepare_images(self, batch: dict[str, Tensor]) -> tuple[list[Tensor], list[Tensor]]:
        images: list[Tensor] = []
        masks: list[Tensor] = []
        for key in self.config.image_features:
            if key not in batch:
                continue
            image = batch[key]
            images.append(image)
            masks.append(torch.ones(image.shape[0], dtype=torch.bool))
        return images, masks

    def prepare_state(self, batch: dict[str, Tensor]) -> Tensor:
        state = batch["observation.state"]
        return nn.functional.pad(state, (0, MAX_STATE - state.shape[-1]))


class TinySmolVLAPreprocessor:
    def __call__(self, sample: dict[str, object]) -> dict[str, Tensor]:
        image = sample["observation.images.image"]
        image2 = sample["observation.images.image2"]
        state = sample["observation.state"]
        task = sample["task"]
        if not isinstance(image, Tensor) or not isinstance(image2, Tensor) or not isinstance(state, Tensor):
            raise TypeError("fixture tensors must be torch.Tensor values")
        if not isinstance(task, str):
            raise TypeError("fixture task must be a string")
        tokens = torch.tensor([ord(char) % VOCAB for char in task][:LANG_LEN], dtype=torch.long)
        tokens = nn.functional.pad(tokens, (0, LANG_LEN - tokens.shape[0]), value=0)[None]
        return {
            "observation.images.camera1": image[None],
            "observation.images.camera2": image2[None],
            "observation.state": state[None],
            "observation.language.tokens": tokens,
            "observation.language.attention_mask": torch.ones_like(tokens, dtype=torch.bool),
        }


class TinySmolVLAPostprocessor:
    def __call__(self, action: Tensor) -> dict[str, Tensor]:
        return {"action": action}


def make_sample(
    *,
    task: str = "alpha task",
    image: Tensor | None = None,
    image2: Tensor | None = None,
    state: Tensor | None = None,
) -> dict[str, object]:
    return {
        "observation.images.image": image if image is not None else torch.zeros(3, 8, 8),
        "observation.images.image2": image2 if image2 is not None else torch.full((3, 8, 8), 0.5),
        "observation.state": state if state is not None else torch.zeros(MAX_STATE),
        "task": task,
    }


def build_fixture_adapter() -> SmolVLAPolicyAdapter:
    context = LeRobotPolicyContext(
        policy_name="smolvla",
        policy=TinySmolVLAPolicy(),
        preprocessor=TinySmolVLAPreprocessor(),
        postprocessor=TinySmolVLAPostprocessor(),
        dataset=SimpleNamespace(repo_id="fixture/libero", revision="fixture-rev"),
    )
    return SmolVLAPolicyAdapter(context, checkpoint=DEFAULT_SMOLVLA_CHECKPOINT)


def build_samples() -> list[dict[str, object]]:
    """Return three samples spaced one full chunk apart so each executes a query."""

    samples: list[dict[str, object]] = []
    for step in (0, CHUNK, 2 * CHUNK):
        state = torch.zeros(4)
        state[0] = float(step) / 10.0
        samples.append(make_sample(state=state))
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/smolvla_policy_representation_benchmark.json"),
    )
    args = parser.parse_args()

    np.random.seed(0)
    torch.manual_seed(0)
    adapter = build_fixture_adapter()
    samples = build_samples()
    noise = np.full((1, CHUNK, MAX_ACTION), 0.25)
    direction = np.array([0.5, -0.5, 0.25, -0.25, 0.1, -0.1], dtype=np.float64)
    intervention = SmolVLAIntervention(direction=direction, strength=2.0)

    adapter.reset()
    baseline = adapter.select_action(samples[0], noise=noise)
    identity = SmolVLAIntervention(direction=direction, strength=0.0)
    adapter.reset()
    identity_selection = adapter.select_action(samples[0], noise=noise, intervention=identity)
    identity_is_exact = bool(np.array_equal(identity_selection.action_array, baseline.action_array))

    measurement = measure_smolvla_intervention(
        adapter,
        samples,
        noise=noise,
        intervention=intervention,
        alternate_prompt_sample=make_sample(task="beta task"),
        camera_swapped_sample=make_sample(image=torch.full((3, 8, 8), 0.5), image2=torch.zeros(3, 8, 8)),
    )
    kind_counts = {
        kind: sum(1 for item in baseline.representations if item.kind == kind)
        for kind in ("vision_context", "language_context", "state_context", "action_expert")
    }
    payload = {
        "claim_scope": (
            "offline fixture evidence for bounded SmolVLA action-expert intervention: "
            "the official-pipeline capture seams, bit-exact identity at strength zero, and "
            "on-target/off-target/drift/prompt/camera measurements are deterministic"
        ),
        "causal_environment_effect": False,
        "checkpoint_pair": DEFAULT_SMOLVLA_CHECKPOINT.to_dict(),
        "hardware_profile": adapter.metadata.hardware_profile.to_dict(),
        "capture_metadata": adapter.metadata.to_dict(),
        "capture_kinds": kind_counts,
        "denoising_steps_per_query": adapter.num_steps,
        "intervention": intervention.to_dict(),
        "acceptance": {
            "identity_at_zero_strength_is_bit_exact": identity_is_exact,
            "action_change_norm_positive": measurement.action_change_norm > 0.0,
            "on_target_fraction_above_linear_fixture_bound": measurement.on_target_fraction >= 0.99,
            "off_target_fraction_below_fixture_bound": (
                measurement.off_target_norm / (measurement.on_target_norm + measurement.off_target_norm) <= 0.01
            ),
            "decomposition_bounds_consistent": (
                measurement.on_target_norm + measurement.off_target_norm >= measurement.action_change_norm - 1e-9
            ),
            "representation_drift_positive": measurement.representation_drift > 0.0,
            "prompt_sensitivity_positive": measurement.prompt_sensitivity > 0.0,
            "camera_order_sensitivity_positive": measurement.camera_order_sensitivity > 0.0,
        },
        "measurement": measurement.to_dict(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload["acceptance"], sort_keys=True))


if __name__ == "__main__":
    main()

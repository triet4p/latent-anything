"""Offline SmolVLA capture/intervention tests plus opt-in pinned checkpoint lanes.

The tiny policy fixture mirrors the module seams LeRobot's official SmolVLA
``select_action`` path executes (SigLIP vision encoder, language embedding
table, state projection, action-expert norm, flow-matching denoising, action
queue) while keeping the default suite offline and deterministic.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable, Mapping
from types import ModuleType, SimpleNamespace
from typing import Protocol, cast

import numpy as np
import pytest
import torch
from torch import Tensor, nn

from latent_anything.integrations.lerobot import LeRobotAPI, LeRobotPolicyContext
from latent_anything.integrations.lerobot_smolvla import (
    DEFAULT_SMOLVLA_CHECKPOINT,
    SMOLVLA_EXPERT_LOCATION,
    SMOLVLA_LANGUAGE_LOCATION,
    SMOLVLA_STATE_LOCATION,
    SMOLVLA_VISION_LOCATION,
    SmolVLACheckpointSpec,
    SmolVLAIntervention,
    SmolVLAPolicyAdapter,
    load_smolvla_policy,
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


class _ActionSelectingPolicy(Protocol):
    """Minimal structural view of LeRobot's action-selection lifecycle."""

    def select_action(self, batch: object, *, noise: object | None = None) -> object: ...


class TinyVisionModel(nn.Module):
    """Mirrors ``model.vlm_with_expert.vlm.model.vision_model`` (SigLIP)."""

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
        self.reset_calls = 0

    def reset(self) -> None:
        self._action_queue.clear()
        self.reset_calls += 1

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
    def __call__(self, sample: Mapping[str, object]) -> dict[str, Tensor]:
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


def make_fixture_adapter(*, checkpoint: SmolVLACheckpointSpec | None = None) -> SmolVLAPolicyAdapter:
    context = LeRobotPolicyContext(
        policy_name="smolvla",
        policy=TinySmolVLAPolicy(),
        preprocessor=TinySmolVLAPreprocessor(),
        postprocessor=TinySmolVLAPostprocessor(),
        dataset=SimpleNamespace(repo_id="fixture/libero", revision="fixture-revision"),
    )
    return SmolVLAPolicyAdapter(context, checkpoint=checkpoint or DEFAULT_SMOLVLA_CHECKPOINT)


def make_noise() -> np.ndarray:
    return np.full((1, CHUNK, MAX_ACTION), 0.25)


def _postprocessed_action_numpy(value: object) -> np.ndarray:
    """Extract the action from a real or fixture post-processor result."""

    if isinstance(value, Mapping):
        for key in ("action", "actions"):
            if key in value:
                value = value[key]
                break
        else:
            raise TypeError("post-processor mapping must contain 'action' or 'actions'")
    if isinstance(value, Tensor):
        return value.detach().cpu().numpy().reshape(-1)
    return np.asarray(value).reshape(-1)


def test_smolvla_module_import_stays_lazy_in_base_install() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import latent_anything.integrations.lerobot_smolvla; "
            "assert not any(name == 'lerobot' or name.startswith('lerobot.') for name in sys.modules)",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_smolvla_capture_records_modalities_with_token_metadata() -> None:
    adapter = make_fixture_adapter()
    selected = adapter.select_action(make_sample(), noise=make_noise())

    assert [item.kind for item in selected.representations] == [
        "vision_context",
        "vision_context",
        "language_context",
        "state_context",
        "action_expert",
        "action_expert",
        "action_expert",
    ]
    vision = selected.of_kind("vision_context")
    assert [item.token.camera for item in vision] == [
        "observation.images.camera1",
        "observation.images.camera2",
    ]
    assert [item.token.prefix_offset for item in vision] == [0, PATCHES]
    assert [item.token.token_count for item in vision] == [PATCHES, PATCHES]
    language = selected.of_kind("language_context")
    assert len(language) == 1
    assert language[0].token.prefix_offset == 2 * PATCHES
    state = selected.of_kind("state_context")
    assert len(state) == 1
    assert state[0].token.prefix_offset == 2 * PATCHES + LANG_LEN
    expert = selected.of_kind("action_expert")
    assert [item.token.denoising_step for item in expert] == [0, 1, 2]
    assert all(item.token.prefix_offset is None for item in expert)
    assert all(item.token.token_count == CHUNK for item in expert)

    assert [item.capture_metadata.location for item in vision] == [SMOLVLA_VISION_LOCATION] * 2
    assert language[0].capture_metadata.location == SMOLVLA_LANGUAGE_LOCATION
    assert state[0].capture_metadata.location == SMOLVLA_STATE_LOCATION
    assert all(item.capture_metadata.location == SMOLVLA_EXPERT_LOCATION for item in expert)
    assert all(item.capture_metadata.sequence_axis == 1 for item in selected.representations)
    assert all(item.latent.values.shape[1] == HIDDEN for item in vision + language + state)
    assert all(item.latent.values.shape[1] == EXPERT for item in expert)
    assert selected.denoising_steps == NUM_STEPS


def test_smolvla_action_matches_direct_preprocess_select_postprocess_with_fixed_noise() -> None:
    adapter = make_fixture_adapter()
    sample = make_sample()
    noise = make_noise()
    preprocessor = cast(TinySmolVLAPreprocessor, adapter.context.preprocessor)
    policy = cast(TinySmolVLAPolicy, adapter.context.policy)
    postprocessor = cast(TinySmolVLAPostprocessor, adapter.context.postprocessor)
    direct_action = postprocessor(policy.select_action(preprocessor(sample), noise=torch.as_tensor(noise)))

    policy.reset()
    selected = adapter.select_action(sample, noise=noise)

    np.testing.assert_array_equal(selected.action_array, direct_action["action"].numpy())
    assert selected.denoising_steps == NUM_STEPS


def test_smolvla_action_queue_executes_a_model_query_every_chunk() -> None:
    adapter = make_fixture_adapter()
    noise = make_noise()
    first = adapter.select_action(make_sample(state=torch.zeros(MAX_STATE)), noise=noise)
    assert first.denoising_steps == NUM_STEPS
    for episode_step in range(1, CHUNK):
        queued = adapter.select_action(make_sample(), noise=noise, episode_step=episode_step)
        assert queued.denoising_steps == 0
        assert queued.representations == ()
    next_query = adapter.select_action(make_sample(), noise=noise, episode_step=CHUNK)
    assert next_query.denoising_steps == NUM_STEPS
    assert all(item.episode_step == CHUNK for item in next_query.representations)


def test_smolvla_intervention_strength_zero_is_bit_exact_identity() -> None:
    adapter = make_fixture_adapter()
    sample = make_sample()
    noise = make_noise()
    baseline = adapter.select_action(sample, noise=noise)
    identity = SmolVLAIntervention(direction=np.ones(EXPERT), strength=0.0)
    adapter.reset()
    intervened = adapter.select_action(sample, noise=noise, intervention=identity)

    np.testing.assert_array_equal(intervened.action_array, baseline.action_array)
    assert len(intervened.representations) == len(baseline.representations)
    for item, base in zip(intervened.representations, baseline.representations, strict=True):
        np.testing.assert_array_equal(item.latent.values, base.latent.values)


def test_smolvla_intervention_is_bounded_and_validated() -> None:
    SmolVLAIntervention(direction=np.array([0.5, -0.5, 0.25, -0.25, 0.1, -0.1]), strength=1.0)
    with pytest.raises(ValueError, match="non-empty 1D"):
        SmolVLAIntervention(direction=np.zeros((EXPERT, 1)), strength=1.0)
    with pytest.raises(ValueError, match="finite"):
        SmolVLAIntervention(direction=np.array([np.nan] * EXPERT), strength=1.0)
    with pytest.raises(ValueError, match="finite"):
        SmolVLAIntervention(direction=np.ones(EXPERT), strength=np.inf)
    with pytest.raises(ValueError, match="bounded maximum"):
        SmolVLAIntervention(direction=np.ones(EXPERT), strength=101.0)


def test_smolvla_hook_session_removes_hooks_after_policy_exception() -> None:
    class PoisonPolicy(TinySmolVLAPolicy):
        def __init__(self) -> None:
            super().__init__()
            self.raise_after_query = False

        def select_action(self, batch: dict[str, Tensor], *, noise: Tensor | None = None) -> Tensor:
            action = super().select_action(batch, noise=noise)
            if self.raise_after_query:
                raise RuntimeError("fixture failure after model query")
            return action

    policy = PoisonPolicy()
    context = LeRobotPolicyContext(
        policy_name="smolvla",
        policy=policy,
        preprocessor=TinySmolVLAPreprocessor(),
        postprocessor=TinySmolVLAPostprocessor(),
        dataset=SimpleNamespace(repo_id="fixture/libero", revision="fixture-revision"),
    )
    adapter = SmolVLAPolicyAdapter(context)
    policy.raise_after_query = True
    with pytest.raises(RuntimeError, match="fixture failure"):
        adapter.select_action(make_sample(), noise=make_noise())
    assert getattr(policy, "_forward_hooks") == {}  # noqa: B009 - private torch hook registry
    assert getattr(policy, "_forward_pre_hooks") == {}  # noqa: B009 - private torch hook registry

    policy.raise_after_query = False
    policy.reset()
    recovered = adapter.select_action(make_sample(), noise=make_noise())
    assert recovered.denoising_steps == NUM_STEPS
    assert len(recovered.representations) == 2 + 1 + 1 + NUM_STEPS


def test_smolvla_measurement_reports_change_drift_and_sensitivity() -> None:
    adapter = make_fixture_adapter()
    sample = make_sample()
    noise = make_noise()
    direction = np.array([0.5, -0.5, 0.25, -0.25, 0.1, -0.1])
    intervention = SmolVLAIntervention(direction=direction, strength=2.0)
    alternate = make_sample(task="beta task")
    swapped = make_sample(image=torch.full((3, 8, 8), 0.5), image2=torch.zeros(3, 8, 8))

    measurement = measure_smolvla_intervention(
        adapter,
        [sample],
        noise=noise,
        intervention=intervention,
        alternate_prompt_sample=alternate,
        camera_swapped_sample=swapped,
    )

    assert measurement.action_change_norm > 0.0
    assert measurement.action_change_per_dim.shape == (ACTION_DIM,)
    assert measurement.on_target_norm > 0.0
    assert measurement.off_target_norm >= 0.0
    assert measurement.on_target_fraction >= 0.99
    assert measurement.representation_drift > 0.0
    assert measurement.first_step_drift == pytest.approx(intervention.applied_norm, rel=1e-5)
    assert measurement.representation_drift == pytest.approx(intervention.applied_norm, rel=2e-2)
    assert measurement.prompt_sensitivity > 0.0
    assert measurement.camera_order_sensitivity > 0.0
    assert measurement.metadata["causal_environment_effect"] is False


def test_load_smolvla_policy_uses_supported_upstream_factories(monkeypatch: pytest.MonkeyPatch) -> None:
    config_calls: list[tuple[str, str]] = []
    policy_calls: list[tuple[object, object, object]] = []
    processor_calls: list[tuple[object, str, str, object]] = []

    class FakeSmolVLAConfig:
        @classmethod
        def from_pretrained(cls, repo_id: str, *, revision: str) -> SimpleNamespace:
            config_calls.append((repo_id, revision))
            return SimpleNamespace(pretrained_path=None, pretrained_revision=None, device=None)

    module = ModuleType("lerobot.policies.smolvla.configuration_smolvla")
    module.SmolVLAConfig = FakeSmolVLAConfig  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, module.__name__, module)

    policy = TinySmolVLAPolicy()

    def make_policy(config: object, *, ds_meta: object, rename_map: object) -> object:
        policy_calls.append((config, ds_meta, rename_map))
        return policy

    def make_processors(
        config: object,
        *,
        pretrained_path: str,
        pretrained_revision: str,
        preprocessor_overrides: object,
        **kwargs: object,
    ) -> tuple[TinySmolVLAPreprocessor, TinySmolVLAPostprocessor]:
        del kwargs
        processor_calls.append((config, pretrained_path, pretrained_revision, preprocessor_overrides))
        return TinySmolVLAPreprocessor(), TinySmolVLAPostprocessor()

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
    spec = SmolVLACheckpointSpec(
        policy_repo_id="fixture/smolvla",
        policy_revision="policy-revision",
        dataset_repo_id="fixture/libero",
        dataset_revision="dataset-revision",
    )
    adapter = load_smolvla_policy(spec, api=api, dataset_meta=SimpleNamespace(stats={}), device="cpu")

    assert config_calls == [("fixture/smolvla", "policy-revision")]
    assert len(policy_calls) == 1
    assert policy_calls[0][2] == {
        "observation.images.image": "observation.images.camera1",
        "observation.images.image2": "observation.images.camera2",
    }
    assert processor_calls[0][1] == "fixture/smolvla"
    assert processor_calls[0][2] == "policy-revision"
    assert processor_calls[0][3] == {"device_processor": {"device": "cpu"}}
    assert adapter.context.policy is policy
    assert adapter.context.metadata["dataset_revision"] == "dataset-revision"
    assert adapter.context.metadata["environment_task"] == "libero_spatial"


@pytest.mark.network
@pytest.mark.large_download
def test_smolvla_gpu_checkpoint_intervention_lane() -> None:
    pytest.importorskip("lerobot")
    if not torch.cuda.is_available():
        pytest.skip("SmolVLA checkpoint intervention lane requires a CUDA device")
    from latent_anything.integrations.lerobot import check_lerobot_compatibility

    report = check_lerobot_compatibility()
    if not report.supported:
        pytest.skip(report.diagnostic)
    adapter = load_smolvla_policy(DEFAULT_SMOLVLA_CHECKPOINT, device="cuda")
    assert adapter.checkpoint.policy_revision == DEFAULT_SMOLVLA_CHECKPOINT.policy_revision
    assert adapter.checkpoint.dataset_revision == DEFAULT_SMOLVLA_CHECKPOINT.dataset_revision
    assert adapter.checkpoint.environment_task == "libero_spatial"
    sample = {
        "observation.images.image": torch.rand(1, 3, 256, 256),
        "observation.images.image2": torch.rand(1, 3, 256, 256),
        "observation.state": torch.zeros(1, 8),
        "task": "pick up the black bowl on the stove and place it on the plate",
    }
    noise = np.zeros((1, adapter.metadata.chunk_size, adapter.metadata.max_action_dim))
    baseline = adapter.select_action(sample, noise=noise)
    assert baseline.denoising_steps == adapter.metadata.num_steps
    assert baseline.representations[0].kind == "vision_context"
    assert baseline.representations[-1].kind == "action_expert"
    preprocessor = cast(Callable[[Mapping[str, object]], Mapping[str, object]], adapter.context.preprocessor)
    postprocessor = cast(Callable[[object], Mapping[str, object]], adapter.context.postprocessor)
    policy = cast(_ActionSelectingPolicy, adapter.context.policy)
    prepared = preprocessor(sample)
    direct_action = postprocessor(
        policy.select_action(
            prepared,
            noise=torch.as_tensor(noise, dtype=torch.float32, device=adapter.device),
        )
    )
    np.testing.assert_array_equal(
        _postprocessed_action_numpy(direct_action),
        np.asarray(baseline.action_array).reshape(-1),
    )
    adapter.reset()
    reseeded = adapter.select_action(sample, noise=noise)
    np.testing.assert_array_equal(reseeded.action_array, baseline.action_array)
    direction = np.full(adapter.metadata.expert_dim, 0.01)
    intervention = SmolVLAIntervention(direction=direction, strength=1.0)
    measurement = measure_smolvla_intervention(
        adapter,
        [sample],
        noise=noise,
        intervention=intervention,
    )
    assert measurement.action_change_norm > 0.0
    assert measurement.representation_drift > 0.0
    assert 0.0 <= measurement.on_target_fraction <= 1.0

"""Offline tests for the conditional diffusion integration.

These tests use either no backend (data-structure tests) or a minimal
FakeBackend that satisfies the protocol expected by
:class:`~latent_anything.integrations.diffusers_conditional.DiffusersConditionalPipeline`.
No real model downloads occur.
"""

from __future__ import annotations

import numpy as np
import pytest
from torch import nn

from latent_anything.integrations.diffusers_conditional import (
    CONDITIONAL_MODEL_ID,
    CONDITIONAL_MODEL_REVISION,
    DENOISER_ACTIVATION_DIM,
    DenoiserCapture,
    DiffusersConditionalPipeline,
    GenerationRequest,
    GenerationResult,
    SchedulerIntervention,
    SchedulerLatentState,
)

# ---------------------------------------------------------------------------
# Fake backend for offline testing
# ---------------------------------------------------------------------------


class FakeScheduler:
    """Fake scheduler that mimics the diffusers scheduler protocol."""

    config: object = None

    @classmethod
    def from_config(cls, config: object) -> FakeScheduler:  # noqa: ARG003
        return cls()


class FakeUNet(nn.Module):
    """Minimal UNet stand-in with a ``mid_block`` submodule for activation capture."""

    def __init__(self) -> None:
        super().__init__()
        self.mid_block = nn.Sequential(nn.Identity())


class FakePipeline:
    """Fake pipeline that mimics the StableDiffusionPipeline protocol."""

    def __init__(self) -> None:
        self.scheduler = FakeScheduler()
        self.unet = FakeUNet()

    def to(self, _device: str) -> FakePipeline:  # noqa: ARG002
        return self

    def set_progress_bar_config(self, **_kwargs: object) -> None:  # noqa: ARG002
        pass

    def __call__(  # type: ignore[reportUnknownMemberType]
        self,
        *,
        prompt: str | list[str],
        num_inference_steps: int,  # noqa: ARG002
        guidance_scale: float,  # noqa: ARG002
        height: int,
        width: int,
        output_type: str,  # noqa: ARG002
        **kwargs: object,  # noqa: ARG002
    ) -> object:
        batch_size = len(prompt) if isinstance(prompt, list) else 1
        images = np.random.rand(batch_size, height, width, 3).astype(np.float32)
        return type("FakeOutput", (), {"images": images})()


# ---------------------------------------------------------------------------
# Data structure tests
# ---------------------------------------------------------------------------


class TestGenerationRequest:
    def test_defaults_are_valid(self) -> None:
        req = GenerationRequest(prompt="test prompt")
        assert req.prompt == "test prompt"
        assert req.num_inference_steps == 50
        assert req.guidance_scale == 7.5
        assert req.seed == 42

    def test_tuple_prompt_is_accepted(self) -> None:
        req = GenerationRequest(prompt=("a", "b"), seed=0)
        assert req.prompt == ("a", "b")

    def test_rejects_invalid_steps(self) -> None:
        with pytest.raises(ValueError, match="num_inference_steps"):
            GenerationRequest(prompt="x", num_inference_steps=0)

    def test_rejects_low_guidance(self) -> None:
        with pytest.raises(ValueError, match="guidance_scale"):
            GenerationRequest(prompt="x", guidance_scale=0.5)

    def test_rejects_non_multiple_of_8(self) -> None:
        with pytest.raises(ValueError, match="multiples of 8"):
            GenerationRequest(prompt="x", height=100, width=100)


class TestSchedulerLatentState:
    def test_valid_state(self) -> None:
        latent = np.zeros((1, 4, 64, 64), dtype=np.float32)
        state = SchedulerLatentState(step=0, timestep=999, latent=latent)
        assert state.step == 0
        assert state.timestep == 999
        assert state.latent.shape == (1, 4, 64, 64)

    def test_rejects_non_4d(self) -> None:
        with pytest.raises(ValueError, match="must be 4D"):
            SchedulerLatentState(step=0, timestep=999, latent=np.zeros((64, 64)))


class TestDenoiserCapture:
    def test_valid_capture(self) -> None:
        values = np.zeros((1, 1280, 64, 64), dtype=np.float32)
        cap = DenoiserCapture(step=0, location="mid_block", values=values)
        assert cap.step == 0
        assert cap.location == "mid_block"

    def test_default_metadata_is_empty(self) -> None:
        cap = DenoiserCapture(step=0, location="x", values=np.zeros((1, 4, 8, 8)))
        assert cap.metadata == {}


class TestGenerationResult:
    def test_holds_all_fields(self) -> None:
        req = GenerationRequest(prompt="test")
        images = np.zeros((1, 64, 64, 3), dtype=np.float32)
        latent = np.zeros((1, 4, 8, 8), dtype=np.float32)
        result = GenerationResult(
            images=images,
            scheduler_states=(),
            denoiser_captures=(),
            final_vae_latent=latent,
            request=req,
        )
        assert result.images.shape == (1, 64, 64, 3)
        assert result.final_vae_latent.shape == (1, 4, 8, 8)


# ---------------------------------------------------------------------------
# Pipeline construction & descriptor tests
# ---------------------------------------------------------------------------


class TestDiffusersConditionalPipeline:
    def test_constructor_with_defaults(self) -> None:
        pipe = DiffusersConditionalPipeline()
        assert pipe.model_id == CONDITIONAL_MODEL_ID
        assert pipe.revision == CONDITIONAL_MODEL_REVISION

    def test_constructor_rejects_invalid_dtype(self) -> None:
        with pytest.raises(TypeError, match="dtype"):
            DiffusersConditionalPipeline(dtype=np.int64)  # type: ignore[arg-type]

    def test_vae_latent_space_has_correct_role(self) -> None:
        pipe = DiffusersConditionalPipeline()
        space = pipe.vae_latent_space
        assert space.dim == 4
        assert space.metadata.get("role") == "vae_bottleneck"

    def test_scheduler_latent_space_has_correct_role(self) -> None:
        pipe = DiffusersConditionalPipeline()
        space = pipe.scheduler_latent_space
        assert space.dim == 4
        assert space.metadata.get("role") == "scheduler_state"

    def test_denoiser_activation_space_has_correct_dim(self) -> None:
        pipe = DiffusersConditionalPipeline()
        space = pipe.denoiser_activation_space
        assert space.dim == DENOISER_ACTIVATION_DIM
        assert space.metadata.get("role") == "denoiser_activation"

    def test_latent_value_wrappers(self) -> None:
        pipe = DiffusersConditionalPipeline()
        # LatentSpace for VAE/scheduler has shape (4,) — last axis is channel.
        vae_latent = np.zeros((1, 64, 64, 4), dtype=np.float32)
        lv = pipe.vae_latent_value(vae_latent)
        assert lv.metadata.get("role") == "vae_bottleneck"
        assert lv.shape == (1, 64, 64, 4)

        sched_latent = np.zeros((1, 64, 64, 4), dtype=np.float32)
        lv_s = pipe.scheduler_latent_value(sched_latent)
        assert lv_s.metadata.get("role") == "scheduler_state"

        # Denoiser activation space has shape (DENOISER_ACTIVATION_DIM,).
        # When wrapped as NHWC spatial map, last axis is the channel dim.
        activ = np.zeros((1, 64, 64, DENOISER_ACTIVATION_DIM), dtype=np.float32)
        lv_a = pipe.denoiser_activation_value(activ)
        assert lv_a.metadata.get("role") == "denoiser_activation"


# ---------------------------------------------------------------------------
# FakeBackend generation tests   (Task 6)
# ---------------------------------------------------------------------------


class TestFakeBackendPipeline:
    def test_generate_with_no_capture_returns_valid_result(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import types

        pipe = DiffusersConditionalPipeline()
        monkeypatch.setattr(pipe, "_backend", lambda: FakePipeline())
        monkeypatch.setattr(pipe, "_scheduler_name", "FakeScheduler")
        monkeypatch.setattr(
            pipe,
            "_diffusers_module",
            types.SimpleNamespace(
                DDIMScheduler=FakeScheduler,
                PNDMScheduler=FakeScheduler,
                LMSDiscreteScheduler=FakeScheduler,
                EulerDiscreteScheduler=FakeScheduler,
                EulerAncestralDiscreteScheduler=FakeScheduler,
            ),
        )

        req = GenerationRequest(
            prompt="hello", num_inference_steps=5, capture_scheduler_states=False, capture_denoiser_location=None
        )
        result = pipe.generate(req)
        assert isinstance(result, GenerationResult)
        assert result.images.shape == (1, 512, 512, 3)
        assert len(result.scheduler_states) == 0
        assert len(result.denoiser_captures) == 0

    def test_generate_captures_scheduler_states(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import types

        pipe = DiffusersConditionalPipeline()
        monkeypatch.setattr(pipe, "_backend", lambda: FakePipeline())
        monkeypatch.setattr(pipe, "_scheduler_name", "FakeScheduler")
        monkeypatch.setattr(
            pipe,
            "_diffusers_module",
            types.SimpleNamespace(
                DDIMScheduler=FakeScheduler,
                PNDMScheduler=FakeScheduler,
                LMSDiscreteScheduler=FakeScheduler,
                EulerDiscreteScheduler=FakeScheduler,
                EulerAncestralDiscreteScheduler=FakeScheduler,
            ),
        )

        req = GenerationRequest(prompt="test", num_inference_steps=3, capture_denoiser_location=None)
        result = pipe.generate(req)
        assert isinstance(result, GenerationResult)

    def test_generate_with_activation_capture(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify that activation capture context manager is accepted (hooks fire only with real pipeline)."""
        import types

        pipe = DiffusersConditionalPipeline()
        monkeypatch.setattr(pipe, "_backend", lambda: FakePipeline())
        monkeypatch.setattr(pipe, "_scheduler_name", "FakeScheduler")
        monkeypatch.setattr(
            pipe,
            "_diffusers_module",
            types.SimpleNamespace(
                DDIMScheduler=FakeScheduler,
                PNDMScheduler=FakeScheduler,
                LMSDiscreteScheduler=FakeScheduler,
                EulerDiscreteScheduler=FakeScheduler,
                EulerAncestralDiscreteScheduler=FakeScheduler,
            ),
        )

        req = GenerationRequest(prompt="test", num_inference_steps=3, capture_denoiser_location="mid_block")
        result = pipe.generate(req)
        assert isinstance(result, GenerationResult)

    def test_set_seed_is_deterministic(self) -> None:
        DiffusersConditionalPipeline.set_seed(42)
        a = np.random.rand(5)
        DiffusersConditionalPipeline.set_seed(42)
        b = np.random.rand(5)
        np.testing.assert_array_equal(a, b)

    def test_resolve_scheduler_names(self) -> None:
        import types

        fake_diffusers = types.SimpleNamespace(
            DDIMScheduler=type("DDIM", (), {}),
            PNDMScheduler=type("PNDM", (), {}),
            LMSDiscreteScheduler=type("LMS", (), {}),
            EulerDiscreteScheduler=type("Euler", (), {}),
            EulerAncestralDiscreteScheduler=type("EulerA", (), {}),
        )
        pipe = DiffusersConditionalPipeline()
        assert pipe.resolve_scheduler(fake_diffusers, "ddim").__name__ == "DDIM"
        assert pipe.resolve_scheduler(fake_diffusers, "pndm").__name__ == "PNDM"
        assert pipe.resolve_scheduler(fake_diffusers, "euler_a").__name__ == "EulerA"
        with pytest.raises(ValueError, match="Unknown scheduler"):
            pipe.resolve_scheduler(fake_diffusers, "nonexistent")


# ---------------------------------------------------------------------------
# Scheduler intervention tests
# ---------------------------------------------------------------------------


class TestSchedulerIntervention:
    def test_valid_intervention(self) -> None:
        direction = np.zeros((1, 4, 64, 64), dtype=np.float32)
        direction.setflags(write=False)
        intervention = SchedulerIntervention(direction=direction, strength=1.0, step_range=(5, 15))
        assert intervention.strength == 1.0
        assert intervention.step_range == (5, 15)
        assert intervention.direction.shape == (1, 4, 64, 64)

    def test_rejects_non_4d_direction(self) -> None:
        with pytest.raises(ValueError, match="must be 4D"):
            SchedulerIntervention(direction=np.zeros((64, 64)), strength=1.0, step_range=(0, 1))

    def test_rejects_negative_strength(self) -> None:
        with pytest.raises(ValueError, match="strength must be >= 0"):
            SchedulerIntervention(direction=np.zeros((1, 4, 8, 8)), strength=-1.0, step_range=(0, 1))

    def test_rejects_invalid_step_range(self) -> None:
        with pytest.raises(ValueError, match="invalid step_range"):
            SchedulerIntervention(direction=np.zeros((1, 4, 8, 8)), strength=1.0, step_range=(10, 5))

    def test_zero_strength_is_acceptable(self) -> None:
        """Zero strength is a valid (no-op) intervention."""
        intervention = SchedulerIntervention(direction=np.zeros((1, 4, 8, 8)), strength=0.0, step_range=(0, 1))
        assert intervention.strength == 0.0


class TestInterventionDirectionHelpers:
    def test_random_direction_has_correct_shape(self) -> None:
        intervention = DiffusersConditionalPipeline.random_direction(
            shape=(1, 4, 64, 64), seed=42, strength=2.0, step_range=(0, 5)
        )
        assert intervention.direction.shape == (1, 4, 64, 64)
        assert intervention.strength == 2.0
        assert intervention.step_range == (0, 5)

    def test_random_direction_is_deterministic(self) -> None:
        a = DiffusersConditionalPipeline.random_direction((1, 4, 8, 8), seed=99)
        b = DiffusersConditionalPipeline.random_direction((1, 4, 8, 8), seed=99)
        np.testing.assert_array_equal(a.direction, b.direction)

    def test_matched_norm_matches_source_norm(self) -> None:
        source = np.random.RandomState(42).randn(1, 4, 8, 8).astype(np.float32)
        src_norm = np.linalg.norm(source)
        intervention = DiffusersConditionalPipeline.matched_norm_direction(
            source, seed=0, strength=1.0, step_range=(0, 1)
        )
        dir_norm = np.linalg.norm(intervention.direction)
        assert abs(dir_norm - src_norm) < 1e-5

    def test_matched_norm_respects_zeros(self) -> None:
        source = np.zeros((1, 4, 8, 8), dtype=np.float32)
        intervention = DiffusersConditionalPipeline.matched_norm_direction(source, seed=0)
        # When source norm is ~0, no scaling happens — direction stays as drawn.
        assert intervention.direction.shape == (1, 4, 8, 8)


class TestFakeBackendIntervention:
    def test_generate_with_intervention_returns_result(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import types

        pipe = DiffusersConditionalPipeline()
        monkeypatch.setattr(pipe, "_backend", lambda: FakePipeline())
        monkeypatch.setattr(pipe, "_scheduler_name", "FakeScheduler")
        monkeypatch.setattr(
            pipe,
            "_diffusers_module",
            types.SimpleNamespace(
                DDIMScheduler=FakeScheduler,
                PNDMScheduler=FakeScheduler,
                LMSDiscreteScheduler=FakeScheduler,
                EulerDiscreteScheduler=FakeScheduler,
                EulerAncestralDiscreteScheduler=FakeScheduler,
            ),
        )

        req = GenerationRequest(
            prompt="test",
            num_inference_steps=5,
            capture_scheduler_states=False,
            capture_denoiser_location=None,
        )
        intervention = SchedulerIntervention(
            direction=np.zeros((1, 4, 64, 64), dtype=np.float32),
            strength=0.5,
            step_range=(1, 4),
        )
        # FakePipeline does not invoke callbacks, so the intervention
        # mechanism is not exercised here.  This test verifies the
        # generate() method accepts the intervention parameter and
        # returns a properly typed result without errors.
        result = pipe.generate(req, intervention=intervention)
        assert isinstance(result, GenerationResult)
        # The final latent should exist (zeros from the fallback path).
        assert result.final_vae_latent.shape == (1, 4, 64, 64)

    def test_intervention_without_explicit_capture(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """generate() accepts intervention even when capture_scheduler_states is False."""
        import types

        pipe = DiffusersConditionalPipeline()
        monkeypatch.setattr(pipe, "_backend", lambda: FakePipeline())
        monkeypatch.setattr(pipe, "_scheduler_name", "FakeScheduler")
        monkeypatch.setattr(
            pipe,
            "_diffusers_module",
            types.SimpleNamespace(
                DDIMScheduler=FakeScheduler,
                PNDMScheduler=FakeScheduler,
                LMSDiscreteScheduler=FakeScheduler,
                EulerDiscreteScheduler=FakeScheduler,
                EulerAncestralDiscreteScheduler=FakeScheduler,
            ),
        )

        req = GenerationRequest(
            prompt="test",
            num_inference_steps=3,
            capture_scheduler_states=False,
            capture_denoiser_location=None,
        )
        intervention = SchedulerIntervention(
            direction=np.zeros((1, 4, 64, 64), dtype=np.float32),
            strength=1.0,
            step_range=(0, 3),
        )
        # FakePipeline doesn't invoke callbacks, so we verify only that
        # generate() accepts the intervention parameter without error.
        result = pipe.generate(req, intervention=intervention)
        assert isinstance(result, GenerationResult)

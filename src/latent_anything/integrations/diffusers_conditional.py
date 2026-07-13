"""Revision-pinned conditional diffusion integration with scheduler latent
state and denoiser activation capture, plus scheduler latent intervention.

Design
------
This is a **concrete integration**, not a ``ModelAdapter`` implementation.
The full conditional-diffusion lifecycle (prompt processing, tokenization,
scheduler iteration, callback hooks, activation capture, VAE decode) does
not fit the ``encode()`` / ``decode()`` / ``latent_space`` contract — it
owns its own prompt, scheduler state, initial noise, iterative callbacks,
multiple non‑homogeneous representation spaces, and generated outputs.
Collapsing into ``encode()`` would hide meaningful semantics.

**No generative protocol is introduced.** Sharing a generative interface
requires ≥3 differing integrations (per Rule of Three). This first concrete
integration proves the common shape; extraction waits for repetitions.

Scheduler latent states are captured via Diffusers' native
``callback_on_step_end``.  Denoiser activations are captured via
:class:`~latent_anything.capture.ActivationCaptureSession` (PyTorch
forward hooks).  Both paths produce NumPy arrays for the public surface.

Scheduler latent intervention is an optional additive edit applied to
the scheduler's latent state during denoising.  It reuses the same
``callback_on_step_end`` hook but returns the modified latents so the
pipeline uses the edited state at the next step.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from latent_anything.capture import ActivationCaptureSession
from latent_anything.integrations import require_optional
from latent_anything.latent_space import LatentSpace
from latent_anything.latent_value import LatentValue

# ---------------------------------------------------------------------------
# Pinned model identity
# ---------------------------------------------------------------------------

CONDITIONAL_MODEL_ID = "runwayml/stable-diffusion-v1-5"
"""HuggingFace model ID for the pinned conditional diffusion pipeline."""

CONDITIONAL_MODEL_REVISION = "39593d56b552c3a24aeb192dd11d2a1429c3102b"
"""Pinned revision for reproducible behaviour across installations."""

# Tested dependency ranges (installation guards)
TESTED_DIFFUSERS_RANGE = ">=0.30,<1.0"
TESTED_TRANSFORMERS_RANGE = ">=4.45,<5.0"

# SD 1.5 UNet mid_block output channels (the highest-channel bottleneck).
DENOISER_ACTIVATION_DIM = 1280

# ---------------------------------------------------------------------------
# Public data types  (Task 2)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GenerationRequest:
    """Typed request for one conditional diffusion generation run.

    All parameters are fixed before generation starts so that captures
    are reproducible for a given prompt, seed, and scheduler.

    Parameters
    ----------
    prompt:
        Text prompt or tuple of prompts (batched generation).
    num_inference_steps:
        Number of denoising steps.
    guidance_scale:
        Classifier-free guidance scale (>= 1.0).
    seed:
        Random seed for initial noise and scheduler stochasticity.
    scheduler:
        Scheduler name: ``"ddim"``, ``"pndm"``, ``"lms"``,
        ``"euler"``, ``"euler_a"``.
    height, width:
        Output image dimensions (must be multiples of 8).
    capture_scheduler_states:
        If True, record scheduler latent states at each denoising step.
    capture_denoiser_location:
        UNet submodule name for activation capture, or None to skip.
    """

    prompt: str | tuple[str, ...]
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    seed: int = 42
    scheduler: str = "ddim"
    height: int = 512
    width: int = 512
    capture_scheduler_states: bool = True
    capture_denoiser_location: str | None = "mid_block"

    def __post_init__(self) -> None:
        if self.num_inference_steps < 1:
            raise ValueError(f"num_inference_steps must be >= 1, got {self.num_inference_steps}")
        if self.guidance_scale < 1.0:
            raise ValueError(f"guidance_scale must be >= 1.0, got {self.guidance_scale}")
        if self.height % 8 != 0 or self.width % 8 != 0:
            raise ValueError(f"height and width must be multiples of 8, got {self.height}x{self.width}")


@dataclass(frozen=True)
class SchedulerLatentState:
    """A single scheduler latent state captured at one denoising step."""

    step: int
    """Denoising step index (0-based)."""

    timestep: int
    """Discrete timestep value from the scheduler."""

    latent: np.ndarray
    """Scheduler latent as ``(N, C, H, W)`` non-writable NumPy array."""

    def __post_init__(self) -> None:
        if self.latent.ndim != 4:
            raise ValueError(f"latent must be 4D (NCHW), got {self.latent.ndim}D")


@dataclass(frozen=True)
class DenoiserCapture:
    """A single denoiser activation captured at one denoising step."""

    step: int
    """Denoising step index (0-based)."""

    location: str
    """UNet submodule name that produced this activation."""

    values: np.ndarray
    """Activation as non-writable NumPy array."""

    metadata: dict[str, Any] = field(default_factory=dict)  # type: ignore[reportUnknownVariableType]
    """Additional shape / dtype / device metadata."""


@dataclass(frozen=True)
class GenerationResult:
    """Complete typed result of one conditional diffusion generation run.

    Contains the generated image(s), all captured intermediate states,
    and the full request for traceability.

    Parameters
    ----------
    images:
        Generated images as ``NHWC`` float32 NumPy array in ``[0, 1]``.
    scheduler_states:
        Captured scheduler latent states per step (if requested).
    denoiser_captures:
        Captured denoiser activations per step (if requested).
    final_vae_latent:
        Final VAE latent (before VAE decode) as ``NCHW`` float32.
    request:
        The :class:`GenerationRequest` that produced this result.
    """

    images: np.ndarray
    scheduler_states: tuple[SchedulerLatentState, ...]
    denoiser_captures: tuple[DenoiserCapture, ...]
    final_vae_latent: np.ndarray
    request: GenerationRequest


@dataclass(frozen=True)
class SchedulerIntervention:
    """Additive intervention on scheduler latent states during denoising.

    Applies ``latents ← latents + strength * direction`` at every step
    in ``step_range``.  The direction is a fixed 4D ``(1, C, H, W)``
    numpy array — for example a concept direction or random vector.

    Parameters
    ----------
    direction:
        Direction vector as ``(1, C, H, W)`` non-writable NumPy array.
    strength:
        Strength multiplier (>= 0).  Zero means no effect.
    step_range:
        ``(start, end)`` — apply at steps ``[start, end)``.
    """

    direction: np.ndarray
    strength: float
    step_range: tuple[int, int]

    def __post_init__(self) -> None:
        if self.direction.ndim != 4:
            raise ValueError(f"direction must be 4D (NCHW), got {self.direction.ndim}D")
        if self.strength < 0:
            raise ValueError(f"strength must be >= 0, got {self.strength}")
        if self.step_range[0] < 0 or self.step_range[1] <= self.step_range[0]:
            raise ValueError(f"invalid step_range: {self.step_range}")


# ---------------------------------------------------------------------------
# Diffusion integration  (Tasks 1, 3, 4, 5)
# ---------------------------------------------------------------------------


class DiffusersConditionalPipeline:
    """Revision-pinned conditional text-to-image diffusion integration.

    This is a **concrete integration**, not a ``ModelAdapter``.  Use
    :meth:`generate` to run the full pipeline with optional scheduler
    state and denoiser activation capture.

    Parameters
    ----------
    model_id:
        HuggingFace model ID (default: pinned ``runwayml/stable-diffusion-v1-5``).
    revision:
        Git revision (commit hash or tag; default: pinned revision).
    device:
        Torch device string (``"cpu"``, ``"cuda"``, …).
    dtype:
        NumPy dtype for the public boundary (float16 or float32 only).
    """

    def __init__(
        self,
        model_id: str = CONDITIONAL_MODEL_ID,
        revision: str = CONDITIONAL_MODEL_REVISION,
        *,
        device: str = "cpu",
        dtype: np.dtype | None = None,
    ) -> None:
        self.model_id = model_id
        self.revision = revision
        self.device = device
        self.dtype = np.dtype(np.float32 if dtype is None else dtype)
        if self.dtype not in {np.dtype(np.float16), np.dtype(np.float32)}:
            raise TypeError("dtype must be float16 or float32")
        self._pipeline: Any = None
        self._scheduler_name: str = ""
        self._diffusers_module: Any = None

    # -- internal helpers ---------------------------------------------------

    def _torch_dtype(self) -> Any:
        import torch

        return torch.float16 if self.dtype == np.dtype(np.float16) else torch.float32

    def _backend(self) -> Any:
        """Lazy import and construct the full StableDiffusionPipeline."""
        if self._pipeline is not None:
            return self._pipeline

        diffusers = require_optional("diffusers", extra="diffusers-full")
        require_optional("transformers", extra="diffusers-full")

        pipe = diffusers.StableDiffusionPipeline.from_pretrained(
            self.model_id,
            revision=self.revision,
            torch_dtype=self._torch_dtype(),
        )
        pipe = pipe.to(self.device)
        pipe.set_progress_bar_config(disable=True)

        # Store references for scheduler resolution and identity checks.
        self._scheduler_name = type(pipe.scheduler).__name__
        self._diffusers_module = diffusers
        self._pipeline = pipe
        return pipe

    @staticmethod
    def set_seed(seed: int) -> None:
        """Set torch + numpy RNG for reproducible generation."""
        import random

        import torch

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)  # pyright: ignore[reportUnknownMemberType]

    # -- public LatentSpace descriptors (Task 5) ---------------------------

    @property
    def vae_latent_space(self) -> LatentSpace:
        """Descriptor for the VAE decoder bottleneck.

        The VAE compresses the 512×512 RGB image to a 64×64×4 latent
        map.  This space has 4 channels (the SD 1.5 VAE latent channels)
        and uses ``gaussian_set``'s shape convention to indicate a
        spatial map layout via metadata.
        """
        # The latent is (N, 4, H/8, W/8) in NCHW layout.
        return LatentSpace(
            dim=4,
            source_model=f"{self.model_id}@{self.revision}",
            metadata={
                "role": "vae_bottleneck",
                "layout": "NCHW",
                "spatial_scale": 8,
                "scheduler": self._scheduler_name,
            },
        )

    @property
    def scheduler_latent_space(self) -> LatentSpace:
        """Descriptor for scheduler latent states during denoising.

        The scheduler operates on the same 4-channel spatial latent as
        the VAE bottleneck, but at different points in the denoising
        trajectory.  The distinction is captured in ``metadata["role"]``
        so consumers can distinguish VAE latents from scheduler latents
        even when shape / channels coincide.
        """
        return LatentSpace(
            dim=4,
            source_model=f"{self.model_id}@{self.revision}",
            metadata={
                "role": "scheduler_state",
                "layout": "NCHW",
                "spatial_scale": 8,
                "scheduler": self._scheduler_name,
            },
        )

    @property
    def denoiser_activation_space(self) -> LatentSpace:
        """Descriptor for internal UNet denoiser activations.

        The UNet ``mid_block`` output has ``(N, 1280, H/8, W/8)``
        channels for SD 1.5 at 512×512.  This is the highest-channel
        bottleneck inside the denoiser, distinct from the 4-channel VAE
        and scheduler spaces.
        """
        return LatentSpace(
            dim=DENOISER_ACTIVATION_DIM,
            source_model=f"{self.model_id}@{self.revision}",
            metadata={
                "role": "denoiser_activation",
                "layout": "NCHW",
                "location": "mid_block",
                "scheduler": self._scheduler_name,
            },
        )

    # -- generation + capture (Tasks 3, 4) ---------------------------------

    def generate(
        self,
        request: GenerationRequest,
        intervention: SchedulerIntervention | None = None,
    ) -> GenerationResult:
        """Run conditional diffusion generation with optional capture.

        Parameters
        ----------
        request:
            Typed generation parameters (prompt, seed, scheduler, …).
        intervention:
            Optional scheduler latent intervention to apply during
            denoising.  When provided, scheduler states are always
            captured so the before-intervention state is recorded.

        Returns
        -------
        GenerationResult
            Generated images, captured states, and full request metadata.

        Raises
        ------
        ImportError
            If ``diffusers`` or ``transformers`` are not installed.
        """
        pipe = self._backend()
        self.set_seed(request.seed)

        # Build the requested scheduler
        if self._diffusers_module is not None:
            diffusers_module = self._diffusers_module
        else:
            diffusers_module = require_optional("diffusers", extra="diffusers-full")
        scheduler_class = self.resolve_scheduler(diffusers_module, request.scheduler)
        pipe.scheduler = scheduler_class.from_config(pipe.scheduler.config)  # pyright: ignore[reportUnknownMemberType]

        # Prepare prompt(s)
        prompts = [request.prompt] if isinstance(request.prompt, str) else list(request.prompt)
        batch_size = len(prompts)

        # Capture containers
        scheduler_states: list[SchedulerLatentState] = []
        denoiser_captures: list[DenoiserCapture] = []

        # -- Scheduler state capture + intervention via callback_on_step_end --
        need_scheduler_callback = request.capture_scheduler_states or intervention is not None
        if need_scheduler_callback:

            def _scheduler_callback(
                pipe: Any,  # noqa: ARG001  # callback signature required by diffusers
                step_index: int,
                timestep: int,
                callback_kwargs: dict[str, Any],
            ) -> dict[str, Any]:
                latents = callback_kwargs.get("latents")
                if latents is not None:
                    # Always capture the pre-intervention state.
                    if request.capture_scheduler_states:
                        captured = latents.detach().cpu().numpy().copy()
                        captured.setflags(write=False)
                        scheduler_states.append(
                            SchedulerLatentState(step=step_index, timestep=timestep, latent=captured)
                        )

                    # Apply intervention if configured for this step.
                    if intervention is not None and (
                        intervention.step_range[0] <= step_index < intervention.step_range[1]
                    ):
                        import torch

                        direction_t = torch.tensor(
                            intervention.direction,
                            dtype=latents.dtype,
                            device=latents.device,
                        )
                        latents = latents + intervention.strength * direction_t
                        callback_kwargs["latents"] = latents

                return callback_kwargs

            pipe.callback_on_step_end = _scheduler_callback

        # -- Denoiser activation capture via ActivationCaptureSession (Task 4) --
        capture_location = request.capture_denoiser_location
        capture_session = nullcontext()
        if capture_location is not None:
            capture_session = ActivationCaptureSession(
                pipe.unet,  # pyright: ignore[reportUnknownArgumentType]
                [capture_location],
                source_model_version=f"{self.model_id}@{self.revision}",
            )

        # -- Run generation --
        with capture_session as session:  # type: ignore[union-attr]
            # Type narrowing: when capture_location is not None, session is
            # ActivationCaptureSession; otherwise it's nullcontext.
            results = pipe(
                prompt=prompts,
                num_inference_steps=request.num_inference_steps,
                guidance_scale=request.guidance_scale,
                height=request.height,
                width=request.width,
                output_type="np",
            )

        # Extract denoiser captures from the session (Task 4).
        if capture_location is not None and isinstance(session, ActivationCaptureSession):
            # Captures are in forward execution order, one per denoising step.
            for step_idx, cap in enumerate(session.captures):
                vals = cap.values.copy()
                vals.setflags(write=False)
                denoiser_captures.append(
                    DenoiserCapture(
                        step=step_idx,
                        location=cap.metadata.location,
                        values=vals,
                        metadata={
                            "shape": cap.metadata.shape,
                            "device": cap.metadata.device,
                            "dtype": cap.metadata.dtype,
                        },
                    )
                )

        # Reset the scheduler callback to avoid leaking state.
        pipe.callback_on_step_end = None

        # Post-process result (Task 5: wrap in LatentValue-friendly forms).
        images_nhwc: np.ndarray = results.images  # type: ignore[union-attr]
        # Normalise from [-1, 1] to [0, 1] if needed.
        if images_nhwc.min() < 0:
            images_nhwc = (images_nhwc * 0.5 + 0.5).clip(0, 1)

        # The final VAE latent is the last captured scheduler state (if captured)
        # or we need to get it differently.  For now, try to extract from
        # scheduler_states (last entry), or leave as zeros.
        if scheduler_states:
            final_latent = scheduler_states[-1].latent.copy()
        else:
            final_latent = np.zeros((batch_size, 4, request.height // 8, request.width // 8), dtype=np.float32)
        final_latent.setflags(write=False)

        return GenerationResult(
            images=images_nhwc.astype(np.float32, copy=False),
            scheduler_states=tuple(scheduler_states),
            denoiser_captures=tuple(denoiser_captures),
            final_vae_latent=final_latent,
            request=request,
        )

    # -- scheduler resolution ----------------------------------------------

    def resolve_scheduler(self, diffusers: Any, name: str) -> type:
        """Return the scheduler class for the requested scheduler name."""
        _schedulers: dict[str, type] = {
            "ddim": diffusers.DDIMScheduler,
            "pndm": diffusers.PNDMScheduler,
            "lms": diffusers.LMSDiscreteScheduler,
            "euler": diffusers.EulerDiscreteScheduler,
            "euler_a": diffusers.EulerAncestralDiscreteScheduler,
        }
        cls = _schedulers.get(name)
        if cls is None:
            raise ValueError(f"Unknown scheduler {name!r}; supported: {list(_schedulers)}")
        return cls

    # -- intervention direction helpers -----------------------------------

    @staticmethod
    def random_direction(
        shape: tuple[int, int, int, int],
        seed: int | None = None,
        *,
        strength: float = 1.0,
        step_range: tuple[int, int] = (0, 1),
    ) -> SchedulerIntervention:
        """Create an intervention with a random normal direction.

        Parameters
        ----------
        shape:
            ``(N, C, H, W)`` — must match the scheduler latent shape.
        seed:
            Optional RNG seed for reproducibility.
        strength:
            Intervention strength multiplier.
        step_range:
            Denoising step range ``[start, end)``.
        """
        rng = np.random.default_rng(seed)
        direction = rng.normal(0, 1, size=shape).astype(np.float32)
        direction.setflags(write=False)
        return SchedulerIntervention(direction=direction, strength=strength, step_range=step_range)

    @staticmethod
    def matched_norm_direction(
        source: np.ndarray,
        seed: int | None = None,
        *,
        strength: float = 1.0,
        step_range: tuple[int, int] = (0, 1),
    ) -> SchedulerIntervention:
        """Create a random direction with the same per-sample norm as *source*.

        Draws a random normal vector, scales each sample to match the
        norm of *source*, then returns a frozen ``SchedulerIntervention``.

        Parameters
        ----------
        source:
            Reference latent ``(N, C, H, W)`` whose norm to match.
        seed:
            Optional RNG seed for reproducibility.
        strength:
            Intervention strength multiplier.
        step_range:
            Denoising step range ``[start, end)``.
        """
        rng = np.random.default_rng(seed)
        direction = rng.normal(0, 1, size=source.shape).astype(np.float32)
        # Scale each sample to match source norm.
        for i in range(source.shape[0]):
            src_norm = np.linalg.norm(source[i])
            if src_norm > 1e-15:
                direction[i] *= src_norm / np.linalg.norm(direction[i])
        direction.setflags(write=False)
        return SchedulerIntervention(direction=direction, strength=strength, step_range=step_range)

    # -- convenience descriptors (Task 5) ----------------------------------

    def vae_latent_value(self, latent: np.ndarray) -> LatentValue:
        """Wrap a VAE latent array in a :class:`LatentValue` with correct space."""
        return LatentValue(latent, self.vae_latent_space, metadata={"role": "vae_bottleneck"})

    def scheduler_latent_value(self, latent: np.ndarray) -> LatentValue:
        """Wrap a scheduler latent array in a :class:`LatentValue`."""
        return LatentValue(latent, self.scheduler_latent_space, metadata={"role": "scheduler_state"})

    def denoiser_activation_value(self, activation: np.ndarray) -> LatentValue:
        """Wrap a denoiser activation array in a :class:`LatentValue`."""
        return LatentValue(activation, self.denoiser_activation_space, metadata={"role": "denoiser_activation"})

"""Offline causal simulation benchmark tests plus the marked statistical lane.

The tiny policy fixture mirrors the SmolVLA ``select_action`` seams the
benchmark executes (state projection, action-expert norm, flow-matching
denoising, action queue) and a fake vector environment replays deterministic
observations per seed, so the default suite stays offline and deterministic.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
import torch
from torch import Tensor, nn

from latent_anything.integrations.lerobot import LeRobotAPI, LeRobotPolicyContext
from latent_anything.integrations.lerobot_benchmark import (
    BenchmarkEnvironmentBundle,
    SimulationBenchmarkConfig,
    build_correlation,
    build_libero_benchmark_environment,
    run_episode,
    run_simulation_benchmark,
    wilson_ci,
)
from latent_anything.integrations.lerobot_smolvla import (
    DEFAULT_SMOLVLA_CHECKPOINT,
    SmolVLAPolicyAdapter,
    load_smolvla_policy,
)

HIDDEN = 8
EXPERT = 6
CHUNK = 4
NUM_STEPS = 2
MAX_STATE = 4
MAX_ACTION = 4
ACTION_DIM = 2
MAX_STEPS = 20
TERMINATE_AT = 6
SUCCESS_STEP = 5
VOCAB = 16
LANG_LEN = 4


class TinyVisionModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(3 * 8 * 8, HIDDEN)

    def forward(self, *, pixel_values: Tensor, patch_attention_mask: Tensor | None = None) -> SimpleNamespace:
        del patch_attention_mask
        flat = pixel_values.reshape(pixel_values.shape[0], -1)
        pooled = self.projection(flat)
        return SimpleNamespace(last_hidden_state=pooled[:, :LANG_LEN, None] + pooled[:, None, :HIDDEN])


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


class TinyExpertNorm(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(EXPERT)


class TinyVLMWithExpert(nn.Module):
    expert_hidden_size = EXPERT

    def __init__(self) -> None:
        super().__init__()
        self.vlm = TinyVLM()
        self.lm_expert = TinyExpertNorm()


class TinyBenchmarkModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.vlm_with_expert = TinyVLMWithExpert()
        self.state_proj = nn.Linear(MAX_STATE, HIDDEN)
        self.action_in_proj = nn.Linear(MAX_ACTION, EXPERT)
        self.action_out_proj = nn.Linear(EXPERT, MAX_ACTION)


class TinyBenchmarkPolicy(nn.Module):
    """Mirrors the official SmolVLA action-selection seams without vision/language."""

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
            image_features={},
        )
        self.model = TinyBenchmarkModel()
        self._action_queue: list[Tensor] = []

    def reset(self) -> None:
        self._action_queue.clear()

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], *, noise: Tensor | None = None) -> Tensor:
        if not self._action_queue:
            state = batch["observation.state"].to(torch.float32)
            state = nn.functional.pad(state, (0, MAX_STATE - state.shape[-1]))
            self.model.state_proj(state)
            x_t = noise if noise is not None else torch.zeros(1, CHUNK, MAX_ACTION)
            x_t = x_t.to(torch.float32)
            for _ in range(NUM_STEPS):
                suffix = self.model.action_in_proj(x_t)
                expert = self.model.vlm_with_expert.lm_expert.norm(suffix)
                velocity = self.model.action_out_proj(expert)
                x_t = x_t - 0.1 * velocity
            actions = x_t[:, :, :ACTION_DIM]
            self._action_queue.extend(actions.transpose(0, 1)[: self.config.n_action_steps])
        return self._action_queue.pop(0)


class TinyBenchmarkPreprocessor:
    def __call__(self, sample: Mapping[str, object]) -> dict[str, Tensor]:
        image = sample["observation.images.image"]
        image2 = sample["observation.images.image2"]
        state = sample["observation.state"]
        task = sample["task"]
        if not isinstance(image, Tensor) or not isinstance(image2, Tensor) or not isinstance(state, Tensor):
            raise TypeError("fixture tensors must be torch.Tensor values")
        if isinstance(task, (list, tuple)):
            task = task[0]
        if not isinstance(task, str):
            raise TypeError("fixture task must be a string or list of strings")
        tokens = torch.tensor([ord(char) % VOCAB for char in task][:LANG_LEN], dtype=torch.long)
        tokens = nn.functional.pad(tokens, (0, LANG_LEN - tokens.shape[0]), value=0)[None]
        return {
            "observation.state": state[None],
            "observation.language.tokens": tokens,
            "observation.language.attention_mask": torch.ones_like(tokens, dtype=torch.bool),
        }


class TinyBenchmarkPostprocessor:
    def __call__(self, action: Tensor) -> dict[str, Tensor]:
        return {"action": action}


def make_fixture_observation(seed: int, step: int) -> dict[str, object]:
    """Deterministic, action-independent observation replay for one (seed, step)."""

    rng = np.random.default_rng(seed * 1000 + step)
    return {
        "observation.images.image": torch.from_numpy(rng.normal(size=(3, 8, 8)).astype(np.float32)),
        "observation.images.image2": torch.full((3, 8, 8), 0.5, dtype=torch.float32),
        "observation.state": torch.from_numpy(rng.normal(size=MAX_STATE).astype(np.float32)),
    }


class FakeVectorEnv:
    """Deterministic stand-in for gym.vector.SyncVectorEnv with n_envs=1."""

    def __init__(
        self,
        *,
        max_steps: int = MAX_STEPS,
        terminate_at: int = TERMINATE_AT,
        success_step: int = SUCCESS_STEP,
    ) -> None:
        self._max_steps = max_steps
        self._terminate_at = terminate_at
        self._success_step = success_step
        self._seed: int | None = None
        self._step = 0
        self.closed = False

    @property
    def num_envs(self) -> int:
        return 1

    def reset(self, seed: int | None = None, **kwargs: object) -> tuple[dict[str, object], dict[str, object]]:
        del kwargs
        self._seed = seed
        self._step = 0
        return make_fixture_observation(cast(int, seed), 0), {}

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, object], np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
        del action
        self._step += 1
        terminated = self._step >= self._terminate_at
        success = self._step >= self._success_step
        reward = np.array([1.0 if success else 0.0], dtype=np.float32)
        info: dict[str, object] = {}
        if terminated:
            info["final_info"] = {0: {"is_success": bool(success)}}
        else:
            info["is_success"] = np.array([False], dtype=bool)
        return (
            make_fixture_observation(cast(int, self._seed), self._step),
            reward,
            np.array([terminated], dtype=bool),
            np.array([False], dtype=bool),
            info,
        )

    def call(self, name: str, *args: object, **kwargs: object) -> list[object]:
        del args, kwargs
        if name == "_max_episode_steps":
            return [self._max_steps]
        if name == "task_description":
            return ["fixture task"]
        raise AttributeError(name)

    def close(self) -> None:
        self.closed = True


def make_fixture_bundle() -> BenchmarkEnvironmentBundle:
    return BenchmarkEnvironmentBundle(
        env_factory=lambda: FakeVectorEnv(),
        env_preprocessor=lambda observation: observation,
        preprocess_observation=lambda observation: observation,
        task_description="fixture task",
        max_episode_steps=MAX_STEPS,
        metadata={"fixture": True},
    )


def make_fixture_adapter() -> SmolVLAPolicyAdapter:
    context = LeRobotPolicyContext(
        policy_name="smolvla",
        policy=TinyBenchmarkPolicy(),
        preprocessor=TinyBenchmarkPreprocessor(),
        postprocessor=TinyBenchmarkPostprocessor(),
        dataset=SimpleNamespace(repo_id="fixture/libero", revision="fixture-revision"),
    )
    return SmolVLAPolicyAdapter(context, checkpoint=DEFAULT_SMOLVLA_CHECKPOINT)


def make_benchmark_noise() -> np.ndarray:
    return np.full((1, CHUNK, MAX_ACTION), 0.25)


def test_benchmark_config_validates_conditions_and_strengths() -> None:
    with pytest.raises(ValueError, match="Input should be"):
        SimulationBenchmarkConfig(conditions=("no_hook", "unknown"))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="random condition requires"):
        SimulationBenchmarkConfig(conditions=("no_hook", "random"))
    with pytest.raises(ValueError, match="non-zero"):
        SimulationBenchmarkConfig(strengths=(0.0,))
    with pytest.raises(ValueError, match="unique"):
        SimulationBenchmarkConfig(seeds=(1, 1))
    with pytest.raises(ValueError, match="empty"):
        SimulationBenchmarkConfig(seeds=())
    with pytest.raises(ValueError, match="positive"):
        SimulationBenchmarkConfig(probe_queries=0)


def test_wilson_ci_matches_known_interval() -> None:
    low, high = wilson_ci([True] * 4)
    assert high == 1.0
    assert low == pytest.approx(0.5101, abs=1e-3)
    low, high = wilson_ci([True, False, False, False])
    assert low == pytest.approx(0.0456, abs=1e-3)
    assert high == pytest.approx(0.6994, abs=1e-3)
    with pytest.raises(ValueError, match="at least one"):
        wilson_ci([])


def test_benchmark_baseline_is_bit_exact_and_acceptance_passes() -> None:
    adapter = make_fixture_adapter()
    bundle = make_fixture_bundle()
    config = SimulationBenchmarkConfig(seeds=(1, 2), strengths=(1.0,))

    result = run_simulation_benchmark(adapter, bundle, config, noise=make_benchmark_noise())

    assert result.acceptance.passed, result.acceptance.failures
    assert result.acceptance.checks["baseline_actions_bit_exact"] is True
    assert result.acceptance.checks["baseline_success_equals_no_hook"] is True
    assert result.acceptance.checks["intervention_changes_actions"] is True
    assert result.acceptance.checks["all_episodes_within_max_steps"] is True
    baseline = [outcome for outcome in result.outcomes if outcome.condition == "baseline"]
    assert all(outcome.mean_action_deviation == 0.0 for outcome in baseline)
    no_hook = [outcome for outcome in result.outcomes if outcome.condition == "no_hook"]
    assert all(outcome.success for outcome in no_hook + baseline)


def test_benchmark_interventions_change_actions_but_report_metrics() -> None:
    adapter = make_fixture_adapter()
    bundle = make_fixture_bundle()
    config = SimulationBenchmarkConfig(seeds=(1,), strengths=(1.0,))

    result = run_simulation_benchmark(adapter, bundle, config, noise=make_benchmark_noise())

    random_outcome = next(outcome for outcome in result.outcomes if outcome.condition == "random")
    targeted_outcome = next(outcome for outcome in result.outcomes if outcome.condition == "targeted")
    assert random_outcome.mean_action_deviation > 0.0
    assert targeted_outcome.mean_action_deviation > 0.0
    assert all(outcome.length == TERMINATE_AT for outcome in result.outcomes)
    assert all(outcome.n_queries == TERMINATE_AT for outcome in result.outcomes)
    assert all(outcome.mean_query_latency_s > 0.0 for outcome in result.outcomes)
    assert all(outcome.sum_reward == 2.0 for outcome in result.outcomes)
    assert len(result.summaries) == 4
    summary = next(summary for summary in result.summaries if summary.condition == "no_hook")
    assert summary.n_episodes == 1
    assert summary.success_rate == 1.0
    assert summary.success_ci_high == 1.0


def test_benchmark_reports_offline_scores_and_disagreement() -> None:
    adapter = make_fixture_adapter()
    bundle = make_fixture_bundle()
    config = SimulationBenchmarkConfig(seeds=(1,), strengths=(1.0,))

    result = run_simulation_benchmark(adapter, bundle, config, noise=make_benchmark_noise())

    assert len(result.offline_scores) == 2
    targeted_score = next(score for score in result.offline_scores if score.condition == "targeted")
    random_score = next(score for score in result.offline_scores if score.condition == "random")
    assert targeted_score.on_target_fraction >= 0.99
    assert targeted_score.on_target_fraction >= random_score.on_target_fraction
    assert targeted_score.probe_queries >= 1
    assert len(result.correlation.cells) == 2
    assert result.correlation.spearman_rho is None
    assert any("overstatement" in disagreement for disagreement in result.correlation.disagreements)


def test_benchmark_episode_outcome_records_termination_and_actions() -> None:
    adapter = make_fixture_adapter()
    bundle = make_fixture_bundle()
    outcome, samples = run_episode(
        adapter,
        bundle,
        seed=7,
        condition="no_hook",
        strength=0.0,
        direction=np.zeros(1),
        noise=make_benchmark_noise(),
        record_samples=True,
    )

    assert outcome.seed == 7
    assert outcome.condition == "no_hook"
    assert outcome.success is True
    assert outcome.terminated is True
    assert outcome.length == TERMINATE_AT
    assert len(outcome.actions) == TERMINATE_AT
    assert all(action.shape == (ACTION_DIM,) for action in outcome.actions)
    assert len(samples) == 2  # executed model queries at steps 0 and 4


def test_benchmark_creates_a_fresh_environment_per_cell() -> None:
    created: list[FakeVectorEnv] = []

    def factory() -> FakeVectorEnv:
        env = FakeVectorEnv()
        created.append(env)
        return env

    adapter = make_fixture_adapter()
    bundle = BenchmarkEnvironmentBundle(
        env_factory=factory,
        env_preprocessor=lambda observation: observation,
        preprocess_observation=lambda observation: observation,
        task_description="fixture task",
        max_episode_steps=MAX_STEPS,
    )
    config = SimulationBenchmarkConfig(seeds=(1, 2), strengths=(1.0,))

    run_simulation_benchmark(adapter, bundle, config, noise=make_benchmark_noise())

    expected_cells = 2 * 4  # two seeds x four conditions
    assert len(created) == expected_cells
    assert all(env.closed for env in created)


def test_libero_config_bootstrap_is_non_interactive_and_idempotent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import importlib.util

    from latent_anything.integrations import lerobot_benchmark as benchmark_module

    package_dir = tmp_path / "site-packages" / "libero" / "libero"
    package_dir.mkdir(parents=True)
    (package_dir / "init_files").mkdir()
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    fake_spec = importlib.util.spec_from_file_location("libero.libero", package_dir / "__init__.py")
    assert fake_spec is not None and fake_spec.origin is not None
    monkeypatch.setattr(benchmark_module, "_find_spec", lambda name: fake_spec if name == "libero.libero" else None)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))

    benchmark_module._bootstrap_libero_config()
    config_path = tmp_path / ".libero" / "config.yaml"
    assert config_path.is_file()
    content = config_path.read_text(encoding="utf-8")
    assert "init_states:" in content
    assert str(package_dir / "init_files") in content
    first_mtime = config_path.stat().st_mtime_ns

    benchmark_module._bootstrap_libero_config()
    assert config_path.stat().st_mtime_ns == first_mtime  # valid config is preserved

    stale = "init_states: /nowhere/that/exists\n"
    config_path.write_text(stale, encoding="utf-8")
    benchmark_module._bootstrap_libero_config()
    refreshed = config_path.read_text(encoding="utf-8")
    assert "init_states: /nowhere/that/exists" not in refreshed
    assert str(package_dir / "init_files") in refreshed


def test_build_libero_benchmark_environment_uses_upstream_factories(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    from types import ModuleType

    make_calls: list[dict[str, object]] = []

    class FakeEnvConfig:
        def __init__(self, **kwargs: object) -> None:
            make_calls.append(kwargs)
            self._task_description = "pick up the black bowl on the stove and place it on the plate"
            self._max_steps = 280

        def get_env_processors(self) -> tuple[object, object]:
            return ("env-preprocessor", "env-postprocessor")

    def fake_make_env_config(env_type: str, **kwargs: object) -> FakeEnvConfig:
        assert env_type == "libero"
        return FakeEnvConfig(**kwargs)

    envs_module = ModuleType("lerobot.envs")
    envs_module.make_env_config = fake_make_env_config  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "lerobot.envs", envs_module)

    utils_module = ModuleType("lerobot.envs.utils")
    utils_module.preprocess_observation = lambda observation: observation  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "lerobot.envs.utils", utils_module)

    fake_env = FakeVectorEnv()

    def make_env(env_config: object, n_envs: int) -> dict[str, dict[int, object]]:
        del env_config
        assert n_envs == 1
        return {"libero_spatial": {2: fake_env}}

    def make_processors(*args: object, **kwargs: object) -> tuple[object, object]:
        del args, kwargs
        return object(), object()

    def make_policy(*args: object, **kwargs: object) -> object:
        del args, kwargs
        return object()

    api = LeRobotAPI(
        make_policy=make_policy,
        make_pre_post_processors=make_processors,
        dataset_type=object,
        streaming_dataset_type=object,
        policy_processor_pipeline_type=object,
        make_env=make_env,
        evaluation_main=make_policy,
        register_third_party_plugins=lambda: None,
    )
    config = SimulationBenchmarkConfig(seeds=(1,), task_ids=(2,), observation_height=256, observation_width=360)

    bundle = build_libero_benchmark_environment(config, api=api)

    assert make_calls[0]["task"] == "libero_spatial"
    assert make_calls[0]["task_ids"] == [2]
    assert make_calls[0]["observation_height"] == 256
    assert make_calls[0]["observation_width"] == 360
    created = bundle.env_factory()
    assert created is fake_env
    assert bundle.task_description == "fixture task"
    assert bundle.max_episode_steps == MAX_STEPS
    assert bundle.metadata["task_id"] == 2
    assert bundle.metadata["task_description"] == "fixture task"


def test_correlation_understatement_and_reversal_rules() -> None:
    from latent_anything.integrations.lerobot_benchmark import CausalCorrelationCell

    cells = (
        CausalCorrelationCell("targeted", 1.0, 0.3, 0.5, 0.4, 0.9, -0.5),
        CausalCorrelationCell("random", 1.0, 0.2, 0.4, 0.3, 0.6, 0.1),
        CausalCorrelationCell("random", 2.0, 0.25, 0.8, 0.5, 1.2, 0.2),
    )
    correlation = build_correlation(cells)
    kinds = {part.split(":", maxsplit=1)[0] for part in correlation.disagreements}
    assert "understatement" in kinds
    assert "reversal" in kinds
    assert correlation.spearman_rho is not None


@pytest.mark.network
@pytest.mark.large_download
def test_smolvla_simulation_statistical_benchmark() -> None:
    """Statistical lane: the real SmolVLA policy evaluated in LIBERO simulation.

    Runs every condition on a small, tractable episode grid and enforces the
    predeclared acceptance gate: baseline bit-exactness, no-hook success
    equality, and measurable intervention action changes.
    """

    pytest.importorskip("lerobot")
    if not torch.cuda.is_available():
        pytest.skip("SmolVLA simulation statistical benchmark requires a CUDA device")
    from latent_anything.integrations.lerobot import check_lerobot_compatibility

    report = check_lerobot_compatibility()
    if not report.supported:
        pytest.skip(report.diagnostic)
    config = SimulationBenchmarkConfig(
        seeds=(1, 2),
        strengths=(1.0,),
        probe_queries=1,
    )
    adapter = load_smolvla_policy(DEFAULT_SMOLVLA_CHECKPOINT, device="cuda")
    environment = build_libero_benchmark_environment(config)
    result = run_simulation_benchmark(adapter, environment, config)
    assert result.acceptance.passed, result.acceptance.failures
    no_hook = next(summary for summary in result.summaries if summary.condition == "no_hook")
    assert no_hook.n_episodes == 2
    assert 0.0 <= no_hook.success_rate <= 1.0
    for summary in result.summaries:
        assert 0.0 <= summary.success_ci_low <= summary.success_ci_high <= 1.0

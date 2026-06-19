"""Tests for the Sprint 13 showcase helpers."""
# pyright: reportPrivateUsage=false

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Protocol, TypedDict, cast

import numpy as np
import numpy.typing as npt
import pytest

from latent_anything import Trajectory
from latent_anything.adapters import VAE
from latent_anything.methods import PCA, ActivationPatch

# Ensure we can import from scripts/
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


class DataConfig(TypedDict):
    n_clusters: int
    n_per_cluster: int
    input_dim: int
    latent_dim: int
    noise_scale: float


class VAEConfig(TypedDict):
    hidden_dim: int | None
    n_epochs: int
    learning_rate: float
    beta: float


class SplitConfig(TypedDict):
    source_clusters: list[int]
    target_clusters: list[int]
    n_held_out: int


class PCAConfig(TypedDict):
    n_components: int


class PatchConfig(TypedDict):
    pass


class LerpConfig(TypedDict):
    n_steps: int


class OutputConfig(TypedDict):
    figure: str
    summary: str
    config_snapshot: str


class ShowcaseConfig(TypedDict):
    seed: int
    data: DataConfig
    vae: VAEConfig
    split: SplitConfig
    pca: PCAConfig
    patch: PatchConfig
    lerp: LerpConfig
    output: OutputConfig


class ClusterInfo(TypedDict):
    cluster_centers: FloatArray
    indices_by_label: dict[int, IntArray]


class SplitResult(TypedDict):
    source_data: FloatArray
    target_data: FloatArray
    failure_data: FloatArray
    test_source_idx: IntArray
    test_target_idx: IntArray
    test_target_ref: FloatArray


class BaselineMetrics(TypedDict):
    recon_mse_failure: float
    recon_mse_source: float
    recon_mse_target: float
    dist_to_target_before: float
    centroid_source_to_target: float


class PostMetrics(TypedDict):
    dist_to_target_after: float
    improvement_ratio: float
    dist_delta: float


class FullRunResult(TypedDict):
    vae: VAE
    patch: ActivationPatch
    baseline: BaselineMetrics
    post: PostMetrics
    edited: FloatArray
    failure_data: FloatArray
    failure_data_original: FloatArray


class PCAResults(TypedDict):
    pca: PCA
    proj_source: FloatArray
    proj_target: FloatArray
    proj_failure: FloatArray


class TrajectoryRun(TypedDict):
    patch: ActivationPatch
    orig_decoded: FloatArray
    patched_decoded: FloatArray
    traj_lerp: Trajectory
    api_decoded: FloatArray


class ConfigModule(Protocol):
    SHOWCASE_CONFIG: ShowcaseConfig


class ShowcaseModule(Protocol):
    def _generate_data(self, cfg: ShowcaseConfig) -> tuple[FloatArray, IntArray, ClusterInfo]: ...

    def _split_data(self, points: FloatArray, labels: IntArray, cfg: ShowcaseConfig) -> SplitResult: ...

    def _compute_baseline_metrics(
        self,
        source_data: FloatArray,
        target_data: FloatArray,
        failure_data: FloatArray,
        target_centroid_data: FloatArray,
        vae: VAE,
    ) -> BaselineMetrics: ...

    def _apply_activation_patch(
        self,
        vae: VAE,
        source_data: FloatArray,
        target_data: FloatArray,
        failure_data: FloatArray,
    ) -> tuple[ActivationPatch, FloatArray]: ...

    def _compute_post_metrics(
        self,
        edited_data: FloatArray,
        target_centroid: FloatArray,
        baseline: BaselineMetrics,
    ) -> PostMetrics: ...

    def _project_latent_pca(
        self,
        encoded_source: FloatArray,
        encoded_target: FloatArray,
        encoded_failure: FloatArray,
        n_components: int = 2,
    ) -> tuple[PCA, FloatArray, FloatArray, FloatArray]: ...

    def _build_trajectory_panel(
        self,
        vae: VAE,
        patch: ActivationPatch,
        source_data: FloatArray,
        target_data: FloatArray,
        n_steps: int = 6,
    ) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, Trajectory]: ...


_SPEC = importlib.util.spec_from_file_location(
    "showcase_demo",
    str(_SCRIPTS_DIR / "end_to_end_showcase_demo.py"),
)
assert _SPEC is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
assert isinstance(_MODULE, ModuleType)
_LOADER = _SPEC.loader
assert _LOADER is not None
_LOADER.exec_module(_MODULE)
_SHOWCASE = cast(ShowcaseModule, _MODULE)

_CONFIG_SPEC = importlib.util.spec_from_file_location(
    "showcase_config",
    str(_SCRIPTS_DIR / "showcase_config.py"),
)
assert _CONFIG_SPEC is not None
_CONFIG_MODULE = importlib.util.module_from_spec(_CONFIG_SPEC)
assert isinstance(_CONFIG_MODULE, ModuleType)
_CONFIG_LOADER = _CONFIG_SPEC.loader
assert _CONFIG_LOADER is not None
_CONFIG_LOADER.exec_module(_CONFIG_MODULE)
_CONFIG = cast(ConfigModule, _CONFIG_MODULE)


@pytest.fixture(scope="module")
def cfg() -> ShowcaseConfig:
    return _CONFIG.SHOWCASE_CONFIG


@pytest.fixture(scope="module")
def generated(cfg: ShowcaseConfig) -> tuple[FloatArray, IntArray, ClusterInfo]:
    """Generate data once per module for all tests."""
    return _SHOWCASE._generate_data(cfg)


@pytest.fixture(scope="module")
def split(generated: tuple[FloatArray, IntArray, ClusterInfo], cfg: ShowcaseConfig) -> SplitResult:
    points, labels, _ = generated
    return _SHOWCASE._split_data(points, labels, cfg)


@pytest.fixture(scope="module")
def baseline(split: SplitResult, cfg: ShowcaseConfig) -> BaselineMetrics:
    vae = VAE(
        input_dim=cfg["data"]["input_dim"],
        latent_dim=cfg["data"]["latent_dim"],
        n_epochs=30,
        random_state=cfg["seed"],
    )
    combined = np.vstack([split["source_data"], split["target_data"]])
    vae.fit(combined)
    return _SHOWCASE._compute_baseline_metrics(
        split["source_data"],
        split["target_data"],
        split["failure_data"],
        split["target_data"],
        vae,
    )


@pytest.fixture(scope="module")
def full_run(split: SplitResult, cfg: ShowcaseConfig) -> FullRunResult:
    vae = VAE(
        input_dim=cfg["data"]["input_dim"],
        latent_dim=cfg["data"]["latent_dim"],
        hidden_dim=cfg["vae"]["hidden_dim"],
        n_epochs=100,
        learning_rate=cfg["vae"]["learning_rate"],
        beta=cfg["vae"]["beta"],
        random_state=cfg["seed"],
    )
    combined = np.vstack([split["source_data"], split["target_data"]])
    vae.fit(combined)

    baseline_metrics = _SHOWCASE._compute_baseline_metrics(
        split["source_data"],
        split["target_data"],
        split["failure_data"],
        split["target_data"],
        vae,
    )

    failure_copy = split["failure_data"].copy()
    patch, edited = _SHOWCASE._apply_activation_patch(
        vae,
        split["source_data"],
        split["target_data"],
        split["failure_data"],
    )

    post = _SHOWCASE._compute_post_metrics(
        edited,
        split["target_data"],
        baseline_metrics,
    )

    return {
        "vae": vae,
        "patch": patch,
        "baseline": baseline_metrics,
        "post": post,
        "edited": edited,
        "failure_data": split["failure_data"],
        "failure_data_original": failure_copy,
    }


@pytest.fixture(scope="module")
def pca_results(split: SplitResult, cfg: ShowcaseConfig) -> PCAResults:
    vae = VAE(
        input_dim=cfg["data"]["input_dim"],
        latent_dim=cfg["data"]["latent_dim"],
        n_epochs=30,
        random_state=cfg["seed"],
    )
    combined = np.vstack([split["source_data"], split["target_data"]])
    vae.fit(combined)

    encoded_source = cast(FloatArray, vae.encode(split["source_data"]))
    encoded_target = cast(FloatArray, vae.encode(split["target_data"]))
    encoded_failure = cast(FloatArray, vae.encode(split["failure_data"]))

    pca, proj_source, proj_target, proj_failure = _SHOWCASE._project_latent_pca(
        encoded_source,
        encoded_target,
        encoded_failure,
        n_components=2,
    )
    return {
        "pca": pca,
        "proj_source": proj_source,
        "proj_target": proj_target,
        "proj_failure": proj_failure,
    }


@pytest.fixture(scope="module")
def traj_results(split: SplitResult, cfg: ShowcaseConfig) -> TrajectoryRun:
    vae = VAE(
        input_dim=cfg["data"]["input_dim"],
        latent_dim=cfg["data"]["latent_dim"],
        n_epochs=50,
        random_state=cfg["seed"],
    )
    combined = np.vstack([split["source_data"], split["target_data"]])
    vae.fit(combined)

    patch, _ = _SHOWCASE._apply_activation_patch(
        vae,
        split["source_data"],
        split["target_data"],
        split["failure_data"],
    )

    _, _, orig_decoded, patched_decoded, traj_lerp = _SHOWCASE._build_trajectory_panel(
        vae,
        patch,
        split["source_data"],
        split["target_data"],
        n_steps=cfg["lerp"]["n_steps"],
    )
    api_decoded = cast(FloatArray, patch.apply_trajectory(traj_lerp))
    return {
        "patch": patch,
        "orig_decoded": orig_decoded,
        "patched_decoded": patched_decoded,
        "traj_lerp": traj_lerp,
        "api_decoded": api_decoded,
    }


class TestShowcaseConfig:
    def test_config_has_keys(self, cfg: ShowcaseConfig) -> None:
        assert "seed" in cfg
        assert "data" in cfg
        assert "vae" in cfg
        assert "split" in cfg
        assert "pca" in cfg
        assert "patch" in cfg
        assert "lerp" in cfg
        assert "output" in cfg

    def test_config_seed_is_int(self, cfg: ShowcaseConfig) -> None:
        assert isinstance(cfg["seed"], int)

    def test_config_output_has_paths(self, cfg: ShowcaseConfig) -> None:
        out = cfg["output"]
        assert "figure" in out
        assert "summary" in out
        assert "config_snapshot" in out


class TestShowcaseGenerateData:
    def test_shapes(self, generated: tuple[FloatArray, IntArray, ClusterInfo], cfg: ShowcaseConfig) -> None:
        points, labels, cluster_info = generated
        dc = cfg["data"]
        expected_n = dc["n_clusters"] * dc["n_per_cluster"]
        assert points.shape == (expected_n, dc["input_dim"])
        assert labels.shape == (expected_n,)
        assert len(cluster_info["cluster_centers"]) == dc["n_clusters"]

    def test_range(self, generated: tuple[FloatArray, IntArray, ClusterInfo]) -> None:
        points, _, _ = generated
        assert points.min() >= 0.0
        assert points.max() <= 1.0

    def test_labels(self, generated: tuple[FloatArray, IntArray, ClusterInfo], cfg: ShowcaseConfig) -> None:
        _, labels, cluster_info = generated
        dc = cfg["data"]
        n_per = dc["n_per_cluster"]
        for i in range(dc["n_clusters"]):
            idx = cluster_info["indices_by_label"][i]
            assert len(idx) == n_per
            assert np.all(labels[idx] == i)


class TestShowcaseSplit:
    def test_split_keys(self, split: SplitResult) -> None:
        for key in ("source_data", "target_data", "failure_data", "test_target_ref"):
            assert key in split

    def test_split_shapes(self, split: SplitResult, cfg: ShowcaseConfig) -> None:
        dc = cfg["data"]
        sc = cfg["split"]
        n_per = dc["n_per_cluster"]
        n_source = len(sc["source_clusters"]) * n_per
        n_target = len(sc["target_clusters"]) * n_per
        n_held = sc["n_held_out"]

        assert len(split["source_data"]) == n_source - n_held
        assert len(split["target_data"]) == n_target - n_held
        assert len(split["failure_data"]) == n_held
        assert len(split["test_target_ref"]) == n_held

    def test_split_dims(self, split: SplitResult, cfg: ShowcaseConfig) -> None:
        input_dim = cfg["data"]["input_dim"]
        assert split["source_data"].shape[1] == input_dim
        assert split["target_data"].shape[1] == input_dim
        assert split["failure_data"].shape[1] == input_dim
        assert split["test_target_ref"].shape[1] == input_dim


class TestShowcaseBaselineMetrics:
    def test_metric_keys(self, baseline: BaselineMetrics) -> None:
        expected_keys = {
            "recon_mse_failure",
            "recon_mse_source",
            "recon_mse_target",
            "dist_to_target_before",
            "centroid_source_to_target",
        }
        assert set(baseline.keys()) == expected_keys

    def test_metrics_are_finite(self, baseline: BaselineMetrics) -> None:
        metric_keys = (
            "recon_mse_failure",
            "recon_mse_source",
            "recon_mse_target",
            "dist_to_target_before",
            "centroid_source_to_target",
        )
        for key in metric_keys:
            val = baseline[key]
            assert np.isfinite(val), f"{key} is not finite: {val}"

    def test_metrics_are_positive(self, baseline: BaselineMetrics) -> None:
        metric_keys = (
            "recon_mse_failure",
            "recon_mse_source",
            "recon_mse_target",
            "dist_to_target_before",
            "centroid_source_to_target",
        )
        for key in metric_keys:
            val = baseline[key]
            assert val >= 0.0, f"{key} is negative: {val}"


class TestShowcasePostMetrics:
    """Verify that ActivationPatch improves distance to target."""

    def test_improvement_is_positive(self, full_run: FullRunResult) -> None:
        assert full_run["post"]["improvement_ratio"] > 0.0

    def test_dist_to_target_decreases(self, full_run: FullRunResult) -> None:
        assert full_run["post"]["dist_to_target_after"] < full_run["baseline"]["dist_to_target_before"]

    def test_edited_shape_matches_failure(self, full_run: FullRunResult) -> None:
        assert full_run["edited"].shape == full_run["failure_data"].shape

    def test_edit_does_not_mutate_input(self, full_run: FullRunResult) -> None:
        assert np.allclose(full_run["failure_data"], full_run["failure_data_original"])


class TestShowcasePCA:
    def test_projection_shapes(self, pca_results: PCAResults) -> None:
        assert pca_results["proj_source"].shape[1] == 2
        assert pca_results["proj_target"].shape[1] == 2
        assert pca_results["proj_failure"].shape[1] == 2
        assert pca_results["pca"].explained_variance_ratio_.shape[0] == 2


class TestShowcaseTrajectory:
    def test_trajectory_length(self, traj_results: TrajectoryRun, cfg: ShowcaseConfig) -> None:
        expected_len = cfg["lerp"]["n_steps"] + 1
        assert len(traj_results["traj_lerp"]) == expected_len
        assert traj_results["orig_decoded"].shape[0] == expected_len
        assert traj_results["patched_decoded"].shape[0] == expected_len

    def test_trajectory_panel_uses_public_patch_api(self, traj_results: TrajectoryRun) -> None:
        assert np.allclose(traj_results["patched_decoded"], traj_results["api_decoded"])

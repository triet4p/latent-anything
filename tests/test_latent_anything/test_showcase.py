"""Tests for the Sprint 13 showcase helpers.

Target: ~10–12 tests covering:
- Config import and key presence
- Data generation: shapes, range, cluster structure
- Split logic: source/target/failure sizes, held-out count
- Baseline metrics: all keys present, values are finite floats
- Post metrics: improvement_ratio is positive (synthetic setup)
- PCA projection: output shapes match input
- ActivationPatch edit: output shapes, non-mutation
- Trajectory panel: correct shapes
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure we can import from scripts/
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

# ---------------------------------------------------------------------------
# Import helpers from the showcase script
# ---------------------------------------------------------------------------
# We import the module directly to access helper functions
import importlib.util  # noqa: E402

from showcase_config import SHOWCASE_CONFIG  # noqa: E402

_SPEC = importlib.util.spec_from_file_location(
    "showcase_demo",
    str(_SCRIPTS_DIR / "end_to_end_showcase_demo.py"),
)
assert _SPEC is not None
_SHOWCASE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_SHOWCASE)  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cfg() -> dict:
    return SHOWCASE_CONFIG


@pytest.fixture(scope="module")
def generated(cfg: dict) -> tuple:
    """Generate data once per module for all tests."""
    points, labels, cluster_info = _SHOWCASE._generate_data(cfg)  # noqa: SLF001
    return points, labels, cluster_info


@pytest.fixture(scope="module")
def split(generated: tuple, cfg: dict) -> dict:
    points, labels, _ = generated
    return _SHOWCASE._split_data(points, labels, cfg)  # noqa: SLF001


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


class TestShowcaseConfig:
    def test_config_has_keys(self, cfg: dict) -> None:
        assert "seed" in cfg
        assert "data" in cfg
        assert "vae" in cfg
        assert "split" in cfg
        assert "pca" in cfg
        assert "patch" in cfg
        assert "lerp" in cfg
        assert "output" in cfg

    def test_config_seed_is_int(self, cfg: dict) -> None:
        assert isinstance(cfg["seed"], int)

    def test_config_output_has_paths(self, cfg: dict) -> None:
        out = cfg["output"]
        assert "figure" in out
        assert "summary" in out
        assert "config_snapshot" in out


# ---------------------------------------------------------------------------
# Data generation tests
# ---------------------------------------------------------------------------


class TestShowcaseGenerateData:
    def test_shapes(self, generated: tuple, cfg: dict) -> None:
        points, labels, cluster_info = generated
        dc = cfg["data"]
        expected_n = dc["n_clusters"] * dc["n_per_cluster"]
        assert points.shape == (expected_n, dc["input_dim"])
        assert labels.shape == (expected_n,)
        assert len(cluster_info["cluster_centers"]) == dc["n_clusters"]

    def test_range(self, generated: tuple) -> None:
        points, _, _ = generated
        assert points.min() >= 0.0
        assert points.max() <= 1.0

    def test_labels(self, generated: tuple, cfg: dict) -> None:
        _, labels, cluster_info = generated
        dc = cfg["data"]
        n_per = dc["n_per_cluster"]
        for i in range(dc["n_clusters"]):
            idx = cluster_info["indices_by_label"][i]
            assert len(idx) == n_per
            assert np.all(labels[idx] == i)


# ---------------------------------------------------------------------------
# Split tests
# ---------------------------------------------------------------------------


class TestShowcaseSplit:
    def test_split_keys(self, split: dict) -> None:
        for key in ("source_data", "target_data", "failure_data", "test_target_ref"):
            assert key in split

    def test_split_shapes(self, split: dict, cfg: dict) -> None:
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

    def test_split_dims(self, split: dict, cfg: dict) -> None:
        input_dim = cfg["data"]["input_dim"]
        assert split["source_data"].shape[1] == input_dim
        assert split["target_data"].shape[1] == input_dim
        assert split["failure_data"].shape[1] == input_dim
        assert split["test_target_ref"].shape[1] == input_dim


# ---------------------------------------------------------------------------
# Baseline metrics tests
# ---------------------------------------------------------------------------


class TestShowcaseBaselineMetrics:
    _baseline: dict = {}  # set by classmethod fixture

    def test_metric_keys(self, baseline: dict) -> None:
        expected_keys = {
            "recon_mse_failure",
            "recon_mse_source",
            "recon_mse_target",
            "dist_to_target_before",
            "centroid_source_to_target",
        }
        assert set(baseline.keys()) == expected_keys

    def test_metrics_are_finite(self, baseline: dict) -> None:
        for key, val in baseline.items():
            assert np.isfinite(val), f"{key} is not finite: {val}"

    def test_metrics_are_positive(self, baseline: dict) -> None:
        for key, val in baseline.items():
            assert val >= 0.0, f"{key} is negative: {val}"

    @pytest.fixture(scope="class")
    @classmethod
    def baseline(cls, split: dict, cfg: dict) -> dict:  # noqa: ANN206
        import latent_anything

        vae = latent_anything.adapters.VAE(
            input_dim=cfg["data"]["input_dim"],
            latent_dim=cfg["data"]["latent_dim"],
            n_epochs=30,
            random_state=cfg["seed"],
        )
        combined = np.vstack([split["source_data"], split["target_data"]])
        vae.fit(combined)
        cls._baseline = _SHOWCASE._compute_baseline_metrics(
            combined,
            split["failure_data"],
            split["target_data"],
            split["target_data"],
            vae,
        )
        return cls._baseline


# ---------------------------------------------------------------------------
# Post metrics test (requires a full VAE + ActivationPatch)
# ---------------------------------------------------------------------------


class TestShowcasePostMetrics:
    """Verify that ActivationPatch improves distance to target."""

    def test_improvement_is_positive(self, full_run: dict) -> None:
        assert full_run["post"]["improvement_ratio"] > 0.0

    def test_dist_to_target_decreases(self, full_run: dict) -> None:
        assert full_run["post"]["dist_to_target_after"] < full_run["baseline"]["dist_to_target_before"]

    def test_edited_shape_matches_failure(self, full_run: dict) -> None:
        assert full_run["edited"].shape == full_run["failure_data"].shape

    def test_edit_does_not_mutate_input(self, full_run: dict) -> None:
        # Check failure_data was not mutated during patch
        assert np.allclose(full_run["failure_data"], full_run["failure_data_original"])

    @pytest.fixture(scope="class")
    @classmethod
    def full_run(cls, split: dict, cfg: dict) -> dict:  # noqa: ANN206
        import latent_anything

        vae = latent_anything.adapters.VAE(
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

        baseline = _SHOWCASE._compute_baseline_metrics(  # noqa: SLF001
            combined,
            split["failure_data"],
            split["target_data"],
            split["target_data"],
            vae,
        )

        failure_copy = split["failure_data"].copy()
        patch, edited = _SHOWCASE._apply_activation_patch(  # noqa: SLF001
            vae,
            split["source_data"],
            split["target_data"],
            split["failure_data"],
        )

        post = _SHOWCASE._compute_post_metrics(  # noqa: SLF001
            edited,
            split["target_data"],
            baseline,
        )

        return {
            "vae": vae,
            "patch": patch,
            "baseline": baseline,
            "post": post,
            "edited": edited,
            "failure_data": split["failure_data"],
            "failure_data_original": failure_copy,
        }


# ---------------------------------------------------------------------------
# PCA projection tests
# ---------------------------------------------------------------------------


class TestShowcasePCA:
    def test_projection_shapes(self, pca_results: tuple) -> None:
        pca, proj_source, proj_target, proj_failure = pca_results
        assert proj_source.shape[1] == 2
        assert proj_target.shape[1] == 2
        assert proj_failure.shape[1] == 2
        assert pca.explained_variance_ratio_ is not None

    @pytest.fixture(scope="class")
    @classmethod
    def pca_results(cls, split: dict, cfg: dict) -> tuple:  # noqa: ANN206
        import latent_anything

        vae = latent_anything.adapters.VAE(
            input_dim=cfg["data"]["input_dim"],
            latent_dim=cfg["data"]["latent_dim"],
            n_epochs=30,
            random_state=cfg["seed"],
        )
        combined = np.vstack([split["source_data"], split["target_data"]])
        vae.fit(combined)

        encoded_source = vae.encode(split["source_data"])
        encoded_target = vae.encode(split["target_data"])
        encoded_failure = vae.encode(split["failure_data"])

        return _SHOWCASE._project_latent_pca(  # noqa: SLF001
            encoded_source,
            encoded_target,
            encoded_failure,
            n_components=2,
        )


# ---------------------------------------------------------------------------
# Trajectory panel tests
# ---------------------------------------------------------------------------


class TestShowcaseTrajectory:
    def test_trajectory_length(self, traj_results: tuple, cfg: dict) -> None:
        _, _, orig_decoded, patched_decoded, traj_lerp = traj_results
        expected_len = cfg["lerp"]["n_steps"] + 1
        assert len(traj_lerp) == expected_len
        assert orig_decoded.shape[0] == expected_len
        assert patched_decoded.shape[0] == expected_len

    @pytest.fixture(scope="class")
    def traj_results(self, split: dict, cfg: dict) -> tuple:
        import latent_anything

        vae = latent_anything.adapters.VAE(
            input_dim=cfg["data"]["input_dim"],
            latent_dim=cfg["data"]["latent_dim"],
            n_epochs=50,
            random_state=cfg["seed"],
        )
        combined = np.vstack([split["source_data"], split["target_data"]])
        vae.fit(combined)

        patch, _ = _SHOWCASE._apply_activation_patch(  # noqa: SLF001
            vae,
            split["source_data"],
            split["target_data"],
            split["failure_data"],
        )

        return _SHOWCASE._build_trajectory_panel(  # noqa: SLF001
            vae,
            patch,
            split["source_data"],
            split["target_data"],
            n_steps=cfg["lerp"]["n_steps"],
        )

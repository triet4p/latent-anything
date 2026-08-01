"""Smoke the interactive real-model visualization walkthrough path."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.interactive_viz_walkthrough import (
    WalkthroughResult,
    _export,  # pyright: ignore[reportPrivateUsage]
    build_digits_views,
    measure_responsiveness,
)


def _small_result() -> WalkthroughResult:
    return build_digits_views(
        n_samples=64,
        vae_epochs=2,
        latent_dim=3,
        n_clusters=4,
        density_components=2,
        sae_components=4,
        sae_epochs=10,
    )


@pytest.fixture(scope="module")
def small_result() -> WalkthroughResult:
    return _small_result()


def test_walkthrough_builds_all_views(small_result: WalkthroughResult) -> None:
    result = small_result
    assert set(result.views) == {"kmeans", "probe", "density", "trajectory", "atlas"}
    assert result.views["kmeans"].n_points > 0
    assert result.views["probe"].n_points > 0
    assert result.views["density"].n_points > 0
    assert result.views["trajectory"].n_points >= 2
    assert result.views["atlas"].n_points > 0


def test_walkthrough_records_quantitative_metrics(small_result: WalkthroughResult) -> None:
    result = small_result
    for group in ("adapter", "kmeans", "probe", "density", "trajectory", "atlas"):
        assert result.metrics[group], f"missing metrics for {group}"
    assert "silhouette_score" in result.metrics["kmeans"]
    assert "accuracy" in result.metrics["probe"]
    assert "auroc" in result.metrics["density"]
    assert "dead_fraction" in result.metrics["atlas"]


def test_walkthrough_trajectory_has_overlay(small_result: WalkthroughResult) -> None:
    result = small_result
    assert len(result.views["trajectory"].trajectories) == 1


def test_responsiveness_target_holds() -> None:
    report = measure_responsiveness()
    assert report["n_input"] == 60_000
    assert report["n_rendered"] <= 50_000
    assert report["n_rendered"] > 0


def test_walkthrough_export_writes_html_and_png(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, small_result: WalkthroughResult
) -> None:
    monkeypatch.chdir(tmp_path)
    result = small_result
    exported = _export(result)
    names = [path.name for path in exported]
    for chart in ("kmeans", "probe", "density", "trajectory", "atlas"):
        assert f"{chart}.html" in names
        assert f"{chart}.png" in names
    html = (tmp_path / "artifacts" / "interactive-viz-walkthrough" / "kmeans.html").read_text(encoding="utf-8")
    assert "<div" in html and "plotly" in html
    png = (tmp_path / "artifacts" / "interactive-viz-walkthrough" / "kmeans.png").read_bytes()
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_walkthrough_artifacts_reproducible() -> None:
    first = _small_result()
    second = _small_result()
    assert first.views["kmeans"].to_dict() == second.views["kmeans"].to_dict()


def test_responsiveness_uses_deterministic_seed() -> None:
    assert measure_responsiveness()["n_rendered"] == measure_responsiveness()["n_rendered"]

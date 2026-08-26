"""Smoke the interactive real-model visualization walkthrough path."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from latent_anything.visualization import build_projection, prepare_view, projection_explorer
from scripts.interactive_viz_walkthrough import (
    WalkthroughResult,
    _export,  # pyright: ignore[reportPrivateUsage]
    build_digits_views,
    measure_responsiveness,
)

pytestmark = pytest.mark.viz


@pytest.fixture(scope="module", autouse=True)
def _require_viz_backend() -> None:  # pyright: ignore[reportUnusedFunction]
    pytest.importorskip("plotly", reason="visualization walkthrough requires the 'viz' extra (uv sync --extra viz)")
    pytest.importorskip("kaleido", reason="visualization walkthrough exports PNGs and requires the 'viz' extra")


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


def _responsiveness_snapshot(
    seed: int = 0,
) -> tuple[dict[str, int], dict[str, Any], tuple[dict[str, Any], ...], str, str]:
    rng = np.random.default_rng(seed)
    view = build_projection(
        rng.random((60_000, 2)),
        categories=[f"c{i % 8}" for i in range(60_000)],
        title="Responsiveness check — 60k points",
    )
    prepared = prepare_view(view)
    figure = projection_explorer(view)
    rendered = sum(len(trace.x) for trace in figure.data if trace.type == "scatter")
    selected_indices = (0, prepared.n_points // 2, prepared.n_points - 1)
    selection_metadata = tuple(
        {"index": index, "point": prepared.points[index].to_dict()} for index in selected_indices
    )
    view_payload = prepared.to_dict()
    selection_payload = {"indices": selected_indices, "points": selection_metadata}
    view_digest = hashlib.sha256(
        json.dumps(view_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    selection_digest = hashlib.sha256(
        json.dumps(selection_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return (
        {"n_input": 60_000, "n_rendered": rendered, "n_kept": prepared.n_points},
        dict(prepared.metadata),
        selection_metadata,
        view_digest,
        selection_digest,
    )


def test_responsiveness_uses_deterministic_seed() -> None:
    first = _responsiveness_snapshot()
    second = _responsiveness_snapshot()
    assert first == second
    assert first[0]["n_rendered"] <= 50_000
    assert first[0]["n_rendered"] > 0
    assert _responsiveness_snapshot(seed=1)[3:] != first[3:]

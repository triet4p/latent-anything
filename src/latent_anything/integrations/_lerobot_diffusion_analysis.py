"""Observational analysis for captured LeRobot Diffusion traces."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from latent_anything.density import GaussianMixtureDensity, GMMConfig
from latent_anything.latent_space import LatentSpace
from latent_anything.methods.pca import PCA
from latent_anything.probes import ControlBaselines, LinearProbe, LinearProbeConfig, LinearProbeResult, compute_controls


@dataclass(frozen=True)
class DiffusionAnalysisResult:
    """Projection, probing, density, controls, and two-axis trajectory metrics."""

    projected_conditioning: np.ndarray
    projection_explained_variance: np.ndarray
    conditioning_probe: LinearProbeResult
    conditioning_controls: ControlBaselines
    conditioning_density_auroc: float
    episode_time_lengths: Mapping[str, float]
    timestep_time_lengths: Mapping[str, Mapping[str, float]]
    metadata: Mapping[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        """Return an observational analysis summary without raw arrays."""
        return {
            "projected_conditioning_shape": list(self.projected_conditioning.shape),
            "projection_explained_variance": self.projection_explained_variance.tolist(),
            "conditioning_probe": self.conditioning_probe.to_dict(),
            "conditioning_controls": asdict(self.conditioning_controls),
            "conditioning_density_auroc": self.conditioning_density_auroc,
            "episode_time_lengths": dict(self.episode_time_lengths),
            "timestep_time_lengths": {key: dict(value) for key, value in self.timestep_time_lengths.items()},
            "metadata": dict(self.metadata),
        }


def analyze_traces(
    traces: Sequence[Any],  # internal seam avoids a facade import cycle
    *,
    n_components: int = 2,
    probe_config: LinearProbeConfig | None = None,
    random_state: int = 0,
) -> DiffusionAnalysisResult:
    """Analyze conditioning labels and keep episode/timestep trajectories distinct."""
    if len(traces) < 2:
        raise ValueError("at least two Diffusion episode traces are required")
    outcomes = {trace.outcome for trace in traces}
    if outcomes != {"success", "failure"}:
        raise ValueError("Diffusion analysis requires both success and failure traces")

    features = np.vstack([trace.conditioning_trajectory.to_numpy() for trace in traces])
    labels = np.concatenate(
        [
            np.full(len(trace.conditioning_trajectory), int(trace.outcome == "success"), dtype=np.int64)
            for trace in traces
        ]
    )
    projection = PCA(n_components=n_components)
    projection.fit(features)
    projected = projection.transform(features)
    probe = LinearProbe(probe_config or LinearProbeConfig(random_state=random_state))
    probe_result = probe.fit(
        features, labels, provenance={"analysis": "diffusion_observational", "label": "episode_outcome"}
    )
    controls = compute_controls(
        features,
        labels,
        train_indices=probe_result.train_indices,
        test_indices=probe_result.test_indices,
        random_state=random_state,
    )

    success_features = np.vstack(
        [trace.conditioning_trajectory.to_numpy() for trace in traces if trace.outcome == "success"]
    )
    failure_features = np.vstack(
        [trace.conditioning_trajectory.to_numpy() for trace in traces if trace.outcome == "failure"]
    )
    split = max(features.shape[1] * 3, success_features.shape[0] // 2)
    if split >= success_features.shape[0]:
        split = success_features.shape[0] - 1
    density = GaussianMixtureDensity(GMMConfig(n_components=1, random_state=random_state)).fit(
        success_features[:split],
        source_representation_identity="diffusion:observation_conditioning",
    )
    density_report = density.evaluate(
        success_features[split:],
        failure_features,
        source_representation_identity="diffusion:observation_conditioning",
        split_provenance={"fit": "success_prefix", "evaluation": "success_suffix_vs_failure"},
    )

    space = LatentSpace(dim=features.shape[1], source_model="diffusion_observation_conditioning")
    episode_lengths: dict[str, float] = {}
    timestep_lengths: dict[str, dict[str, float]] = {}
    for trace in traces:
        condition_values = trace.conditioning_trajectory.to_numpy()
        episode_lengths[trace.episode_id] = _trajectory_length(space, condition_values)
        per_timestep: dict[str, float] = {}
        timesteps = sorted(
            {
                int(representation.diffusion_timestep)
                for selection in trace.selections
                for representation in selection.representations
                if representation.kind == "denoising_action" and representation.diffusion_timestep is not None
            }
        )
        for timestep in timesteps:
            values = trace.denoising_by_timestep(timestep)
            denoising_space = LatentSpace(dim=values.shape[1], source_model="diffusion_denoising_action")
            per_timestep[str(timestep)] = _trajectory_length(denoising_space, values)
        timestep_lengths[trace.episode_id] = per_timestep

    return DiffusionAnalysisResult(
        projected_conditioning=projected,
        projection_explained_variance=projection.explained_variance_ratio_,
        conditioning_probe=probe_result,
        conditioning_controls=controls,
        conditioning_density_auroc=density_report.metrics.auroc,
        episode_time_lengths=episode_lengths,
        timestep_time_lengths=timestep_lengths,
        metadata={
            "analysis": "observational_diffusion_representation",
            "axes": ["episode_time", "action_chunk_position", "diffusion_timestep"],
            "negative_controls": ["majority_class", "shuffled_label", "raw_input_not_used"],
            "causal_intervention": False,
        },
    )


def _trajectory_length(space: LatentSpace, values: np.ndarray) -> float:
    """Compute consecutive path length on one explicitly chosen space."""
    if len(values) < 2:
        return 0.0
    return float(sum(space.distance(before, after) for before, after in zip(values[:-1], values[1:], strict=True)))

"""Latent Understanding, Manipulation & Execution Network.

A Python framework that treats latent space as a first-class object:
load latent representations from any model, inspect them, manipulate
them, and execute pipelines efficiently.

Plugin-first architecture with three pillars:
- **Introspection (A)** — Visualization, probing, clustering, sparse
  decomposition, trajectory analysis.
- **Manipulation (B)** — Interpolation, arithmetic, steering, activation
  patching, composition, constrained editing.
- **Runtime (C)** — Batching, caching, async execution, streaming,
  profiling.
"""

# Trigger built-in registration into GLOBAL_REGISTRY before any
# registry-dependent imports (like config).
from latent_anything import _plugin_builtins as _plugin_builtins  # noqa: F401  # trigger registration
from latent_anything.clustering import ClusterStabilityReport as ClusterStabilityReport
from latent_anything.clustering import KMeans as KMeans
from latent_anything.clustering import KMeansConfig as KMeansConfig
from latent_anything.clustering import KMeansResult as KMeansResult
from latent_anything.clustering import check_clustering_geometry as check_clustering_geometry
from latent_anything.clustering import cluster_stability_analysis as cluster_stability_analysis
from latent_anything.clustering import compare_with_labels as compare_with_labels
from latent_anything.config import ObjectSpec as ObjectSpec
from latent_anything.config import build_from_config as build_from_config
from latent_anything.config import build_from_dict as build_from_dict
from latent_anything.covariance import CovarianceConfig as CovarianceConfig
from latent_anything.covariance import CovarianceState as CovarianceState
from latent_anything.covariance import fit_covariance_state as fit_covariance_state
from latent_anything.density import (
    DensityEvaluationReport as DensityEvaluationReport,
)
from latent_anything.density import (
    DensityMetrics as DensityMetrics,
)
from latent_anything.density import (
    DensityResult as DensityResult,
)
from latent_anything.density import (
    DensityStabilityReport as DensityStabilityReport,
)
from latent_anything.density import (
    GaussianMixtureDensity as GaussianMixtureDensity,
)
from latent_anything.density import (
    GMMConfig as GMMConfig,
)
from latent_anything.density import (
    cross_seed_evaluation as density_cross_seed_evaluation,
)
from latent_anything.density import (
    mahalanobis_baseline as mahalanobis_baseline,
)
from latent_anything.dtw import DTWConfig as DTWConfig
from latent_anything.dtw import DTWCostSummary as DTWCostSummary
from latent_anything.dtw import DTWResult as DTWResult
from latent_anything.dtw import compute_dtw as compute_dtw
from latent_anything.dtw import indexwise_distance as indexwise_distance
from latent_anything.geodesic import DensityGeodesic as DensityGeodesic
from latent_anything.geodesic import GeodesicConfig as GeodesicConfig
from latent_anything.geodesic import GeodesicPath as GeodesicPath
from latent_anything.geodesic import PathOptimizationStatus as PathOptimizationStatus
from latent_anything.integrated_gradients import (
    IntegratedGradients as IntegratedGradients,
)
from latent_anything.integrated_gradients import (
    IntegratedGradientsConfig as IntegratedGradientsConfig,
)
from latent_anything.integrated_gradients import (
    IntegratedGradientsResult as IntegratedGradientsResult,
)
from latent_anything.integrated_gradients import (
    SensitivityReport as SensitivityReport,
)
from latent_anything.integrated_gradients import (
    compute_integrated_gradients as compute_integrated_gradients,
)
from latent_anything.integrated_gradients import (
    evaluate_sensitivity as evaluate_sensitivity,
)
from latent_anything.latent_space import LatentSpace as LatentSpace
from latent_anything.latent_value import LatentValue as LatentValue
from latent_anything.latent_value import (
    assert_arithmetic_compatible as assert_arithmetic_compatible,
)
from latent_anything.latent_value import coordinate_identity as coordinate_identity
from latent_anything.methods import Method as Method
from latent_anything.mlp_probe import MLPProbe as MLPProbe
from latent_anything.mlp_probe import MLPProbeConfig as MLPProbeConfig
from latent_anything.mlp_probe import MLPProbeResult as MLPProbeResult
from latent_anything.mlp_probe import ProbeComparison as ProbeComparison
from latent_anything.mlp_probe import compare_probes as compare_probes
from latent_anything.mlp_probe import nonlinear_memorization_test as nonlinear_memorization_test
from latent_anything.pipeline import AnalysisPipeline as AnalysisPipeline
from latent_anything.pipeline import ManipulationPipeline as ManipulationPipeline
from latent_anything.pipeline import ManipulationPipelineSpec as ManipulationPipelineSpec
from latent_anything.pipeline import PipelineResult as PipelineResult
from latent_anything.pipeline import PipelineSpec as PipelineSpec
from latent_anything.pipeline import build_manipulation_pipeline_from_config as build_manipulation_pipeline_from_config
from latent_anything.pipeline import build_pipeline_from_config as build_pipeline_from_config
from latent_anything.pose import SE3 as SE3
from latent_anything.pose import SO3 as SO3
from latent_anything.pose import PoseConfig as PoseConfig
from latent_anything.pose import PoseMetadata as PoseMetadata
from latent_anything.pose import PoseTrajectory as PoseTrajectory
from latent_anything.probes import ControlBaselines as ControlBaselines
from latent_anything.probes import CrossSeedReport as CrossSeedReport
from latent_anything.probes import LinearProbe as LinearProbe
from latent_anything.probes import LinearProbeConfig as LinearProbeConfig
from latent_anything.probes import LinearProbeResult as LinearProbeResult
from latent_anything.probes import cross_seed_evaluation as cross_seed_evaluation
from latent_anything.probes import evaluate_layers as evaluate_layers
from latent_anything.projection import OrthonormalSubspace as OrthonormalSubspace
from latent_anything.projection import SubspaceProjection as SubspaceProjection
from latent_anything.projection import SubspaceProjectionConfig as SubspaceProjectionConfig
from latent_anything.registry import GLOBAL_REGISTRY as GLOBAL_REGISTRY
from latent_anything.registry import Registry as Registry
from latent_anything.registry import RegistryEntry as RegistryEntry
from latent_anything.registry import list_entries as list_entries
from latent_anything.registry import lookup as lookup_entry
from latent_anything.registry import register as register_entry
from latent_anything.run_record import ArtifactRef as ArtifactRef
from latent_anything.run_record import DuplicateRunError as DuplicateRunError
from latent_anything.run_record import FileSystemRunRecorder as FileSystemRunRecorder
from latent_anything.run_record import RunComparisonReport as RunComparisonReport
from latent_anything.run_record import RunRecord as RunRecord
from latent_anything.run_record import build_comparison_report as build_comparison_report
from latent_anything.run_record import compute_run_identity as compute_run_identity
from latent_anything.run_record import migrate_run_record as migrate_run_record
from latent_anything.runtime import BatchExecutor as BatchExecutor
from latent_anything.runtime import CacheKey as CacheKey
from latent_anything.runtime import CacheStats as CacheStats
from latent_anything.runtime import InMemoryCache as InMemoryCache
from latent_anything.runtime import ProfileEvent as ProfileEvent
from latent_anything.runtime import RuntimeProfile as RuntimeProfile
from latent_anything.runtime import RuntimeProfiler as RuntimeProfiler
from latent_anything.sae_evaluation import (
    FeatureAtlas as FeatureAtlas,
)
from latent_anything.sae_evaluation import (
    FeatureAtlasEntry as FeatureAtlasEntry,
)
from latent_anything.sae_evaluation import (
    FeatureCrossCheck as FeatureCrossCheck,
)
from latent_anything.sae_evaluation import (
    FeatureRanking as FeatureRanking,
)
from latent_anything.sae_evaluation import (
    SAEConfig as SAEConfig,
)
from latent_anything.sae_evaluation import (
    SAEEvaluationResult as SAEEvaluationResult,
)
from latent_anything.sae_evaluation import (
    SAEFeatureEvaluation as SAEFeatureEvaluation,
)
from latent_anything.sae_evaluation import (
    SAEFeatureMetrics as SAEFeatureMetrics,
)
from latent_anything.sae_evaluation import (
    SAEStabilityResult as SAEStabilityResult,
)
from latent_anything.sae_evaluation import (
    build_feature_atlas as build_feature_atlas,
)
from latent_anything.sae_evaluation import (
    cross_check_feature as cross_check_feature,
)
from latent_anything.sae_evaluation import (
    cross_seed_sae_stability as cross_seed_sae_stability,
)
from latent_anything.sae_evaluation import (
    evaluate_sae_features as evaluate_sae_features,
)
from latent_anything.sae_evaluation import (
    load_feature_atlas as load_feature_atlas,
)
from latent_anything.sae_evaluation import (
    rank_feature_examples as rank_feature_examples,
)
from latent_anything.sae_evaluation import (
    save_feature_atlas as save_feature_atlas,
)
from latent_anything.tcav import TCAV as TCAV
from latent_anything.tcav import ConceptDataset as ConceptDataset
from latent_anything.tcav import ConceptDirectionResult as ConceptDirectionResult
from latent_anything.tcav import TCAVConfig as TCAVConfig
from latent_anything.tcav import TCAVResult as TCAVResult
from latent_anything.tcav import TCAVScore as TCAVScore
from latent_anything.tcav import TransformerLogitTarget as TransformerLogitTarget
from latent_anything.tcav import compute_tcav as compute_tcav
from latent_anything.tcav import intervention_agreement as intervention_agreement
from latent_anything.tcav import learn_linear_separator_direction as learn_linear_separator_direction
from latent_anything.tcav import learn_mean_diff_direction as learn_mean_diff_direction
from latent_anything.temporal import (
    BoundaryMetrics as BoundaryMetrics,
)
from latent_anything.temporal import (
    ChangePointResult as ChangePointResult,
)
from latent_anything.temporal import (
    Segment as Segment,
)
from latent_anything.temporal import (
    SegmentationConfig as SegmentationConfig,
)
from latent_anything.temporal import (
    SmoothedTrajectory as SmoothedTrajectory,
)
from latent_anything.temporal import (
    SmoothingConfig as SmoothingConfig,
)
from latent_anything.temporal import (
    detect_change_points as detect_change_points,
)
from latent_anything.temporal import (
    evaluate_boundaries as evaluate_boundaries,
)
from latent_anything.temporal import (
    smooth_trajectory as smooth_trajectory,
)
from latent_anything.temporal import (
    smoothing_distortion as smoothing_distortion,
)
from latent_anything.trajectory import Trajectory as Trajectory
from latent_anything.transition import (
    DeterministicLatentTransition as DeterministicLatentTransition,
)
from latent_anything.transition import GaussianPrediction as GaussianPrediction
from latent_anything.transition import OneStepMetrics as OneStepMetrics
from latent_anything.transition import RolloutMetrics as RolloutMetrics
from latent_anything.transition import StochasticGaussianLatentTransition as StochasticGaussianLatentTransition
from latent_anything.transition import StochasticOneStepMetrics as StochasticOneStepMetrics
from latent_anything.transition import StochasticRollout as StochasticRollout
from latent_anything.transition import StochasticRolloutMetrics as StochasticRolloutMetrics

__version__ = "0.1.0b1"

__all__ = [
    "AnalysisPipeline",
    "BatchExecutor",
    "CacheKey",
    "CacheStats",
    "ClusterStabilityReport",
    "ConceptDataset",
    "ConceptDirectionResult",
    "ControlBaselines",
    "CovarianceConfig",
    "CovarianceState",
    "CrossSeedReport",
    "FeatureAtlas",
    "FeatureAtlasEntry",
    "FeatureCrossCheck",
    "FeatureRanking",
    "GLOBAL_REGISTRY",
    "InMemoryCache",
    "IntegratedGradients",
    "IntegratedGradientsConfig",
    "IntegratedGradientsResult",
    "KMeans",
    "KMeansConfig",
    "KMeansResult",
    "LatentSpace",
    "LatentValue",
    "LinearProbe",
    "LinearProbeConfig",
    "LinearProbeResult",
    "MLPProbe",
    "MLPProbeConfig",
    "MLPProbeResult",
    "ManipulationPipeline",
    "ManipulationPipelineSpec",
    "Method",
    "ObjectSpec",
    "OrthonormalSubspace",
    "PipelineResult",
    "PipelineSpec",
    "PoseConfig",
    "PoseMetadata",
    "PoseTrajectory",
    "ProbeComparison",
    "SubspaceProjection",
    "SubspaceProjectionConfig",
    "TCAV",
    "TCAVConfig",
    "TCAVResult",
    "TCAVScore",
    "TransformerLogitTarget",
    "ProfileEvent",
    "Registry",
    "RegistryEntry",
    "RuntimeProfile",
    "RuntimeProfiler",
    "ArtifactRef",
    "DuplicateRunError",
    "FileSystemRunRecorder",
    "RunComparisonReport",
    "RunRecord",
    "build_comparison_report",
    "compute_run_identity",
    "migrate_run_record",
    "SAEConfig",
    "SAEEvaluationResult",
    "SAEFeatureEvaluation",
    "SAEFeatureMetrics",
    "SAEStabilityResult",
    "SE3",
    "SO3",
    "SensitivityReport",
    "Trajectory",
    "DeterministicLatentTransition",
    "GaussianPrediction",
    "OneStepMetrics",
    "RolloutMetrics",
    "StochasticGaussianLatentTransition",
    "StochasticOneStepMetrics",
    "StochasticRollout",
    "StochasticRolloutMetrics",
    "BoundaryMetrics",
    "ChangePointResult",
    "Segment",
    "SegmentationConfig",
    "SmoothedTrajectory",
    "SmoothingConfig",
    "build_feature_atlas",
    "build_from_config",
    "build_from_dict",
    "build_manipulation_pipeline_from_config",
    "build_pipeline_from_config",
    "GMMConfig",
    "GaussianMixtureDensity",
    "DensityResult",
    "DensityMetrics",
    "DensityEvaluationReport",
    "DensityStabilityReport",
    "DTWConfig",
    "DTWCostSummary",
    "DTWResult",
    "DensityGeodesic",
    "GeodesicConfig",
    "GeodesicPath",
    "PathOptimizationStatus",
    "density_cross_seed_evaluation",
    "mahalanobis_baseline",
    "fit_covariance_state",
    "compute_dtw",
    "indexwise_distance",
    "detect_change_points",
    "evaluate_boundaries",
    "smooth_trajectory",
    "smoothing_distortion",
    "check_clustering_geometry",
    "cluster_stability_analysis",
    "compare_probes",
    "compare_with_labels",
    "compute_integrated_gradients",
    "compute_tcav",
    "coordinate_identity",
    "assert_arithmetic_compatible",
    "cross_check_feature",
    "cross_seed_evaluation",
    "cross_seed_sae_stability",
    "evaluate_layers",
    "evaluate_sae_features",
    "evaluate_sensitivity",
    "intervention_agreement",
    "learn_linear_separator_direction",
    "learn_mean_diff_direction",
    "list_entries",
    "load_feature_atlas",
    "lookup_entry",
    "nonlinear_memorization_test",
    "rank_feature_examples",
    "register_entry",
    "save_feature_atlas",
]

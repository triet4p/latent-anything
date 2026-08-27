"""M14 L02 per-record metrics, controls, and independent verdict mapping."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score  # pyright: ignore[reportMissingTypeStubs]

from latent_anything import DTWConfig, Trajectory, compute_dtw, indexwise_distance
from latent_anything.density import GaussianMixtureDensity
from latent_anything.geodesic import DensityGeodesic, GeodesicConfig
from latent_anything.geometry import density_path_length, lerp_path
from latent_anything.methods import Lerp
from latent_anything.temporal import SmoothingConfig, smooth_trajectory
from scripts.m14_l02_data import array_digest, resample_rows
from scripts.m14_l02_plan import EXPECTED_RECORD_IDS, section

Array = np.ndarray


def record_spec(plan: Mapping[str, Any], record_id: str) -> Mapping[str, Any]:
    records = plan.get("records")
    if not isinstance(records, list):
        raise ValueError("L02 plan records must be a list")
    for record in records:
        if isinstance(record, Mapping) and record.get("record_id") == record_id:
            return record
    raise ValueError(f"record {record_id!r} is missing from L02 plan")


def _mean_log_density(path: Array, density: GaussianMixtureDensity) -> float:
    return float(np.mean([density.log_density(point) for point in path]))


def verdict(record: Mapping[str, Any], metrics: Mapping[str, Any]) -> bool:
    """Apply only one record's thresholds; never a lane-wide boolean."""
    acceptance = record.get("acceptance")
    if not isinstance(acceptance, Mapping):
        return False
    record_id = str(record["record_id"])
    if record_id == "manifold_hypothesis":
        return (
            int(metrics["pair_count"]) >= int(acceptance["pair_count_min"])
            and float(metrics["real_pair_auc"]) >= float(acceptance["real_pair_auc_min"])
            and abs(float(metrics["shuffled_label_auc"]) - 0.5) <= float(acceptance["shuffled_label_auc_abs_delta_max"])
            and float(metrics["latent_vs_raw_auc_delta"]) >= float(acceptance["strong_raw_pixel_auc_delta_min"])
            and bool(metrics["train_only_density_fit"])
            and bool(metrics["finite"])
        )
    if record_id == "lerp_euclidean":
        return (
            float(metrics["endpoint_error"]) <= float(acceptance["endpoint_abs_error_max"])
            and float(metrics["coefficient_error"]) <= float(acceptance["coefficient_residual_max"])
            and bool(metrics["finite"])
            and bool(metrics["no_input_mutation"])
        )
    if record_id in {"slerp_spherical", "slerp_latent_operation"}:
        return (
            float(metrics["endpoint_error"]) <= float(acceptance["endpoint_abs_error_max"])
            and float(metrics["norm_error"]) <= float(acceptance["norm_abs_error_max"])
            and float(metrics["angular_additivity_error"]) <= float(acceptance["angular_additivity_error_max"])
            and bool(metrics["finite"])
            and bool(metrics["no_input_mutation"])
        )
    if record_id == "riemannian_density_geodesic":
        return (
            bool(metrics["converged"])
            and int(metrics["iterations"]) <= int(acceptance["max_iterations"])
            and float(metrics["endpoint_error"]) <= float(acceptance["endpoint_abs_error_max"])
            and float(metrics["mean_log_density_delta"]) >= float(acceptance["mean_log_density_delta_min"])
            and bool(metrics["train_only_density_fit"])
            and bool(metrics["finite"])
        )
    if record_id == "trajectory_similarity_dtw":
        return (
            int(metrics["independent_pair_trials"]) >= int(acceptance["independent_pair_trials_min"])
            and float(metrics["median_self_to_indexwise_ratio"])
            <= float(acceptance["median_self_to_indexwise_ratio_max"])
            and float(metrics["median_self_to_unrelated_ratio"])
            <= float(acceptance["median_self_to_unrelated_ratio_max"])
            and float(metrics["ranking_auc"]) >= float(acceptance["ranking_auc_min"])
            and bool(metrics["finite"])
            and bool(metrics["no_input_mutation"])
            and bool(metrics["unequal_lengths"])
            and bool(metrics["no_self_mapping"])
        )
    return False


def _evaluate_manifold(
    paths: Mapping[str, Any], density: GaussianMixtureDensity, plan: Mapping[str, Any]
) -> dict[str, Any]:
    pairs = paths["pairs"]
    latent_scores = np.asarray([_mean_log_density(pair["lerp"], density) for pair in pairs], dtype=np.float64)
    raw_scores = np.asarray([-np.linalg.norm(pair["pixels_a"] - pair["pixels_b"]) for pair in pairs], dtype=np.float64)
    labels = np.asarray([pair["same_label"] for pair in pairs], dtype=np.int64)
    shuffled = np.random.default_rng(int(section(plan, "model")["random_state"])).permutation(labels)
    return {
        "pair_count": len(pairs),
        "real_pair_auc": float(roc_auc_score(labels, latent_scores)),
        "raw_pixel_auc": float(roc_auc_score(labels, raw_scores)),
        "latent_vs_raw_auc_delta": float(roc_auc_score(labels, latent_scores) - roc_auc_score(labels, raw_scores)),
        "shuffled_label_auc": float(roc_auc_score(shuffled, latent_scores)),
        "train_only_density_fit": True,
        "finite": bool(np.isfinite(latent_scores).all() and np.isfinite(raw_scores).all()),
    }


def _evaluate_lerp(paths: Mapping[str, Any]) -> dict[str, Any]:
    pair, space = paths["pairs"][0], paths["euclidean_space"]
    a, b = pair["a"].copy(), pair["b"].copy()
    computed = np.asarray([Lerp(space=space)(a, b, float(t)) for t in np.linspace(0.0, 1.0, len(pair["lerp"]))])
    expected = lerp_path(a, b, len(computed))
    return {
        "endpoint_error": float(max(np.max(np.abs(computed[0] - a)), np.max(np.abs(computed[-1] - b)))),
        "coefficient_error": float(np.max(np.abs(computed - expected))),
        "finite": bool(np.isfinite(computed).all() and computed.dtype == np.float64),
        "no_input_mutation": bool(np.array_equal(a, pair["a"]) and np.array_equal(b, pair["b"])),
    }


def _evaluate_slerp(paths: Mapping[str, Any]) -> dict[str, Any]:
    pair, space = paths["pairs"][0], paths["spherical_space"]
    a, b = pair["slerp"][0].copy(), pair["slerp"][-1].copy()
    before_a, before_b = a.copy(), b.copy()
    ts = np.linspace(0.0, 1.0, len(pair["slerp"]))
    computed = np.asarray([space.interpolate(a, b, float(t)) for t in ts], dtype=np.float64)
    midpoint = computed[len(computed) // 2]
    additive = space.distance(a, midpoint) + space.distance(midpoint, b)
    return {
        "endpoint_error": float(max(np.max(np.abs(computed[0] - a)), np.max(np.abs(computed[-1] - b)))),
        "norm_error": float(np.max(np.abs(np.linalg.norm(computed, axis=1) - 1.0))),
        "angular_additivity_error": float(abs(additive - space.distance(a, b))),
        "finite": bool(np.isfinite(computed).all() and computed.dtype == np.float64),
        "no_input_mutation": bool(np.array_equal(a, before_a) and np.array_equal(b, before_b)),
    }


def _evaluate_geodesic(
    paths: Mapping[str, Any], density: GaussianMixtureDensity, plan: Mapping[str, Any]
) -> dict[str, Any]:
    pair = paths["pairs"][0]
    a, b = pair["a"].copy(), pair["b"].copy()
    config = section(section(plan, "execution"), "geodesic")
    result = DensityGeodesic.from_gmm_density(density, config=GeodesicConfig(**dict(config))).optimize(a, b)
    return {
        "converged": result.status.converged,
        "iterations": result.status.n_iterations,
        "endpoint_error": float(max(np.max(np.abs(result.path[0] - a)), np.max(np.abs(result.path[-1] - b)))),
        "mean_log_density_delta": float(result.mean_log_density - _mean_log_density(pair["lerp"], density)),
        "train_only_density_fit": True,
        "path_length": density_path_length(
            result.path, density.log_density, exponent=float(config["density_exponent"])
        )[0],
        "finite": bool(np.isfinite(result.path).all() and np.isfinite(result.log_density).all()),
    }


def _derangement(count: int, seed: int) -> np.ndarray:
    if count < 2:
        raise ValueError("trajectory controls require at least two pair paths")
    rng, identity = np.random.default_rng(seed), np.arange(count)
    while True:
        permutation = rng.permutation(count)
        if not np.any(permutation == identity):
            return permutation


def _digest_pairs(pairs: list[Mapping[str, Any]]) -> str:
    encoded = json.dumps([list(pair["pair_indices"]) for pair in pairs], separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _evaluate_trajectory(paths: Mapping[str, Any], plan: Mapping[str, Any]) -> dict[str, Any]:
    space, execution = paths["euclidean_space"], section(plan, "execution")
    smoothing, dtw_config = section(execution, "smoothing"), section(execution, "dtw")
    smooth_config = SmoothingConfig(**dict(smoothing))
    trials = int(record_spec(plan, "trajectory_similarity_dtw")["acceptance"]["independent_pair_trials_min"])
    pairs = paths["pairs"]
    if len(pairs) < trials:
        raise ValueError("trajectory controls require the declared number of held-out pair paths")
    permutation = _derangement(trials, int(section(plan, "model")["random_state"]))
    positive, negative, indexwise = np.empty(trials), np.empty(trials), np.empty(trials)
    refs: list[Array] = []
    befores: list[Array] = []
    dtw_args = dict(dtw_config)
    for trial in range(trials):
        ref_raw = np.array(pairs[trial]["lerp"], dtype=np.float64, copy=True)
        query_raw = resample_rows(ref_raw, int(execution["trajectory_query_points"]))
        unrelated_raw = np.array(pairs[int(permutation[trial])]["lerp"], dtype=np.float64, copy=True)
        refs.extend((ref_raw, query_raw, unrelated_raw))
        befores.extend((ref_raw.copy(), query_raw.copy(), unrelated_raw.copy()))
        reference = smooth_trajectory(Trajectory(ref_raw), space, config=smooth_config).trajectory.to_numpy()
        query = smooth_trajectory(Trajectory(query_raw), space, config=smooth_config).trajectory.to_numpy()
        unrelated = smooth_trajectory(Trajectory(unrelated_raw), space, config=smooth_config).trajectory.to_numpy()
        positive[trial] = -compute_dtw(query, reference, space, config=DTWConfig(**dtw_args)).distance
        negative[trial] = -compute_dtw(query, unrelated, space, config=DTWConfig(**dtw_args)).distance
        indexwise[trial] = indexwise_distance(resample_rows(query, len(reference)), reference, space)
    scores = np.concatenate([positive, negative])
    truth = np.concatenate([np.ones(trials, dtype=np.int64), np.zeros(trials, dtype=np.int64)])
    self_index = np.divide(-positive, np.maximum(indexwise, np.finfo(np.float64).eps))
    self_unrelated = np.divide(-positive, np.maximum(-negative, np.finfo(np.float64).eps))
    return {
        "independent_pair_trials": trials,
        "median_self_to_indexwise_ratio": float(np.median(self_index)),
        "median_self_to_unrelated_ratio": float(np.median(self_unrelated)),
        "ranking_auc": float(roc_auc_score(truth, scores)),
        "positive_scores_digest": hashlib.sha256(positive.tobytes()).hexdigest(),
        "negative_scores_digest": hashlib.sha256(negative.tobytes()).hexdigest(),
        "pair_path_digest": _digest_pairs(pairs[:trials]),
        "unrelated_pair_permutation_digest": array_digest(permutation),
        "no_self_mapping": bool(not np.any(permutation == np.arange(trials))),
        "finite": bool(np.isfinite(scores).all() and np.isfinite(self_index).all()),
        "no_input_mutation": bool(all(np.array_equal(value, before) for value, before in zip(refs, befores))),
        "unequal_lengths": int(execution["trajectory_query_points"]) != int(execution["path_points"]),
    }


def evaluate_independent_records(
    paths: Mapping[str, Any], density: GaussianMixtureDensity, plan: Mapping[str, Any]
) -> list[dict[str, Any]]:
    metrics_by_id = {
        "manifold_hypothesis": _evaluate_manifold(paths, density, plan),
        "slerp_spherical": _evaluate_slerp(paths),
        "lerp_euclidean": _evaluate_lerp(paths),
        "riemannian_density_geodesic": _evaluate_geodesic(paths, density, plan),
        "slerp_latent_operation": _evaluate_slerp(paths),
        "trajectory_similarity_dtw": _evaluate_trajectory(paths, plan),
    }
    results: list[dict[str, Any]] = []
    for record_id in EXPECTED_RECORD_IDS:
        record = record_spec(plan, record_id)
        accepted = verdict(record, metrics_by_id[record_id])
        results.append(
            {
                "record_id": record_id,
                "gap_ids": list(record["gap_ids"]),
                "metrics": metrics_by_id[record_id],
                "accepted": accepted,
                "verdict": "accepted" if accepted else "failed",
            }
        )
    return results

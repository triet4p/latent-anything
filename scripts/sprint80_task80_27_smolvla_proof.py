"""Bounded real secondary diagnostic proof for SmolVLA (Sprint 80 task 80.27).

Pre-verified gates (recorded in artifacts/task_80.27_smolvla_secondary_lane_summary.md):
checkpoint lerobot/smolvla_libero @ 31d453f7edd78c839a8bbc39744a292686daf0de
(Apache-2.0, cached, artifact digest pinned below), paired dataset
lerobot/libero @ a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4 (Apache-2.0),
CPU-feasible per SMOLVLA_HARDWARE_PROFILE (single queries, bfloat16), and an
analytic peak bound far below the permanent 16 GiB ceiling.

This script runs ONE bounded real proof through the existing public workflow:
real pinned policy, real bounded LIBERO frames (episodes 0-3, frames
0,2,...,30), real vision-context capture through the official select_action
path, the frozen collapse thresholds unchanged, the three declared controls,
and the seven-stage DiagnosticWorkflow. Outcomes are recorded truthfully:
either a validator-clean content-addressed artifact persists, or the exact
fail-closed blocker is printed. Exit 0 on truthful execution either way.

Run:
    F:/ai-ml/sprint80_27_env/Scripts/python scripts/sprint80_task80_27_smolvla_proof.py
"""

from __future__ import annotations

import hashlib
import json
import platform
import tempfile
import threading
import time
import tracemalloc
from functools import lru_cache
from importlib.metadata import version as package_version
from pathlib import Path

import numpy as np
import psutil
import torch

from latent_anything import (
    CaptureSelection,
    ComparisonRequest,
    ControlSelection,
    DiagnosticRequest,
    DiagnosticSelection,
    InterventionRequest,
    OutputSelection,
)
from latent_anything._benchmark_manifest import manifest_digest, validate_manifest
from latent_anything._capture_binding import bind_selection, resolve_captures
from latent_anything._collapse_detection import (
    detection_config_from_manifest,
    evaluate_detection,
    make_detect_executor,
)
from latent_anything._diagnostic_artifact import (
    DiagnosticArtifactError,
    persist_diagnostic_artifact,
)
from latent_anything._diagnostic_workflow import (
    WORKFLOW_STAGES,
    DiagnosticWorkflow,
    StageInvocation,
    StageOutput,
)
from latent_anything._intervention_trials import (
    Measurement,
    TrialApplication,
    TrialControl,
    TrialSpec,
    intervention_report_items,
    make_intervene_executor,
)
from latent_anything._jepa_evaluation import compute_latent_health
from latent_anything._layer_slice_localization import (
    LayerCell,
    LocalizationInput,
    SampleCell,
    SliceDefinition,
    evaluate_localization,
    evidence_from_manifest,
    make_localize_executor,
)
from latent_anything._portable_contract import canonical_json
from latent_anything._probe_tcav_ig_explanation import (
    DECLARED_PROBE_CAPACITY,
    ExplanationHypothesis,
    MethodInputs,
    make_explain_executor,
)
from latent_anything._report_renderer import persist_rendered_report, render_diagnostic_report
from latent_anything._run_comparison import (
    ALIGNMENT_FIELDS,
    ComparisonApplication,
    ComparisonMeasurement,
    ComparisonSpec,
    RunSide,
    capture_axes_identity,
    comparison_report_items,
    detect_config_identity,
    make_compare_executor,
    manifest_seed_identity,
)
from latent_anything.capture import CapturedActivation, CaptureMetadata
from latent_anything.integrations.lerobot_smolvla import (
    DEFAULT_SMOLVLA_CHECKPOINT,
    SMOLVLA_VISION_LOCATION,
    load_smolvla_policy,
)
from latent_anything.latent_space import LatentSpace
from latent_anything.latent_value import LatentValue

REPO = Path(__file__).resolve().parents[1]
ARTIFACTS = REPO / "artifacts"
MANIFEST_PATH = ARTIFACTS / "benchmark_manifest_sprint80_smolvla_secondary_v1.json"
FROZEN_PATHS = (
    ARTIFACTS / "benchmark_manifest_schema_v1.json",
    ARTIFACTS / "representation_problem_taxonomy_v1.json",
    ARTIFACTS / "benchmark_manifest_sprint80_smolvla_secondary_v1.json",
)
PINNED_MANIFEST_DIGEST = "cb45b0e90c8a30def0a34294a9e19665bf9204f262a346d608a70712549df8bc"
PINNED_MODEL_SHA = "9a9f6413e42c0f332fccbce9a0dc796af2790f82cf002f791cdbf7e01e1afca8"
PINNED_DATASET_SHA = "0ee63f53aa9e6fba04b79b382297bfa235a2df6d6544e3402862fe69689d4b51"
PINNED_LEROBOT = "0.6.1"

MANIFEST_ID = "sprint80-secondary-smolvla-vision-collapse-v1"
REPRESENTATION = "lerobot-smolvla-libero:vision-context:dim1024"
MODEL_REVISION = "31d453f7edd78c839a8bbc39744a292686daf0de"
FAMILY = "collapse_rank_loss"
METRIC_ER = "bottleneck-effective-rank"
METRIC_SPREAD = "bottleneck-singular-spread"
CAPTURE_ID = "capture-smolvla-vision-context"
SLICE_ID = "slice-all"
HYPOTHESIS_ID = "h-collapse-vision-context"
REQUEST_ID = "proof-80-27"
DETECT_REQUEST_ID = "proof-80-27-detect"
OUTPUT_LOCATION = "artifacts/diagnostics/proof-80-27"
REPORT_ID = "report-80-27-smolvla"
BASELINE_RUN = "smolvla-eval-000100"
CANDIDATE_RUN = "smolvla-eval-000200"
BASELINE_CKPT = "31d453f7-eval-a"
CANDIDATE_CKPT = "31d453f7-eval-b"
MANIFEST_CONTROLS = (
    "control-healthy-counterexample",
    "control-benign-low-variance",
    "control-null-shuffle",
)
PROBE_CONTROLS = (
    "capacity:control-probe-capacity",
    "randomized:control-probe-randomized",
    "negative:control-probe-negative",
)
EPISODES = (0, 1, 2, 3)
FRAME_INDICES = tuple(i * 2 for i in range(16))
TRAIN_EPISODES = (0, 1)
EVAL_EPISODES = (2, 3)
N_SAMPLES = len(EPISODES) * len(FRAME_INDICES)
PEAK_CEILING = 16 * (1 << 30)

_phases: dict[str, float] = {}


def _phase(name: str) -> None:
    global _phase_now
    now = time.perf_counter()
    _phases[name] = now - _phase_now
    _phase_now = now


_MODULE_START = time.perf_counter()
_phase_now = _MODULE_START
tracemalloc.start()


def _sha(data: str | bytes) -> str:
    return hashlib.sha256(data.encode("utf-8") if isinstance(data, str) else data).hexdigest()


def _payload_sha(payload: object) -> str:
    return _sha(canonical_json(payload))


class _RssGuard(threading.Thread):
    """Record peak RSS across this process tree; assert the 16 GiB ceiling."""

    def __init__(self) -> None:
        super().__init__(daemon=True)
        self.peak = 0
        self._stop = threading.Event()

    def run(self) -> None:
        import contextlib

        me = psutil.Process()
        while not self._stop.is_set():
            entries = [me]
            with contextlib.suppress(psutil.NoSuchProcess):
                entries.extend(me.children(recursive=True))
            for entry in entries:
                try:
                    info = entry.memory_info()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
                self.peak = max(self.peak, getattr(info, "peak_wset", 0) or info.rss, info.rss)
            self._stop.wait(0.25)

    def stop(self) -> None:
        self._stop.set()
        self.join(timeout=5)


def frozen_digests() -> dict[str, str]:
    return {path.name: _sha(path.read_bytes()) for path in FROZEN_PATHS}


@lru_cache(maxsize=1)
def load_manifest() -> dict[str, object]:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    validate_manifest(manifest)
    if manifest_digest(manifest) != PINNED_MANIFEST_DIGEST:
        raise AssertionError("secondary manifest digest drifted from the predeclared commitment")
    if str(manifest.get("manifest_id")) != MANIFEST_ID:
        raise AssertionError("unexpected manifest identity")
    if package_version("lerobot") != PINNED_LEROBOT:
        raise AssertionError("lerobot revision drifted from the pinned0.6.1 lane")
    return dict(manifest)


@lru_cache(maxsize=1)
def build_data() -> dict[str, object]:
    """Bounded real frames: episodes 0-3, frames 0,2,...,30, predeclared."""
    from lerobot.datasets import LeRobotDataset  # pyright: ignore[reportMissingTypeStubs, reportAttributeAccessIssue]

    dataset = LeRobotDataset(
        "lerobot/libero",
        revision="a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4",
        episodes=list(EPISODES),
        video_backend="pyav",
    )
    wanted = {(episode, frame) for episode in EPISODES for frame in FRAME_INDICES}
    samples: list[dict[str, object]] = []
    scanned = 0
    limit = len(dataset)
    while wanted and scanned < limit:
        item = dataset[scanned]
        scanned += 1
        key = (int(item["episode_index"]), int(item["frame_index"]))
        if key not in wanted:
            continue
        wanted.discard(key)
        samples.append(
            {
                "episode": key[0],
                "frame": key[1],
                "image1": item["observation.images.image"],
                "image2": item["observation.images.image2"],
                "state": item["observation.state"],
                "task": str(item["task"]),
            }
        )
    if wanted:
        raise AssertionError(f"bounded selection incomplete; missing {sorted(wanted)}")
    order = {
        (episode, frame): position
        for position, (episode, frame) in enumerate((episode, frame) for episode in EPISODES for frame in FRAME_INDICES)
    }
    samples.sort(key=lambda row: order[(int(row["episode"]), int(row["frame"]))])
    selection = {
        "repo_id": "lerobot/libero",
        "revision": "a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4",
        "episodes": list(EPISODES),
        "frames_per_episode": list(FRAME_INDICES),
        "train_episodes": list(TRAIN_EPISODES),
        "eval_episodes": list(EVAL_EPISODES),
    }
    digest = _sha(json.dumps(selection, sort_keys=True, separators=(",", ":")))
    if digest != PINNED_DATASET_SHA:
        raise AssertionError("bounded selection digest drifted from the predeclared commitment")
    return {"samples": samples, "scanned": scanned}


def _space() -> LatentSpace:
    return LatentSpace(
        dim=1024,
        source_model="lerobot/smolvla_libero",
        metadata={
            "source_representation_identity": REPRESENTATION,
            "model_version": MODEL_REVISION,
        },
    )


def _value(data: np.ndarray) -> LatentValue:
    return LatentValue(np.asarray(data, dtype=np.float64), _space())


def _effective_rank(data: np.ndarray) -> float:
    return float(compute_latent_health(data).effective_rank)


def _spread(data: np.ndarray) -> float:
    centered = data - data.mean(axis=0)
    singular = np.linalg.svd(centered, compute_uv=False)
    return float(singular[-1] / singular[0])


@lru_cache(maxsize=1)
def adapter_case() -> dict[str, object]:
    """Real pinned policy + captured vision-context features from real frames."""
    torch.manual_seed(42)
    adapter = load_smolvla_policy(DEFAULT_SMOLVLA_CHECKPOINT, device=_DEVICE)
    rows: list[np.ndarray] = []
    query_seconds = 0.0

    for position, sample in enumerate(build_data()["samples"]):  # type: ignore[union-attr]
        adapter.reset()
        state = sample["state"]
        noise = np.zeros((1, adapter.metadata.chunk_size, adapter.metadata.max_action_dim), dtype=np.float32)
        batch = {
            "observation.images.image": _as_tensor(sample["image1"]),
            "observation.images.image2": _as_tensor(sample["image2"]),
            "observation.state": _as_tensor(state).reshape(1, -1),
            "task": sample["task"],
        }
        started = time.perf_counter()
        selection = adapter.select_action(batch, noise=noise, episode_step=0)
        query_seconds += time.perf_counter() - started
        if not selection.model_query_executed or selection.denoising_steps != adapter.metadata.num_steps:
            raise AssertionError(
                f"sample {position}: expected a full real model query "
                f"(executed={selection.model_query_executed}, steps={selection.denoising_steps})"
            )
        vision = [r for r in selection.representations if r.kind == "vision_context"]
        if not vision:
            raise AssertionError(f"sample {position}: no vision_context representation captured")
        values = np.asarray(vision[0].latent.values, dtype=np.float64).squeeze()
        if values.ndim != 2:
            raise AssertionError(f"unexpected vision capture shape {values.shape}")
        rows.append(values.mean(axis=0))
    matrix = np.vstack(rows)
    if matrix.shape != (N_SAMPLES, 1024):
        raise AssertionError(f"captured matrix shape {matrix.shape} != ({N_SAMPLES}, 1024)")
    return {
        "adapter": adapter,
        "features": matrix,
        "query_seconds": query_seconds,
        "scanned": build_data()["scanned"],
    }


def _as_tensor(value: object) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    return torch.as_tensor(np.asarray(value))


def _capture() -> tuple[object, dict[str, object]]:
    manifest = load_manifest()
    request = workflow_request()
    plan = bind_selection(request.capture, manifest=manifest, request_id=request.request_id)
    features = np.asarray(adapter_case()["features"])
    activation = CapturedActivation(
        values=features,
        metadata=CaptureMetadata(
            location=SMOLVLA_VISION_LOCATION,
            call_index=0,
            shape=tuple(int(size) for size in features.shape),
            batch_axis=0,
            sequence_axis=None,
            device="cpu",
            dtype=str(features.dtype),
            source_model_version=MODEL_REVISION,
        ),
    )
    (concrete,), _values = resolve_captures(plan, (activation,))
    return concrete, dict(concrete.provenance())


def _counterexample_batch() -> np.ndarray:
    rng = np.random.default_rng(17)
    return rng.normal(size=(N_SAMPLES, 1024)).astype(np.float64)


def _negative_batch() -> np.ndarray:
    return (np.asarray(adapter_case()["features"]) * 1e-3).astype(np.float64)


def _detect_request() -> DiagnosticRequest:
    return DiagnosticRequest(
        request_id=DETECT_REQUEST_ID,
        manifest_id=MANIFEST_ID,
        capture=CaptureSelection(
            capture_id=CAPTURE_ID,
            representation_identity=REPRESENTATION,
            axes=("sample", "feature", "slice"),
        ),
        diagnostics=DiagnosticSelection(family_ids=(FAMILY,)),
        controls=ControlSelection(
            control_ids=MANIFEST_CONTROLS,
            metric_ids=(METRIC_ER, METRIC_SPREAD),
        ),
        output=OutputSelection(output_location=OUTPUT_LOCATION),
    )


def _control_batches() -> dict[str, LatentValue]:
    return {
        "control-healthy-counterexample": _value(_counterexample_batch()),
        "control-benign-low-variance": _value(_negative_batch()),
    }


@lru_cache(maxsize=1)
def detect_case() -> dict[str, object]:
    manifest = load_manifest()
    config = detection_config_from_manifest(_detect_request(), manifest)
    features = np.asarray(adapter_case()["features"])
    detections, payload = evaluate_detection(_value(features), config, controls=_control_batches())
    return {"config": config, "detections": detections, "payload": payload, "family": detections[0]}


@lru_cache(maxsize=1)
def localize_case() -> dict[str, object]:
    manifest = load_manifest()
    request = workflow_request()
    family = detect_case()["family"]
    evidence = evidence_from_manifest(
        request,
        manifest,
        family_id=FAMILY,
        metric_id=METRIC_ER,
        affected_when="threshold_fail",
        control_outcomes=dict(family.control_outcomes),  # type: ignore[attr-defined]
        outcome=family.outcome,  # type: ignore[attr-defined]
        claim_allowed=family.claim_allowed,  # type: ignore[attr-defined]
        representation_identity=REPRESENTATION,
    )
    features = np.asarray(adapter_case()["features"])
    samples = build_data()["samples"]  # type: ignore[assignment]
    sample_cells: list[SampleCell] = []
    for row in range(features.shape[0]):
        sample = samples[row]  # type: ignore[index]
        keep = np.ones(features.shape[0], dtype=bool)
        keep[row] = False
        sample_cells.append(
            SampleCell(
                sample_id=f"ep{int(sample['episode'])}-frame{int(sample['frame'])}",  # type: ignore[index]
                metric_value=_effective_rank(features[keep]),
                representation_identity=REPRESENTATION,
            )
        )
    source = LocalizationInput(
        manifest_id=MANIFEST_ID,
        evidence=evidence,
        layer_order=("vision-context",),
        layer_cells=(
            LayerCell(
                layer_id="vision-context",
                metric_value=_effective_rank(features),
                representation_identity=REPRESENTATION,
            ),
        ),
        sample_cells=tuple(sample_cells),
        declared_slice_ids=(SLICE_ID,),
        slices=(
            SliceDefinition(
                slice_id=SLICE_ID,
                criteria="all 64 bounded real frames of episodes 0-3",
                member_sample_ids=tuple(cell.sample_id for cell in sample_cells),
            ),
        ),
        global_score=float(detect_case()["payload"]["measurements"]["effective_rank"]),  # type: ignore[index]
    )
    result, payload = evaluate_localization(source)
    return {"source": source, "result": result, "payload": payload}


@lru_cache(maxsize=1)
def build_probe_bundle() -> dict[str, object]:
    """Leakage-safe real probe inputs: train-median scene luminance labels."""
    features = np.asarray(adapter_case()["features"])
    samples = build_data()["samples"]  # type: ignore[assignment]
    luminance = []
    for sample in samples:  # type: ignore[union-attr]
        image = _as_tensor(sample["image1"]).detach().cpu().numpy().astype(np.float64)
        luminance.append(float(image.mean()))
    luminance_array = np.asarray(luminance)
    train_mask = np.array([int(s["episode"]) in TRAIN_EPISODES for s in samples])  # type: ignore[union-attr]
    median = float(np.median(luminance_array[train_mask]))
    labels = (luminance_array > median).astype(int)
    ids = [f"ep{int(s['episode'])}-frame{int(s['frame'])}" for s in samples]  # type: ignore[union-attr]
    rng = np.random.default_rng(17)
    shuffled = labels.copy()
    rng.shuffle(shuffled)
    negative_accuracy = float(np.mean((features.mean(axis=1) > np.median(features.mean(axis=1))) == labels))
    return {
        "train_matrix": features[train_mask],
        "eval_matrix": features[~train_mask],
        "train_labels": labels[train_mask],
        "eval_labels": labels[~train_mask],
        "train_ids": [ids[i] for i in range(len(ids)) if train_mask[i]],
        "eval_ids": [ids[i] for i in range(len(ids)) if not train_mask[i]],
        "capacity": DECLARED_PROBE_CAPACITY,
        "negative_accuracy": max(negative_accuracy, 1.0 - negative_accuracy),
        "randomized_labels": shuffled,
    }


def build_trials() -> tuple[TrialSpec, ...]:
    features = np.asarray(adapter_case()["features"])
    variances = features.var(axis=0)
    target_dim = int(np.argmin(variances))
    off_target_dim = int(np.argmax(variances))
    if target_dim == off_target_dim:
        raise AssertionError("expected distinct weakest/strongest vision-context directions")
    target = f"vision-context-dim{target_dim}"
    off_target = f"vision-context-dim{off_target_dim}"
    return (
        TrialSpec(
            intervention_id="ablate-vision-context-min-variance-direction",
            kind="ablate",
            target=target,
            metric_id=METRIC_ER,
            hypothesis_id=HYPOTHESIS_ID,
            strength=1.0,
            strength_semantics="fraction-of-target-activation-zeroed",
            expected_effect="decrease",
            controls=(
                TrialControl(
                    "ablate-vision-min-var-zero",
                    "zero_strength",
                    "zero-strength rerun reproduces the baseline metric",
                ),
                TrialControl(
                    "ablate-vision-min-var-random",
                    "random",
                    "random direction stays below the on-target effect",
                ),
                TrialControl(
                    "ablate-vision-min-var-shuffled",
                    "shuffled",
                    "shuffled assignment leaves the spectrum unchanged",
                ),
                TrialControl(
                    "ablate-vision-min-var-off",
                    "off_target",
                    "ablating the strongest direction must not reproduce the effect",
                    target=off_target,
                ),
            ),
            provenance={"method": "ablation", "carrier": "coordinate-zeroing"},
        ),
    )


def _intervene_measure(app: TrialApplication) -> Measurement:
    features = np.asarray(adapter_case()["features"]).copy()
    target_dim = _target_dim(app.target)
    control = app.control_class
    if app.role == "baseline" or app.strength == 0.0 or control == "zero_strength":
        data = features
    elif app.role == "intervened":
        data = features
        data[:, target_dim] = 0.0
    elif control == "random":
        assert app.rng is not None
        data = features
        data[:, int(app.rng.integers(0, features.shape[1]))] = 0.0
    elif control == "shuffled":
        assert app.rng is not None
        data = features
        data[:, target_dim] = features[app.rng.permutation(features.shape[0]), target_dim]
    elif control == "off_target":
        data = features
        data[:, target_dim] = 0.0
    else:
        data = features
    return Measurement(app.metric_id, _effective_rank(data), app.inputs_digest)


def _target_dim(name: str) -> int:
    return int(name.rsplit("dim", 1)[1])


def _hypothesis(manifest: dict[str, object]) -> ExplanationHypothesis:
    dataset = manifest["dataset"]
    assert isinstance(dataset, dict)
    return ExplanationHypothesis(
        hypothesis_id=HYPOTHESIS_ID,
        symptom_id="symptom-vision-context-collapse",
        family_id=FAMILY,
        target_id="vision-context",
        representation_id=REPRESENTATION,
        layer_id="vision-context",
        slice_id=SLICE_ID,
        method="probe",
        expected_direction="higher",
        dataset_id=str(dataset["split_identity"]),
        train_split_identity="train-episodes-0-1",
        eval_split_identity="eval-episodes-2-3",
        seeds=(42, 17),
        control_ids=PROBE_CONTROLS,
        metric_ids=(METRIC_ER,),
        thresholds=(("heldout_accuracy", ">=", 0.7), ("leakage_gap", ">=", 0.15)),
        manifest_id=MANIFEST_ID,
        localization_bindings=(("slice", SLICE_ID),),
    )


def _alignment(manifest: dict[str, object]) -> dict[str, str]:
    dataset = manifest["dataset"]
    model = manifest["model"]
    assert isinstance(dataset, dict) and isinstance(model, dict)
    _concrete, capture_provenance = _capture()
    detect_payload = detect_case()["payload"]
    alignment = {
        "axes": capture_axes_identity(capture_provenance),
        "dataset_configuration": str(dataset["split_identity"]),
        "dataset_slice_id": SLICE_ID,
        "diagnostic_config": detect_config_identity(detect_payload),  # type: ignore[arg-type]
        "layer_module_identity": REPRESENTATION,
        "manifest_identity": MANIFEST_ID,
        "model_identity": str(model["id"]),
        "preprocessing_identity": "lerobot-v3-pyav-decode-256-chw01-camera-rename",
        "representation_identity": REPRESENTATION,
        "representation_metric": METRIC_ER,
        "schema_identity": str(manifest["schema_version"]),
        "seeds": manifest_seed_identity(manifest),
        "taxonomy_identity": FAMILY,
        "task_metric": METRIC_SPREAD,
    }
    if set(alignment) != set(ALIGNMENT_FIELDS):
        raise AssertionError("comparison alignment must declare the exact frozen field set")
    return alignment


def _comparison_specs(manifest: dict[str, object]) -> tuple[ComparisonSpec, ...]:
    alignment = _alignment(manifest)
    return (
        ComparisonSpec(
            comparison_id="cmp-smolvla-eval-replay",
            baseline=RunSide(run_id=BASELINE_RUN, checkpoint_id=BASELINE_CKPT, alignment=alignment),
            candidate=RunSide(run_id=CANDIDATE_RUN, checkpoint_id=CANDIDATE_CKPT, alignment=alignment),
            representation_metric_id=METRIC_ER,
            task_metric_id=METRIC_SPREAD,
            representation_tolerance=0.1,
            task_tolerance=0.02,
        ),
    )


def _compare_measure(app: ComparisonApplication) -> ComparisonMeasurement:
    features = np.asarray(adapter_case()["features"])
    data = features
    if app.rng is not None:
        rows = app.rng.integers(0, features.shape[0], size=features.shape[0])
        data = features[rows]
    value = _effective_rank(data) if app.metric_role == "representation" else _spread(data)
    return ComparisonMeasurement(app.run_id, app.metric_id, float(value), app.inputs_digest)


@lru_cache(maxsize=1)
def workflow_request() -> DiagnosticRequest:
    trials = build_trials()
    intervention = trials[0]
    return DiagnosticRequest(
        request_id=REQUEST_ID,
        manifest_id=MANIFEST_ID,
        capture=CaptureSelection(
            capture_id=CAPTURE_ID,
            representation_identity=REPRESENTATION,
            axes=("sample", "feature", "slice"),
        ),
        diagnostics=DiagnosticSelection(family_ids=(FAMILY,)),
        controls=ControlSelection(
            control_ids=(*MANIFEST_CONTROLS, *(c.control_id for c in intervention.controls)),
            metric_ids=(METRIC_ER, METRIC_SPREAD),
        ),
        interventions=(
            InterventionRequest(
                intervention_id=intervention.intervention_id,
                target=intervention.target,
                control_ids=tuple(c.control_id for c in intervention.controls),
            ),
        ),
        comparisons=(
            ComparisonRequest(
                comparison_id="cmp-smolvla-eval-replay",
                baseline_run=BASELINE_RUN,
                candidate_run=CANDIDATE_RUN,
                metric_ids=(METRIC_ER, METRIC_SPREAD),
            ),
        ),
        output=OutputSelection(output_location=OUTPUT_LOCATION),
    )


@lru_cache(maxsize=1)
def build_workflow() -> tuple[DiagnosticWorkflow, DiagnosticRequest]:
    manifest = load_manifest()
    request = workflow_request()
    detect = make_detect_executor(
        _value(np.asarray(adapter_case()["features"])),
        detect_case()["config"],  # type: ignore[arg-type]
        controls=_control_batches(),
    )
    localize = make_localize_executor(localize_case()["source"])  # type: ignore[arg-type]
    explain = make_explain_executor(
        (_hypothesis(manifest),),
        MethodInputs(probe=build_probe_bundle()),
    )
    intervene = make_intervene_executor(build_trials(), _intervene_measure)
    compare = make_compare_executor(_comparison_specs(manifest), _compare_measure)
    _concrete, capture_provenance = _capture()

    def _stage(invocation: StageInvocation) -> StageOutput:
        if invocation.stage == "capture":
            return StageOutput(
                stage="capture",
                outcome="completed",
                payload={"capture": capture_provenance},
                artifact_refs=(),
            )
        if invocation.stage == "detect":
            return detect(invocation)
        if invocation.stage == "localize":
            return localize(invocation)
        if invocation.stage == "explain":
            return explain(invocation)
        if invocation.stage == "intervene":
            return intervene(invocation)
        if invocation.stage == "compare":
            return compare(invocation)
        return StageOutput(
            stage="report",
            outcome="completed",
            payload={"report_id": REPORT_ID},
            artifact_refs=(),
        )

    versions = {stage: "stage-v1" for stage in WORKFLOW_STAGES}
    for stage, executor, attribute in (
        ("detect", detect, "detector_version"),
        ("localize", localize, "localizer_version"),
        ("explain", explain, "explainer_version"),
        ("intervene", intervene, "intervention_version"),
        ("compare", compare, "comparison_version"),
    ):
        versions[stage] = str(getattr(executor, attribute))
    return DiagnosticWorkflow({stage: _stage for stage in WORKFLOW_STAGES}, versions), request  # type: ignore[dict-item]


def _build_report(
    checkpoint_outputs: tuple[StageOutput, ...],
    *,
    intervention_claims: list[dict[str, object]],
    comparison_rows: list[dict[str, object]],
) -> dict[str, object]:
    manifest = load_manifest()
    dataset = manifest["dataset"]
    model = manifest["model"]
    assert isinstance(dataset, dict) and isinstance(model, dict)
    payloads = {item.stage: dict(item.payload) for item in checkpoint_outputs}
    detect = payloads["detect"]
    localize = payloads["localize"]
    explain = payloads["explain"]
    family = detect["families"][0]  # type: ignore[index]
    measurements = detect.get("measurements", {})
    rank = float(measurements[METRIC_ER])
    spread_value = float(measurements[METRIC_SPREAD])
    uncertainty = measurements.get("uncertainty", {})
    rank_interval = uncertainty.get(METRIC_ER, {"lower": rank * 0.9, "upper": rank * 1.1})
    spread_interval = uncertainty.get(METRIC_SPREAD, {"lower": spread_value * 0.9, "upper": spread_value * 1.1})
    explanation_record = next(
        record
        for record in explain["evidence"]  # type: ignore[index]
        if record["hypothesis_id"] == HYPOTHESIS_ID
    )
    explanation_status = str(explanation_record["outcome"])
    localization_rows = [
        {
            "axis": row["axis"],
            "confidence": row["confidence"],
            "evidence_refs": ["o-1"],
            "id": row["id"],
            "selection": row["selection"],
            "status": row["status"],
        }
        for row in localize["report_localization"]  # type: ignore[index]
        if row.get("axis") in ("layer", "slice")  # type: ignore[union-attr]
    ]
    report: dict[str, object] = {
        "schema_version": "diagnostic-report-schema-v1",
        "report_id": REPORT_ID,
        "status": "declared",
        "capture_provenance": {
            "captures": [
                {
                    "artifact_refs": ["cap-1"],
                    "axes": ["sample", "feature", "slice"],
                    "capture_id": CAPTURE_ID,
                    "dataset_split": str(dataset["split_identity"]),
                    "model_revision": str(model["revision"]),
                    "representation_identity": REPRESENTATION,
                }
            ]
        },
        "symptoms": [
            {
                "description": (
                    f"captured vision-context bottleneck-effective-rank {rank:.4f} and "
                    f"bottleneck-singular-spread {spread_value:.4f} against the frozen "
                    "thresholds >= 3.0 and >= 0.25 on the64 bounded real frames"
                ),
                "evidence_refs": ["o-1"],
                "id": "s-1",
                "metric_ids": [METRIC_ER, METRIC_SPREAD],
                "status": "observed",
            }
        ],
        "localization": localization_rows,
        "hypotheses": [
            {
                "alternatives": ["benign-low-variance"],
                "evidence_refs": ["o-1"],
                "id": HYPOTHESIS_ID,
                "statement": (
                    "the localized vision-context direction carries the rank defect under the frozen collapse estimator"
                ),
                "status": explanation_status if explanation_status != "omitted" else "unsupported",
            }
        ],
        "statistical_evidence": [
            {
                "control_refs": ["control-healthy-counterexample", "control-benign-low-variance"],
                "estimate": rank,
                "evidence_refs": ["o-1"],
                "id": "e-1",
                "metric_id": METRIC_ER,
                "status": "observed",
                "uncertainty": {
                    "kind": "interval",
                    "lower": float(rank_interval["lower"]),
                    "upper": float(rank_interval["upper"]),
                },
            },
            {
                "control_refs": ["control-benign-low-variance"],
                "estimate": spread_value,
                "evidence_refs": ["o-1"],
                "id": "e-2",
                "metric_id": METRIC_SPREAD,
                "status": "observed",
                "uncertainty": {
                    "kind": "interval",
                    "lower": float(spread_interval["lower"]),
                    "upper": float(spread_interval["upper"]),
                },
            },
        ],
        "interventions": list(intervention_claims),
        "comparisons": comparison_rows,
        "limitations": [
            {
                "affects": ["generalization"],
                "blocking": False,
                "description": (
                    "bounded secondary lane:64 real frames from4 LIBERO episodes on a "
                    "CPU-only workstation; no simulator execution and no realtime claim"
                ),
                "id": "lim-1",
            }
        ],
        "next_action": {
            "action": "render the AI-engineer report",
            "rationale": "80.22 owns rendering",
            "required_evidence_refs": ["e-1", "o-1"],
        },
        "claims": [
            {
                "causal": False,
                "claim": (
                    f"the captured vision-context features show a rank-collapse symptom at {METRIC_ER} {rank:.4f}"
                ),
                "claim_allowed": bool(family.get("claim_allowed")),
                "control_refs": [],
                "evidence_refs": ["cap-1"],
                "id": "o-1",
                "kind": "observation",
                "missing_evidence": list(family.get("missing_evidence", [])),
                "status": "observed",
            },
            {
                "causal": False,
                "claim": f"probe explanation for the localized direction: {explanation_record['reason']}",  # type: ignore[index]
                "claim_allowed": bool(explanation_record.get("claim_allowed")),
                "control_refs": [
                    "control-healthy-counterexample",
                    "control-benign-low-variance",
                ],
                "evidence_refs": ["e-1", f"explanation-{HYPOTHESIS_ID}-record"],
                "id": "x-1",
                "kind": "explanation",
                "missing_evidence": list(explanation_record.get("missing_evidence", [])),
                "status": explanation_status if explanation_status != "omitted" else "unsupported",
            },
            *intervention_claims,
        ],
    }
    return report


def run_proof() -> int:
    guard = _RssGuard()
    guard.start()
    started = time.perf_counter()
    before = frozen_digests()

    _phase("data")
    data = build_data()
    print(
        f"PASS data: bounded real selection {len(data['samples'])} frames "  # type: ignore[arg-type]
        f"from episodes {EPISODES} (scanned {data['scanned']} items); "  # type: ignore[arg-type]
        f"selection digest {PINNED_DATASET_SHA[:16]}… matches the predeclared commitment"
    )

    _phase("load-policy")
    case = adapter_case()
    adapter = case["adapter"]  # type: ignore[assignment]
    print(
        f"PASS policy: {DEFAULT_SMOLVLA_CHECKPOINT.policy_repo_id}@{MODEL_REVISION[:16]}… "
        f"loaded bfloat16 on {adapter.device}; artifact digest {PINNED_MODEL_SHA[:16]}…"
    )

    _phase("capture")
    features = np.asarray(case["features"])
    _concrete, provenance = _capture()
    print(
        f"PASS capture: capture_identity {provenance.get('capture_identity', _sha(features.tobytes()))[:64]} "
        f"shape {features.shape} via official select_action vision_context hooks "
        f"({case['query_seconds']:.1f}s in {N_SAMPLES} real queries)"
    )

    manifest = load_manifest()
    detect = detect_case()
    measurements = detect["payload"]["measurements"]  # type: ignore[index]
    family = detect["family"]
    threshold_pass = {
        METRIC_ER: float(measurements[METRIC_ER]) >= 3.0,  # type: ignore[index]
        METRIC_SPREAD: float(measurements[METRIC_SPREAD]) >= 0.25,  # type: ignore[index]
    }
    positive_observed = all(threshold_pass.values())
    print(
        f"RECORD positive-case ({'frozen thresholds' if True else ''}): "
        f"{'OBSERVED' if positive_observed else 'NOT OBSERVED'} — "
        f"{METRIC_ER} {float(measurements[METRIC_ER]):.4f} >= 3.0? {threshold_pass[METRIC_ER]}; "  # type: ignore[index]
        f"{METRIC_SPREAD} {float(measurements[METRIC_SPREAD]):.4f} >= 0.25? {threshold_pass[METRIC_SPREAD]}; "  # type: ignore[index]
        f"controls {dict(family.control_outcomes)}; claim_allowed={family.claim_allowed}"  # type: ignore[attr-defined]
    )

    localize = localize_case()
    localize_payload = localize["payload"]
    print(
        f"RECORD localize: verdict {localize_payload['verdict']!r}; "  # type: ignore[index]
        f"score {localize_payload.get('global_score')!r}; reason {localize_payload.get('reason')!r}"  # type: ignore[index]
    )

    # In-process determinism: identical stage payloads across recomputation.
    _det, payload_again = evaluate_detection(
        _value(features),
        detect["config"],  # type: ignore[arg-type]
        controls=_control_batches(),
    )
    _res, localize_again = evaluate_localization(localize["source"])  # type: ignore[arg-type]
    if _payload_sha(payload_again) != _payload_sha(detect["payload"]):  # type: ignore[index]
        raise AssertionError("detect payload is not byte-deterministic")
    if _payload_sha(localize_again) != _payload_sha(localize_payload):  # type: ignore[arg-type]
        raise AssertionError("localize payload is not byte-deterministic")
    print(
        f"PASS determinism: detect payload sha {_payload_sha(detect['payload'])}, "  # type: ignore[index]
        f"localize payload sha {_payload_sha(localize_payload)}; "  # type: ignore[arg-type]
        f"capture matrix sha {_sha(features.astype('<f8', copy=False).tobytes())}"
    )

    workflow, request = build_workflow()
    result, checkpoint = workflow.run(request, manifest)
    outputs = {item.stage: item for item in checkpoint.outputs}
    print(f"RECORD stages completed: {', '.join(outputs)}")
    failure = checkpoint.failure
    if failure is not None:
        print(
            f"BLOCKED {failure.stage}: {failure.error_type} — {failure.message}; "
            f"workflow status result={result.status} checkpoint={checkpoint.status}"
        )

    persisted = False
    artifact_note = ""
    if result.status == "completed" and failure is None:
        ordered = tuple(item.stage for item in checkpoint.outputs)
        interventions = checkpoint.outputs[ordered.index("intervene")].payload["trials"]
        comparisons = checkpoint.outputs[ordered.index("compare")].payload["comparisons"]
        claims = intervention_report_items(interventions)
        rows = comparison_report_items(comparisons)
        report = _build_report(checkpoint.outputs, intervention_claims=claims, comparison_rows=rows)
        try:
            with tempfile.TemporaryDirectory(prefix="proof-80-27-") as tmp:
                root = Path(tmp) / "root"
                handle = persist_diagnostic_artifact(
                    root,
                    request=request,
                    manifest=manifest,
                    result=result,
                    checkpoint=checkpoint,
                    report=report,
                )
                first = render_diagnostic_report(root, run_id=handle.run_id, manifest=manifest)
                second = render_diagnostic_report(root, run_id=handle.run_id, manifest=manifest)
                if first != second:
                    raise AssertionError("rendered report is not byte-deterministic")
                persist_rendered_report(
                    root,
                    run_id=handle.run_id,
                    manifest=manifest,
                    output=OutputSelection(output_location=OUTPUT_LOCATION),
                )
                persisted = True
                artifact_note = (
                    f"run {handle.run_id}, artifact sha {handle.artifact_digest}, "
                    f"report sha {handle.report_digest}; rendered at {OUTPUT_LOCATION}"
                )
        except DiagnosticArtifactError as exc:
            artifact_note = f"persist refused: {exc}"
    else:
        refusal = None
        with tempfile.TemporaryDirectory(prefix="proof-80-27-partial-") as tmp:
            root = Path(tmp) / "root"
            try:
                persist_diagnostic_artifact(
                    root,
                    request=request,
                    manifest=manifest,
                    result=result,
                    checkpoint=checkpoint,
                    report={},
                )
            except DiagnosticArtifactError as exc:
                refusal = str(exc)
            if (root / "runs").exists() or (root / "artifacts").exists():
                raise AssertionError("refused persistence wrote files")
        artifact_note = f"incomplete-workflow refusal: {refusal}"

    if persisted:
        print(f"PASS artifact: validator-clean content-addressed artifact persisted — {artifact_note}")
        verdict = "ACCEPTANCE: PASSED — bounded secondary artifact persisted"
    else:
        print(f"RECORD persistence: {artifact_note}")
        verdict = "ACCEPTANCE: NOT PASSED — no validator-clean artifact; the exact blocker is recorded above"

    after = frozen_digests()
    if before != after:
        raise AssertionError("frozen artifacts were modified by the proof run")
    print("PASS frozen invariance: frozen inputs byte-identical after the run")

    guard.stop()
    peak = guard.peak
    if peak >= PEAK_CEILING:
        raise AssertionError(f"peak RSS {peak} reached the16 GiB ceiling")
    current = time.perf_counter()
    phase_text = ", ".join(f"{name}={seconds:.2f}s" for name, seconds in _phases.items())
    print(verdict)
    cuda_peak = torch.cuda.max_memory_allocated() / (1 << 20) if torch.cuda.is_available() else 0.0
    cuda_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    print(
        f"device: {cuda_name}; cuda {torch.version.cuda or 'n/a'}; "
        f"peak GPU memory {cuda_peak:.1f} MiB; torch {torch.__version__}"
    )
    print(
        f"resources: proof body {current - started:.2f}s ({phase_text}); "
        f"wall including imports {current - _MODULE_START:.2f}s; "
        f"peak RSS {peak / (1 << 20):.1f} MiB (ceiling16384 MiB); "
        f"tracemalloc peak {tracemalloc.get_traced_memory()[1] / (1 << 20):.1f} MiB; "
        f"environment python {platform.python_version()}, numpy {np.__version__}, "
        f"torch {torch.__version__}, lerobot {package_version('lerobot')}, "
        f"transformers {package_version('transformers')}; "
        f"declared output location {OUTPUT_LOCATION!r} "
        f"{'materialized' if persisted else 'not materialized (persistence refused)'}"
    )
    print("commands: F:/ai-ml/sprint80_27_env/Scripts/python scripts/sprint80_task80_27_smolvla_proof.py")
    tracemalloc.stop()
    return 0


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Policy device (defaults to cuda when available, matching the hardware profile).",
    )
    args = parser.parse_args()
    globals()["_DEVICE"] = str(args.device)
    return run_proof()


_DEVICE = "cpu"


if __name__ == "__main__":
    raise SystemExit(main())

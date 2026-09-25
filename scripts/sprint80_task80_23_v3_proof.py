"""Frozen prospective model-weight-lesion proof for Sprint 80.23 v3.

Run ``--prepare-model`` once before writing the v3 manifest. It fits the
baseline linear autoencoder on the frozen train split only and saves the
checkpoint whose digest is committed by that manifest. After the manifest is
frozen, run this file without arguments to execute the complete diagnostic
workflow and persist its independently validated rendered report.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import cast

import numpy as np
import sklearn

from latent_anything._benchmark_manifest import manifest_digest, validate_manifest
from latent_anything._capture_binding import bind_selection, resolve_captures
from latent_anything._collapse_detection import (
    detection_config_from_manifest,
    evaluate_detection,
    make_detect_executor,
)
from latent_anything._diagnostic_artifact import (
    load_diagnostic_artifact,
    persist_diagnostic_artifact,
    registered_artifact_bytes,
)
from latent_anything._diagnostic_report import SCHEMA_VERSION, validate_report_shape
from latent_anything._diagnostic_validator import validate_diagnostic_report
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
from latent_anything._layer_slice_localization import (
    AxialLocalizationInput,
    FeatureAxisDeclaration,
    FeatureCell,
    evidence_from_manifest,
    make_axial_localize_executor,
    manifest_axis_lookup,
)
from latent_anything._probe_tcav_ig_explanation import (
    ExplanationHypothesis,
    MethodInputs,
    make_explain_executor,
)
from latent_anything._report_renderer import (
    persist_rendered_report,
    render_diagnostic_report,
)
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
from latent_anything.diagnostics import (
    CaptureSelection,
    ComparisonRequest,
    ControlSelection,
    DiagnosticRequest,
    DiagnosticSelection,
    InterventionRequest,
    OutputSelection,
)
from latent_anything.latent_space import LatentSpace
from latent_anything.latent_value import LatentValue

REPO = Path(__file__).resolve().parents[1]
ARTIFACTS = REPO / "artifacts"
MODEL_PATH = ARTIFACTS / "benchmark_model_sprint80_encoder_autoencoder_v3_baseline.npz"
MANIFEST_PATH = ARTIFACTS / "benchmark_manifest_sprint80_encoder_autoencoder_v3_model_weight_lesion.json"
OUTPUT_LOCATION = "artifacts/diagnostics/proof-80-23-v3-model-weight-lesion"
OUTPUT_ROOT = REPO / OUTPUT_LOCATION
MANIFEST_ID = "sprint80-core-encoder-autoencoder-collapse-model-lesion-v3"
FAMILY = "collapse_rank_loss"
METRIC_ER = "bottleneck-effective-rank"
METRIC_SPREAD = "bottleneck-singular-spread"
METRIC_FEATURE_RATIO = "bottleneck-feature-variance-ratio"
METRIC_TASK_ACCURACY = "heldout-brightness-bin-accuracy"
MODEL_ID = "scripts.sprint80_task80_23_v3_proof.LinearBrightnessPCAAutoencoder"
MODEL_REVISION = "linear-pca-autoencoder-4d-train-only-v1"
REPRESENTATION = "linear_pca_autoencoder:bottleneck:latent_dim=4"
FEATURE_IDS = tuple(f"brightness-axis-dim{index}" for index in range(4))
CAPTURE_ID = "capture-linear-autoencoder-bottleneck-v3"
REQUEST_ID = "proof-80-23-v3-model-lesion"
HYPOTHESIS_ID = "h-brightness-axis-dim0-v3"
TRIAL_ID = "restore-lesioned-encoder-dim0-v3"
COMPARISON_ID = "cmp-healthy-vs-model-lesion-v3"
REPORT_ID = "report-80-23-encoder-model-lesion-v3"
MANIFEST_CONTROLS = (
    "control-healthy-counterexample",
    "control-benign-low-variance",
    "control-null-shuffle",
)
TRIAL_CONTROLS = (
    "model-zero-strength",
    "model-random-direction",
    "model-shuffled-pairing",
    "model-off-target-row",
)
TRAIN_SPLIT_ID = "digits-train-1437"
HELDOUT_SPLIT_ID = "digits-heldout-360"
SPLIT_IDENTITY = "default_rng-42-permutation-train1437-heldout360"
DATA_DIGEST = "b3ed6f2d420b7dc8de35bb8adc0088b49f76a3b05d31057a0a48e37430966148"


@dataclass
class LinearBrightnessPCAAutoencoder:
    """Four-dimensional linear encoder/decoder with a fixed brightness readout."""

    mean: np.ndarray
    encoder_weight: np.ndarray
    encoder_bias: np.ndarray
    decoder_weight: np.ndarray
    task_cutoff: float
    task_threshold_raw: float

    def encode(self, images: np.ndarray) -> np.ndarray:
        values = np.asarray(images, dtype=np.float64).reshape(len(images), -1)
        return values @ self.encoder_weight.T + self.encoder_bias

    def decode(self, codes: np.ndarray) -> np.ndarray:
        return np.asarray(codes, dtype=np.float64) @ self.decoder_weight.T + self.mean

    def copy(self) -> LinearBrightnessPCAAutoencoder:
        return LinearBrightnessPCAAutoencoder(
            mean=self.mean.copy(),
            encoder_weight=self.encoder_weight.copy(),
            encoder_bias=self.encoder_bias.copy(),
            decoder_weight=self.decoder_weight.copy(),
            task_cutoff=float(self.task_cutoff),
            task_threshold_raw=float(self.task_threshold_raw),
        )


def _load_v1_module() -> ModuleType:
    path = REPO / "scripts" / "sprint80_task80_23_proof.py"
    spec = importlib.util.spec_from_file_location("_sprint80_task80_23_v1_frozen_data", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the frozen v1 split helper")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_V1 = _load_v1_module()


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _array_digest(values: np.ndarray) -> str:
    return _sha(np.ascontiguousarray(values).tobytes())


def _fit_train_only_model() -> tuple[LinearBrightnessPCAAutoencoder, dict[str, object]]:
    """Fit the fixed brightness-plus-residual-PCA basis on training rows only."""
    data = _V1.build_data()
    train = np.asarray(data["train"], dtype=np.float64).reshape(len(data["train"]), -1)
    if train.shape != (1437, 64):
        raise ValueError(f"frozen train partition has unexpected shape {train.shape!r}")
    mean = train.mean(axis=0)
    centered = train - mean
    brightness = np.full(train.shape[1], 1.0 / np.sqrt(train.shape[1]), dtype=np.float64)
    brightness_scores = centered @ brightness
    residual = centered - np.outer(brightness_scores, brightness)
    _, _, right = np.linalg.svd(residual, full_matrices=False)
    basis_rows = [brightness]
    for candidate in right:
        orthogonal = np.asarray(candidate, dtype=np.float64).copy()
        for existing in basis_rows:
            orthogonal -= float(orthogonal @ existing) * existing
        norm = float(np.linalg.norm(orthogonal))
        if norm > 1e-10:
            basis_rows.append(orthogonal / norm)
        if len(basis_rows) == 4:
            break
    if len(basis_rows) != 4:
        raise ValueError("training-only PCA did not provide four independent directions")
    basis = np.stack(basis_rows)
    raw_scores = centered @ basis.T
    scales = np.std(raw_scores, axis=0, ddof=1)
    if not np.isfinite(scales).all() or np.any(scales <= 0.0):
        raise ValueError("training-only latent scales are not finite and positive")
    encoder_weight = basis / scales[:, None]
    encoder_bias = -(mean @ encoder_weight.T)
    decoder_weight = basis.T * scales[None, :]
    task_threshold_raw = float(np.median(train.mean(axis=1)))
    brightness_mean = float(mean @ brightness)
    task_cutoff = float((task_threshold_raw * np.sqrt(train.shape[1]) - brightness_mean) / scales[0])
    model = LinearBrightnessPCAAutoencoder(
        mean=mean,
        encoder_weight=encoder_weight,
        encoder_bias=encoder_bias,
        decoder_weight=decoder_weight,
        task_cutoff=task_cutoff,
        task_threshold_raw=task_threshold_raw,
    )
    provenance = {
        "dataset_data_digest": DATA_DIGEST,
        "model_revision": MODEL_REVISION,
        "sklearn_version": sklearn.__version__,
        "split_identity": SPLIT_IDENTITY,
        "train_index_digest": _array_digest(np.asarray(data["train_indices"], dtype=np.int64)),
        "train_rows": int(len(train)),
        "train_tensor_digest": _array_digest(train),
    }
    return model, provenance


def _save_model(
    path: Path,
    model: LinearBrightnessPCAAutoencoder,
    provenance: Mapping[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        np.savez(
            handle,
            mean=model.mean,
            encoder_weight=model.encoder_weight,
            encoder_bias=model.encoder_bias,
            decoder_weight=model.decoder_weight,
            task_cutoff=np.asarray(model.task_cutoff, dtype=np.float64),
            task_threshold_raw=np.asarray(model.task_threshold_raw, dtype=np.float64),
            dataset_data_digest=np.asarray(str(provenance["dataset_data_digest"])),
            model_revision=np.asarray(str(provenance["model_revision"])),
            sklearn_version=np.asarray(str(provenance["sklearn_version"])),
            split_identity=np.asarray(str(provenance["split_identity"])),
            train_index_digest=np.asarray(str(provenance["train_index_digest"])),
            train_rows=np.asarray(int(provenance["train_rows"]), dtype=np.int64),
            train_tensor_digest=np.asarray(str(provenance["train_tensor_digest"])),
        )


def _load_model(path: Path) -> tuple[LinearBrightnessPCAAutoencoder, dict[str, object]]:
    with np.load(path, allow_pickle=False) as saved:
        model = LinearBrightnessPCAAutoencoder(
            mean=np.asarray(saved["mean"], dtype=np.float64).copy(),
            encoder_weight=np.asarray(saved["encoder_weight"], dtype=np.float64).copy(),
            encoder_bias=np.asarray(saved["encoder_bias"], dtype=np.float64).copy(),
            decoder_weight=np.asarray(saved["decoder_weight"], dtype=np.float64).copy(),
            task_cutoff=float(saved["task_cutoff"]),
            task_threshold_raw=float(saved["task_threshold_raw"]),
        )
        provenance: dict[str, object] = {
            "dataset_data_digest": str(saved["dataset_data_digest"].item()),
            "model_revision": str(saved["model_revision"].item()),
            "sklearn_version": str(saved["sklearn_version"].item()),
            "split_identity": str(saved["split_identity"].item()),
            "train_index_digest": str(saved["train_index_digest"].item()),
            "train_rows": int(saved["train_rows"]),
            "train_tensor_digest": str(saved["train_tensor_digest"].item()),
        }
    return model, provenance


def _prepare_model() -> dict[str, object]:
    if MODEL_PATH.exists():
        raise FileExistsError(f"baseline model artifact already exists; refusing replacement: {MODEL_PATH}")
    model, provenance = _fit_train_only_model()
    _save_model(MODEL_PATH, model, provenance)
    return {
        "stage": "train-only-model-preparation",
        "checkpoint": str(MODEL_PATH.relative_to(REPO)),
        "checkpoint_sha256": _sha(MODEL_PATH.read_bytes()),
        "train_only": True,
        "provenance": provenance,
    }


def _load_manifest() -> dict[str, object]:
    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(f"prospective v3 manifest must be frozen before proof run: {MANIFEST_PATH}")
    raw = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    validate_manifest(raw)
    if raw.get("manifest_id") != MANIFEST_ID:
        raise ValueError("unexpected prospective v3 manifest identity")
    if manifest_digest(raw) != cast(Mapping[str, object], raw["commitment"])["manifest_sha256"]:
        raise ValueError("prospective v3 manifest commitment does not match its canonical content")
    model = cast(Mapping[str, object], raw["model"])
    if model.get("artifact_digest") != _sha(MODEL_PATH.read_bytes()):
        raise ValueError("baseline model checkpoint digest differs from the frozen manifest")
    return dict(raw)


def _validate_frozen_inputs() -> tuple[dict[str, object], LinearBrightnessPCAAutoencoder, dict[str, object]]:
    manifest = _load_manifest()
    baseline, provenance = _load_model(MODEL_PATH)
    model_decl = cast(Mapping[str, object], manifest["model"])
    dataset = cast(Mapping[str, object], manifest["dataset"])
    expected = {
        "dataset_data_digest": dataset["data_digest"],
        "model_revision": model_decl["revision"],
        "sklearn_version": str(dataset["revision"]).removeprefix("scikit-learn=="),
        "split_identity": dataset["split_identity"],
    }
    for key, value in expected.items():
        if provenance.get(key) != value:
            raise ValueError(f"baseline checkpoint {key} disagrees with the frozen manifest")
    if provenance.get("train_rows") != 1437:
        raise ValueError("baseline checkpoint was not fitted on the complete frozen training split")
    return manifest, baseline, provenance


def _manifest_validation_record() -> dict[str, object]:
    manifest, _baseline, provenance = _validate_frozen_inputs()
    return {
        "manifest_id": manifest["manifest_id"],
        "manifest_digest": manifest_digest(manifest),
        "manifest_locked": cast(Mapping[str, object], manifest["commitment"])["locked"],
        "baseline_checkpoint_sha256": _sha(MODEL_PATH.read_bytes()),
        "baseline_training_provenance": provenance,
        "validation": "passed",
        "proof_outputs_generated": False,
    }


def _request(*, include_trial_controls: bool) -> DiagnosticRequest:
    trial_controls = TRIAL_CONTROLS if include_trial_controls else ()
    interventions = (
        (
            InterventionRequest(
                intervention_id=TRIAL_ID,
                target=FEATURE_IDS[0],
                control_ids=TRIAL_CONTROLS,
            ),
        )
        if include_trial_controls
        else ()
    )
    comparisons = (
        (
            ComparisonRequest(
                comparison_id=COMPARISON_ID,
                baseline_run="linear-autoencoder-heldout-healthy-v3",
                candidate_run="linear-autoencoder-heldout-lesion-v3",
                metric_ids=(METRIC_FEATURE_RATIO, METRIC_TASK_ACCURACY),
            ),
        )
        if include_trial_controls
        else ()
    )
    return DiagnosticRequest(
        request_id=REQUEST_ID if include_trial_controls else f"{REQUEST_ID}-detect",
        manifest_id=MANIFEST_ID,
        capture=CaptureSelection(
            capture_id=CAPTURE_ID,
            representation_identity=REPRESENTATION,
            axes=("sample", "feature", "slice"),
        ),
        diagnostics=DiagnosticSelection(family_ids=(FAMILY,)),
        controls=ControlSelection(
            control_ids=(*MANIFEST_CONTROLS, *trial_controls),
            metric_ids=(METRIC_ER, METRIC_SPREAD, METRIC_FEATURE_RATIO),
        ),
        interventions=interventions,
        comparisons=comparisons,
        output=OutputSelection(
            output_location=OUTPUT_LOCATION,
            artifact_name="encoder-diagnosis-v3.md",
            include_report=True,
        ),
    )


def _space() -> LatentSpace:
    model = cast(Mapping[str, object], _FROZEN_MANIFEST["model"])
    return LatentSpace(
        dim=4,
        source_model=str(model["id"]),
        metadata={
            "source_representation_identity": REPRESENTATION,
            "model_version": str(model["revision"]),
        },
    )


def _value(data: np.ndarray) -> LatentValue:
    return LatentValue(np.asarray(data, dtype=np.float64), _space())


def _feature_ratio(matrix: np.ndarray) -> float:
    variances = np.var(matrix, axis=0, ddof=1)
    median = float(np.median(variances))
    if not np.isfinite(median) or median <= 0.0:
        raise ValueError("minimum-to-median feature variance ratio is undefined")
    return float(np.min(variances) / median)


def _task_accuracy(model: LinearBrightnessPCAAutoencoder, images: np.ndarray, labels: np.ndarray) -> float:
    predictions = model.encode(images)[:, 0] >= model.task_cutoff
    return float(np.mean(predictions == np.asarray(labels, dtype=bool)))


def _capture(
    manifest: Mapping[str, object],
    request: DiagnosticRequest,
    damaged_latents: np.ndarray,
) -> tuple[object, dict[str, object], LatentValue]:
    plan = bind_selection(request.capture, manifest=manifest, request_id=request.request_id)
    activation = CapturedActivation(
        values=damaged_latents,
        metadata=CaptureMetadata(
            location="linear-pca-autoencoder.encode:bottleneck:encoder-row-0-zeroed-in-model-copy",
            call_index=0,
            shape=tuple(int(size) for size in damaged_latents.shape),
            batch_axis=0,
            sequence_axis=None,
            device="cpu",
            dtype=str(damaged_latents.dtype),
            source_model_version=MODEL_REVISION,
        ),
    )
    (concrete,), (value,) = resolve_captures(plan, (activation,))
    return concrete, dict(concrete.provenance()), value


def _trial() -> TrialSpec:
    return TrialSpec(
        intervention_id=TRIAL_ID,
        kind="patch",
        target=FEATURE_IDS[0],
        metric_id=METRIC_TASK_ACCURACY,
        hypothesis_id=HYPOTHESIS_ID,
        strength=1.0,
        strength_semantics="fraction-of-paired-baseline-encoder-row-restored",
        expected_effect="increase",
        controls=(
            TrialControl(
                TRIAL_CONTROLS[0],
                "zero_strength",
                "zero-strength leaves the lesioned encoder row unchanged",
            ),
            TrialControl(
                TRIAL_CONTROLS[1],
                "random",
                "a seeded brightness-orthogonal random row does not recover task accuracy",
            ),
            TrialControl(
                TRIAL_CONTROLS[2],
                "shuffled",
                "restoring the row with shuffled heldout pairing does not recover accuracy",
            ),
            TrialControl(
                TRIAL_CONTROLS[3],
                "off_target",
                "restoring feature dim1 leaves the brightness-task lesion unchanged",
                target=FEATURE_IDS[1],
            ),
        ),
        provenance={"method": "encoder-weight-restoration", "carrier": "paired-baseline-encoder-row"},
    )


def _intervention_measure(
    healthy: LinearBrightnessPCAAutoencoder,
    damaged: LinearBrightnessPCAAutoencoder,
    heldout: np.ndarray,
    labels: np.ndarray,
) -> Callable[[TrialApplication], Measurement]:
    def measure(application: TrialApplication) -> Measurement:
        model = damaged.copy()
        target = int(application.target.rsplit("dim", 1)[1])
        control = application.control_class
        sample = heldout
        sample_labels = labels
        if application.role == "intervened":
            model.encoder_weight[target] = healthy.encoder_weight[target]
            model.encoder_bias[target] = healthy.encoder_bias[target]
        elif application.role == "control" and control == "random":
            if application.rng is None:
                raise ValueError("random model control requires its declared RNG stream")
            brightness = np.full(model.mean.size, 1.0 / np.sqrt(model.mean.size), dtype=np.float64)
            direction = application.rng.normal(size=model.mean.size)
            direction -= float(direction @ brightness) * brightness
            norm = float(np.linalg.norm(direction))
            if norm <= 1e-12:
                raise ValueError("random control direction is degenerate")
            direction *= float(np.linalg.norm(healthy.encoder_weight[target])) / norm
            model.encoder_weight[target] = direction
            model.encoder_bias[target] = -float(model.mean @ direction)
        elif application.role == "control" and control == "shuffled":
            if application.rng is None:
                raise ValueError("shuffled model control requires its declared RNG stream")
            model.encoder_weight[target] = healthy.encoder_weight[target]
            model.encoder_bias[target] = healthy.encoder_bias[target]
            sample = heldout[application.rng.permutation(len(heldout))]
        elif application.role == "control" and control == "off_target":
            model.encoder_weight[target] = healthy.encoder_weight[target]
            model.encoder_bias[target] = healthy.encoder_bias[target]
        if application.rng is not None and control != "shuffled":
            positions = application.rng.integers(0, len(sample), size=len(sample))
            sample = sample[positions]
            sample_labels = labels[positions]
        elif control == "shuffled":
            # The shuffled activation source is compared with labels in original order.
            sample_labels = labels
        value = _task_accuracy(model, sample, sample_labels)
        return Measurement(application.metric_id, value, application.inputs_digest)

    return measure


def _comparison_specs(
    manifest: Mapping[str, object],
    capture_provenance: Mapping[str, object],
    detection_payload: Mapping[str, object],
    baseline_digest: str,
    lesion_digest: str,
) -> tuple[ComparisonSpec, ...]:
    dataset = cast(Mapping[str, object], manifest["dataset"])
    model = cast(Mapping[str, object], manifest["model"])
    alignment = {
        "axes": capture_axes_identity(capture_provenance),
        "dataset_configuration": str(dataset["split_identity"]),
        "dataset_slice_id": FEATURE_IDS[0],
        "diagnostic_config": detect_config_identity(detection_payload),
        "layer_module_identity": REPRESENTATION,
        "manifest_identity": MANIFEST_ID,
        "model_identity": str(model["id"]),
        "preprocessing_identity": "digits-pixels-0-to-16-scaled-by-16",
        "representation_identity": REPRESENTATION,
        "representation_metric": METRIC_FEATURE_RATIO,
        "schema_identity": str(manifest["schema_version"]),
        "seeds": manifest_seed_identity(manifest),
        "taxonomy_identity": FAMILY,
        "task_metric": METRIC_TASK_ACCURACY,
    }
    if set(alignment) != set(ALIGNMENT_FIELDS):
        raise ValueError("comparison alignment fields disagree with the frozen contract")
    return (
        ComparisonSpec(
            comparison_id=COMPARISON_ID,
            baseline=RunSide(
                run_id="linear-autoencoder-heldout-healthy-v3",
                checkpoint_id=f"baseline-{baseline_digest[:16]}",
                alignment=alignment,
            ),
            candidate=RunSide(
                run_id="linear-autoencoder-heldout-lesion-v3",
                checkpoint_id=f"lesion-{lesion_digest[:16]}",
                alignment=alignment,
            ),
            representation_metric_id=METRIC_FEATURE_RATIO,
            task_metric_id=METRIC_TASK_ACCURACY,
            representation_tolerance=0.05,
            task_tolerance=0.05,
        ),
    )


def _probe_bundle(
    train_healthy: np.ndarray,
    heldout_healthy: np.ndarray,
    train_labels: np.ndarray,
    heldout_labels: np.ndarray,
    train_indices: np.ndarray,
    heldout_indices: np.ndarray,
) -> dict[str, object]:
    rng = np.random.default_rng(17)
    noise_train = rng.normal(size=(len(train_healthy), 1))
    noise_heldout = rng.normal(size=(len(heldout_healthy), 1))
    negative_accuracy = float(
        _V1._fast_probe(
            noise_train,
            train_labels,
            noise_heldout,
            heldout_labels,
            17,
        ).accuracy
    )
    return {
        "train_matrix": train_healthy[:, :1],
        "eval_matrix": heldout_healthy[:, :1],
        "train_labels": train_labels,
        "eval_labels": heldout_labels,
        "train_ids": [f"train-{int(index)}" for index in train_indices],
        "eval_ids": [f"heldout-{int(index)}" for index in heldout_indices],
        "capacity": "linear-logreg-C1.0-standardized",
        "negative_accuracy": negative_accuracy,
    }


def _build_report(
    checkpoint_outputs: Sequence[StageOutput],
    *,
    manifest: Mapping[str, object],
) -> dict[str, object]:
    payloads = {item.stage: dict(item.payload) for item in checkpoint_outputs}
    detect = payloads["detect"]
    localize = payloads["localize"]
    explain = payloads["explain"]
    family = cast(Mapping[str, object], cast(Sequence[object], detect["families"])[0])
    observed = cast(Mapping[str, object], family["observed_metrics"])
    measurements = cast(Mapping[str, object], detect["measurements"])
    uncertainty = cast(Mapping[str, object], measurements["uncertainty"])
    evidence_ref = "symptom-model-weight-lesion-v3"
    required_controls = ["control-healthy-counterexample", "control-benign-low-variance"]
    metric_ids = (METRIC_ER, METRIC_SPREAD, METRIC_FEATURE_RATIO)
    statistical_rows: list[dict[str, object]] = []
    for metric_id in metric_ids:
        interval = cast(Mapping[str, object], uncertainty[metric_id])
        statistical_rows.append(
            {
                "control_refs": required_controls,
                "estimate": float(observed[metric_id]),
                "evidence_refs": [evidence_ref],
                "id": f"stat-{metric_id}",
                "metric_id": metric_id,
                "status": "observed",
                "uncertainty": {
                    "kind": "interval",
                    "lower": float(interval["lower"]),
                    "upper": float(interval["upper"]),
                },
            }
        )
    raw_localization = cast(Sequence[object], localize["report_localization"])
    localization_rows = [
        {
            "axis": str(row["axis"]),
            "confidence": float(row["confidence"]),
            "evidence_refs": [evidence_ref],
            "id": str(row["id"]),
            "selection": str(row["selection"]),
            "status": str(row["status"]),
        }
        for row in (cast(Mapping[str, object], value) for value in raw_localization)
    ]
    explain_records = cast(Sequence[object], explain["evidence"])
    explanation = next(
        cast(Mapping[str, object], item)
        for item in explain_records
        if cast(Mapping[str, object], item).get("hypothesis_id") == HYPOTHESIS_ID
    )
    explanation_status = str(explanation["outcome"])
    if explanation_status == "omitted":
        explanation_status = "unsupported"
    explanation_evidence_ref = f"explanation-{HYPOTHESIS_ID}-record"
    dataset = cast(Mapping[str, object], manifest["dataset"])
    model = cast(Mapping[str, object], manifest["model"])
    comparison_records = cast(Sequence[Mapping[str, object]], payloads["compare"]["comparisons"])
    intervention_records = cast(Sequence[Mapping[str, object]], payloads["intervene"]["trials"])
    comparison = comparison_records[0]
    task = cast(Mapping[str, object], comparison["task"])
    signed_task_delta = float(task["signed_delta"])
    comparison_classification = str(comparison["classification"])
    comparison_claim_status = (
        "observed" if comparison_classification not in ("inconclusive", "unsupported") else comparison_classification
    )
    explanation_claim = {
        "causal": False,
        "claim": (
            "A leakage-safe probe using only the predeclared brightness-axis dim0 predicts the heldout high/low "
            f"brightness bin from the intact baseline encoder: {explanation.get('reason', explanation_status)}"
        ),
        "claim_allowed": bool(explanation.get("claim_allowed")) and explanation_status == "supported",
        "control_refs": required_controls,
        "evidence_refs": [f"stat-{METRIC_FEATURE_RATIO}", explanation_evidence_ref],
        "id": f"claim-{HYPOTHESIS_ID}",
        "kind": "explanation",
        "missing_evidence": list(explanation.get("missing_evidence", [])),
        "status": explanation_status,
    }
    comparison_claim = {
        "causal": False,
        "claim": (
            "The predeclared aligned comparison measured heldout brightness-bin accuracy on both the healthy and "
            f"zero-row model checkpoints; candidate-minus-baseline task delta={signed_task_delta:.8g} "
            f"({comparison['classification']})."
        ),
        "claim_allowed": comparison_claim_status == "observed",
        "control_refs": [],
        "evidence_refs": [f"comparison-{COMPARISON_ID}-record"],
        "id": "claim-downstream-task-utility-v3",
        "kind": "observation",
        "missing_evidence": [],
        "status": comparison_claim_status,
    }
    report: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "report_id": REPORT_ID,
        "status": "declared",
        "capture_provenance": {
            "captures": [
                {
                    "artifact_refs": ["cap-v3"],
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
                    "A prospective model-level lesion zeroed encoder weight row 0 and its bias in a copy of the "
                    "train-only linear autoencoder before heldout encoding; the frozen per-feature variance ratio "
                    f"was {float(observed[METRIC_FEATURE_RATIO]):.8g}."
                ),
                "evidence_refs": ["cap-v3"],
                "id": evidence_ref,
                "metric_ids": [METRIC_FEATURE_RATIO],
                "status": str(family["outcome"]),
            }
        ],
        "localization": localization_rows,
        "hypotheses": [
            {
                "alternatives": ["healthy train-only encoder counterexample", "benign global low-gain negative"],
                "evidence_refs": [explanation_evidence_ref],
                "id": HYPOTHESIS_ID,
                "statement": (
                    "The localized dim0 coordinate is the model's predeclared linear image-mean direction: the "
                    "train-only encoder emits centered global brightness on this coordinate and the fixed downstream "
                    "readout thresholds it using the training median."
                ),
                "status": explanation_status,
            }
        ],
        "statistical_evidence": statistical_rows,
        "interventions": [],
        "comparisons": comparison_report_items(comparison_records),
        "limitations": [
            {
                "affects": ["detect", "localization", "intervene", "compare"],
                "blocking": False,
                "description": (
                    "This is a prospective controlled linear-autoencoder weight lesion on one frozen digits split, "
                    "not evidence that the historical ConvVAE or a production encoder has the same defect."
                ),
                "id": "lim-controlled-model-scope-v3",
            },
            {
                "affects": ["explain", "compare"],
                "blocking": False,
                "description": (
                    "The downstream task is a predeclared train-median brightness bin, not a general-purpose digit "
                    "recognition or cross-dataset utility measure."
                ),
                "id": "lim-task-scope-v3",
            },
        ],
        "next_action": {
            "action": "retain the frozen v3 proof as prospective evidence and keep historical v1/v2 results unchanged",
            "rationale": (
                "the artifact binds the model-level lesion, causal controls, task-utility comparison, "
                "and validated report"
            ),
            "required_evidence_refs": [evidence_ref, f"stat-{METRIC_FEATURE_RATIO}", COMPARISON_ID],
        },
        "claims": [
            {
                "causal": False,
                "claim": (
                    "The frozen workflow captured and diagnosed heldout bottleneck outputs from a model copy whose "
                    "encoder dim0 weights and bias were zeroed before encoding; the original checkpoint and input "
                    "images were not changed."
                ),
                "claim_allowed": bool(family["claim_allowed"]) and str(family["outcome"]) == "supported",
                "control_refs": [],
                "evidence_refs": ["cap-v3", evidence_ref],
                "id": "claim-prospective-model-lesion-v3",
                "kind": "observation",
                "missing_evidence": list(family.get("missing_evidence", [])),
                "status": "observed" if str(family["outcome"]) == "supported" else str(family["outcome"]),
            },
            explanation_claim,
            *intervention_report_items(intervention_records),
            comparison_claim,
        ],
    }
    validate_report_shape(report)
    return report


def _write_run_record(root: Path, record: Mapping[str, object]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "proof-run.json"
    encoded = (json.dumps(record, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    with path.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
    return path


def _run() -> dict[str, object]:
    if OUTPUT_ROOT.exists():
        raise FileExistsError(f"proof output already exists; refusing to overwrite frozen evidence: {OUTPUT_ROOT}")
    manifest_raw = MANIFEST_PATH.read_bytes()
    manifest, baseline, baseline_provenance = _validate_frozen_inputs()
    manifest_hash = manifest_digest(manifest)
    manifest_raw_sha = _sha(manifest_raw)
    data = _V1.build_data()
    train_images = np.asarray(data["train"], dtype=np.float64)
    heldout_images = np.asarray(data["heldout"], dtype=np.float64)
    train_indices = np.asarray(data["train_indices"], dtype=np.int64)
    heldout_indices = np.asarray(data["heldout_indices"], dtype=np.int64)
    train_labels = train_images.reshape(len(train_images), -1).mean(axis=1) >= baseline.task_threshold_raw
    heldout_labels = heldout_images.reshape(len(heldout_images), -1).mean(axis=1) >= baseline.task_threshold_raw
    train_healthy = baseline.encode(train_images)
    heldout_healthy = baseline.encode(heldout_images)
    damaged = baseline.copy()
    damaged.encoder_weight[0, :] = 0.0
    damaged.encoder_bias[0] = 0.0
    damaged_latents = damaged.encode(heldout_images)
    lesion_record = {
        "baseline_checkpoint_sha256": _sha(MODEL_PATH.read_bytes()),
        "defect_boundary": "model.encoder_weight[0,:] and model.encoder_bias[0] before heldout encode",
        "lesion_operation": "zero_encoder_row_and_bias_in_model_copy",
        "lesioned_state_sha256": _sha(
            b"".join(
                np.ascontiguousarray(array).tobytes()
                for array in (damaged.mean, damaged.encoder_weight, damaged.encoder_bias, damaged.decoder_weight)
            )
        ),
        "manifest_digest": manifest_hash,
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
    }
    baseline_digest = _sha(MODEL_PATH.read_bytes())
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
    candidate_dir = OUTPUT_ROOT / "model"
    candidate_dir.mkdir()
    candidate_path = candidate_dir / "lesioned_encoder.npz"
    _save_model(candidate_path, damaged, baseline_provenance)
    lesion_digest = _sha(candidate_path.read_bytes())
    lesion_record["lesioned_checkpoint_sha256"] = lesion_digest
    lesion_record_path = OUTPUT_ROOT / "model-lesion.json"
    lesion_record_path.write_text(json.dumps(lesion_record, sort_keys=True, indent=2) + "\n", encoding="utf-8")

    request = _request(include_trial_controls=True)
    detect_request = _request(include_trial_controls=False)
    concrete_capture, capture_provenance, bound_value = _capture(manifest, request, damaged_latents)
    config = detection_config_from_manifest(detect_request, manifest)
    benign_negative = heldout_healthy * 1e-3
    controls = {
        MANIFEST_CONTROLS[0]: _value(heldout_healthy),
        MANIFEST_CONTROLS[1]: _value(benign_negative),
    }
    detections, detect_payload = evaluate_detection(bound_value, config, controls)
    family = detections[0]
    evidence = evidence_from_manifest(
        request,
        manifest,
        family_id=FAMILY,
        metric_id=METRIC_FEATURE_RATIO,
        affected_when="threshold_fail",
        control_outcomes=dict(family.control_outcomes),
        outcome=family.outcome,
        claim_allowed=family.claim_allowed,
        representation_identity=REPRESENTATION,
    )
    axes = manifest_axis_lookup(manifest)
    feature_axis = axes["feature"]
    feature_order = tuple(str(feature) for feature in FEATURE_IDS)
    feature_indices = {feature_id: index for index, feature_id in enumerate(feature_order)}
    metric_ratios = cast(
        Sequence[object],
        cast(Mapping[str, object], detect_payload["measurements"])["feature_variance_ratios"],
    )
    feature_cells = tuple(
        FeatureCell(
            feature_id=feature_id,
            feature_index=feature_indices[feature_id],
            metric_value=float(metric_ratios[feature_indices[feature_id]]),
            representation_identity=REPRESENTATION,
            axis_identity=str(feature_axis["identity"]),
        )
        for feature_id in feature_order
    )
    localization_source = AxialLocalizationInput(
        manifest_id=MANIFEST_ID,
        manifest_axes=axes,
        evidence=evidence,
        requested_axes=("feature",),
        feature_declaration=FeatureAxisDeclaration(
            feature_order=feature_order,
            feature_indices=feature_indices,
            axis_identity=str(feature_axis["identity"]),
            selection=str(feature_axis["selection"]),
            representation_identity=REPRESENTATION,
        ),
        feature_cells=feature_cells,
        global_score=float(family.observed_metrics[METRIC_FEATURE_RATIO]),
    )
    from latent_anything._layer_slice_localization import evaluate_axial_localization

    localized, _localize_payload = evaluate_axial_localization(localization_source)
    dataset = cast(Mapping[str, object], manifest["dataset"])
    hypothesis = ExplanationHypothesis(
        hypothesis_id=HYPOTHESIS_ID,
        symptom_id="symptom-model-weight-lesion-v3",
        family_id=FAMILY,
        target_id=FEATURE_IDS[0],
        representation_id=REPRESENTATION,
        layer_id="linear-autoencoder-bottleneck",
        slice_id="slice-all",
        method="probe",
        expected_direction="higher",
        dataset_id=str(dataset["split_identity"]),
        train_split_identity=TRAIN_SPLIT_ID,
        eval_split_identity=HELDOUT_SPLIT_ID,
        seeds=(42, 17),
        control_ids=tuple(_V1.PROBE_CONTROLS),
        metric_ids=(METRIC_FEATURE_RATIO,),
        thresholds=(
            ("heldout_accuracy", ">=", 0.7),
            ("coef_stability", ">=", 0.8),
            ("leakage_gap", ">=", 0.15),
        ),
        manifest_id=MANIFEST_ID,
        localization_bindings=(("feature", FEATURE_IDS[0]),),
    )
    probe = _probe_bundle(
        train_healthy,
        heldout_healthy,
        train_labels,
        heldout_labels,
        train_indices,
        heldout_indices,
    )
    detect_executor = make_detect_executor(bound_value, config, controls=controls)
    localize_executor = make_axial_localize_executor(localization_source)
    explain_executor = make_explain_executor((hypothesis,), MethodInputs(probe=probe))
    intervene_executor = make_intervene_executor(
        (_trial(),),
        _intervention_measure(baseline, damaged, heldout_images, heldout_labels),
    )
    comparison_specs = _comparison_specs(
        manifest,
        capture_provenance,
        detect_payload,
        baseline_digest,
        lesion_digest,
    )

    def compare_measure(application: ComparisonApplication) -> ComparisonMeasurement:
        side_model = baseline if application.side == "baseline" else damaged
        sample = heldout_images
        labels = heldout_labels
        if application.rng is not None:
            positions = application.rng.integers(0, len(heldout_images), size=len(heldout_images))
            sample = heldout_images[positions]
            labels = heldout_labels[positions]
        encoded = side_model.encode(sample)
        value = (
            _feature_ratio(encoded)
            if application.metric_role == "representation"
            else _task_accuracy(side_model, sample, labels)
        )
        return ComparisonMeasurement(application.run_id, application.metric_id, value, application.inputs_digest)

    compare_executor = make_compare_executor(comparison_specs, compare_measure)

    def capture_executor(invocation: StageInvocation) -> StageOutput:
        if invocation.stage != "capture":
            raise ValueError(f"capture executor received stage {invocation.stage!r}")
        return StageOutput(
            stage="capture",
            outcome="completed",
            payload={"capture": dict(concrete_capture.provenance())},
            artifact_refs=(),
        )

    def report_executor(invocation: StageInvocation) -> StageOutput:
        if invocation.stage != "report":
            raise ValueError(f"report executor received stage {invocation.stage!r}")
        return StageOutput(stage="report", outcome="completed", payload={"report_id": REPORT_ID}, artifact_refs=())

    executors = {
        "capture": capture_executor,
        "detect": detect_executor,
        "localize": localize_executor,
        "explain": explain_executor,
        "intervene": intervene_executor,
        "compare": compare_executor,
        "report": report_executor,
    }
    versions = {stage: "stage-v1" for stage in WORKFLOW_STAGES}
    for stage, executor, attribute in (
        ("detect", detect_executor, "detector_version"),
        ("localize", localize_executor, "localizer_version"),
        ("explain", explain_executor, "explainer_version"),
        ("intervene", intervene_executor, "intervention_version"),
        ("compare", compare_executor, "comparison_version"),
    ):
        versions[stage] = str(getattr(executor, attribute))
    workflow = DiagnosticWorkflow(executors, versions)
    result, checkpoint = workflow.run(request, manifest)
    if result.status != "completed" or checkpoint.status != "completed":
        failure = checkpoint.failure
        record = {
            "acceptance": "not_passed",
            "checkpoint_status": checkpoint.status,
            "error_type": failure.error_type if failure else None,
            "failed_stage": checkpoint.next_stage,
            "manifest_digest": manifest_hash,
            "manifest_raw_sha256": manifest_raw_sha,
            "message": failure.message if failure else result.message,
            "workflow_status": result.status,
        }
        _write_run_record(OUTPUT_ROOT, record)
        return record

    stage_payloads = {output.stage: dict(output.payload) for output in checkpoint.outputs}
    trial_records = cast(Sequence[Mapping[str, object]], stage_payloads["intervene"]["trials"])
    comparison_records = cast(Sequence[Mapping[str, object]], stage_payloads["compare"]["comparisons"])
    report = _build_report(checkpoint.outputs, manifest=manifest)
    handle = persist_diagnostic_artifact(
        OUTPUT_ROOT,
        request=request,
        manifest=manifest,
        result=result,
        checkpoint=checkpoint,
        report=report,
    )
    loaded = load_diagnostic_artifact(OUTPUT_ROOT, run_id=handle.run_id, manifest=manifest)
    validator_input = cast(Mapping[str, object], loaded.document["validator_input"])
    validate_diagnostic_report(
        loaded.report,
        manifest,
        applicability=cast(Mapping[str, str], validator_input["applicability"]),
        family_evidence=cast(Mapping[str, Mapping[str, str]], validator_input["family_evidence"]),
        control_outcomes=cast(Mapping[str, str], validator_input["control_outcomes"]),
        artifacts=registered_artifact_bytes(loaded, OUTPUT_ROOT),
        artifact_digests=cast(Mapping[str, str], validator_input["artifact_digests"]),
    )
    rendered = render_diagnostic_report(OUTPUT_ROOT, run_id=handle.run_id, manifest=manifest)
    rendered_replay = render_diagnostic_report(OUTPUT_ROOT, run_id=handle.run_id, manifest=manifest)
    deterministic_render = rendered == rendered_replay
    render_handle = persist_rendered_report(
        OUTPUT_ROOT,
        run_id=handle.run_id,
        manifest=manifest,
        output=request.output,
    )
    report_path = Path(render_handle.declared_path)
    if not report_path.is_absolute():
        report_path = REPO / report_path
    report_bytes = report_path.read_bytes()
    rendered_sha = _sha(rendered)
    persisted_render_sha = _sha(report_bytes)
    final_loaded = load_diagnostic_artifact(OUTPUT_ROOT, run_id=handle.run_id, manifest=manifest)
    final_validator_input = cast(Mapping[str, object], final_loaded.document["validator_input"])
    validate_diagnostic_report(
        final_loaded.report,
        manifest,
        applicability=cast(Mapping[str, str], final_validator_input["applicability"]),
        family_evidence=cast(Mapping[str, Mapping[str, str]], final_validator_input["family_evidence"]),
        control_outcomes=cast(Mapping[str, str], final_validator_input["control_outcomes"]),
        artifacts=registered_artifact_bytes(final_loaded, OUTPUT_ROOT),
        artifact_digests=cast(Mapping[str, str], final_validator_input["artifact_digests"]),
    )
    explanation_record = cast(Mapping[str, object], stage_payloads["explain"]["evidence"][0])
    detection_supported = family.outcome == "supported" and bool(family.claim_allowed)
    localization_supported = localized.verdict == "localized" and tuple(localized.axes[0].affected) == (FEATURE_IDS[0],)
    explanation_supported = explanation_record.get("outcome") == "supported" and bool(
        explanation_record.get("claim_allowed")
    )
    intervention_supported = bool(trial_records) and trial_records[0].get("conclusion") == "supported"
    task = cast(Mapping[str, object], comparison_records[0]["task"])
    comparison_supported = (
        comparison_records[0].get("classification") not in ("inconclusive", "unsupported", "neither")
        and bool(task.get("changed"))
        and float(task.get("signed_delta", 0.0)) < -0.05
    )
    controls_passed = not (set(family.control_outcomes.values()) - {"passed"})
    validator_passed = final_loaded.document["validator_result"]["status"] == "passed"
    acceptance_passed = all(
        (
            result.status == "completed",
            detection_supported,
            localization_supported,
            explanation_supported,
            intervention_supported,
            comparison_supported,
            controls_passed,
            validator_passed,
            deterministic_render,
            rendered_sha == persisted_render_sha,
        )
    )
    record: dict[str, object] = {
        "acceptance": "passed" if acceptance_passed else "not_passed",
        "manifest_id": MANIFEST_ID,
        "manifest_digest": manifest_hash,
        "manifest_raw_sha256": manifest_raw_sha,
        "manifest_raw_unchanged_during_run": MANIFEST_PATH.read_bytes() == manifest_raw,
        "baseline_checkpoint_sha256": baseline_digest,
        "lesioned_checkpoint_sha256": lesion_digest,
        "lesion_record_sha256": _sha(lesion_record_path.read_bytes()),
        "lesion_boundary": "encoder row 0 and bias zeroed before heldout encoding",
        "detection": {
            "outcome": family.outcome,
            "claim_allowed": family.claim_allowed,
            "observed_metrics": dict(family.observed_metrics),
            "threshold_pass": dict(family.threshold_pass),
            "control_outcomes": dict(family.control_outcomes),
        },
        "localization": localized.to_dict(),
        "explanation": {
            "outcome": explanation_record.get("outcome"),
            "claim_allowed": explanation_record.get("claim_allowed"),
            "reason": explanation_record.get("reason"),
            "fidelity": explanation_record.get("fidelity"),
            "stability": explanation_record.get("stability"),
            "selectivity": explanation_record.get("selectivity"),
            "leakage": explanation_record.get("leakage"),
            "control_outcomes": explanation_record.get("control_outcomes"),
        },
        "intervention": dict(trial_records[0]),
        "comparison": dict(comparison_records[0]),
        "workflow_status": result.status,
        "workflow_stages": list(result.completed_stages),
        "artifact": {
            "run_id": handle.run_id,
            "artifact_digest": handle.artifact_digest,
            "report_digest": handle.report_digest,
            "independent_validator": final_loaded.document["validator_result"],
            "render_sha256": rendered_sha,
            "persisted_render_sha256": persisted_render_sha,
            "render_deterministic": deterministic_render,
            "rendered_report_path": str(report_path.relative_to(REPO)),
            "rendered_report_artifact": render_handle.to_dict(),
        },
        "output_root": str(OUTPUT_ROOT.relative_to(REPO)),
        "proof_started_at": _RUN_STARTED_AT,
        "proof_completed_at": datetime.now(UTC).isoformat(),
    }
    _write_run_record(OUTPUT_ROOT, record)
    return record


_FROZEN_MANIFEST: dict[str, object] = {}
_RUN_STARTED_AT = ""


def main() -> int:
    global _FROZEN_MANIFEST, _RUN_STARTED_AT
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--prepare-model", action="store_true", help="fit/save the train-only baseline checkpoint")
    modes.add_argument("--run", action="store_true", help="execute the frozen end-to-end proof")
    modes.add_argument("--validate-manifest", action="store_true", help="validate frozen inputs without proof outputs")
    arguments = parser.parse_args()
    if arguments.prepare_model:
        output = _prepare_model()
    elif arguments.validate_manifest:
        output = _manifest_validation_record()
    else:
        _FROZEN_MANIFEST, _baseline, _provenance = _validate_frozen_inputs()
        _RUN_STARTED_AT = datetime.now(UTC).isoformat()
        output = _run()
    print(json.dumps(output, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

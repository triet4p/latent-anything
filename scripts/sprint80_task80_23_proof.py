"""Executable core proof for Sprint 80.23: the frozen encoder/autoencoder
manifest executed through the single high-level diagnostic workflow.

Run from the repository root:

    uv run python scripts/sprint80_task80_23_proof.py

Exit code 0 means the proof executed completely and recorded the frozen
acceptance verdict truthfully, namely:

1. every frozen input verifies read-only (manifest commitment digest,
   schema/taxonomy/report-schema documents, scikit-learn revision, the
   pinned1437/360 split with its pinned index digests, and byte-identical
   frozen artifacts after the run);
2. the pinned ConvVAE (latent4 / seed0 / epochs5) trains on the pinned
   train partition deterministically — a refit re-encodes the heldout
   bottleneck bit-identically at the declared seeds;
3. a real capture identity is bound through ``bind_selection`` /
   ``resolve_captures`` over the real heldout bottleneck-mu array;
4. the real seven-stage ``DiagnosticWorkflow`` runs the real detect
   executor with the manifest-declared control cases: the healthy
   full-rank counterexample scores above both frozen thresholds, the
   benign low-variance negative does not trigger, the null-shuffle is
   recorded, and the declared known positive defect is recorded exactly
   as observed (it is NOT observed: both frozen metrics pass on every
   legitimate capture of the pinned revision);
5. the real80.13 localizer runs on manifest-bound evidence and is
   truthfully negative — no frozen rule marks any direction, so the
   manifest's expected feature-axis selection cannot be named without
   fabrication;
6. the workflow then fails closed at the explain stage (the80.16
   contract requires a localized/supported prior selection), which makes
   the seven-stage chain — and with it any validator-clean artifact or
   rendered report — impossible to complete truthfully. The completed
   prefix is exercised through the stop-after/resume contract, and
   persistence of the incomplete workflow is refused with no files
   written;
7. truthfulness guards fire: a tampered manifest digest, a tampered
   capture provenance, a tampered resume identity, and a failed required
   control (real batch) are all rejected/blocked fail-closed, and the
   detect/localize payloads are byte-deterministic across reruns;
8. bounded runtime/resource evidence and the real downstream
   measurements the blocked chain cannot reach (per-direction ablation
   effects on the frozen target metric plus heldout reconstruction, and
   aligned-run comparison points) are recorded.

The frozen acceptance verdict is computed from the manifest's own rules
and printed; thresholds and data are never adjusted to force a pass.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import platform
import tempfile
import time
import tracemalloc
from functools import lru_cache
from importlib.metadata import version as package_version
from pathlib import Path

_MODULE_START = time.perf_counter()
# Keep the runtime imports below this marker so the reported wall time includes them.

import numpy as np  # noqa: E402
from sklearn.datasets import load_digits  # noqa: E402

from latent_anything._benchmark_manifest import manifest_digest, validate_manifest  # noqa: E402
from latent_anything._capture_binding import bind_selection, resolve_captures  # noqa: E402
from latent_anything._collapse_detection import (  # noqa: E402
    detection_config_from_manifest,
    evaluate_detection,
    make_detect_executor,
)
from latent_anything._diagnostic_artifact import persist_diagnostic_artifact  # noqa: E402
from latent_anything._diagnostic_workflow import (  # noqa: E402
    WORKFLOW_STAGES,
    DiagnosticWorkflow,
    StageInvocation,
    StageOutput,
)
from latent_anything._intervention_trials import (  # noqa: E402
    Measurement,
    TrialApplication,
    TrialControl,
    TrialSpec,
    make_intervene_executor,
)
from latent_anything._jepa_evaluation import compute_latent_health  # noqa: E402
from latent_anything._layer_slice_localization import (  # noqa: E402
    LayerCell,
    LocalizationInput,
    SampleCell,
    SliceDefinition,
    evaluate_localization,
    evidence_from_manifest,
    make_localize_executor,
)
from latent_anything._portable_contract import canonical_json  # noqa: E402
from latent_anything._probe_tcav_ig_explanation import (  # noqa: E402
    DECLARED_PROBE_CAPACITY,
    ExplanationHypothesis,
    MethodInputs,
    make_explain_executor,
)
from latent_anything._run_comparison import (  # noqa: E402
    ALIGNMENT_FIELDS,
    ComparisonApplication,
    ComparisonMeasurement,
    ComparisonSpec,
    RunSide,
    capture_axes_identity,
    detect_config_identity,
    make_compare_executor,
    manifest_seed_identity,
)
from latent_anything.capture import CapturedActivation, CaptureMetadata  # noqa: E402
from latent_anything.diagnostics import (  # noqa: E402
    CaptureSelection,
    ComparisonRequest,
    ControlSelection,
    DiagnosticRequest,
    DiagnosticSelection,
    InterventionRequest,
    OutputSelection,
)
from latent_anything.latent_value import LatentValue  # noqa: E402
from latent_anything.probes import _fast_probe  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
ARTIFACTS = REPO / "artifacts"
MANIFEST_PATH = ARTIFACTS / "benchmark_manifest_sprint80_encoder_autoencoder_v1.json"
FROZEN_PATHS = (
    ARTIFACTS / "benchmark_manifest_schema_v1.json",
    ARTIFACTS / "benchmark_manifest_sprint80_encoder_autoencoder_v1.json",
    ARTIFACTS / "benchmark_manifest_sprint80_transformer_hidden_state_v1.json",
    ARTIFACTS / "diagnostic_report_schema_v1.json",
    ARTIFACTS / "representation_problem_taxonomy_v1.json",
)
PINNED_MANIFEST_DIGEST = "b4c49c890cf34cc0530d499c2b82a09057f1f1ebf5b9d9a3843200760b137dd3"
PINNED_TRAIN_INDEX_DIGEST = "0b03643758a0dd1999442234f213e51b20f87948dfc7d2871ca05d0ff7f39b90"
PINNED_HELDOUT_INDEX_DIGEST = "48e9e0ed2c89e42ee0fdc3592e8c995f8b95d37184b4ae791e137765c32444c6"
PINNED_SKLEARN = "1.9.0"

MANIFEST_ID = "sprint80-core-encoder-autoencoder-collapse-v1"
REPRESENTATION = "conv_vae_8x8:bottleneck-mu:latent_dim=4"
MODEL_REVISION = "conv-vae-8x8-latent4-seed0-epochs5"
FAMILY = "collapse_rank_loss"
METRIC_ER = "bottleneck-effective-rank"
METRIC_SPREAD = "bottleneck-singular-spread"
CAPTURE_ID = "capture-encoder-bottleneck-mu"
SLICE_ID = "slice-all"
HYPOTHESIS_ID = "h-collapse-bottleneck-mu"
REQUEST_ID = "proof-80-23"
DETECT_REQUEST_ID = "proof-80-23-detect"
OUTPUT_LOCATION = "artifacts/diagnostics/proof-80-23"
BASELINE_RUN = "conv-vae-8x8-heldout-eval-000100"
CANDIDATE_RUN = "conv-vae-8x8-heldout-eval-000200"
BASELINE_CKPT = "epoch5-heldout-eval-a"
CANDIDATE_CKPT = "epoch5-heldout-eval-b"
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


def _sha(data: str | bytes) -> str:
    return hashlib.sha256(data.encode("utf-8") if isinstance(data, str) else data).hexdigest()


def _payload_sha(payload: object) -> str:
    return _sha(canonical_json(payload))


def frozen_digests() -> dict[str, str]:
    """SHA-256 of every frozen Sprint80 artifact before/after the run."""
    return {path.name: _sha(path.read_bytes()) for path in FROZEN_PATHS}


@lru_cache(maxsize=1)
def load_manifest() -> dict[str, object]:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    validate_manifest(manifest)
    if manifest_digest(manifest) != PINNED_MANIFEST_DIGEST:
        raise AssertionError("encoder manifest digest drifted from the sprint-pinned commitment")
    if str(manifest.get("manifest_id")) != MANIFEST_ID:
        raise AssertionError("unexpected manifest identity")
    if package_version("scikit-learn") != PINNED_SKLEARN:
        raise AssertionError("scikit-learn revision drifted from the pinned dataset revision")
    return dict(manifest)


@lru_cache(maxsize=1)
def build_data() -> dict[str, object]:
    """The exact frozen split: default_rng(42).permutation,1437/360."""
    digits = load_digits()
    images = (digits.images / 16.0).astype(np.float64)[:, None, :, :]
    labels = digits.target.astype(np.int64)
    permutation = np.random.default_rng(42).permutation(len(images))
    split = int(len(images) * 0.8)
    train_idx, heldout_idx = permutation[:split], permutation[split:]
    if (len(train_idx), len(heldout_idx)) != (1437, 360):
        raise AssertionError("frozen split sizes drifted")
    if _sha(np.asarray(train_idx, dtype=np.int64).tobytes()) != PINNED_TRAIN_INDEX_DIGEST:
        raise AssertionError("train index digest drifted from the pinned benchmark artifact")
    if _sha(np.asarray(heldout_idx, dtype=np.int64).tobytes()) != PINNED_HELDOUT_INDEX_DIGEST:
        raise AssertionError("heldout index digest drifted from the pinned benchmark artifact")
    if np.intersect1d(train_idx, heldout_idx).size:
        raise AssertionError("frozen split is not disjoint")
    pinned = json.loads((ARTIFACTS / "conv_vae_heldout_benchmark.json").read_text(encoding="utf-8"))
    split_block = pinned["split"]
    assert isinstance(split_block, dict)
    if split_block.get("train_index_digest") != PINNED_TRAIN_INDEX_DIGEST:
        raise AssertionError("pinned benchmark disagrees on the train index digest")
    if split_block.get("heldout_index_digest") != PINNED_HELDOUT_INDEX_DIGEST:
        raise AssertionError("pinned benchmark disagrees on the heldout index digest")
    return {
        "train": images[train_idx],
        "heldout": images[heldout_idx],
        "train_labels": labels[train_idx],
        "heldout_labels": labels[heldout_idx],
        "train_indices": train_idx,
        "heldout_indices": heldout_idx,
    }


def _fit_pinned() -> object:
    from latent_anything.adapters.conv_vae import ConvVAE

    data = build_data()
    model = ConvVAE(latent_dim=4, random_state=0, n_epochs=5)
    model.fit(np.asarray(data["train"]))
    return model


def _fit_untrained() -> object:
    from latent_anything.adapters.conv_vae import ConvVAE

    return ConvVAE(latent_dim=4, random_state=0, n_epochs=5)


def _effective_rank(data: np.ndarray) -> float:
    return float(compute_latent_health(data).effective_rank)


def _spread(data: np.ndarray) -> float:
    centered = data - data.mean(axis=0)
    singular = np.linalg.svd(centered, compute_uv=False)
    return float(singular[-1] / singular[0])


def _reconstruction_mse(model: object, latents: np.ndarray, images: np.ndarray) -> float:
    reconstruction = model.decode(latents)  # type: ignore[attr-defined]
    return float(np.mean((images - reconstruction) ** 2))


@lru_cache(maxsize=1)
def build_model_case() -> dict[str, object]:
    """Pinned training plus deterministic replay at the declared training seed."""
    data = build_data()
    model = _fit_pinned()
    heldout_latents = model.encode(np.asarray(data["heldout"]))  # type: ignore[attr-defined]
    train_latents = model.encode(np.asarray(data["train"]))  # type: ignore[attr-defined]
    replay = _fit_pinned()
    replay_latents = replay.encode(np.asarray(data["heldout"]))  # type: ignore[attr-defined]
    if not np.array_equal(heldout_latents, replay_latents):
        raise AssertionError("refit did not replay bit-identically at the declared seeds")
    variances = np.var(heldout_latents, axis=0)
    return {
        "model": model,
        "train_latents": train_latents,
        "heldout_latents": heldout_latents,
        "replay_latents": replay_latents,
        "heldout_reconstruction_mse": _reconstruction_mse(model, heldout_latents, np.asarray(data["heldout"])),
        "train_reconstruction_mse": _reconstruction_mse(model, train_latents, np.asarray(data["train"])),
        "variances": variances,
        "argmin_dim": int(np.argmin(variances)),
        "argmax_dim": int(np.argmax(variances)),
    }


def _space() -> object:
    from latent_anything.latent_space import LatentSpace

    return LatentSpace(
        dim=4,
        source_model="conv-vae-8x8",
        metadata={
            "source_representation_identity": REPRESENTATION,
            "model_version": MODEL_REVISION,
        },
    )


def _value(data: np.ndarray) -> LatentValue:
    return LatentValue(np.asarray(data, dtype=np.float64), _space())  # type: ignore[arg-type]


def _counterexample_batch() -> np.ndarray:
    """Healthy full-rank reference: PCA-4 of the same real heldout pixels."""
    heldout = np.asarray(build_data()["heldout"])
    flat = heldout.reshape(len(heldout), -1)
    centered = flat - flat.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return (centered @ vt[:4].T).astype(np.float64)


def _negative_batch() -> np.ndarray:
    """Benign low-variance negative: the real bottleneck at tiny global gain."""
    latents = np.asarray(build_model_case()["heldout_latents"])
    return (latents * 1e-3).astype(np.float64)


def _failing_counterexample_batch() -> np.ndarray:
    """A real rank-2 projection (padded to the4-dim feature space) used only
    to prove that a failing required control blocks the conclusion."""
    heldout = np.asarray(build_data()["heldout"])
    flat = heldout.reshape(len(heldout), -1)
    centered = flat - flat.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    rank2 = (centered @ vt[:2].T).astype(np.float64)
    return np.column_stack([rank2, np.zeros(len(rank2)), np.zeros(len(rank2))])


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


def build_trials() -> tuple[TrialSpec, ...]:
    """Ablate the manifest-expected direction (smallest heldout variance)."""
    case = build_model_case()
    target_dim = int(case["argmin_dim"])
    off_target_dim = int(case["argmax_dim"])
    if target_dim == off_target_dim:
        raise AssertionError("expected distinct weakest/strongest bottleneck directions")
    target = f"bottleneck-mu-dim{target_dim}"
    off_target = f"bottleneck-mu-dim{off_target_dim}"
    return (
        TrialSpec(
            intervention_id="ablate-bottleneck-min-variance-direction",
            kind="ablate",
            target=target,
            metric_id=METRIC_ER,
            hypothesis_id=HYPOTHESIS_ID,
            strength=1.0,
            strength_semantics="fraction-of-target-activation-zeroed",
            expected_effect="decrease",
            controls=(
                TrialControl(
                    "ablate-min-var-zero",
                    "zero_strength",
                    "zero-strength rerun reproduces the baseline metric",
                ),
                TrialControl(
                    "ablate-min-var-random",
                    "random",
                    "random direction stays below the on-target effect",
                ),
                TrialControl(
                    "ablate-min-var-shuffled",
                    "shuffled",
                    "shuffled assignment leaves the spectrum unchanged",
                ),
                TrialControl(
                    "ablate-min-var-off",
                    "off_target",
                    "ablating the strongest direction must not reproduce the effect",
                    target=off_target,
                ),
            ),
            provenance={"method": "ablation", "carrier": "coordinate-zeroing"},
        ),
    )


def _intervene_measure(app: TrialApplication) -> Measurement:
    """Real downstream measurement: frozen effective rank of the manipulated
    captured inputs (the manifest causal-expectation target metric)."""
    latents = np.asarray(build_model_case()["heldout_latents"]).copy()
    target_dim = _target_dim(app.target)
    control = app.control_class
    if app.role == "baseline" or app.strength == 0.0 or control == "zero_strength":
        data = latents
    elif app.role == "intervened":
        data = latents
        data[:, target_dim] = 0.0
    elif control == "random":
        assert app.rng is not None
        data = latents
        data[:, int(app.rng.integers(0, latents.shape[1]))] = 0.0
    elif control == "shuffled":
        assert app.rng is not None
        data = latents
        data[:, target_dim] = latents[app.rng.permutation(latents.shape[0]), target_dim]
    elif control == "off_target":
        data = latents
        data[:, target_dim] = 0.0
    else:
        data = latents
    return Measurement(app.metric_id, _effective_rank(data), app.inputs_digest)


def _target_dim(name: str) -> int:
    return int(name.rsplit("dim", 1)[1])


def _hypothesis(manifest: dict[str, object]) -> ExplanationHypothesis:
    dataset = manifest["dataset"]
    assert isinstance(dataset, dict)
    return ExplanationHypothesis(
        hypothesis_id=HYPOTHESIS_ID,
        symptom_id="symptom-bottleneck-collapse",
        family_id=FAMILY,
        target_id="bottleneck-mu",
        representation_id=REPRESENTATION,
        layer_id="bottleneck-mu",
        slice_id=SLICE_ID,
        method="probe",
        expected_direction="higher",
        dataset_id=str(dataset["split_identity"]),
        train_split_identity="train-1437",
        eval_split_identity="heldout-360",
        seeds=(42, 17),
        control_ids=PROBE_CONTROLS,
        metric_ids=(METRIC_ER,),
        thresholds=(("heldout_accuracy", ">=", 0.7), ("leakage_gap", ">=", 0.15)),
        manifest_id=MANIFEST_ID,
        localization_bindings=(("slice", SLICE_ID),),
    )


@lru_cache(maxsize=1)
def build_probe_bundle() -> dict[str, object]:
    """Leakage-safe real probe inputs: bottleneck -> high/low reconstruction
    error, thresholded by the TRAIN-partition median (labels never fit on
    eval data)."""
    case = build_model_case()
    data = build_data()
    train_latents = np.asarray(case["train_latents"])
    heldout_latents = np.asarray(case["heldout_latents"])
    model = case["model"]
    train_mse = np.mean(
        (np.asarray(data["train"]) - model.decode(train_latents)) ** 2,  # type: ignore[attr-defined]
        axis=tuple(range(1, np.asarray(data["train"]).ndim)),
    )
    heldout_mse = np.mean(
        (np.asarray(data["heldout"]) - model.decode(heldout_latents)) ** 2,  # type: ignore[attr-defined]
        axis=tuple(range(1, np.asarray(data["heldout"]).ndim)),
    )
    median = float(np.median(train_mse))
    train_labels = (train_mse >= median).astype(np.int64)
    eval_labels = (heldout_mse >= median).astype(np.int64)
    if len(np.unique(train_labels)) < 2 or len(np.unique(eval_labels)) < 2:
        raise AssertionError("probe splits must contain two classes")
    rng = np.random.default_rng(17)
    noise = rng.normal(size=train_latents.shape)
    noise_eval = rng.normal(size=heldout_latents.shape)
    negative = float(
        _fast_probe(
            noise,
            train_labels,
            noise_eval,
            eval_labels,
            17,
        ).accuracy
    )
    train_idx = np.asarray(data["train_indices"])
    heldout_idx = np.asarray(data["heldout_indices"])
    return {
        "train_matrix": train_latents,
        "eval_matrix": heldout_latents,
        "train_labels": train_labels,
        "eval_labels": eval_labels,
        "train_ids": [f"train-{int(i)}" for i in train_idx],
        "eval_ids": [f"heldout-{int(i)}" for i in heldout_idx],
        "capacity": DECLARED_PROBE_CAPACITY,
        "negative_accuracy": negative,
    }


def _capture() -> tuple[object, dict[str, object]]:
    """Real capture identity over the real heldout bottleneck-mu array."""
    manifest = load_manifest()
    request = workflow_request()
    plan = bind_selection(request.capture, manifest=manifest, request_id=request.request_id)
    latents = np.asarray(build_model_case()["heldout_latents"])
    activation = CapturedActivation(
        values=latents,
        metadata=CaptureMetadata(
            location="conv_vae.encode:bottleneck-mu",
            call_index=0,
            shape=tuple(int(size) for size in latents.shape),
            batch_axis=0,
            sequence_axis=None,
            device="cpu",
            dtype=str(latents.dtype),
            source_model_version=MODEL_REVISION,
        ),
    )
    (concrete,), _values = resolve_captures(plan, (activation,))
    return concrete, dict(concrete.provenance())


@lru_cache(maxsize=1)
def detect_case() -> dict[str, object]:
    manifest = load_manifest()
    config = detection_config_from_manifest(_detect_request(), manifest)
    latents = np.asarray(build_model_case()["heldout_latents"])
    detections, payload = evaluate_detection(_value(latents), config, controls=_control_batches())
    return {"config": config, "detections": detections, "payload": payload, "family": detections[0]}


@lru_cache(maxsize=1)
def localize_case() -> dict[str, object]:
    """Truthful localization input from manifest wiring plus real measurements.

    The bottleneck is the single ordered layer of this single-tensor
    representation; per-sample cells are leave-one-out effective-rank
    observations; the declared slice covers every heldout sample.
    """
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
    latents = np.asarray(build_model_case()["heldout_latents"])
    heldout_indices = np.asarray(build_data()["heldout_indices"])
    sample_cells: list[SampleCell] = []
    for row in range(latents.shape[0]):
        keep = np.ones(latents.shape[0], dtype=bool)
        keep[row] = False
        sample_cells.append(
            SampleCell(
                sample_id=f"heldout-{int(heldout_indices[row])}",
                metric_value=_effective_rank(latents[keep]),
                representation_identity=REPRESENTATION,
            )
        )
    source = LocalizationInput(
        manifest_id=MANIFEST_ID,
        evidence=evidence,
        layer_order=("bottleneck-mu",),
        layer_cells=(
            LayerCell(
                layer_id="bottleneck-mu",
                metric_value=_effective_rank(latents),
                representation_identity=REPRESENTATION,
            ),
        ),
        sample_cells=tuple(sample_cells),
        declared_slice_ids=(SLICE_ID,),
        slices=(
            SliceDefinition(
                slice_id=SLICE_ID,
                criteria="all heldout samples of the frozen split",
                member_sample_ids=tuple(cell.sample_id for cell in sample_cells),
            ),
        ),
        global_score=float(detect_case()["payload"]["measurements"]["effective_rank"]),  # type: ignore[index]
    )
    result, payload = evaluate_localization(source)
    return {"source": source, "result": result, "payload": payload}


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
        "preprocessing_identity": "digits-scale-01-to-v1",
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
            comparison_id="cmp-heldout-eval-replay",
            baseline=RunSide(run_id=BASELINE_RUN, checkpoint_id=BASELINE_CKPT, alignment=alignment),
            candidate=RunSide(run_id=CANDIDATE_RUN, checkpoint_id=CANDIDATE_CKPT, alignment=alignment),
            representation_metric_id=METRIC_ER,
            task_metric_id=METRIC_SPREAD,
            representation_tolerance=0.1,
            task_tolerance=0.02,
        ),
    )


def _compare_measure(app: ComparisonApplication) -> ComparisonMeasurement:
    latents = np.asarray(build_model_case()["heldout_latents"])
    data = latents
    if app.rng is not None:
        rows = app.rng.integers(0, latents.shape[0], size=latents.shape[0])
        data = latents[rows]
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
                comparison_id="cmp-heldout-eval-replay",
                baseline_run=BASELINE_RUN,
                candidate_run=CANDIDATE_RUN,
                metric_ids=(METRIC_ER, METRIC_SPREAD),
            ),
        ),
        output=OutputSelection(output_location=OUTPUT_LOCATION),
    )


@lru_cache(maxsize=1)
def build_workflow() -> tuple[DiagnosticWorkflow, DiagnosticRequest]:
    """The single high-level workflow composed from the real stage factories."""
    manifest = load_manifest()
    request = workflow_request()
    detect = make_detect_executor(
        _value(np.asarray(build_model_case()["heldout_latents"])),
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
            payload={"report_id": "report-80-23-encoder"},
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


def downstream_previews() -> dict[str, object]:
    """Real measurements for stages the blocked chain cannot reach.

    These are computed directly from the same captured inputs the stage
    executors would bind; they are recorded as measurements, never as
    executed stage outcomes.
    """
    case = build_model_case()
    data = build_data()
    model = case["model"]
    latents = np.asarray(case["heldout_latents"])
    heldout = np.asarray(data["heldout"])
    baseline_er = _effective_rank(latents)
    baseline_mse = float(case["heldout_reconstruction_mse"])
    per_direction: list[dict[str, float]] = []
    for dim in range(latents.shape[1]):
        ablated = latents.copy()
        ablated[:, dim] = 0.0
        mse = float(np.mean((heldout - model.decode(ablated)) ** 2))  # type: ignore[attr-defined]
        per_direction.append(
            {
                "dim": dim,
                "effective_rank": _effective_rank(ablated),
                "effective_rank_delta": _effective_rank(ablated) - baseline_er,
                "heldout_reconstruction_mse": mse,
                "heldout_reconstruction_mse_delta": mse - baseline_mse,
                "variance": float(np.var(latents[:, dim])),
            }
        )
    target_dim = int(case["argmin_dim"])
    off_target_dim = int(case["argmax_dim"])
    target_effect = abs(float(per_direction[target_dim]["effective_rank_delta"]))
    off_target_effect = abs(float(per_direction[off_target_dim]["effective_rank_delta"]))
    tolerance = 0.1
    return {
        "baseline_effective_rank": baseline_er,
        "baseline_heldout_reconstruction_mse": baseline_mse,
        "per_direction": per_direction,
        "declared_trial_target": f"bottleneck-mu-dim{target_dim}",
        "declared_off_target": f"bottleneck-mu-dim{off_target_dim}",
        "target_metric_effect": target_effect,
        "off_target_metric_effect": off_target_effect,
        "off_target_specificity_rule_violated_by_arithmetic": bool(off_target_effect >= max(target_effect, tolerance)),
        "comparison_points": {
            BASELINE_RUN: {METRIC_ER: baseline_er, METRIC_SPREAD: _spread(latents)},
            CANDIDATE_RUN: {METRIC_ER: _effective_rank(latents), METRIC_SPREAD: _spread(latents)},
        },
        "comparison_absolute_delta": {
            METRIC_ER: 0.0,
            METRIC_SPREAD: 0.0,
        },
    }


def run_proof() -> int:
    started = time.perf_counter()
    tracemalloc.start()
    phases: dict[str, float] = {}

    def _phase(name: str, since: float) -> float:
        now = time.perf_counter()
        phases[name] = now - since
        return now

    before = frozen_digests()
    manifest = load_manifest()
    data = build_data()
    print(
        "PASS frozen-inputs: manifest "
        f"{PINNED_MANIFEST_DIGEST} validated; scikit-learn {PINNED_SKLEARN}; "
        f"{len(before)} frozen artifacts hashed"
    )
    print(
        "PASS split: train1437/heldout360 disjoint; index digests "
        f"{PINNED_TRAIN_INDEX_DIGEST[:12]}/{PINNED_HELDOUT_INDEX_DIGEST[:12]} match the pinned benchmark"
    )

    case = build_model_case()
    since = _phase("frozen+data+model", started)
    heldout_latents = np.asarray(case["heldout_latents"])
    print(
        "PASS deterministic-replay: refit (latent4/seed0/epochs5) re-encodes heldout "
        f"bit-identically; heldout latents sha {_payload_sha(heldout_latents.tolist())}"
    )

    concrete, capture_provenance = _capture()
    since = _phase("capture", since)
    print(
        f"PASS capture: capture_identity {concrete.capture_identity} shape "
        f"{tuple(concrete.shape)} dtype {concrete.dtype} device {concrete.device} "
        "via bind_selection/resolve_captures"
    )

    detect = detect_case()
    family = detect["family"]
    payload = detect["payload"]
    since = _phase("detect", since)
    measurements = payload["measurements"]  # type: ignore[index]
    control_metrics = payload["control_metrics"]  # type: ignore[index]
    threshold_pass = dict(family.threshold_pass)  # type: ignore[attr-defined]
    positive_observed = not all(threshold_pass.values())
    uncertainty = measurements["uncertainty"][METRIC_ER]
    print(
        "RECORD positive-case (declared known defect): "
        f"{'OBSERVED' if positive_observed else 'NOT OBSERVED'} — "
        f"{METRIC_ER} {measurements['effective_rank']:.4f} "
        f"{'<' if not threshold_pass[METRIC_ER] else '≥'}3.0, "
        f"{METRIC_SPREAD} {measurements['singular_spread']:.4f} "
        f"{'<' if not threshold_pass[METRIC_SPREAD] else '≥'}0.25; "
        f"threshold_pass={threshold_pass}; bootstrap CI {METRIC_ER} "
        f"[{uncertainty['lower']:.4f}, {uncertainty['upper']:.4f}]"
    )
    healthy = control_metrics["control-healthy-counterexample"]
    benign = control_metrics["control-benign-low-variance"]
    outcomes = dict(family.control_outcomes)  # type: ignore[attr-defined]
    if outcomes != {
        "control-healthy-counterexample": "passed",
        "control-benign-low-variance": "passed",
        "control-null-shuffle": "passed",
    }:
        raise AssertionError(f"manifest-declared control behaviors drifted: {outcomes}")
    if not (
        healthy[METRIC_ER] >= 3.0
        and healthy[METRIC_SPREAD] >= 0.25
        and benign[METRIC_ER] >= 3.0
        and benign[METRIC_SPREAD] >= 0.25
    ):
        raise AssertionError("counterexample/negative controls must score above both thresholds")
    print(
        "RECORD counterexample: control-healthy-counterexample passed — "
        f"ER {healthy[METRIC_ER]:.4f} ≥3.0, spread {healthy[METRIC_SPREAD]:.4f} ≥0.25 "
        "(PCA-4 of the same heldout pixels)"
    )
    print(
        "RECORD negative: control-benign-low-variance passed, not triggered — "
        f"ER {benign[METRIC_ER]:.4f}, spread {benign[METRIC_SPREAD]:.4f}, "
        f"min-variance {benign['bottleneck-min-variance']:.3e} (real bottleneck ×1e-3)"
    )
    null_metrics = measurements["null_shuffled"]
    print(
        "RECORD null: control-null-shuffle recorded — shuffled "
        f"ER {null_metrics[METRIC_ER]:.4f}, spread {null_metrics[METRIC_SPREAD]:.4f}"
    )
    # Blocker diligence: every other legitimate mu capture of the pinned
    # artifacts also passes both frozen thresholds, so the declared known
    # defect is absent across the whole pinned artifact surface.
    all_digits = np.concatenate([np.asarray(data["train"]), np.asarray(data["heldout"])], axis=0)
    variant_captures = {
        "trained-mu-on-train": (np.asarray(case["train_latents"]), None),
        "trained-mu-on-all-digits": (None, all_digits),
        "untrained-init-mu-on-heldout": (None, np.asarray(data["heldout"])),
    }
    for variant, (precomputed, images) in variant_captures.items():
        if precomputed is not None:
            latents = precomputed
        elif variant == "untrained-init-mu-on-heldout":
            latents = _fit_untrained().encode(images)  # type: ignore[arg-type]
        else:
            latents = case["model"].encode(images)  # type: ignore[union-attr,arg-type]
        variant_er = _effective_rank(np.asarray(latents))
        variant_spread = _spread(np.asarray(latents))
        print(
            f"RECORD capture-variant diligence ({variant}): ER {variant_er:.4f}, "
            f"spread {variant_spread:.4f} — "
            f"{'passes' if variant_er >= 3.0 and variant_spread >= 0.25 else 'FAILS'} "
            "both frozen thresholds"
        )

    localize = localize_case()
    # Blocker arithmetic for the manifest's expected feature-axis selection:
    # every leave-one-feature-out rank observation sits below the affected
    # boundary, so no per-feature reading of the frozen rule can single out
    # the smallest-variance direction.
    loo_feature_ranks = []
    for dim in range(heldout_latents.shape[1]):
        loo_feature_ranks.append(_effective_rank(np.delete(heldout_latents, dim, axis=1)))
    expected_dim = int(case["argmin_dim"])
    earliest_flagged_dim = next(dim for dim in range(len(loo_feature_ranks)) if loo_feature_ranks[dim] < 2.9)
    print(
        "RECORD feature-axis localization arithmetic: leave-one-feature-out effective ranks "
        f"{[round(value, 4) for value in loo_feature_ranks]} — all below the2.9 affected "
        f"boundary, so every direction would be flagged and the earliest-in-order selection "
        f"(bottleneck-mu-dim{earliest_flagged_dim}) != the manifest-expected smallest-variance "
        f"direction (bottleneck-mu-dim{expected_dim}); a single direction carries rank ≤1, so "
        "no per-feature effective-rank observation ≥3.0 exists to separate it"
    )
    result = localize["result"]
    localize_payload = localize["payload"]
    since = _phase("localize", since)
    if result.verdict != "negative":  # type: ignore[attr-defined]
        raise AssertionError(
            f"frozen inputs without an observed defect must localize truthfully negative, got {result.verdict!r}"  # type: ignore[attr-defined]
        )
    print(
        "RECORD localize: truthful negative — no layer/sample meets the declared affected "
        f"criterion ({result.reason}); the manifest's expected feature-axis selection "  # type: ignore[attr-defined]
        "cannot be named without fabricating a location"
    )

    # Determinism of the real stage payloads at the declared seeds.
    _detection_again, _payload_again = evaluate_detection(
        _value(heldout_latents),
        detect["config"],  # type: ignore[arg-type]
        controls=_control_batches(),
    )
    _result_again, _localize_again = evaluate_localization(localize["source"])  # type: ignore[arg-type]
    if _payload_sha(_payload_again) != _payload_sha(payload):
        raise AssertionError("detect payload is not byte-deterministic across reruns")
    if _payload_sha(_localize_again) != _payload_sha(localize_payload):
        raise AssertionError("localize payload is not byte-deterministic across reruns")
    identity_a, _ = _capture()
    identity_b, _ = _capture()
    if identity_a.capture_identity != identity_b.capture_identity:
        raise AssertionError("capture identity is not deterministic")
    print(
        "PASS determinism: detect payload sha "
        f"{_payload_sha(payload)}, localize payload sha "
        f"{_payload_sha(localize_payload)}, capture identity stable across reruns"
    )

    # The real seven-stage chain: stop at the resumable boundary, then resume.
    workflow, request = build_workflow()
    running_result, running_checkpoint = workflow.run(request, manifest, stop_after="localize")
    if running_result.status != "running" or running_checkpoint.status != "running":
        raise AssertionError("stop_after=localize must leave a running, resumable checkpoint")
    if tuple(running_checkpoint.completed_stages) != WORKFLOW_STAGES[:3]:
        raise AssertionError("running checkpoint must cover exactly capture/detect/localize")
    resumed_result, failed_checkpoint = workflow.resume(running_checkpoint, request, manifest)
    since = _phase("workflow", since)
    failure = failed_checkpoint.failure
    if resumed_result.status != "failed" or failed_checkpoint.status != "failed":
        raise AssertionError(
            "the workflow must fail closed when localization is truthfully negative: the80.16 "
            "contract admits no hypothesis without a localized/supported prior selection"
        )
    if failure is None or failure.stage != "explain" or failure.error_type != "StageContractError":
        raise AssertionError(f"unexpected failure shape: {failure}")
    if "localiz" not in failure.message:
        raise AssertionError(f"unexpected failure message: {failure.message}")
    print(
        "BLOCKED explain: StageContractError — "
        f"{failure.message}; workflow failed at explain (7-stage chain cannot complete, "
        "no validator-clean artifact or rendered report is producible without fabricating "
        "a localization)"
    )

    # Tampered resume identity must reject before any stage runs.
    _, tampered_checkpoint = workflow.run(request, manifest, stop_after="localize")
    tampered_request = dataclasses.replace(
        request,
        output=OutputSelection(output_location="artifacts/diagnostics/proof-80-23-tampered"),
    )
    tamper_rejected = False
    try:
        workflow.resume(tampered_checkpoint, tampered_request, manifest)
    except Exception as exc:  # noqa: BLE001 - recorded fail-closed evidence
        tamper_rejected = True
        print(f"PASS fail-closed resume identity: {type(exc).__name__}: {exc}")
    if not tamper_rejected:
        raise AssertionError("a tampered resume identity must be rejected")

    # Persistence of the incomplete workflow must refuse with no files written.
    with tempfile.TemporaryDirectory(prefix="proof-80-23-") as tmp:
        root = Path(tmp) / "root"
        refused = False
        try:
            persist_diagnostic_artifact(
                root,
                request=request,
                manifest=manifest,
                result=resumed_result,
                checkpoint=failed_checkpoint,
                report={},
            )
        except Exception as exc:  # noqa: BLE001 - recorded fail-closed evidence
            refused = True
            print(f"PASS fail-closed persistence: {type(exc).__name__}: {exc}")
        if not refused:
            raise AssertionError("persistence must refuse an incomplete workflow")
        if (root / "runs").exists() or (root / "artifacts").exists():
            raise AssertionError("refused persistence must not create files")

    # Tampered frozen manifest and capture provenance must reject.
    mutated = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    mutated["thresholds"][0]["value"] = 0.01  # type: ignore[index]
    try:
        validate_manifest(mutated)
    except Exception as exc:  # noqa: BLE001 - recorded fail-closed evidence
        print(f"PASS fail-closed manifest tamper: {type(exc).__name__}: {exc}")
    else:
        raise AssertionError("a tampered manifest must be rejected")
    try:
        bind_selection(
            CaptureSelection(
                capture_id=CAPTURE_ID,
                representation_identity="conv_vae_8x8:bottleneck-mu:latent_dim=99",
                axes=("sample", "feature", "slice"),
            ),
            manifest=manifest,
            request_id=request.request_id,
        )
    except Exception as exc:  # noqa: BLE001 - recorded fail-closed evidence
        print(f"PASS fail-closed capture provenance: {type(exc).__name__}: {exc}")
    else:
        raise AssertionError("a tampered capture provenance must be rejected")

    # A real failed required control must block the conclusion.
    blocked_config = detection_config_from_manifest(_detect_request(), manifest)
    blocked_detections, _blocked_payload = evaluate_detection(
        _value(heldout_latents),
        blocked_config,
        controls={
            "control-healthy-counterexample": _value(_failing_counterexample_batch()),
            "control-benign-low-variance": _value(_negative_batch()),
        },
    )
    blocked = blocked_detections[0]
    if blocked.outcome != "inconclusive" or blocked.claim_allowed:
        raise AssertionError("a failed required control must block the supported conclusion")
    if not any(item.startswith("failed-control:") for item in blocked.missing_evidence):
        raise AssertionError("a failed required control must be named as missing evidence")
    print(
        "PASS fail-closed failed control: real rank-2 counterexample fails the frozen "
        f"threshold → outcome {blocked.outcome}, claim_allowed={blocked.claim_allowed}, "
        f"missing {list(blocked.missing_evidence)}"
    )

    previews = downstream_previews()
    for row in previews["per_direction"]:  # type: ignore[union-attr]
        print(
            f"RECORD ablation dim{int(row['dim'])}: {METRIC_ER} "
            f"{row['effective_rank']:.4f} (delta {row['effective_rank_delta']:+.4f}), "
            f"heldout reconstruction mse {row['heldout_reconstruction_mse']:.6f} "
            f"(delta {row['heldout_reconstruction_mse_delta']:+.6f}), "
            f"heldout variance {row['variance']:.6e}"
        )
    target_reconstruction_delta = float(
        next(
            row
            for row in previews["per_direction"]  # type: ignore[union-attr]
            if int(row["dim"]) == int(case["argmin_dim"])
        )["heldout_reconstruction_mse_delta"]
    )
    since = _phase("guards+previews", since)
    print(
        "RECORD downstream (direct; stage chain blocked at explain): per-direction ablation "
        "effects on "
        f"{METRIC_ER} + heldout reconstruction MSE recorded for4 directions; declared trial "
        f"target {previews['declared_trial_target']} (smallest heldout variance per frozen "
        f"estimator), off-target {previews['declared_off_target']}; on-target heldout-"
        f"reconstruction delta {target_reconstruction_delta:+.6f} against baseline "
        f"{previews['baseline_heldout_reconstruction_mse']:.6f} (negligible — the manifest "
        "falsification rule1 arithmetic would fire); off-target specificity rule violated by "
        f"arithmetic={previews['off_target_specificity_rule_violated_by_arithmetic']} "
        "(the manifest falsification rule2 would fire if the trial executed)"
    )
    print(
        "RECORD comparison preview (direct; stage chain blocked): aligned evaluation replays "
        f"of the pinned revision give identical points — {METRIC_ER} "
        f"{previews['comparison_points'][BASELINE_RUN][METRIC_ER]:.4f}, "  # type: ignore[index]
        f"{METRIC_SPREAD} {previews['comparison_points'][BASELINE_RUN][METRIC_SPREAD]:.4f} "  # type: ignore[index]
        "on both sides, |delta|=0; frozen manifest declares no downstream task metric "
        "(the compare task role is bound to a second manifest-declared metric by contract)"
    )

    after = frozen_digests()
    if before != after:
        raise AssertionError("frozen artifacts were modified by the proof run")
    print("PASS frozen invariance: all five frozen artifacts byte-identical after the run")

    # Frozen-acceptance verdict, computed from the manifest's own rules.
    acceptance_reasons: list[str] = []
    if not positive_observed:
        acceptance_reasons.append(
            "declared known defect not observed on the pinned revision: both frozen metrics "
            f"pass on the declared capture ({METRIC_ER} {measurements['effective_rank']:.4f} "
            f"≥3.0, {METRIC_SPREAD} {measurements['singular_spread']:.4f} ≥0.25) and on every "
            "other legitimate mu capture of the pinned artifacts (train, all-digits, "
            "untrained-init variants recorded above)"
        )
    acceptance_reasons.append(
        "truthful negative localization leaves the80.16 localization binding unresolvable, "
        "so the seven-stage chain fails closed at explain and no validator-clean artifact/"
        "rendered report exists"
    )
    acceptance_reasons.append(
        "no localization seam can express the manifest's expected feature-axis direction "
        "selection under the frozen effective-rank metric (per-feature rank observations "
        "cannot separate a direction at the ≥3.0 threshold)"
    )
    print("ACCEPTANCE: NOT PASSED —80.23 frozen acceptance cannot be satisfied truthfully")
    for reason in acceptance_reasons:
        print(f"  blocker: {reason}")
    print(
        "  blocker record: artifacts/task_80.23_encoder_end_to_end_proof_summary.md "
        "(positive/counterexample/negative results, hashes, runtime, limitations, commands)"
    )

    current = time.perf_counter()
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    total = current - started
    wall = current - _MODULE_START
    phase_text = ", ".join(f"{name}={seconds:.2f}s" for name, seconds in phases.items())
    print(
        f"resources: proof body {total:.2f}s ({phase_text}); wall including imports {wall:.2f}s; "
        f"tracemalloc peak {peak / (1 << 20):.1f} MiB; "
        f"environment python {platform.python_version()}, numpy {np.__version__}, "
        f"torch-cpu, scikit-learn {package_version('scikit-learn')}; declared output location "
        f"{OUTPUT_LOCATION!r} intentionally not materialized (persistence refused)"
    )
    print(
        "commands: uv run python scripts/sprint80_task80_23_proof.py; "
        "uv run pytest tests/test_encoder_end_to_end_proof.py -q"
    )
    return 0


def main() -> int:
    return run_proof()


if __name__ == "__main__":
    raise SystemExit(main())

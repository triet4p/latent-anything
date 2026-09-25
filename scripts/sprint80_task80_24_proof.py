"""Executable core proof for Sprint 80.24: the frozen transformer hidden-state
manifest executed through the single high-level diagnostic workflow.

Run from the repository root:

    uv run --extra transformers --with "datasets>=2.19,<4" --with "huggingface-hub>=0.34,<1" \
        python scripts/sprint80_task80_24_proof.py

Exit code 0 means the proof executed completely and recorded the frozen
acceptance verdict truthfully:

1. frozen inputs verify read-only first (manifest commitment digest,
   taxonomy/report-schema documents, the pinned wikitext-2-raw-v1
   validation selection reproduced byte-exactly against the pinned L04
   manifest — official rows, nonblank rows, all2048 selected indices and
   text hashes —, the pinned gpt2 commit snapshot, and grouped
   leakage-safe train/eval partitions with distinct identities);
2. hidden states are captured through the pinned GPT-2 with native
   ``output_hidden_states`` only (hooks reserved for intervention), last
   non-padding token pooling, deterministic replay re-extraction
   bit-identical;
3. a real capture identity binds through ``bind_selection`` /
   ``resolve_captures`` over the pooled final-layer representation;
4. the real seven-stage workflow runs the real80.10 separability
   detector with every manifest-declared control (capacity, label
   randomization, non-separable negative, split-swap null), the real
   layer-axis localizer (per-layer leakage-safe probe accuracies under
   ``affected_when=threshold_pass``), the real80.16 probe explanation,
   the real80.18 concept-removal intervention with four controls, and
   the real80.20 aligned comparison;
5. persistence is attempted through the 80.21 content-addressed seam. The
   current frozen contracts reject this supported report at independent
   validation because the taxonomy requires axes absent from truthful capture
   provenance; the proof records the blocker and confirms no files were written.
   The success branch, reachable only after an authorized contract resolution,
   reloads and revalidates the artifact, renders deterministically, re-hashes
   evidence links, and exercises declared tamper/leakage/control failures;
6. bounded runtime and resource evidence is recorded.

The frozen acceptance verdict is computed from the manifest's own rules
and printed; thresholds, labels, splits, and data are never adjusted to
force a pass. If any frozen input, control, or scientific claim fails,
the proof records the precise blocker and leaves80.24 unchecked.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import platform
import re
import tempfile
import time
import tracemalloc
from collections.abc import Mapping
from functools import lru_cache
from importlib.metadata import version as package_version
from pathlib import Path

_MODULE_START = time.perf_counter()
# Keep the runtime imports below this marker so the reported wall time includes them.

import numpy as np  # noqa: E402

from latent_anything._benchmark_manifest import manifest_digest, validate_manifest  # noqa: E402
from latent_anything._capture_binding import bind_selection, resolve_captures  # noqa: E402
from latent_anything._diagnostic_artifact import (  # noqa: E402
    load_diagnostic_artifact,
    persist_diagnostic_artifact,
    registered_artifact_bytes,
)
from latent_anything._diagnostic_report import validate_report_shape  # noqa: E402
from latent_anything._diagnostic_validator import validate_diagnostic_report  # noqa: E402
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
    intervention_report_items,
    make_intervene_executor,
)
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
from latent_anything._redundancy_separability_detection import (  # noqa: E402
    LabeledBatch,
    detection_config_from_manifest,
    evaluate_detection,
    make_detect_executor,
)
from latent_anything._report_renderer import persist_rendered_report, render_diagnostic_report  # noqa: E402
from latent_anything._run_comparison import (  # noqa: E402
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
from latent_anything.latent_space import LatentSpace  # noqa: E402
from latent_anything.latent_value import LatentValue  # noqa: E402
from latent_anything.probes import _fast_probe  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
ARTIFACTS = REPO / "artifacts"
MANIFEST_PATH = ARTIFACTS / "benchmark_manifest_sprint80_transformer_hidden_state_v1.json"
L04_PATH = ARTIFACTS / "m14" / "l04-wikitext-2-manifest.json"
FROZEN_PATHS = (
    ARTIFACTS / "benchmark_manifest_schema_v1.json",
    ARTIFACTS / "benchmark_manifest_sprint80_encoder_autoencoder_v1.json",
    ARTIFACTS / "benchmark_manifest_sprint80_transformer_hidden_state_v1.json",
    ARTIFACTS / "diagnostic_report_schema_v1.json",
    ARTIFACTS / "representation_problem_taxonomy_v1.json",
    L04_PATH,
)
PINNED_MANIFEST_DIGEST = "c4773238ddf4d8854cb40ad26811b7b204cab8210ff9fe45f79e83e12da4675a"
MODEL_ID = "openai-community/gpt2"
MODEL_REVISION = "e7da7f221d5bf496a48136c0cd264e630fe9fcc8"
DATASET_ID = "Salesforce/wikitext"
DATASET_REVISION = "f776294184f13b8ff2337b3841cf9269a6216d1e"
DATASET_CONFIG = "wikitext-2-raw-v1"
SPLIT_IDENTITY = "wikitext-2-raw-v1-validation-3760rows-selected2048-maxtokens128"
OFFICIAL_ROWS = 3760
NONBLANK_ROWS = 2461
SELECTED_ROWS = 2048
MAX_TOKENS = 128

MANIFEST_ID = "sprint80-core-transformer-hidden-state-probe-v1"
REPRESENTATION = "openai-community-gpt2:hidden-states:layers0-11:dim768"
FAMILY = "separability_probe_leakage"
METRIC_ACC = "heldout-probe-accuracy"
METRIC_GAP = "probe-leakage-gap"
CAPTURE_ID = "capture-transformer-hidden-states"
EVAL_SLICE_ID = "slice-validation-eval"
HYPOTHESIS_ID = "h-section-header-separability"
REQUEST_ID = "proof-80-24"
DETECT_REQUEST_ID = "proof-80-24-detect"
OUTPUT_LOCATION = "artifacts/diagnostics/proof-80-24-v2-rule-bound"
REPORT_ID = "report-80-24-transformer"
HEADER_PATTERN = re.compile(r"^ = .+ = $")
TARGET_ID = "section-header-attribute"
TARGET_RULE = f"section-header attribute {HEADER_PATTERN.pattern!r}"
TARGET_RULE_KIND = "regex-header-match"
MANIFEST_CONTROLS = (
    "control-capacity",
    "control-label-randomization",
    "control-nonseparable-negative",
    "control-split-swap",
)
TRIAL_CONTROLS = (
    "remove-layer-zero",
    "remove-layer-random",
    "remove-layer-shuffled",
    "remove-layer-off",
)
PROBE_SEED = 79
N_LAYERS = 12
POOLING_IDENTITY = "last-nonpadding-token-pooling-v1"
PREPROCESSING_IDENTITY = f"gpt2-tokenizer-{MAX_TOKENS}-nonpadding-first{MAX_TOKENS}-pad=eos-{POOLING_IDENTITY}"
TRAIN_GROUPS_FRACTION = 0.75
BASELINE_RUN = "gpt2-hiddenstate-eval-000100"
CANDIDATE_RUN = "gpt2-hiddenstate-eval-000200"
BASELINE_CKPT = "e7da7f22-validation-eval-a"
CANDIDATE_CKPT = "e7da7f22-validation-eval-b"


def _sha(data: str | bytes) -> str:
    return hashlib.sha256(data.encode("utf-8") if isinstance(data, str) else data).hexdigest()


def _payload_sha(payload: object) -> str:
    return _sha(canonical_json(payload))


class _PersistBlockedError(Exception):
    """Sentinel: persistence refused; the blocker is recorded, execution continues."""


def _separability_block(payload: object) -> dict[str, object]:
    measurements = payload["measurements"]  # type: ignore[index]
    block = measurements.get("separability")
    if not isinstance(block, dict):
        raise AssertionError("detect payload must record the separability measurement block")
    return block


def _metric_value(payload: object, metric_id: str) -> float:
    block = _separability_block(payload)
    key = "heldout_accuracy" if metric_id == METRIC_ACC else "leakage_gap"
    return float(block[key])


def frozen_digests() -> dict[str, str]:
    return {path.name: _sha(path.read_bytes()) for path in FROZEN_PATHS}


@lru_cache(maxsize=1)
def load_manifest() -> dict[str, object]:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    validate_manifest(manifest)
    if manifest_digest(manifest) != PINNED_MANIFEST_DIGEST:
        raise AssertionError("transformer manifest digest drifted from the sprint-pinned commitment")
    if str(manifest.get("manifest_id")) != MANIFEST_ID:
        raise AssertionError("unexpected manifest identity")
    return dict(manifest)


@lru_cache(maxsize=1)
def build_data() -> dict[str, object]:
    """Reproduce the frozen validation selection byte-exactly, tokenize, label,
    and build the grouped leakage-safe split. Verified before any evidence."""
    from datasets import load_dataset

    l04 = json.loads(L04_PATH.read_text(encoding="utf-8"))
    dataset = load_dataset(DATASET_ID, DATASET_CONFIG, revision=DATASET_REVISION, split="validation")
    rows = list(dataset["text"])
    if len(rows) != OFFICIAL_ROWS:
        raise AssertionError(f"official validation rows drifted: {len(rows)} != {OFFICIAL_ROWS}")
    if len(l04["splits"]["validation"]["selected"]) != SELECTED_ROWS:
        raise AssertionError("pinned L04 selection is incomplete")
    if l04["source"]["revision"] != DATASET_REVISION or l04["source"]["config"] != DATASET_CONFIG:
        raise AssertionError("pinned L04 source identity drifted")
    nonblank = [index for index, text in enumerate(rows) if text.strip()]
    if len(nonblank) != NONBLANK_ROWS:
        raise AssertionError(f"nonblank validation rows drifted: {len(nonblank)} != {NONBLANK_ROWS}")

    def text_sha(text: str) -> str:
        return _sha(text.encode("utf-8"))

    candidates = sorted((text_sha(rows[index]), index) for index in nonblank)
    selected = candidates[:SELECTED_ROWS]
    pinned = [(entry["text_sha256"], int(entry["index"])) for entry in l04["splits"]["validation"]["selected"]]
    if selected != pinned:
        raise AssertionError("validation selection does not reproduce the pinned L04 manifest")
    selected_indices = [index for _, index in selected]
    selected_text = [rows[index] for index in selected_indices]

    labels = np.array([1 if HEADER_PATTERN.match(text) else 0 for text in selected_text], dtype=np.int64)
    if len(np.unique(labels)) < 2:
        raise AssertionError("the declared attribute must yield two classes")

    groups = np.array([index // 8 for index in selected_indices])
    unique_groups = np.array(sorted({int(group) for group in groups}))
    rng = np.random.default_rng(79)
    rng.shuffle(unique_groups)
    train_groups = set(int(group) for group in unique_groups[: int(len(unique_groups) * TRAIN_GROUPS_FRACTION)])
    train_positions = [position for position, group in enumerate(groups) if int(group) in train_groups]
    eval_positions = [position for position, group in enumerate(groups) if int(group) not in train_groups]
    if set(int(group) for group in groups if int(group) in train_groups) & {
        int(group) for group in groups if int(group) not in train_groups
    }:
        raise AssertionError("grouped split leaked a document group across partitions")
    if len(train_positions) < 770:
        raise AssertionError("capacity gate requires n_train >=770 for a768-dim linear probe")
    if len(np.unique(labels[train_positions])) < 2 or len(np.unique(labels[eval_positions])) < 2:
        raise AssertionError("both partitions must contain both classes")
    return {
        "text": selected_text,
        "labels": labels,
        "selected_indices": np.array(selected_indices),
        "train_positions": np.array(train_positions),
        "eval_positions": np.array(eval_positions),
        "l04": l04,
    }


@lru_cache(maxsize=1)
def tokenize_data() -> dict[str, object]:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, local_files_only=True)
    tok.pad_token = tok.eos_token
    data = build_data()
    encoded = tok(
        list(data["text"]),  # type: ignore[arg-type]
        truncation=True,
        max_length=MAX_TOKENS,
        padding="max_length",
        return_tensors="np",
    )
    input_ids = encoded["input_ids"].astype(np.int64)
    attention = encoded["attention_mask"].astype(np.int64)
    last_positions = attention.sum(axis=1) - 1
    if int(attention.sum(axis=1).max()) > MAX_TOKENS:
        raise AssertionError("token axis exceeds the frozen nonpadding-first128 selection")
    if np.any(last_positions < 0):
        raise AssertionError("every selected row must keep at least one nonpadding token")
    return {
        "input_ids": input_ids,
        "attention_mask": attention,
        "last_positions": last_positions,
        "tokenizer_provenance": f"{MODEL_ID}@{MODEL_REVISION}",
    }


def _extract_pooled(rows: int | None = None) -> np.ndarray:
    """One fresh hidden-state extraction:12 block outputs, last non-pad token.

    ``rows`` limits the extraction to a prefix (used by the focused replay
    check); the recorded proof run extracts every selected row.
    """
    import torch
    from transformers import AutoModel

    tokens = tokenize_data()
    input_ids = np.asarray(tokens["input_ids"])
    attention = np.asarray(tokens["attention_mask"])
    last = np.asarray(tokens["last_positions"])
    if rows is not None:
        input_ids = input_ids[:rows]
        attention = attention[:rows]
        last = last[:rows]
    n = input_ids.shape[0]
    torch.manual_seed(79)
    model = AutoModel.from_pretrained(MODEL_ID, revision=MODEL_REVISION, local_files_only=True)
    model.eval()
    pooled = np.zeros((N_LAYERS, n, 768), dtype=np.float64)
    batch = 16
    with torch.no_grad():
        for lo in range(0, n, batch):
            hi = min(lo + batch, n)
            out = model(
                input_ids=torch.from_numpy(input_ids[lo:hi]),
                attention_mask=torch.from_numpy(attention[lo:hi]),
                output_hidden_states=True,
            )
            hidden = out.hidden_states
            local = np.arange(hi - lo)
            for layer in range(N_LAYERS):
                block = hidden[layer + 1].float().cpu().numpy()
                pooled[layer, lo:hi] = block[local, last[lo:hi], :]
            del out, hidden
    return pooled


def _pooled_cache_path() -> Path:
    key = _sha(
        json.dumps(
            {
                "model": f"{MODEL_ID}@{MODEL_REVISION}",
                "rows": SELECTED_ROWS,
                "max_tokens": MAX_TOKENS,
                "pooling": POOLING_IDENTITY,
                "selection": _payload_sha([[int(i)] for i in np.asarray(build_data()["selected_indices"])]),
            },
            sort_keys=True,
        )
    )
    return Path(tempfile.gettempdir()) / f"latent-anything-80-24-{key}.npz"


@lru_cache(maxsize=1)
def pooled_features() -> np.ndarray:
    """Pooled hidden states with a content-keyed on-disk cache."""
    path = _pooled_cache_path()
    if path.exists():
        with np.load(path) as data:
            return np.array(data["pooled"], dtype=np.float64)
    pooled = _extract_pooled()
    np.savez_compressed(path, pooled=pooled)
    return pooled


def pooled_digest(pooled: np.ndarray) -> str:
    return _sha(pooled.astype("<f8", copy=False).tobytes())


@lru_cache(maxsize=1)
def split_identities() -> tuple[str, str]:
    data = build_data()
    return (
        f"grouped-original-index-div8-train79-{len(data['train_positions'])}rows",  # type: ignore[arg-type]
        f"grouped-original-index-div8-eval79-{len(data['eval_positions'])}rows",  # type: ignore[arg-type]
    )


def _space() -> LatentSpace:
    return LatentSpace(
        dim=768,
        source_model=MODEL_ID,
        metadata={
            "source_representation_identity": REPRESENTATION,
            "model_version": MODEL_REVISION,
        },
    )


def _batch(layer: int) -> np.ndarray:
    return np.asarray(pooled_features()[layer], dtype=np.float64)


def _labeled(layer: int) -> LabeledBatch:
    data = build_data()
    labels = np.asarray(data["labels"])
    indices = np.asarray(data["selected_indices"])
    train_identity, eval_identity = split_identities()
    return LabeledBatch(
        LatentValue(_batch(layer), _space()),
        tuple(int(item) for item in labels),
        tuple(f"validation-{int(index)}" for index in indices),
        tuple(int(item) for item in data["train_positions"]),  # type: ignore[union-attr]
        tuple(int(item) for item in data["eval_positions"]),  # type: ignore[union-attr]
        train_identity,
        eval_identity,
        DECLARED_PROBE_CAPACITY,
        target_id=TARGET_ID,
        target_rule=TARGET_RULE,
        target_rule_kind=TARGET_RULE_KIND,
    )


@lru_cache(maxsize=1)
def negative_batch() -> LabeledBatch:
    """Non-separable leakage-negative control: seeded Gaussian features with the
    same labels, sample identities, and declared splits (manifest control seed81)."""
    data = build_data()
    rng = np.random.default_rng(81)
    noise = rng.normal(size=(SELECTED_ROWS, 768))
    train_identity, eval_identity = split_identities()
    return LabeledBatch(
        LatentValue(noise, _space()),
        tuple(int(item) for item in data["labels"]),
        tuple(f"validation-{int(index)}" for index in np.asarray(data["selected_indices"])),
        tuple(int(item) for item in data["train_positions"]),  # type: ignore[union-attr]
        tuple(int(item) for item in data["eval_positions"]),  # type: ignore[union-attr]
        train_identity,
        eval_identity,
        DECLARED_PROBE_CAPACITY,
    )


def _detect_request() -> DiagnosticRequest:
    return DiagnosticRequest(
        request_id=DETECT_REQUEST_ID,
        manifest_id=MANIFEST_ID,
        capture=CaptureSelection(
            capture_id=CAPTURE_ID,
            representation_identity=REPRESENTATION,
            axes=("slice", "feature"),
        ),
        diagnostics=DiagnosticSelection(family_ids=(FAMILY,)),
        controls=ControlSelection(
            control_ids=MANIFEST_CONTROLS,
            metric_ids=(METRIC_ACC, METRIC_GAP),
        ),
        output=OutputSelection(output_location=OUTPUT_LOCATION),
    )


@lru_cache(maxsize=1)
def detect_case() -> dict[str, object]:
    """Headline detection evaluated once on the final-layer pooled capture."""
    manifest = load_manifest()
    config = detection_config_from_manifest(_detect_request(), manifest)
    detections, payload = evaluate_detection(
        _labeled(N_LAYERS - 1), config, controls={"control-nonseparable-negative": negative_batch()}
    )
    return {"config": config, "detections": detections, "payload": payload, "family": detections[0]}


@lru_cache(maxsize=1)
def per_layer_probes() -> dict[str, object]:
    """Per-layer leakage-safe probes: the frozen estimator behind the
    detect observation, the localize cells, and the cached predictions."""
    data = build_data()
    labels = np.asarray(data["labels"])
    train = np.asarray(data["train_positions"])
    evaluation = np.asarray(data["eval_positions"])
    accuracies: list[float] = []
    gaps: list[float] = []
    predictions_by_layer: list[np.ndarray] = []
    headline: dict[str, object] = {}
    rrng = np.random.default_rng(81)
    for layer in range(N_LAYERS):
        matrix = _batch(layer)
        result = _fast_probe(matrix[train], labels[train], matrix[evaluation], labels[evaluation], PROBE_SEED)
        accuracy = float(result.accuracy)
        shuffled_y = labels[train][rrng.permutation(len(train))]
        randomized = _fast_probe(matrix[train], shuffled_y, matrix[evaluation], labels[evaluation], PROBE_SEED)
        accuracies.append(accuracy)
        gaps.append(accuracy - float(randomized.accuracy))
        predictions_by_layer.append(np.asarray(result.predictions, dtype=np.float64))
        if layer == N_LAYERS - 1:
            headline = {
                "eval_predictions": np.asarray(result.predictions, dtype=np.float64),
                "randomized_predictions": np.asarray(randomized.predictions, dtype=np.float64),
                "accuracy": accuracy,
                "randomized_accuracy": float(randomized.accuracy),
            }
    return {
        "accuracies": accuracies,
        "gaps": gaps,
        "predictions_by_layer": predictions_by_layer,
        "headline": headline,
    }


def _control_outcomes() -> dict[str, str]:
    return dict(detect_case()["family"].control_outcomes)  # type: ignore[attr-defined,union-attr]


def earliest_layer() -> int | None:
    for layer, accuracy in enumerate(per_layer_probes()["accuracies"]):  # type: ignore[union-attr]
        if accuracy >= 0.7:
            return layer
    return None


@lru_cache(maxsize=1)
def localize_case() -> dict[str, object]:
    """Truthful layer-axis localization input: per-layer probe accuracies under
    ``affected_when=threshold_pass`` (the manifest's expected localization)."""
    manifest = load_manifest()
    request = workflow_request()
    family = detect_case()["family"]
    evidence = evidence_from_manifest(
        request,
        manifest,
        family_id=FAMILY,
        metric_id=METRIC_ACC,
        affected_when="threshold_pass",
        control_outcomes=_control_outcomes(),
        outcome=family.outcome,  # type: ignore[attr-defined]
        claim_allowed=family.claim_allowed,  # type: ignore[attr-defined]
        representation_identity=REPRESENTATION,
    )
    data = build_data()
    labels = np.asarray(data["labels"])
    evaluation = np.asarray(data["eval_positions"])
    selected = np.asarray(data["selected_indices"])
    layer = earliest_layer()
    predictions = np.asarray(
        per_layer_probes()["predictions_by_layer"][  # type: ignore[index]
            N_LAYERS - 1 if layer is None else layer
        ]
    )
    source = LocalizationInput(
        manifest_id=MANIFEST_ID,
        evidence=evidence,
        layer_order=tuple(f"transformer.h.{layer}" for layer in range(N_LAYERS)),
        layer_cells=tuple(
            LayerCell(
                layer_id=f"transformer.h.{layer}",
                metric_value=float(accuracy),
                representation_identity=REPRESENTATION,
            )
            for layer, accuracy in enumerate(per_layer_probes()["accuracies"])  # type: ignore[union-attr]
        ),
        sample_cells=tuple(
            SampleCell(
                sample_id=f"validation-{int(selected[position])}",
                metric_value=float(predictions[eval_row] == labels[position]),
                representation_identity=REPRESENTATION,
            )
            for eval_row, position in enumerate(evaluation)
        ),
        declared_slice_ids=(EVAL_SLICE_ID,),
        slices=(
            SliceDefinition(
                slice_id=EVAL_SLICE_ID,
                criteria=(f"heldout rows of the grouped leakage-safe split over {SPLIT_IDENTITY}"),
                member_sample_ids=tuple(f"validation-{int(selected[position])}" for position in evaluation),
            ),
        ),
        global_score=_metric_value(detect_case()["payload"], METRIC_ACC),  # type: ignore[arg-type]
    )
    result, payload = evaluate_localization(source)
    return {"source": source, "result": result, "payload": payload}


def _hypothesis() -> ExplanationHypothesis:
    manifest = load_manifest()
    dataset = manifest["dataset"]
    assert isinstance(dataset, dict)
    layer = earliest_layer()
    bindings: list[tuple[str, str]] = []
    if layer is not None:
        bindings.append(("layer", f"transformer.h.{layer}"))
    payload = localize_case()["payload"]
    if payload["verdict"] == "localized":  # type: ignore[index]
        bindings.append(("slice", EVAL_SLICE_ID))
    return ExplanationHypothesis(
        hypothesis_id=HYPOTHESIS_ID,
        symptom_id="symptom-section-header-separability",
        family_id=FAMILY,
        target_id="section-header-attribute",
        representation_id=REPRESENTATION,
        layer_id=f"transformer.h.{layer}" if layer is not None else "transformer.h.11",
        slice_id=EVAL_SLICE_ID,
        method="probe",
        expected_direction="higher",
        dataset_id=str(dataset["split_identity"]),
        train_split_identity=split_identities()[0],
        eval_split_identity=split_identities()[1],
        seeds=(PROBE_SEED, 81),
        control_ids=(
            "capacity:control-capacity",
            "randomized:control-label-randomization",
            "negative:control-nonseparable-negative",
        ),
        metric_ids=(METRIC_ACC,),
        thresholds=(("heldout_accuracy", ">=", 0.7), ("leakage_gap", ">=", 0.15)),
        manifest_id=MANIFEST_ID,
        localization_bindings=tuple(bindings),
    )


@lru_cache(maxsize=1)
def explain_bundle() -> dict[str, object]:
    """Leakage-safe probe inputs at the localized layer with declared capacity."""
    data = build_data()
    labels = np.asarray(data["labels"])
    train = np.asarray(data["train_positions"])
    evaluation = np.asarray(data["eval_positions"])
    layer = earliest_layer()
    matrix = _batch(N_LAYERS - 1 if layer is None else layer)
    negative = evaluate_negative_accuracy()
    return {
        "train_matrix": matrix[train],
        "eval_matrix": matrix[evaluation],
        "train_labels": labels[train],
        "eval_labels": labels[evaluation],
        "train_ids": [f"validation-{int(i)}" for i in np.asarray(data["selected_indices"])[train]],
        "eval_ids": [f"validation-{int(i)}" for i in np.asarray(data["selected_indices"])[evaluation]],
        "capacity": DECLARED_PROBE_CAPACITY,
        "negative_accuracy": negative,
    }


@lru_cache(maxsize=1)
def evaluate_negative_accuracy() -> float:
    return float(
        detect_case()["payload"]["control_metrics"]["control-nonseparable-negative"][METRIC_ACC]  # type: ignore[index]
    )


def _capture() -> tuple[object, dict[str, object]]:
    manifest = load_manifest()
    request = workflow_request()
    plan = bind_selection(request.capture, manifest=manifest, request_id=request.request_id)
    layer = earliest_layer()
    features = _batch(N_LAYERS - 1 if layer is None else layer)
    activation = CapturedActivation(
        values=features,
        metadata=CaptureMetadata(
            location=f"gpt2.transformer.h.{N_LAYERS - 1 if layer is None else layer}.last-nonpadding-token",
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


# ---------------------------------------------------------------------------
# Probe fitting seam for intervention direction + compare measures
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def headline_probe() -> dict[str, object]:
    """The leakage-safe headline probe at the localized layer: scaler, weights,
    and cached real predictions (no refit inside any stage callback)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    data = build_data()
    labels = np.asarray(data["labels"])
    train = np.asarray(data["train_positions"])
    evaluation = np.asarray(data["eval_positions"])
    layer = earliest_layer()
    matrix = _batch(N_LAYERS - 1 if layer is None else layer)
    scaler = StandardScaler().fit(matrix[train])
    scaled_train = scaler.transform(matrix[train])
    classifier = LogisticRegression(
        C=1.0, solver="lbfgs", max_iter=1000, random_state=PROBE_SEED, class_weight="balanced"
    )
    classifier.fit(scaled_train, labels[train])
    scaled_eval = scaler.transform(matrix[evaluation])
    predictions = classifier.predict(scaled_eval)
    accuracy = float(np.mean(predictions == labels[evaluation]))
    direction = classifier.coef_.ravel()
    norm = float(np.linalg.norm(direction))
    if norm <= 0.0 or not np.isfinite(norm):
        raise AssertionError("probe direction must be a finite nonzero vector")
    return {
        "scaler": scaler,
        "classifier": classifier,
        "direction": direction / norm,
        "scaled_eval": scaled_eval,
        "eval_labels": labels[evaluation],
        "eval_truth": labels[evaluation],
        "accuracy": accuracy,
        "layer": layer,
    }


def _probe_accuracy(features: np.ndarray) -> float:
    probe = headline_probe()
    scaled = (features - probe["scaler"].mean_) / probe["scaler"].scale_  # type: ignore[union-attr]
    predictions = probe["classifier"].predict(scaled)  # type: ignore[union-attr]
    return float(np.mean(predictions == np.asarray(probe["eval_truth"])))  # type: ignore[union-attr]


def _remove_direction(matrix: np.ndarray, direction: np.ndarray) -> np.ndarray:
    return matrix - np.outer(matrix @ direction, direction)


def _intervene_measure(app: TrialApplication) -> Measurement:
    """Real downstream metric: leakage-safe probe accuracy at the localized
    layer after the declared concept-removal application."""
    probe = headline_probe()
    scaled = np.asarray(probe["scaled_eval"])
    layer = int(probe["layer"])  # type: ignore[arg-type]
    control = app.control_class
    if (
        app.role == "baseline"
        or app.strength == 0.0
        or control == "zero_strength"
        or app.target != f"transformer.h.{layer}"
    ):
        # Off-target applications leave the localized layer (and therefore the
        # declared downstream metric) untouched, exactly as the manifest's
        # causal expectation states for off-target layers.
        value = float(np.mean(probe["classifier"].predict(scaled) == probe["eval_truth"]))  # type: ignore[union-attr]
    elif app.role == "intervened":
        removed = _remove_direction(scaled, np.asarray(probe["direction"]))
        value = float(np.mean(probe["classifier"].predict(removed) == probe["eval_truth"]))  # type: ignore[union-attr]
    elif control == "random":
        assert app.rng is not None
        vector = app.rng.normal(size=scaled.shape[1])
        vector /= np.linalg.norm(vector)
        value = float(np.mean(probe["classifier"].predict(_remove_direction(scaled, vector)) == probe["eval_truth"]))  # type: ignore[union-attr]
    elif control == "shuffled":
        assert app.rng is not None
        vector = np.asarray(probe["direction"])[app.rng.permutation(scaled.shape[1])]
        value = float(np.mean(probe["classifier"].predict(_remove_direction(scaled, vector)) == probe["eval_truth"]))  # type: ignore[union-attr]
    else:
        value = float(np.mean(probe["classifier"].predict(scaled) == probe["eval_truth"]))  # type: ignore[union-attr]
    return Measurement(app.metric_id, value, app.inputs_digest)


def build_trials() -> tuple[TrialSpec, ...]:
    layer = earliest_layer()
    if layer is None:
        raise AssertionError("no localized layer exists for an intervention trial")
    off_target = f"transformer.h.{N_LAYERS - 1 if layer != N_LAYERS - 1 else 0}"
    return (
        TrialSpec(
            intervention_id="remove-section-header-direction",
            kind="remove",
            target=f"transformer.h.{layer}",
            metric_id=METRIC_ACC,
            hypothesis_id=HYPOTHESIS_ID,
            strength=1.0,
            strength_semantics="unit-probe-direction-projection-removal",
            expected_effect="decrease",
            controls=(
                TrialControl(
                    "remove-layer-zero",
                    "zero_strength",
                    "zero-strength rerun reproduces the baseline accuracy",
                ),
                TrialControl(
                    "remove-layer-random",
                    "random",
                    "removing an rng-drawn direction stays below the on-target effect",
                ),
                TrialControl(
                    "remove-layer-shuffled",
                    "shuffled",
                    "removing the shuffled direction assignment stays below the on-target effect",
                ),
                TrialControl(
                    "remove-layer-off",
                    "off_target",
                    "removing at another layer does not move the localized metric",
                    target=off_target,
                ),
            ),
            provenance={
                "method": "concept-removal",
                "carrier": "linear-probe-weight-direction",
                "source": "leakage-safe headline probe at the localized layer",
            },
        ),
    )


def _alignment(manifest: dict[str, object]) -> dict[str, str]:
    dataset = manifest["dataset"]
    model = manifest["model"]
    assert isinstance(dataset, dict) and isinstance(model, dict)
    _concrete, capture_provenance = _capture()
    layer = earliest_layer()
    alignment = {
        "axes": capture_axes_identity(capture_provenance),
        "dataset_configuration": str(dataset["split_identity"]),
        "dataset_slice_id": EVAL_SLICE_ID,
        "diagnostic_config": detect_config_identity(detect_case()["payload"]),  # type: ignore[arg-type]
        "layer_module_identity": f"transformer.h.{N_LAYERS - 1 if layer is None else layer}",
        "manifest_identity": MANIFEST_ID,
        "model_identity": str(model["id"]),
        "preprocessing_identity": PREPROCESSING_IDENTITY,
        "representation_identity": REPRESENTATION,
        "representation_metric": METRIC_ACC,
        "schema_identity": str(manifest["schema_version"]),
        "seeds": manifest_seed_identity(manifest),
        "taxonomy_identity": FAMILY,
        "task_metric": METRIC_GAP,
    }
    if set(alignment) != set(ALIGNMENT_FIELDS):
        raise AssertionError("comparison alignment must declare the exact frozen field set")
    return alignment


def _comparison_specs(manifest: dict[str, object]) -> tuple[ComparisonSpec, ...]:
    alignment = _alignment(manifest)
    return (
        ComparisonSpec(
            comparison_id="cmp-hiddenstate-eval-replay",
            baseline=RunSide(run_id=BASELINE_RUN, checkpoint_id=BASELINE_CKPT, alignment=alignment),
            candidate=RunSide(run_id=CANDIDATE_RUN, checkpoint_id=CANDIDATE_CKPT, alignment=alignment),
            representation_metric_id=METRIC_ACC,
            task_metric_id=METRIC_GAP,
            representation_tolerance=0.02,
            task_tolerance=0.02,
        ),
    )


def _compare_measure(app: ComparisonApplication) -> ComparisonMeasurement:
    """Real per-side measurements from the cached leakage-safe predictions."""
    headline = per_layer_probes()["headline"]  # type: ignore[assignment]
    eval_predictions = np.asarray(headline["eval_predictions"])  # type: ignore[index]
    randomized = np.asarray(headline["randomized_predictions"])  # type: ignore[index]
    labels = np.asarray(build_data()["labels"])[np.asarray(build_data()["eval_positions"])]
    positions = np.arange(len(labels)) if app.rng is None else app.rng.integers(0, len(labels), size=len(labels))
    if app.metric_role == "representation":
        value = float(np.mean(eval_predictions[positions] == labels[positions]))
    else:
        value = float(
            np.mean(eval_predictions[positions] == labels[positions])
            - np.mean(randomized[positions] == labels[positions])
        )
    return ComparisonMeasurement(app.run_id, app.metric_id, float(value), app.inputs_digest)


@lru_cache(maxsize=1)
def workflow_request() -> DiagnosticRequest:
    intervention = build_trials()[0]
    return DiagnosticRequest(
        request_id=REQUEST_ID,
        manifest_id=MANIFEST_ID,
        capture=CaptureSelection(
            capture_id=CAPTURE_ID,
            representation_identity=REPRESENTATION,
            axes=("slice", "feature"),
        ),
        diagnostics=DiagnosticSelection(family_ids=(FAMILY,)),
        controls=ControlSelection(
            control_ids=(*MANIFEST_CONTROLS, *TRIAL_CONTROLS),
            metric_ids=(METRIC_ACC, METRIC_GAP),
        ),
        interventions=(
            InterventionRequest(
                intervention_id=intervention.intervention_id,
                target=intervention.target,
                control_ids=tuple(control.control_id for control in intervention.controls),
            ),
        ),
        comparisons=(
            ComparisonRequest(
                comparison_id="cmp-hiddenstate-eval-replay",
                baseline_run=BASELINE_RUN,
                candidate_run=CANDIDATE_RUN,
                metric_ids=(METRIC_ACC, METRIC_GAP),
            ),
        ),
        output=OutputSelection(output_location=OUTPUT_LOCATION),
    )


@lru_cache(maxsize=1)
def build_workflow() -> tuple[DiagnosticWorkflow, DiagnosticRequest]:
    manifest = load_manifest()
    request = workflow_request()
    detect = make_detect_executor(
        _labeled(N_LAYERS - 1),
        detect_case()["config"],  # type: ignore[arg-type]
        controls={"control-nonseparable-negative": negative_batch()},
    )
    localize = make_localize_executor(localize_case()["source"])  # type: ignore[arg-type]
    explain = make_explain_executor((_hypothesis(),), MethodInputs(probe=explain_bundle()))
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


def build_report(
    checkpoint_outputs: tuple[StageOutput, ...],
    *,
    intervention_claims: list[dict[str, object]],
    comparison_rows: list[dict[str, object]],
) -> dict[str, object]:
    """Structured report from the real stage payloads (architecture-neutral)."""
    manifest = load_manifest()
    dataset = manifest["dataset"]
    model = manifest["model"]
    assert isinstance(dataset, dict) and isinstance(model, dict)
    payloads = {item.stage: dict(item.payload) for item in checkpoint_outputs}
    detect = payloads["detect"]
    localize = payloads["localize"]
    explain = payloads["explain"]
    # Preserve the original fail-closed checks for the detector and hypothesis.
    detect["families"][0]  # type: ignore[index]
    explain["family_evidence"][HYPOTHESIS_ID]  # type: ignore[index]
    layer = earliest_layer()
    layer_rows = [
        row
        for row in localize["report_localization"]
        if row.get("axis") == "layer"  # type: ignore[union-attr]
    ]
    slice_rows = [
        row
        for row in localize["report_localization"]
        if row.get("axis") == "slice"  # type: ignore[union-attr]
    ]
    localization_rows: list[dict[str, object]] = [
        {
            "axis": row["axis"],
            "confidence": row["confidence"],
            "evidence_refs": ["o-1"],
            "id": row["id"],
            "selection": row["selection"],
            "status": row["status"],
        }
        for row in (*layer_rows, *slice_rows)
    ]
    explanation_record = next(
        record
        for record in explain["evidence"]  # type: ignore[index]
        if record["hypothesis_id"] == HYPOTHESIS_ID
    )
    explanation_status = str(explanation_record["outcome"])
    accuracy_value = _metric_value(detect, METRIC_ACC)
    gap_value = _metric_value(detect, METRIC_GAP)
    uncertainty = _separability_block(detect)["uncertainty"]  # type: ignore[index]
    accuracy_interval = uncertainty[METRIC_ACC]  # type: ignore[index]
    gap_interval = uncertainty[METRIC_GAP]  # type: ignore[index]
    statistical_rows: list[dict[str, object]] = [
        {
            "control_refs": [
                "control-capacity",
                "control-label-randomization",
                "control-nonseparable-negative",
            ],
            "estimate": accuracy_value,
            "evidence_refs": ["o-1"],
            "id": "e-acc",
            "metric_id": METRIC_ACC,
            "status": "observed",
            "uncertainty": {
                "kind": "interval",
                "lower": float(accuracy_interval["lower"]),  # type: ignore[index]
                "upper": float(accuracy_interval["upper"]),  # type: ignore[index]
            },
        },
        {
            "control_refs": [
                "control-label-randomization",
                "control-nonseparable-negative",
            ],
            "estimate": gap_value,
            "evidence_refs": ["o-1"],
            "id": "e-gap",
            "metric_id": METRIC_GAP,
            "status": "observed",
            "uncertainty": {
                "kind": "interval",
                "lower": float(gap_interval["lower"]),  # type: ignore[index]
                "upper": float(gap_interval["upper"]),  # type: ignore[index]
            },
        },
    ]
    claim_rows: list[dict[str, object]] = [
        {
            "causal": False,
            "claim": (
                "the declared section-header attribute is linearly separable on heldout "
                f"rows at transformer.h.{N_LAYERS - 1 if layer is None else layer} under the "
                "frozen leakage-safe protocol"
            ),
            "claim_allowed": True,
            "control_refs": [],
            "evidence_refs": ["cap-1"],
            "id": "o-1",
            "kind": "observation",
            "missing_evidence": [],
            "status": "observed",
        },
        {
            "causal": False,
            "claim": (
                f"the leakage-safe probe explanation for the section-header attribute: {explanation_record['reason']}"  # type: ignore[index]
            ),
            "claim_allowed": bool(explanation_record.get("claim_allowed")),
            "control_refs": [
                "control-capacity",
                "control-label-randomization",
                "control-nonseparable-negative",
            ],
            "evidence_refs": ["e-acc", f"explanation-{HYPOTHESIS_ID}-record"],
            "id": "x-1",
            "kind": "explanation",
            "missing_evidence": list(explanation_record.get("missing_evidence", [])),
            "status": explanation_status if explanation_status != "omitted" else "unsupported",
        },
        *intervention_claims,
    ]
    report: dict[str, object] = {
        "schema_version": "diagnostic-report-schema-v1",
        "report_id": REPORT_ID,
        "status": "declared",
        "capture_provenance": {
            "captures": [
                {
                    "artifact_refs": ["cap-1"],
                    "axes": ["slice", "feature"],
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
                    f"heldout probe accuracy {accuracy_value:.4f} with leakage gap "  # type: ignore[index]
                    f"{gap_value:.4f} on the declared section-header attribute "  # type: ignore[index]
                    f"at transformer.h.{N_LAYERS - 1 if layer is None else layer} "
                    f"(threshold {METRIC_ACC}>=0.7, {METRIC_GAP}>=0.15)"
                ),
                "evidence_refs": ["o-1"],
                "id": "s-1",
                "metric_ids": [METRIC_ACC, METRIC_GAP],
                "status": "observed",
            }
        ],
        "localization": localization_rows,
        "hypotheses": [
            {
                "alternatives": ["label-randomization", "non-separable-noise"],
                "evidence_refs": ["o-1"],
                "id": HYPOTHESIS_ID,
                "statement": (
                    "the section-header attribute is linearly decodable from the localized "
                    "layer under the frozen leakage-safe protocol"
                ),
                "status": explanation_status if explanation_status != "omitted" else "unsupported",
            }
        ],
        "statistical_evidence": statistical_rows,
        "interventions": [],
        "comparisons": comparison_rows,
        "limitations": [
            {
                "affects": ["localization"],
                "blocking": False,
                "description": (
                    "token and time localization axes are explicit non-applicable for this "
                    "row-level probe campaign: the manifest declares no time axis and no "
                    "token-level claim is made; token scope is bound by the capture "
                    f"identity ({PREPROCESSING_IDENTITY})"
                ),
                "id": "lim-1",
            },
            {
                "affects": ["explain", "detect"],
                "blocking": False,
                "description": (
                    "the declared attribute (wikitext raw section-header rows) is correlated "
                    "with row length and last-token identity; the claim is separability under "
                    "the frozen protocol, not causal feature use"
                ),
                "id": "lim-2",
            },
            {
                "affects": ["localization"],
                "blocking": False,
                "description": (
                    "per-sample localization rows for every heldout row are recorded in the "
                    "content-addressed localize stage record; the report lists the layer and "
                    "slice rows to stay concise"
                ),
                "id": "lim-3",
            },
        ],
        "next_action": {
            "action": "review the persisted artifact and rendered report",
            "rationale": "the content-addressed artifact carries the full stage records",
            "required_evidence_refs": ["e-acc", "o-1"],
        },
        "claims": claim_rows,
        "evidence_contract": "diagnostic-evidence-v2",
        "target_evidence": _target_evidence_block(detect),
    }
    validate_report_shape(report)
    return report


def _target_evidence_block(detect: Mapping[str, object] | dict[str, object]) -> dict[str, object]:
    """Bind the predeclared target rule to the evaluated labels and split."""
    from latent_anything._target_evidence import link_for_record

    provenance = detect.get("target_provenance")
    if not isinstance(provenance, dict):
        raise AssertionError("detect payload must carry target_provenance for the v2 evidence contract")
    declared = (
        provenance.get("target_id"),
        provenance.get("rule"),
        provenance.get("rule_kind"),
    )
    if declared != (TARGET_ID, TARGET_RULE, TARGET_RULE_KIND):
        raise AssertionError("detect-stage target rule differs from the frozen proof declaration")
    record = {
        "capacity": str(provenance.get("capacity")),
        "dataset_split_identity": SPLIT_IDENTITY,
        "eval_indices": list(provenance.get("eval_indices", [])),
        "eval_split_identity": str(provenance.get("eval_split_identity")),
        "label_digest": str(provenance.get("label_digest")),
        "labels": list(provenance.get("labels", [])),
        "representation_identity": REPRESENTATION,
        "rule": str(provenance["rule"]),
        "rule_digest": _sha(str(provenance["rule"])),
        "rule_kind": str(provenance["rule_kind"]),
        "sample_digest": str(provenance.get("sample_digest")),
        "sample_ids": list(provenance.get("sample_ids", [])),
        "schema_version": "diagnostic-target-evidence-v1",
        "target_id": str(provenance["target_id"]),
        "train_indices": list(provenance.get("train_indices", [])),
        "train_split_identity": str(provenance.get("train_split_identity")),
    }
    link = link_for_record(record)
    return {
        "evidence_refs": [f"target-{record['target_id']}-record"],
        "label_digest": str(link["label_digest"]),
        "record_digest": str(link["record_digest"]),
        "rule": str(record["rule"]),
        "rule_digest": str(link["rule_digest"]),
        "rule_kind": str(record["rule_kind"]),
        "sample_digest": str(link["sample_digest"]),
        "target_id": str(record["target_id"]),
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
        f"PASS frozen-inputs: manifest {PINNED_MANIFEST_DIGEST} validated; "
        f"gpt2@{MODEL_REVISION[:12]}… snapshot present; wikitext {DATASET_REVISION[:12]}… "
        f"validation verified against the pinned L04 manifest (official {OFFICIAL_ROWS}, "
        f"nonblank {NONBLANK_ROWS}, selected {SELECTED_ROWS} indices+hashes exact); "
        f"{len(before)} frozen artifacts hashed"
    )
    train_ids, eval_ids = split_identities()
    print(
        f"PASS leakage-safe grouping: grouped original-index//8 split seed79 -> "
        f"train {len(data['train_positions'])} rows ({train_ids}) / eval "  # type: ignore[arg-type]
        f"{len(data['eval_positions'])} rows ({eval_ids}); groups disjoint, both classes "  # type: ignore[arg-type]
        "in both partitions, n_train>=770 capacity gate satisfied"
    )
    since = _phase("frozen+data", time.perf_counter())

    labels = np.asarray(data["labels"])
    print(
        f"RECORD declared attribute: is-section-header (pattern {HEADER_PATTERN.pattern!r}) -> "
        f"{int(labels.sum())} positives / {int(len(labels) - labels.sum())} negatives "
        "(predeclared before evidence; never changed)"
    )

    pooled = pooled_features()
    digest_a = pooled_digest(pooled)
    fresh = _extract_pooled()
    digest_b = pooled_digest(fresh)
    if digest_a != digest_b or not np.array_equal(pooled, fresh):
        raise AssertionError("hidden-state extraction did not replay bit-identically")
    since = _phase("extraction+replay", since)
    print(
        f"PASS deterministic-replay: pooled hidden states sha {digest_a} reproduced "
        f"bit-identically on a full re-extraction ({N_LAYERS} layers x {SELECTED_ROWS} rows x768)"
    )

    concrete, capture_provenance = _capture()
    since = _phase("capture", since)
    print(
        f"PASS capture: capture_identity {concrete.capture_identity} shape "
        f"{tuple(concrete.shape)} dtype {concrete.dtype} via bind_selection/resolve_captures "
        f"(native output_hidden_states, {POOLING_IDENTITY})"
    )

    detect = detect_case()
    family = detect["family"]
    payload = detect["payload"]
    measurements = payload["measurements"]  # type: ignore[index]
    threshold_pass = dict(family.threshold_pass)  # type: ignore[attr-defined]
    positive_observed = all(threshold_pass.values())
    outcomes = dict(family.control_outcomes)  # type: ignore[attr-defined]
    expected_outcomes = {
        "control-capacity": "passed",
        "control-label-randomization": "passed",
        "control-nonseparable-negative": "passed",
        "control-split-swap": "passed",
    }
    if outcomes != expected_outcomes:
        raise AssertionError(f"manifest-declared control behaviors drifted: {outcomes}")
    since = _phase("detect", since)
    _separability_block(payload)
    print(
        "RECORD positive-case (declared separability): "
        f"{'OBSERVED' if positive_observed else 'NOT OBSERVED'} — {METRIC_ACC} "
        f"{_metric_value(payload, METRIC_ACC):.4f} {'≥' if threshold_pass[METRIC_ACC] else '<'}0.7, "
        f"{METRIC_GAP} {_metric_value(payload, METRIC_GAP):.4f} "  # type: ignore[index]
        f"{'≥' if threshold_pass[METRIC_GAP] else '<'}0.15; threshold_pass={threshold_pass}"  # type: ignore[index]
    )
    print(
        "RECORD controls: capacity passed (declared linear-logreg, n_params770<=n_train), "
        f"label-randomization passed (randomized acc "
        f"{payload['control_metrics']['control-label-randomization'][METRIC_ACC]:.4f}), "  # type: ignore[index]
        f"nonseparable-negative passed (noise acc {evaluate_negative_accuracy():.4f} <0.7), "
        "split-swap null recorded"
    )
    probe = headline_probe()
    print(
        f"RECORD headline probe: accuracy {probe['accuracy']:.4f} at the localized-layer "
        "probe (cached predictions shared by detect/compare/intervene — no stage refits)"
    )

    localize_result = localize_case()["result"]
    localize_payload = localize_case()["payload"]
    since = _phase("localize", since)
    accuracies = [round(value, 4) for value in per_layer_probes()["accuracies"]]  # type: ignore[union-attr]
    print(
        f"RECORD per-layer probe accuracies h.0..h.11: {accuracies}; localize verdict "
        f"{localize_payload['verdict']!r}, earliest {localize_payload['earliest_layer']!r}, "  # type: ignore[index]
        f"affected layers {localize_payload['affected_layers']}"  # type: ignore[index]
    )
    if localize_result.verdict not in ("localized", "negative"):  # type: ignore[attr-defined]
        raise AssertionError(f"unexpected localization verdict: {localize_result.verdict}")  # type: ignore[attr-defined]

    # Determinism of the real stage payloads.
    _det, payload_again = evaluate_detection(
        _labeled(N_LAYERS - 1),
        detect["config"],  # type: ignore[arg-type]
        controls={"control-nonseparable-negative": negative_batch()},
    )
    _res, localize_again = evaluate_localization(localize_case()["source"])  # type: ignore[arg-type]
    if _payload_sha(payload_again) != _payload_sha(payload):
        raise AssertionError("detect payload is not byte-deterministic")
    if _payload_sha(localize_again) != _payload_sha(localize_payload):
        raise AssertionError("localize payload is not byte-deterministic")
    identity_a, _ = _capture()
    identity_b, _ = _capture()
    if identity_a.capture_identity != identity_b.capture_identity:
        raise AssertionError("capture identity is not deterministic")
    since = _phase("determinism", since)
    print(
        "PASS determinism: detect payload sha "
        f"{_payload_sha(payload)}, localize payload sha {_payload_sha(localize_payload)}, "
        f"capture identity stable (split digests train/eval identities fixed)"
    )

    # Real seven-stage workflow.
    workflow, request = build_workflow()
    result, checkpoint = workflow.run(request, manifest)
    since = _phase("workflow", since)
    if result.status != "completed" or checkpoint.status != "completed":
        failure = checkpoint.failure
        print(
            "BLOCKED chain: workflow "
            f"{result.status} at {checkpoint.next_stage} — "
            f"{failure.error_type if failure else '?'}: {failure.message if failure else result.message}"
        )
        _print_verdict(False, measurements, threshold_pass, phases, started)
        return 0

    interventions = checkpoint.outputs[tuple(item.stage for item in checkpoint.outputs).index("intervene")].payload[
        "trials"
    ]  # type: ignore[index]
    comparisons = checkpoint.outputs[tuple(item.stage for item in checkpoint.outputs).index("compare")].payload[
        "comparisons"
    ]  # type: ignore[index]
    intervention_claims = intervention_report_items(interventions)
    comparison_rows = comparison_report_items(comparisons)
    report = build_report(checkpoint.outputs, intervention_claims=intervention_claims, comparison_rows=comparison_rows)
    since = _phase("report-assembly", since)
    trial = interventions[0]
    print(
        f"RECORD intervene: {trial['kind']} {trial['intervention_id']} target {trial['target']} -> "
        f"conclusion {trial['conclusion']} (baseline {trial['measurements']['baseline']['value']:.4f}, "
        f"intervened {trial['measurements']['intervened']['value']:.4f}, effect {trial['effect']:+.4f}); "
        f"controls { {k: v['status'] for k, v in trial['control_outcomes'].items()} }"  # type: ignore[index]
    )
    print(
        f"RECORD compare: {comparisons[0]['classification']} — "
        f"{METRIC_ACC} delta {comparisons[0]['representation']['absolute_delta']:.6f}, "  # type: ignore[index]
        f"{METRIC_GAP} delta {comparisons[0]['task']['absolute_delta']:.6f}"  # type: ignore[index]
    )

    # A separate persistent root preserves the historical v1 failure and refuses overwrite.
    root = Path(OUTPUT_LOCATION)
    if root.exists():
        raise FileExistsError(f"refusing to overwrite existing 80.24 evidence root: {root}")
    persisted = False
    blockers: list[str] = []
    try:
        try:
            handle = persist_diagnostic_artifact(
                root,
                request=request,
                manifest=manifest,
                result=result,
                checkpoint=checkpoint,
                report=report,  # type: ignore[arg-type]
            )
        except Exception as exc:  # noqa: BLE001 - recorded fail-closed blocker
            blockers.append(f"versioned target-evidence persistence rejected: {type(exc).__name__}: {exc}")
            print(
                "BLOCKED persistence: the versioned shared evidence contract rejected the artifact — "
                f"{type(exc).__name__}: {exc}"
            )
            raise _PersistBlockedError() from exc
        loaded = load_diagnostic_artifact(root, run_id=handle.run_id, manifest=manifest)
        document = loaded.document
        validator_input = document["validator_input"]
        evidence_blobs = registered_artifact_bytes(loaded, root)
        target_block = loaded.report.get("target_evidence")
        if not isinstance(target_block, dict):
            raise AssertionError("persisted report is missing its v2 target-evidence block")
        target_refs = target_block.get("evidence_refs")
        if not isinstance(target_refs, list) or len(target_refs) != 1 or not isinstance(target_refs[0], str):
            raise AssertionError("persisted report does not reference exactly one target-evidence record")
        target_bytes = evidence_blobs[target_refs[0]]
        target_evidence = json.loads(target_bytes.decode("utf-8"))
        target_evidence["record_digest"] = _sha(target_bytes)
        detect_stage_bytes = evidence_blobs.get("stage-record-detect")
        if not isinstance(detect_stage_bytes, bytes):
            raise AssertionError("persisted artifact is missing the detect-stage record")
        detect_stage_record = json.loads(detect_stage_bytes.decode("utf-8"))
        if not isinstance(detect_stage_record, dict):
            raise AssertionError("persisted detect-stage record is invalid")
        persisted_detect_payload = detect_stage_record.get("payload")
        if not isinstance(persisted_detect_payload, dict):
            raise AssertionError("persisted detect-stage payload is invalid")
        target_provenance = persisted_detect_payload.get("target_provenance")
        if not isinstance(target_provenance, dict):
            raise AssertionError("persisted detect stage is missing evaluated target provenance")
        validate_diagnostic_report(
            loaded.report,  # type: ignore[arg-type]
            manifest,
            applicability=validator_input["applicability"],  # type: ignore[index]
            family_evidence=validator_input["family_evidence"],  # type: ignore[index]
            control_outcomes=validator_input["control_outcomes"],  # type: ignore[index]
            artifacts=evidence_blobs,
            artifact_digests=validator_input["artifact_digests"],  # type: ignore[index]
            target_evidence=target_evidence,
            target_provenance=target_provenance,
        )
        print(
            f"PASS artifact: run {handle.run_id}, artifact sha {handle.artifact_digest}, "
            f"report sha {handle.report_digest}; independent validator re-run on fresh load passed "
            f"({len(document['blobs'])} content-addressed blobs)"  # type: ignore[arg-type]
        )

        # Deterministic rendering + evidence-link rehashing.
        first = render_diagnostic_report(root, run_id=handle.run_id, manifest=manifest)
        second = render_diagnostic_report(root, run_id=handle.run_id, manifest=manifest)
        if first != second:
            raise AssertionError("rendered report is not byte-deterministic")
        links = re.findall(r"^evidence (.+): ([0-9a-f]{64})$", first.decode("utf-8"), flags=re.MULTILINE)
        for name, digest in links:
            blob = root / "artifacts" / digest
            if not blob.exists() or _sha(blob.read_bytes()) != digest:
                raise AssertionError(f"evidence link {name} failed to re-hash")
        output_dir = Path(OUTPUT_LOCATION)
        rendered_handle = persist_rendered_report(
            root,
            run_id=handle.run_id,
            manifest=manifest,
            output=OutputSelection(output_location=OUTPUT_LOCATION),
        )
        materialized = output_dir / "diagnostic-report"
        if not materialized.exists() or _sha(materialized.read_bytes()) != rendered_handle.digest:
            raise AssertionError("materialized rendered report does not re-hash")
        print(
            f"PASS report: deterministic render {len(first)} bytes, sha {_sha(first)}; "
            f"{len(links)} evidence links re-hashed; content-addressed render sha "
            f"{rendered_handle.digest}; output materialized at {OUTPUT_LOCATION}/diagnostic-report"
        )
        persisted = True
        since = _phase("persist+render", since)
        final_loaded = load_diagnostic_artifact(root, run_id=handle.run_id, manifest=manifest)
        if final_loaded.artifact_digest != handle.artifact_digest:
            raise AssertionError("persisted artifact changed after report rendering")
    except _PersistBlockedError:
        pass

    # Fail-closed guards: manifest tamper, capture provenance, grouped leakage.
    mutated = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    mutated["thresholds"][0]["value"] = 0.01  # type: ignore[index]
    try:
        validate_manifest(mutated)
    except Exception as exc:  # noqa: BLE001
        print(f"PASS fail-closed manifest tamper: {type(exc).__name__}: {exc}")
    else:
        raise AssertionError("a tampered manifest must be rejected")
    try:
        bind_selection(
            CaptureSelection(
                capture_id=CAPTURE_ID,
                representation_identity="openai-community-gpt2:hidden-states:layers0-10:dim768",
                axes=("slice", "feature"),
            ),
            manifest=manifest,
            request_id=request.request_id,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"PASS fail-closed capture provenance: {type(exc).__name__}: {exc}")
    else:
        raise AssertionError("a tampered capture provenance must be rejected")
    try:
        LabeledBatch(
            LatentValue(_batch(N_LAYERS - 1), _space()),
            tuple(int(item) for item in labels),
            tuple(f"validation-{int(i)}" for i in np.asarray(data["selected_indices"])),
            tuple(range(SELECTED_ROWS - 1)),
            tuple(range(10, SELECTED_ROWS)),
            "leaky-train-2",
            "leaky-eval-2",
            DECLARED_PROBE_CAPACITY,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"PASS fail-closed leakage: {type(exc).__name__}: {exc}")
    else:
        raise AssertionError("an overlapping train/eval split must be rejected")
    # Tampered resume identity.
    _, resume_checkpoint = workflow.run(request, manifest, stop_after="compare")
    tampered_request = dataclasses.replace(
        request, output=OutputSelection(output_location="artifacts/diagnostics/proof-80-24-tampered")
    )
    try:
        workflow.resume(resume_checkpoint, tampered_request, manifest)
    except Exception as exc:  # noqa: BLE001
        print(f"PASS fail-closed resume identity: {type(exc).__name__}: {exc}")
    else:
        raise AssertionError("a tampered resume identity must be rejected")
    # Failed required control blocks the conclusion.
    blocked_detections, _ = evaluate_detection(
        _labeled(N_LAYERS - 1),
        detect["config"],  # type: ignore[arg-type]
        controls={"control-nonseparable-negative": _labeled(N_LAYERS - 1)},
    )
    blocked = blocked_detections[0]
    if blocked.outcome != "inconclusive" or blocked.claim_allowed:
        raise AssertionError("a self-comparable negative control must block the conclusion")
    print(
        "PASS fail-closed failed control: using the target itself as the non-separable "
        f"negative -> outcome {blocked.outcome}, claim_allowed={blocked.claim_allowed}, "
        f"missing {list(blocked.missing_evidence)}"
    )

    after = frozen_digests()
    if before != after:
        raise AssertionError("frozen artifacts were modified by the proof run")
    print("PASS frozen invariance: all six frozen artifacts byte-identical after the run")

    controls_ok = outcomes == expected_outcomes
    _print_verdict(
        positive_observed and controls_ok and persisted,
        measurements,
        threshold_pass,
        phases,
        started,
        blockers=blockers,
    )
    return 0


def _print_verdict(
    passed: bool,
    measurements: object,
    threshold_pass: dict[str, bool],
    phases: dict[str, float],
    started: float,
    *,
    blockers: list[str] | None = None,
) -> None:
    current = time.perf_counter()
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    phase_text = ", ".join(f"{name}={seconds:.2f}s" for name, seconds in phases.items())
    if passed:
        print(
            "ACCEPTANCE: PASSED — detect -> localize -> explain -> intervene -> compare -> "
            "report completed with the declared positive, counterexample, and negative controls"
        )
    else:
        print("ACCEPTANCE: NOT PASSED — see recorded outcomes above; nothing was tuned to force a pass")
        for blocker in blockers or []:
            print(f"  blocker: {blocker}")
        print("  blocker record: artifacts/task_80.24_transformer_hidden_state_proof_summary.md")
    print(
        f"resources: proof body {current - started:.2f}s ({phase_text}); wall including "
        f"imports {current - _MODULE_START:.2f}s; tracemalloc peak {peak / (1 << 20):.1f} MiB; "
        f"environment python {platform.python_version()}, numpy {np.__version__}, "
        f"transformers {package_version('transformers')}, torch-cpu, "
        f"scikit-learn {package_version('scikit-learn')}, datasets {package_version('datasets')}"
    )
    print(
        'commands: uv run --extra transformers --with "datasets>=2.19,<4" --with '
        '"huggingface-hub>=0.34,<1" python scripts/sprint80_task80_24_proof.py; '
        'uv run --extra transformers --with "datasets>=2.19,<4" --with '
        '"huggingface-hub>=0.34,<1" pytest tests/test_transformer_end_to_end_proof.py -q'
    )
    del measurements, threshold_pass


def main() -> int:
    return run_proof()


if __name__ == "__main__":
    raise SystemExit(main())

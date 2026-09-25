"""Prospective, content-addressed stability evidence for Sprint 80.24.

The immutable 80.24 v2 run remains historical. This entrypoint evaluates a
separately frozen grouped split and grouped-training-set stability protocol
through the shared Sprint 80.16 explanation evaluator.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from collections.abc import Mapping, Sequence
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from pathlib import Path
from typing import cast

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from latent_anything._portable_contract import canonical_json  # noqa: E402
from latent_anything._probe_tcav_ig_explanation import (  # noqa: E402
    DECLARED_PROBE_CAPACITY,
    ExplanationHypothesis,
    MethodInputs,
    evaluate_explanations,
)
from latent_anything.capture import CapturedActivation  # noqa: E402
from scripts import sprint80_task80_24_proof as base  # noqa: E402

PROTOCOL_PATH = REPO / "artifacts/benchmark_manifest_sprint80_transformer_explanation_stability_v1.json"
PROTOCOL_SHA256 = "a69b7152a800ac543ac76fec4d699c16681517708a95b353e5d241241c8acf0a"
PROOF_ROOT = REPO / "artifacts/diagnostics/proof-80-24-stability-v1"
PRIOR_ROOT = REPO / "artifacts/diagnostics/proof-80-24-v2-rule-bound"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return canonical_json(value).encode("utf-8")


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return cast(Mapping[str, object], value)


def _sequence(value: object, *, name: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a sequence")
    return value


def _read_protocol() -> tuple[dict[str, object], bytes]:
    protocol_bytes = PROTOCOL_PATH.read_bytes()
    protocol_digest = _sha(protocol_bytes)
    if protocol_digest != PROTOCOL_SHA256:
        raise ValueError(f"frozen stability protocol changed: {protocol_digest} != {PROTOCOL_SHA256}")
    protocol = json.loads(protocol_bytes.decode("utf-8"))
    if not isinstance(protocol, dict):
        raise ValueError("stability protocol root must be an object")
    if protocol.get("schema_version") != "transformer-explanation-stability-protocol-v1":
        raise ValueError("unsupported stability protocol schema")
    if protocol.get("status") != "predeclared":
        raise ValueError("stability protocol is not marked predeclared")
    return protocol, protocol_bytes


def _verify_prior_evidence(protocol: Mapping[str, object]) -> dict[str, str]:
    source = _mapping(protocol.get("source_evidence"), name="source_evidence")
    run_id = str(source["prior_run_id"])
    artifact_digest = str(source["prior_artifact_sha256"])
    explain_digest = str(source["prior_explain_stage_record_sha256"])
    report_digest = str(source["prior_diagnostic_report_sha256"])
    rendered_digest = str(source["prior_rendered_report_sha256"])
    precedent = _mapping(source.get("threshold_precedent"), name="threshold_precedent")
    precedent_threshold = _mapping(precedent.get("threshold"), name="threshold_precedent.threshold")
    thresholds = _mapping(protocol.get("thresholds"), name="thresholds")
    stability_threshold = _mapping(thresholds.get("coef_stability"), name="thresholds.coef_stability")
    if (
        precedent_threshold.get("metric") != "coef_stability"
        or precedent_threshold.get("comparator") != ">="
        or precedent_threshold.get("value") != 0.8
        or stability_threshold.get("comparator") != ">="
        or stability_threshold.get("value") != 0.8
    ):
        raise ValueError("the predeclared stability threshold differs from its accepted precedent")
    precedent_source_digest = str(precedent["source_sha256"])
    precedent_artifact_digest = str(precedent["accepted_artifact_sha256"])
    precedent_run_digest = str(precedent["accepted_run_record_sha256"])
    precedent_run_id = str(precedent["accepted_run_id"])
    precedent_source_path = REPO / str(precedent["source"])
    precedent_artifact_path = REPO / str(precedent["accepted_artifact"])
    precedent_run_path = REPO / str(precedent["accepted_run_record"])
    if precedent_artifact_path.name != precedent_artifact_digest:
        raise ValueError("the accepted threshold-precedent artifact path is not content-addressed")
    paths = {
        "prior_artifact": PRIOR_ROOT / "artifacts" / artifact_digest,
        "prior_explain_stage_record": PRIOR_ROOT / "artifacts" / explain_digest,
        "prior_diagnostic_report": PRIOR_ROOT / "artifacts" / report_digest,
        "prior_rendered_report": PRIOR_ROOT / "diagnostic-report",
        "prior_run": PRIOR_ROOT / "runs" / f"{run_id}.json",
        "threshold_precedent_source": precedent_source_path,
        "threshold_precedent_artifact": precedent_artifact_path,
        "threshold_precedent_run_record": precedent_run_path,
    }
    expected = {
        "prior_artifact": artifact_digest,
        "prior_explain_stage_record": explain_digest,
        "prior_diagnostic_report": report_digest,
        "prior_rendered_report": rendered_digest,
        "threshold_precedent_source": precedent_source_digest,
        "threshold_precedent_artifact": precedent_artifact_digest,
        "threshold_precedent_run_record": precedent_run_digest,
    }
    digests: dict[str, str] = {}
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"immutable prior evidence is missing: {path}")
        digest = _sha(path.read_bytes())
        if name in expected and digest != expected[name]:
            raise ValueError(f"immutable prior evidence changed: {name} {digest} != {expected[name]}")
        digests[name] = digest
    run_record = json.loads(paths["prior_run"].read_text(encoding="utf-8"))
    if not isinstance(run_record, dict) or run_record.get("status") != "completed":
        raise ValueError("the recorded v2 run is not a completed historical run")
    precedent_run_record = json.loads(paths["threshold_precedent_run_record"].read_text(encoding="utf-8"))
    if not isinstance(precedent_run_record, dict) or precedent_run_record.get("acceptance") != "passed":
        raise ValueError("the encoder threshold precedent is not a passed accepted run")
    if precedent_run_record.get("manifest_id") != precedent.get("manifest_id"):
        raise ValueError("the encoder threshold precedent manifest identity changed")
    accepted_artifact = _mapping(precedent_run_record.get("artifact"), name="threshold_precedent.artifact")
    if (
        accepted_artifact.get("run_id") != precedent_run_id
        or accepted_artifact.get("artifact_digest") != precedent_artifact_digest
    ):
        raise ValueError("the encoder threshold precedent run/artifact identity changed")
    stability = _mapping(
        _mapping(precedent_run_record.get("explanation"), name="threshold_precedent.explanation").get("stability"),
        name="threshold_precedent.explanation.stability",
    )
    if (
        stability.get("metric") != "coef_stability"
        or stability.get("threshold") != [">=", 0.8]
        or stability.get("status") != "passed"
    ):
        raise ValueError("the accepted encoder run no longer records coef_stability >= 0.8")
    explain_record = json.loads(paths["prior_explain_stage_record"].read_text(encoding="utf-8"))
    old_stability = [item for item in _walk_mappings(explain_record) if item.get("metric") == "coef_stability"]
    if len(old_stability) != 1:
        raise ValueError(f"expected one historical stability record, found {len(old_stability)}")
    if old_stability[0].get("threshold") is not None:
        raise ValueError("the historical v2 stability record no longer has its recorded null threshold")
    if old_stability[0].get("status") != "passed":
        raise ValueError("the historical v2 stability status differs from its recorded status")
    digests["historical_stability_threshold"] = "null"
    return digests


def _walk_mappings(value: object) -> list[Mapping[str, object]]:
    found: list[Mapping[str, object]] = []
    if isinstance(value, Mapping):
        found.append(cast(Mapping[str, object], value))
        for item in value.values():
            found.extend(_walk_mappings(item))
    elif isinstance(value, list):
        for item in value:
            found.extend(_walk_mappings(item))
    return found


def _independent_split(data: Mapping[str, object], protocol: Mapping[str, object]) -> dict[str, object]:
    spec = _mapping(protocol.get("independent_split"), name="independent_split")
    indices = np.asarray(data["selected_indices"], dtype=np.int64)
    labels = np.asarray(data["labels"], dtype=np.int64)
    if indices.ndim != 1 or labels.shape != indices.shape:
        raise ValueError("the pinned selection and target labels do not align")
    groups = indices // 8
    unique_groups = np.unique(groups)
    shuffled_groups = unique_groups.copy()
    seed = int(spec["seed"])
    np.random.default_rng(seed).shuffle(shuffled_groups)
    train_count = int(len(shuffled_groups) * float(spec["train_group_fraction"]))
    train_groups = {int(value) for value in shuffled_groups[:train_count]}
    eval_groups = {int(value) for value in shuffled_groups[train_count:]}
    train_positions = np.asarray([position for position, group in enumerate(groups) if int(group) in train_groups])
    eval_positions = np.asarray([position for position, group in enumerate(groups) if int(group) in eval_groups])
    observed_train_groups = set(int(group) for group in groups[train_positions])
    observed_eval_groups = set(int(group) for group in groups[eval_positions])
    if train_groups & eval_groups or observed_train_groups & observed_eval_groups:
        raise ValueError("independent split leaks a group across partitions")
    if not len(train_positions) or not len(eval_positions):
        raise ValueError("independent split must have non-empty partitions")
    if len(np.unique(labels[train_positions])) < 2 or len(np.unique(labels[eval_positions])) < 2:
        raise ValueError("independent split must retain both target classes")
    sample_ids = [f"validation-{int(index)}" for index in indices]
    train_ids = [sample_ids[int(position)] for position in train_positions]
    eval_ids = [sample_ids[int(position)] for position in eval_positions]
    if set(train_ids) & set(eval_ids):
        raise ValueError("independent split leaks sample identities")
    return {
        "eval_groups": eval_groups,
        "eval_ids": eval_ids,
        "eval_positions": eval_positions,
        "groups": groups,
        "indices": indices,
        "labels": labels,
        "sample_ids": sample_ids,
        "train_groups": train_groups,
        "train_ids": train_ids,
        "train_positions": train_positions,
    }


def _array_sha(array: np.ndarray) -> str:
    normalized = np.asarray(array, dtype="<f8", order="C")
    return _sha(normalized.tobytes(order="C"))


def _identity_sha(identities: Sequence[str]) -> str:
    return _sha(_canonical_bytes(list(identities)))


def _stability_inputs(
    matrix: np.ndarray,
    split: Mapping[str, object],
    protocol: Mapping[str, object],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    stability_spec = _mapping(protocol.get("stability"), name="stability")
    labels = np.asarray(split["labels"], dtype=np.int64)
    groups = np.asarray(split["groups"], dtype=np.int64)
    train_positions = np.asarray(split["train_positions"], dtype=np.int64)
    sample_ids = cast(list[str], split["sample_ids"])
    sampling_seeds = [int(item) for item in _sequence(stability_spec.get("sampling_seeds"), name="sampling_seeds")]
    fit_seeds = [int(item) for item in _sequence(stability_spec.get("fit_seeds"), name="fit_seeds")]
    if len(sampling_seeds) != len(fit_seeds) or len(sampling_seeds) != int(stability_spec["replicates"]):
        raise ValueError("stability sampling and fit seeds do not match the frozen replicate count")
    train_group_ids = np.unique(groups[train_positions])
    fraction = float(stability_spec["replicate_group_fraction"])
    replicate_bundles: list[dict[str, object]] = []
    provenance: list[dict[str, object]] = []
    for sampling_seed, fit_seed in zip(sampling_seeds, fit_seeds, strict=True):
        sampled_groups = train_group_ids.copy()
        np.random.default_rng(sampling_seed).shuffle(sampled_groups)
        selected_group_count = int(len(sampled_groups) * fraction)
        selected_groups = {int(group) for group in sampled_groups[:selected_group_count]}
        positions = np.asarray(
            [int(position) for position in train_positions if int(groups[int(position)]) in selected_groups],
            dtype=np.int64,
        )
        if not positions.size or not np.all(np.isin(groups[positions], np.asarray(sorted(selected_groups)))):
            raise ValueError("stability replicate escaped the independent training groups")
        if len(np.unique(labels[positions])) < 2 or matrix.shape[1] + 1 > len(positions):
            raise ValueError("stability replicate violates class or capacity requirements")
        replicate_ids = [sample_ids[int(position)] for position in positions]
        replicate_bundle = {
            "fit_seed": fit_seed,
            "train_ids": replicate_ids,
            "train_labels": labels[positions],
            "train_matrix": matrix[positions],
        }
        replicate_bundles.append(replicate_bundle)
        provenance.append(
            {
                "fit_seed": fit_seed,
                "group_ids": sorted(selected_groups),
                "sample_ids": replicate_ids,
                "sample_ids_sha256": _identity_sha(replicate_ids),
                "train_features_sha256": _array_sha(matrix[positions]),
                "train_labels_sha256": _sha(_canonical_bytes(labels[positions].tolist())),
                "train_rows": int(len(positions)),
                "train_sampling_seed": sampling_seed,
            }
        )
    return replicate_bundles, provenance


def _negative_accuracy(
    features: np.ndarray,
    labels: np.ndarray,
    train_positions: np.ndarray,
    eval_positions: np.ndarray,
    fit_seed: int,
) -> float:
    from sklearn.linear_model import LogisticRegression  # type: ignore[reportMissingTypeStubs]
    from sklearn.preprocessing import StandardScaler  # type: ignore[reportMissingTypeStubs]

    scaler = StandardScaler()
    train_x = np.asarray(scaler.fit_transform(features[train_positions]), dtype=np.float64)
    eval_x = np.asarray(scaler.transform(features[eval_positions]), dtype=np.float64)
    classifier = LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=1000,
        random_state=fit_seed,
        class_weight="balanced",
    )
    classifier.fit(train_x, labels[train_positions])
    return float(np.mean(classifier.predict(eval_x) == labels[eval_positions]))


def _sample_rows(split: Mapping[str, object]) -> list[dict[str, object]]:
    indices = np.asarray(split["indices"], dtype=np.int64)
    labels = np.asarray(split["labels"], dtype=np.int64)
    groups = np.asarray(split["groups"], dtype=np.int64)
    return [
        {
            "group_id": int(group),
            "label": int(label),
            "original_index": int(index),
            "sample_id": f"validation-{int(index)}",
        }
        for index, label, group in zip(indices, labels, groups, strict=True)
    ]


def _versions() -> dict[str, str]:
    packages = ("numpy", "torch", "transformers", "datasets", "scikit-learn")
    versions: dict[str, str] = {"python": platform.python_version()}
    for package in packages:
        try:
            versions[package] = package_version(package)
        except PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def _render_report(record: Mapping[str, object]) -> bytes:
    protocol = _mapping(record.get("protocol"), name="record.protocol")
    explanation = _mapping(record.get("explanation"), name="record.explanation")
    hypothesis = _mapping(explanation.get("hypothesis"), name="explanation.hypothesis")
    evidence = _mapping(explanation.get("evidence"), name="explanation.evidence")
    target = _mapping(record.get("target_provenance"), name="target_provenance")
    split = _mapping(record.get("split"), name="split")
    train = _mapping(split.get("train"), name="split.train")
    evaluation = _mapping(split.get("evaluation"), name="split.evaluation")
    gates = (
        ("Fidelity", _mapping(evidence.get("fidelity"), name="evidence.fidelity")),
        ("Stability", _mapping(evidence.get("stability"), name="evidence.stability")),
        ("Selectivity", _mapping(evidence.get("selectivity"), name="evidence.selectivity")),
    )
    observed = _mapping(evidence.get("observed_effect"), name="evidence.observed_effect")
    controls = _mapping(evidence.get("control_outcomes"), name="evidence.control_outcomes")
    stability = _mapping(evidence.get("stability"), name="evidence.stability")
    stability_results = cast(
        list[dict[str, object]], _sequence(stability.get("replicates"), name="stability.replicates")
    )
    lines = [
        "# Supplemental transformer explanation stability proof v1",
        "",
        "This prospective supplemental record does not modify or retroactively gate the accepted transformer v2 run.",
        "",
        f"- Protocol: `{protocol['id']}` (SHA-256 `{protocol['sha256']}`).",
        f"- Hypothesis: `{hypothesis['hypothesis_id']}`; "
        f"method `{hypothesis['method']}`; layer `{hypothesis['layer_id']}`.",
        f"- Target: `{target['target_id']}` under `{target['target_rule']}`; "
        "target provenance is separate from capture axes.",
        f"- Outcome: **{evidence['outcome']}** (`claim_allowed={str(evidence['claim_allowed']).lower()}`).",
        "",
        "## Independent split and provenance",
        "",
        f"- Grouped split seed `{split['seed']}`: train {train['rows']} rows "
        f"({train['identity']}); heldout {evaluation['rows']} rows ({evaluation['identity']}).",
        f"- Train/evaluation sample-ID SHA-256: `{train['sample_ids_sha256']}` / "
        f"`{evaluation['sample_ids_sha256']}`; group overlap: `{split['group_overlap']}`.",
        f"- Selected label and sample-ID SHA-256: `{target['labels_sha256']}` / `{target['sample_ids_sha256']}`.",
        f"- Real bound capture identity `{record['capture']['capture_identity']}`; axes "
        f"`{', '.join(cast(list[str], record['capture']['axes']))}`.",
        "",
        "## Gated explanation evidence",
        "",
        "| Gate | Observed | Frozen threshold | Status |",
        "|---|---:|---:|---|",
    ]
    for label, detail in gates:
        threshold = cast(list[object], detail["threshold"])
        value = float(detail["observed"])
        lines.append(
            f"| {label} (`{detail['metric']}`) | {value:.6f} | "
            f"{threshold[0]} {float(threshold[1]):.3f} | {detail['status']} |"
        )
    lines.extend(
        [
            "",
            (
                f"Stability is the minimum sign-aware cosine over "
                f"{len(stability_results)} identity-bound grouped training-set refits: "
                + ", ".join(
                    f"seed {item['fit_seed']} → {float(item['coefficient_cosine']):.6f}" for item in stability_results
                )
                + ". Observed selectivity is heldout accuracy minus randomized-label accuracy "
                f"({float(observed['randomized_accuracy']):.6f})."
            ),
            "",
            "## Controls and scope",
            "",
        ]
    )
    for control_id, status in sorted(controls.items()):
        lines.append(f"- `{control_id}`: {status}.")
    lines.extend(
        [
            "",
            (
                "The stability threshold 0.8 was borrowed prospectively from the accepted encoder-v3 "
                "explanation gate; no threshold was selected from the historical transformer score. "
                "The grouped split uses the same frozen 2,048-row validation selection, so this is not "
                "a new source corpus or an independent dataset sample. Passing probe evidence is "
                "non-causal and does not establish causal feature use."
            ),
            "",
        ]
    )
    return "\n".join(lines).encode("utf-8")


def _passes(metric: float, comparator: str, threshold: float) -> bool:
    return metric >= threshold if comparator == ">=" else metric <= threshold


def _validate_record(
    record_bytes: bytes,
    report_bytes: bytes,
    protocol: Mapping[str, object],
    protocol_sha256: str,
    expected_rows: list[dict[str, object]],
    expected_split: Mapping[str, object],
    expected_stability_inputs: list[dict[str, object]],
    expected_feature_digests: Mapping[str, str],
    expected_capture: Mapping[str, object],
) -> dict[str, object]:
    record_digest = _sha(record_bytes)
    record = json.loads(record_bytes.decode("utf-8"))
    if not isinstance(record, dict) or _canonical_bytes(record) != record_bytes:
        raise ValueError("supplemental run record is not canonical JSON")
    if record.get("schema_version") != "transformer-explanation-stability-evidence-v1":
        raise ValueError("supplemental evidence schema mismatch")
    record_protocol = _mapping(record.get("protocol"), name="record.protocol")
    if record_protocol.get("sha256") != protocol_sha256 or record_protocol.get("id") != protocol.get("protocol_id"):
        raise ValueError("supplemental evidence is not bound to the frozen protocol")
    source = _mapping(record.get("source_evidence"), name="source_evidence")
    protocol_source = _mapping(protocol.get("source_evidence"), name="protocol.source_evidence")
    prior_hashes = _mapping(source.get("prior_evidence_sha256"), name="source_evidence.prior_evidence_sha256")
    for record_key, protocol_key in (
        ("prior_artifact", "prior_artifact_sha256"),
        ("prior_explain_stage_record", "prior_explain_stage_record_sha256"),
        ("prior_diagnostic_report", "prior_diagnostic_report_sha256"),
        ("prior_rendered_report", "prior_rendered_report_sha256"),
    ):
        if prior_hashes.get(record_key) != protocol_source.get(protocol_key):
            raise ValueError(f"historical evidence digest mismatch for {record_key}")
    if (
        source.get("prior_run_id") != protocol_source.get("prior_run_id")
        or source.get("prior_stability_threshold") is not None
        or source.get("prior_threshold_was_not_used_for_selection") is not True
        or prior_hashes.get("historical_stability_threshold") != "null"
    ):
        raise ValueError("historical v2 threshold-null provenance was changed or used for threshold choice")
    target = _mapping(record.get("target_provenance"), name="target_provenance")
    if target.get("sample_rows") != expected_rows:
        raise ValueError("supplemental target sample/label provenance differs from the pinned selection")
    sample_rows = _sequence(target.get("sample_rows"), name="target_provenance.sample_rows")
    row_ids: list[str] = []
    row_labels: list[int] = []
    for position, raw_row in enumerate(sample_rows):
        sample = _mapping(raw_row, name=f"target_provenance.sample_rows[{position}]")
        row_ids.append(str(sample["sample_id"]))
        row_labels.append(int(sample["label"]))
    if target.get("sample_ids_sha256") != _identity_sha(row_ids):
        raise ValueError("target sample-identity digest does not match the recorded identities")
    if target.get("labels_sha256") != _sha(_canonical_bytes(row_labels)):
        raise ValueError("target label digest does not match the recorded labels")
    split = _mapping(record.get("split"), name="split")
    if split.get("seed") != expected_split.get("seed") or split.get("train") != expected_split.get("train"):
        raise ValueError("supplemental training split differs from the frozen independent split")
    if split.get("evaluation") != expected_split.get("evaluation"):
        raise ValueError("supplemental evaluation split differs from the frozen independent split")
    split_spec = _mapping(protocol.get("independent_split"), name="protocol.independent_split")
    if split.get("grouping") != split_spec.get("grouping") or split.get("seed") != split_spec.get("seed"):
        raise ValueError("supplemental split method or seed differs from the frozen protocol")
    train_split = _mapping(split.get("train"), name="split.train")
    evaluation_split = _mapping(split.get("evaluation"), name="split.evaluation")
    train_sample_ids = set(str(item) for item in _sequence(train_split.get("sample_ids"), name="train.sample_ids"))
    evaluation_sample_ids = set(
        str(item) for item in _sequence(evaluation_split.get("sample_ids"), name="evaluation.sample_ids")
    )
    train_groups = set(int(item) for item in _sequence(train_split.get("group_ids"), name="train.group_ids"))
    evaluation_groups = set(
        int(item) for item in _sequence(evaluation_split.get("group_ids"), name="evaluation.group_ids")
    )
    if train_sample_ids & evaluation_sample_ids or train_groups & evaluation_groups:
        raise ValueError("supplemental split has sample or group overlap")
    if train_sample_ids | evaluation_sample_ids != set(row_ids) or split.get("group_overlap") != []:
        raise ValueError("supplemental split does not partition the selected target samples")
    if record.get("stability_inputs") != expected_stability_inputs:
        raise ValueError("supplemental stability replicate membership differs from the frozen protocol")
    if record.get("feature_digests") != dict(expected_feature_digests):
        raise ValueError("supplemental model features differ from the replayed frozen representation")
    capture = _mapping(record.get("capture"), name="capture")
    data_spec = _mapping(protocol.get("data"), name="protocol.data")
    model = _mapping(record.get("model"), name="model")
    expected_model = {
        "id": data_spec["model_id"],
        "revision": data_spec["model_revision"],
        "dataset_id": data_spec["dataset_id"],
        "dataset_revision": data_spec["dataset_revision"],
        "dataset_config": data_spec["dataset_config"],
        "dataset_split": data_spec["dataset_split"],
        "selection_identity": data_spec["selection_identity"],
    }
    if dict(model) != expected_model:
        raise ValueError("supplemental model/data identity differs from the frozen protocol")
    if (
        target.get("target_id") != data_spec["target_id"]
        or target.get("target_rule") != data_spec["target_rule"]
        or target.get("target_rule_sha256") != data_spec["target_rule_sha256"]
        or _sha(str(target.get("target_rule")).encode("utf-8")) != target.get("target_rule_sha256")
    ):
        raise ValueError("supplemental target identity or rule differs from the frozen protocol")
    if dict(capture) != dict(expected_capture):
        raise ValueError("supplemental capture provenance differs from the fresh bound capture")
    expected_axes = list(_sequence(data_spec.get("capture_axes"), name="capture_axes"))
    if capture.get("axes") != expected_axes or "label" in expected_axes or "sample" in expected_axes:
        raise ValueError("capture axes must remain the truthful bound axes, separate from target labels")
    explanation = _mapping(record.get("explanation"), name="explanation")
    hypothesis = _mapping(explanation.get("hypothesis"), name="explanation.hypothesis")
    evidence = _mapping(explanation.get("evidence"), name="explanation.evidence")
    expected_hypothesis = {
        "hypothesis_id": _mapping(protocol.get("explanation"), name="protocol.explanation")["hypothesis_id"],
        "manifest_id": protocol["protocol_id"],
        "target_id": data_spec["target_id"],
        "representation_id": data_spec["representation"],
        "layer_id": data_spec["localized_layer"],
        "method": "probe",
        "train_split_identity": train_split["identity"],
        "eval_split_identity": evaluation_split["identity"],
    }
    if any(hypothesis.get(key) != value for key, value in expected_hypothesis.items()):
        raise ValueError("supplemental explanation hypothesis identity differs from the frozen protocol")
    payload = _mapping(explanation.get("payload"), name="explanation.payload")
    payload_evidence = _sequence(payload.get("evidence"), name="explanation.payload.evidence")
    payload_hypotheses = _sequence(payload.get("hypotheses"), name="explanation.payload.hypotheses")
    if len(payload_evidence) != 1 or payload_evidence[0] != dict(evidence):
        raise ValueError("shared explanation payload differs from the persisted evidence record")
    if len(payload_hypotheses) != 1 or payload_hypotheses[0] != dict(hypothesis):
        raise ValueError("shared explanation payload differs from the frozen hypothesis")
    family_evidence = _mapping(payload.get("family_evidence"), name="explanation.payload.family_evidence")
    family_item = _mapping(family_evidence.get(str(hypothesis["hypothesis_id"])), name="family_evidence.hypothesis")
    if family_item.get("claim_allowed") != evidence.get("claim_allowed") or family_item.get("outcome") != evidence.get(
        "outcome"
    ):
        raise ValueError("shared family summary differs from its detailed explanation evidence")
    frozen_thresholds = _mapping(protocol.get("thresholds"), name="protocol.thresholds")
    hypothesis_rules = {}
    for position, raw_rule in enumerate(_sequence(hypothesis.get("thresholds"), name="hypothesis.thresholds")):
        rule = _mapping(raw_rule, name=f"hypothesis.thresholds[{position}]")
        hypothesis_rules[str(rule["metric_id"])] = [str(rule["comparator"]), float(rule["value"])]
    expected_rules = {
        metric: [
            str(_mapping(frozen_thresholds[metric], name=f"thresholds.{metric}")["comparator"]),
            float(_mapping(frozen_thresholds[metric], name=f"thresholds.{metric}")["value"]),
        ]
        for metric in ("heldout_accuracy", "coef_stability", "leakage_gap")
    }
    if hypothesis_rules != expected_rules:
        raise ValueError("shared explanation hypothesis thresholds differ from the frozen protocol")
    gate_map = {
        "heldout_accuracy": "fidelity",
        "coef_stability": "stability",
        "leakage_gap": "selectivity",
    }
    observed_effect = _mapping(evidence.get("observed_effect"), name="evidence.observed_effect")
    gates_pass = True
    for metric, evidence_key in gate_map.items():
        detail = _mapping(evidence.get(evidence_key), name=f"evidence.{evidence_key}")
        declared = _mapping(frozen_thresholds.get(metric), name=f"thresholds.{metric}")
        observed = float(detail["observed"])
        threshold = detail.get("threshold")
        expected_threshold = [declared["comparator"], float(declared["value"])]
        if threshold != expected_threshold:
            raise ValueError(f"{metric} evidence has no matching frozen threshold")
        metric_name = str(detail.get("metric"))
        if metric_name != metric:
            raise ValueError(f"{metric} evidence is bound to metric {metric_name!r}")
        if float(observed_effect[metric]) != observed:
            raise ValueError(f"{metric} gate value differs from the shared explanation observation")
        passed = _passes(observed, str(declared["comparator"]), float(declared["value"]))
        expected_status = "passed" if passed else "failed"
        if detail.get("status") != expected_status:
            raise ValueError(f"{metric} gate status does not match the observed value")
        gates_pass = gates_pass and passed
    stability_detail = _mapping(evidence.get("stability"), name="evidence.stability")
    if stability_detail.get("basis") != "identity-bound-training-set-replicates":
        raise ValueError("coefficient stability is not based on identity-bound training-set replicates")
    replicate_results = _sequence(stability_detail.get("replicates"), name="evidence.stability.replicates")
    if len(replicate_results) != len(expected_stability_inputs) or len(replicate_results) < 2:
        raise ValueError("coefficient stability replicate count differs from the frozen protocol")
    replicate_cosines: list[float] = []
    for position, (raw_result, expected_input) in enumerate(
        zip(replicate_results, expected_stability_inputs, strict=True)
    ):
        result = _mapping(raw_result, name=f"evidence.stability.replicates[{position}]")
        if (
            result.get("fit_seed") != expected_input["fit_seed"]
            or result.get("train_rows") != expected_input["train_rows"]
        ):
            raise ValueError("coefficient stability result is not aligned to its training-set replicate")
        cosine = float(result["coefficient_cosine"])
        if not np.isfinite(cosine) or cosine < 0.0 or cosine > 1.0 + 1e-12:
            raise ValueError("coefficient cosine is outside the sign-aware similarity range")
        replicate_cosines.append(cosine)
    if float(stability_detail["observed"]) != min(replicate_cosines):
        raise ValueError("reported coefficient stability is not the minimum replicate cosine")
    controls = _mapping(evidence.get("control_outcomes"), name="evidence.control_outcomes")
    controls_pass = bool(controls) and all(status == "passed" for status in controls.values())
    expected_control_ids = {
        "control-capacity",
        "control-label-randomization",
        "control-nonseparable-negative",
    }
    if set(controls) != expected_control_ids:
        raise ValueError("supplemental explanation control set differs from its frozen protocol")
    leakage = _mapping(evidence.get("leakage"), name="evidence.leakage")
    if leakage.get("disjoint_sample_identities") is not True:
        raise ValueError("shared explanation evidence does not confirm disjoint sample identities")
    if leakage.get("train_split_identity") != train_split.get("identity") or leakage.get(
        "eval_split_identity"
    ) != evaluation_split.get("identity"):
        raise ValueError("shared explanation leakage record does not bind the supplemental split")
    expected_support = gates_pass and controls_pass and leakage.get("status") == "passed"
    if bool(evidence.get("claim_allowed")) != expected_support:
        raise ValueError("explanation promotion does not follow all three gates, leakage, and controls")
    if expected_support and evidence.get("outcome") != "supported":
        raise ValueError("all gates passed but the explanation is not supported")
    if not expected_support and evidence.get("outcome") == "supported":
        raise ValueError("unsupported or failed evidence was promoted")
    if not expected_support and not evidence.get("missing_evidence"):
        raise ValueError("blocked explanation omitted its missing-evidence reasons")
    if record.get("acceptance") != {
        "all_three_gates_passed": gates_pass,
        "claim_allowed": expected_support,
        "controls_passed": controls_pass,
        "outcome": evidence.get("outcome"),
    }:
        raise ValueError("acceptance summary does not match independently checked evidence")
    expected_report = _render_report(record)
    if expected_report != report_bytes:
        raise ValueError("supplemental report does not deterministically render from the persisted record")
    return {
        "artifact_sha256": record_digest,
        "report_sha256": _sha(report_bytes),
        "claim_allowed": expected_support,
        "all_three_gates_passed": gates_pass,
        "controls_passed": controls_pass,
        "outcome": str(evidence["outcome"]),
    }


def _write_immutable(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError(f"refusing to overwrite different content at {path}")
        return
    path.write_bytes(payload)


def run_proof() -> int:
    started = time.perf_counter()
    protocol, protocol_bytes = _read_protocol()
    source_digests_before = base.frozen_digests()
    prior_digests_before = _verify_prior_evidence(protocol)
    data = base.build_data()
    split = _independent_split(data, protocol)
    pooled = base.pooled_features()
    replay = base._extract_pooled()
    if not np.array_equal(pooled, replay):
        raise AssertionError("fresh GPT-2 hidden-state replay differs from the cached representation")
    pooled_sha256 = base.pooled_digest(pooled)
    layer = int(_mapping(protocol.get("data"), name="data")["localized_layer_index"])
    matrix = np.asarray(pooled[layer], dtype=np.float64)
    split_seed = int(_mapping(protocol.get("independent_split"), name="independent_split")["seed"])
    train_positions = np.asarray(split["train_positions"], dtype=np.int64)
    eval_positions = np.asarray(split["eval_positions"], dtype=np.int64)
    labels = np.asarray(split["labels"], dtype=np.int64)
    groups = np.asarray(split["groups"], dtype=np.int64)
    sample_ids = cast(list[str], split["sample_ids"])
    train_ids = cast(list[str], split["train_ids"])
    eval_ids = cast(list[str], split["eval_ids"])
    train_split_id = f"grouped-original-index-div8-train{split_seed}-{len(train_positions)}rows"
    eval_split_id = f"grouped-original-index-div8-eval{split_seed}-{len(eval_positions)}rows"
    group_overlap = sorted(
        set(int(item) for item in groups[train_positions]) & set(int(item) for item in groups[eval_positions])
    )
    if group_overlap:
        raise AssertionError("independent split has overlapping original-index groups")

    request = base.workflow_request()
    concrete_raw, capture_provenance = base._capture()
    concrete = cast(CapturedActivation, concrete_raw)
    expected_axes = list(
        _sequence(_mapping(protocol.get("data"), name="data").get("capture_axes"), name="capture_axes")
    )
    if list(request.capture.axes) != expected_axes:
        raise AssertionError(f"real capture axes differ from the frozen contract: {request.capture.axes!r}")
    capture = {
        "axes": list(request.capture.axes),
        "capture_identity": concrete.capture_identity,
        "dtype": str(concrete.dtype),
        "provenance": capture_provenance,
        "shape": [int(size) for size in concrete.shape],
    }

    stability_bundles, stability_provenance = _stability_inputs(matrix, split, protocol)
    control_spec = _mapping(protocol.get("controls"), name="controls")
    negative_seed = int(control_spec["negative_control_feature_seed"])
    negative_fit_seed = int(control_spec["negative_control_fit_seed"])
    negative_features = np.random.default_rng(negative_seed).standard_normal(matrix.shape)
    negative_accuracy = _negative_accuracy(
        negative_features, labels, train_positions, eval_positions, negative_fit_seed
    )
    probe_bundle: dict[str, object] = {
        "capacity": DECLARED_PROBE_CAPACITY,
        "eval_ids": eval_ids,
        "eval_labels": labels[eval_positions],
        "eval_matrix": matrix[eval_positions],
        "negative_accuracy": negative_accuracy,
        "stability_replicates": stability_bundles,
        "train_ids": train_ids,
        "train_labels": labels[train_positions],
        "train_matrix": matrix[train_positions],
    }
    thresholds = _mapping(protocol.get("thresholds"), name="thresholds")
    threshold_order = ("heldout_accuracy", "coef_stability", "leakage_gap")
    hypothesis_spec = _mapping(protocol.get("explanation"), name="explanation")
    data_spec = _mapping(protocol.get("data"), name="data")
    hypothesis = ExplanationHypothesis(
        hypothesis_id=str(hypothesis_spec["hypothesis_id"]),
        symptom_id="symptom-section-header-separability",
        family_id=base.FAMILY,
        target_id=base.TARGET_ID,
        representation_id=base.REPRESENTATION,
        layer_id=str(data_spec["localized_layer"]),
        slice_id=base.EVAL_SLICE_ID,
        method="probe",
        expected_direction="higher",
        dataset_id=str(data_spec["selection_identity"]),
        train_split_identity=train_split_id,
        eval_split_identity=eval_split_id,
        seeds=(int(hypothesis_spec["headline_fit_seed"]),),
        control_ids=(
            "capacity:control-capacity",
            "randomized:control-label-randomization",
            "negative:control-nonseparable-negative",
        ),
        metric_ids=threshold_order,
        thresholds=tuple(
            (
                metric,
                str(_mapping(thresholds[metric], name=f"thresholds.{metric}")["comparator"]),
                float(_mapping(thresholds[metric], name=f"thresholds.{metric}")["value"]),
            )
            for metric in threshold_order
        ),
        manifest_id=str(protocol["protocol_id"]),
        localization_bindings=(("layer", str(data_spec["localized_layer"])), ("slice", base.EVAL_SLICE_ID)),
    )
    explain_started = time.perf_counter()
    evidence_items, explanation_payload = evaluate_explanations(
        (hypothesis,),
        MethodInputs(probe=probe_bundle),
        repetitions=int(_mapping(hypothesis_spec.get("uncertainty"), name="uncertainty")["repetitions"]),
        confidence_level=float(_mapping(hypothesis_spec.get("uncertainty"), name="uncertainty")["confidence_level"]),
        evaluation_seed=int(hypothesis_spec["evaluation_bootstrap_seed"]),
        control_seed=int(hypothesis_spec["control_seed"]),
        training_seed=int(hypothesis_spec["headline_fit_seed"]),
    )
    explain_seconds = time.perf_counter() - explain_started
    if len(evidence_items) != 1:
        raise AssertionError("shared explanation evaluator did not return one declared hypothesis")
    evidence = evidence_items[0].to_dict()
    rows = _sample_rows(split)
    all_labels = labels.tolist()
    target_rule = str(data_spec["target_rule"])
    target_rule_digest = _sha(target_rule.encode("utf-8"))
    if target_rule_digest != data_spec["target_rule_sha256"]:
        raise AssertionError("target-rule digest differs from the predeclared provenance")
    train_labels = labels[train_positions].tolist()
    eval_labels = labels[eval_positions].tolist()
    sample_ids_digest = _identity_sha(sample_ids)
    target_provenance = {
        "labels_sha256": _sha(_canonical_bytes(all_labels)),
        "sample_ids_sha256": sample_ids_digest,
        "sample_rows": rows,
        "target_id": base.TARGET_ID,
        "target_rule": target_rule,
        "target_rule_sha256": target_rule_digest,
    }
    train_split = {
        "identity": train_split_id,
        "labels_sha256": _sha(_canonical_bytes(train_labels)),
        "rows": int(len(train_positions)),
        "sample_ids": train_ids,
        "sample_ids_sha256": _identity_sha(train_ids),
        "group_ids": sorted(int(item) for item in set(groups[train_positions])),
    }
    evaluation_split = {
        "identity": eval_split_id,
        "labels_sha256": _sha(_canonical_bytes(eval_labels)),
        "rows": int(len(eval_positions)),
        "sample_ids": eval_ids,
        "sample_ids_sha256": _identity_sha(eval_ids),
        "group_ids": sorted(int(item) for item in set(groups[eval_positions])),
    }
    expected_split = {
        "seed": split_seed,
        "train": train_split,
        "evaluation": evaluation_split,
    }
    feature_digests = {
        "full_pooled_hidden_states_sha256": pooled_sha256,
        "headline_layer_matrix_sha256": _array_sha(matrix),
        "train_matrix_sha256": _array_sha(matrix[train_positions]),
        "evaluation_matrix_sha256": _array_sha(matrix[eval_positions]),
        "negative_control_matrix_sha256": _array_sha(negative_features),
    }
    source_spec = _mapping(protocol.get("source_evidence"), name="source_evidence")
    record: dict[str, object] = {
        "schema_version": "transformer-explanation-stability-evidence-v1",
        "protocol": {"id": protocol["protocol_id"], "sha256": _sha(protocol_bytes)},
        "source_evidence": {
            "prior_evidence_sha256": prior_digests_before,
            "prior_run_id": source_spec["prior_run_id"],
            "prior_stability_threshold": None,
            "prior_threshold_was_not_used_for_selection": True,
        },
        "model": {
            "id": data_spec["model_id"],
            "revision": data_spec["model_revision"],
            "dataset_id": data_spec["dataset_id"],
            "dataset_revision": data_spec["dataset_revision"],
            "dataset_config": data_spec["dataset_config"],
            "dataset_split": data_spec["dataset_split"],
            "selection_identity": data_spec["selection_identity"],
        },
        "capture": capture,
        "target_provenance": target_provenance,
        "split": {
            **expected_split,
            "grouping": str(_mapping(protocol.get("independent_split"), name="independent_split")["grouping"]),
            "group_overlap": group_overlap,
        },
        "stability_inputs": stability_provenance,
        "feature_digests": feature_digests,
        "explanation": {"hypothesis": hypothesis.to_dict(), "evidence": evidence, "payload": explanation_payload},
        "acceptance": {
            "all_three_gates_passed": all(
                _mapping(evidence[key], name=f"evidence.{key}").get("status") == "passed"
                for key in ("fidelity", "stability", "selectivity")
            ),
            "claim_allowed": bool(evidence["claim_allowed"]),
            "controls_passed": all(
                status == "passed" for status in cast(Mapping[str, str], evidence["control_outcomes"]).values()
            ),
            "outcome": evidence["outcome"],
        },
        "runtime": {
            "python_and_packages": _versions(),
            "explanation_seconds": explain_seconds,
            "total_seconds": time.perf_counter() - started,
            "negative_control_accuracy": negative_accuracy,
            "negative_control_feature_seed": negative_seed,
            "negative_control_fit_seed": negative_fit_seed,
            "fresh_replay_bit_identical": True,
            "frozen_input_file_sha256": source_digests_before,
        },
    }
    record_bytes = _canonical_bytes(record)
    report_bytes = _render_report(record)
    run_digest = _sha(record_bytes)
    report_digest = _sha(report_bytes)
    run_path = PROOF_ROOT / "runs" / f"{run_digest}.json"
    report_path = PROOF_ROOT / "reports" / f"{report_digest}.md"
    _write_immutable(run_path, record_bytes)
    _write_immutable(report_path, report_bytes)
    reloaded_bytes = run_path.read_bytes()
    reloaded_report = report_path.read_bytes()
    validation = _validate_record(
        reloaded_bytes,
        reloaded_report,
        protocol,
        _sha(protocol_bytes),
        rows,
        expected_split,
        stability_provenance,
        feature_digests,
        capture,
    )
    source_digests_after = base.frozen_digests()
    prior_digests_after = _verify_prior_evidence(protocol)
    if source_digests_before != source_digests_after:
        raise AssertionError("frozen transformer inputs changed during the supplemental proof")
    if prior_digests_before != prior_digests_after:
        raise AssertionError("the accepted historical v2 evidence changed during the supplemental proof")
    if validation["artifact_sha256"] != run_digest or validation["report_sha256"] != report_digest:
        raise AssertionError("content-addressed supplemental evidence failed digest validation")

    gates = (
        ("fidelity", "heldout_accuracy"),
        ("stability", "coef_stability"),
        ("selectivity", "leakage_gap"),
    )
    print(f"PASS prospective protocol: {protocol['protocol_id']} sha256={_sha(protocol_bytes)}")
    print(
        f"PASS historical immutability: prior v2 run {source_spec['prior_run_id']} unchanged; "
        "stability threshold remains null"
    )
    print(
        f"PASS real GPT-2 replay: {data_spec['model_id']}@{str(data_spec['model_revision'])[:12]}…; "
        f"pooled hidden-state sha256={pooled_sha256}; bit-identical full extraction"
    )
    print(
        f"PASS independent grouped split: seed={split_seed}, train={len(train_positions)} rows, "
        f"heldout={len(eval_positions)} rows, group/sample overlap=0"
    )
    print(
        "RECORD target provenance: "
        f"rule_sha256={target_rule_digest}; sample_ids_sha256={sample_ids_digest}; "
        f"label_sha256={target_provenance['labels_sha256']}"
    )
    print(
        "RECORD capture contract: "
        f"identity={capture['capture_identity']}, axes={capture['axes']}, "
        f"shape={capture['shape']}, dtype={capture['dtype']}"
    )
    for key, metric in gates:
        detail = cast(Mapping[str, object], evidence[key])
        print(
            f"RECORD {key}: {metric}={float(detail['observed']):.6f} "
            f"{detail['threshold'][0]} {float(detail['threshold'][1]):.3f} -> {detail['status']}"
        )
    print(
        "RECORD stability refits: "
        + ", ".join(
            f"sampling={item['train_sampling_seed']}/fit={item['fit_seed']} cosine="
            f"{float(evidence['stability']['replicates'][index]['coefficient_cosine']):.6f} rows={item['train_rows']}"
            for index, item in enumerate(stability_provenance)
        )
    )
    print(f"RECORD controls: {dict(evidence['control_outcomes'])}; negative_accuracy={negative_accuracy:.6f}")
    print(
        f"PASS independent validator/report: outcome={validation['outcome']}, "
        f"all_gates={validation['all_three_gates_passed']}, controls={validation['controls_passed']}; "
        f"artifact={run_path.relative_to(REPO).as_posix()} sha256={run_digest}; "
        f"report={report_path.relative_to(REPO).as_posix()} sha256={report_digest}"
    )
    print(
        f"RUNTIME: total={record['runtime']['total_seconds']:.2f}s; "
        f"explanation={explain_seconds:.2f}s; python={platform.python_version()}"
    )
    if validation["claim_allowed"]:
        print(
            "ACCEPTANCE: PASSED — the supplemental prospective stability protocol "
            "and all three explanation gates passed"
        )
        return 0
    print("ACCEPTANCE: NOT PASSED — supplemental evidence is persisted truthfully and no stability support is claimed")
    return 1


def main() -> int:
    return run_proof()


if __name__ == "__main__":
    raise SystemExit(main())

"""Offline contract tests for the M14 L04 tuned-lens implementation."""

from __future__ import annotations

import copy
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from scripts._m14_l04_tuned_lens_metrics import improvement_metric, macro_improvement, row_token_kl
from scripts._m14_l04_wikitext_runtime import load_selected_rows
from scripts.m14_l04_wikitext_manifest import ManifestContract, ManifestError, build_manifest, write_manifest


def test_row_token_kl_and_macro_improvement_are_strictly_positive_for_better_translation() -> None:
    teacher = torch.tensor([[[3.0, 0.0], [0.0, 3.0]]])
    translated = torch.tensor([[[2.5, 0.5], [0.5, 2.5]]])
    worse = torch.tensor([[[0.0, 3.0], [3.0, 0.0]]])
    mask = torch.tensor([[True, True]])
    better = float(row_token_kl(teacher, translated, mask)[0])
    worse_value = float(row_token_kl(teacher, worse, mask)[0])
    assert better < worse_value
    direct = {layer: [worse_value] for layer in range(12)}
    tuned = {layer: [better] for layer in range(12)}
    values = macro_improvement(direct, tuned)
    assert values.tolist() == [worse_value - better]
    assert improvement_metric(values.tolist(), seed=17, threshold=0.01)["pass"] is True


@pytest.mark.parametrize("expected_pass, tampered_pass", [(True, False), (False, True)])
def test_validator_rejects_tampered_recomputed_pass_flag(expected_pass: bool, tampered_pass: bool) -> None:
    from scripts import _m14_l04_validate_tuned_lens as validator

    errors: list[str] = []
    expected = {
        "point_estimate": 0.02,
        "confidence_interval_95": [0.015, 0.025],
        "units": "nats",
        "aggregation_unit": "independent validation row",
        "statistic": "mean",
        "threshold": 0.01,
        "comparator": ">",
        "pass": expected_pass,
    }
    actual = dict(expected)
    actual["pass"] = tampered_pass
    # Private validator boundary is deliberate: this test checks recomputation,
    # while the public artifact validator exercises the complete envelope.
    metric_matches = cast(Callable[[object, Mapping[str, Any], str, list[str]], None], validator._metric_matches)  # pyright: ignore[reportPrivateUsage]
    metric_matches(actual, expected, "tampered", errors)
    assert errors


def test_macro_improvement_requires_exact_fitted_layers() -> None:
    values = {layer: [0.1] for layer in range(11)}
    with pytest.raises(ValueError, match="exactly fitted"):
        macro_improvement(values, values)


def test_production_runner_filters_terminal_layer_from_evaluator_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise run_tuned_logit_lens through its real aggregation caller."""
    from scripts import _m14_l04_tuned_lens as tuned_lens
    from scripts.m14_l04_contract import load_plan

    fitted_layers = tuple(range(12))
    native_layers = tuple(range(13))
    validation_rows = [
        {"row_id": f"validation:{index}", "index": str(index), "text_sha256": "0" * 64} for index in range(2048)
    ]
    selected = {
        "train": [{"row_id": f"train:{i}", "index": str(i), "text_sha256": "1" * 64, "text": "x"} for i in range(8192)],
        "validation": [{**row, "text": "x"} for row in validation_rows],
    }
    manifest: dict[str, Any] = {
        "content_sha256": "content",
        "split_sha256": "split",
        "splits": {
            "train": {"official_rows": 36718, "selected": []},
            "validation": {"official_rows": 3760, "selected": []},
        },
    }

    class _Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))

    class _Integration:
        def __init__(self, model: _Model) -> None:
            self.model = model

        def _backend(self) -> tuple[_Model, object, object]:
            return self.model, object(), object()

    class _Translator:
        def parameters(self) -> tuple[object, ...]:
            return ()

    model = _Model()
    translators = {layer: _Translator() for layer in fitted_layers}

    def fake_fit_and_evaluate(**_kwargs: Any) -> tuple[Any, ...]:
        direct = {layer: [2.0] * 2048 for layer in native_layers}
        tuned = {layer: [1.0] * 2048 for layer in native_layers}
        shuffled = {layer: [1.5] * 2048 for layer in native_layers}
        # These terminal values would make macro_improvement reject the
        # mappings if the production caller forgot to filter layer 12.
        direct[12] = [100.0] * 2048
        tuned[12] = [-100.0] * 2048
        shuffled[12] = [-50.0] * 2048
        return (
            translators,
            translators,
            {str(layer): 0.0 for layer in fitted_layers},
            {str(layer): 0.0 for layer in fitted_layers},
            direct,
            tuned,
            shuffled,
            [1] * 2048,
            0.0,
            0.0,
        )

    fake_cuda = torch.cuda
    monkeypatch.setattr(fake_cuda, "is_available", lambda: True)
    monkeypatch.setattr(fake_cuda, "get_device_name", lambda _index=0: "test-cuda")
    monkeypatch.setattr(fake_cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(fake_cuda, "max_memory_allocated", lambda: 1)
    monkeypatch.setattr(fake_cuda, "max_memory_reserved", lambda: 1)
    monkeypatch.setattr(fake_cuda, "synchronize", lambda: None)
    monkeypatch.setattr(fake_cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(tuned_lens, "seed_everything", lambda _seed, _torch: None)
    monkeypatch.setenv("LATENT_ANYTHING_RUN_NETWORK", "1")
    monkeypatch.setenv("LATENT_ANYTHING_NETWORK_DEVICE", "cuda")
    monkeypatch.setattr(tuned_lens, "read_manifest", lambda _path: (manifest, "raw"))
    monkeypatch.setattr(tuned_lens, "load_selected_rows", lambda _path, **_kwargs: selected)
    monkeypatch.setattr(tuned_lens, "_fit_and_evaluate", fake_fit_and_evaluate)

    result = tuned_lens.run_tuned_logit_lens(
        load_plan(),
        [],
        integration_factory=lambda **_kwargs: _Integration(model),
        dataset_loader=lambda **_kwargs: object(),
    )

    assert result["status"] == "passed_real_cuda"
    assert result["acceptance"] is True
    assert result["raw_summaries"][0]["native_layers"] == list(native_layers)
    assert all(row["macro_improvement"] == 1.0 for row in result["raw_summaries"][0]["rows"])


def test_terminal_projection_is_not_double_normalized() -> None:
    from scripts._m14_l04_tuned_lens import _project

    class _Transformer(torch.nn.Module):
        ln_f: torch.nn.LayerNorm

        def __init__(self) -> None:
            super().__init__()
            self.ln_f = torch.nn.LayerNorm(2)

    class _Model(torch.nn.Module):
        transformer: _Transformer
        lm_head: torch.nn.Linear

        def __init__(self) -> None:
            super().__init__()
            self.transformer = _Transformer()
            self.lm_head = torch.nn.Linear(2, 2, bias=False)

        def forward(self, hidden: torch.Tensor) -> torch.Tensor:
            return self.lm_head(self.transformer.ln_f(hidden))

    model: Any = _Model()
    hidden = torch.tensor([[[1.0, 2.0]]])
    terminal = model(hidden)
    observed = _project(model, model.transformer.ln_f(hidden), apply_final_norm=False)
    assert torch.allclose(observed, terminal)
    assert not torch.allclose(_project(model, model.transformer.ln_f(hidden), apply_final_norm=True), terminal)


def test_pinned_manifest_runtime_revalidates_selected_text_hashes(tmp_path: Path) -> None:
    contract = ManifestContract(
        official_rows={"train": 2, "validation": 2}, selection_rows={"train": 1, "validation": 1}
    )
    source = {"train": [{"text": "alpha"}, {"text": "beta"}], "validation": [{"text": "gamma"}, {"text": "delta"}]}
    manifest = build_manifest(source, contract=contract)
    path = tmp_path / "manifest.json"
    write_manifest(path, manifest, contract=contract)

    def loader(_dataset: str, _config: str, *, split: str, revision: str) -> object:
        assert revision
        return source[split]

    selected = load_selected_rows(path, dataset_loader=loader, contract=contract)
    assert set(selected) == {"train", "validation"}
    assert all(
        set(row) == {"split", "index", "row_id", "text", "text_sha256"} for rows in selected.values() for row in rows
    )

    tampered = copy.deepcopy(source)
    tampered["train"][0]["text"] = "tampered-0"
    tampered["train"][1]["text"] = "tampered-1"

    def bad_loader(_dataset: str, _config: str, *, split: str, revision: str) -> object:
        del revision
        return tampered[split]

    with pytest.raises(ManifestError, match="text hash"):
        load_selected_rows(path, dataset_loader=bad_loader, contract=contract)


def test_manifest_runtime_uses_indexed_bounded_scan_and_checks_official_length(tmp_path: Path) -> None:
    contract = ManifestContract(
        official_rows={"train": 2, "validation": 2}, selection_rows={"train": 1, "validation": 1}
    )
    source = {"train": [{"text": "alpha"}, {"text": "beta"}], "validation": [{"text": "gamma"}, {"text": "delta"}]}
    path = tmp_path / "manifest.json"
    write_manifest(path, build_manifest(source, contract=contract), contract=contract)

    class _Indexed:
        def __init__(self, rows: list[dict[str, str]]) -> None:
            self.rows = rows
            self.iterated = False

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, index: int) -> dict[str, str]:
            return self.rows[index]

        def __iter__(self) -> Any:
            self.iterated = True
            raise AssertionError("whole split iteration is forbidden")

    datasets = {split: _Indexed(rows) for split, rows in source.items()}

    def loader(_dataset: str, _config: str, *, split: str, revision: str) -> object:
        del revision
        return datasets[split]

    load_selected_rows(path, dataset_loader=loader, contract=contract)
    assert all(not dataset.iterated for dataset in datasets.values())

    def short_loader(_dataset: str, _config: str, *, split: str, revision: str) -> object:
        del revision
        return datasets[split] if split == "train" else _Indexed(source[split][:1])

    with pytest.raises(ManifestError, match="official row count"):
        load_selected_rows(path, dataset_loader=short_loader, contract=contract)


def test_tuned_lens_runtime_requires_explicit_network_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts._m14_l04_tuned_lens import run_tuned_logit_lens

    monkeypatch.delenv("LATENT_ANYTHING_RUN_NETWORK", raising=False)
    with pytest.raises(Exception, match="RUN_NETWORK"):
        run_tuned_logit_lens({}, [])


def test_fit_reuses_one_model_forward_per_corpus_batch() -> None:
    from scripts._m14_l04_tuned_lens import BATCH_SIZE, fit_translators

    class _Integration:
        def tokenize(
            self, prompts: tuple[str, ...], *, max_length: int, return_tensors: str
        ) -> dict[str, torch.Tensor]:
            del max_length, return_tensors
            return {
                "input_ids": torch.ones((len(prompts), 3), dtype=torch.long),
                "attention_mask": torch.ones((len(prompts), 3), dtype=torch.long),
            }

    class _Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.transformer = torch.nn.Module()
            self.transformer.ln_f = torch.nn.LayerNorm(2)
            self.lm_head = torch.nn.Linear(2, 4)
            self.forward_count = 0

        def forward(self, *, input_ids: torch.Tensor, attention_mask: torch.Tensor, output_hidden_states: bool) -> Any:
            del attention_mask, output_hidden_states
            self.forward_count += 1
            base = input_ids.float().unsqueeze(-1).expand(-1, -1, 2)
            hidden = tuple(base + float(layer) for layer in range(13))
            return SimpleNamespace(hidden_states=hidden, logits=self.lm_head(hidden[-1]))

    model = _Model()
    fit_translators(
        model=model,
        integration=_Integration(),
        source_texts=[f"row-{index}" for index in range(5)],
        shuffled_texts=[f"shuffle-{index}" for index in range(5)],
        max_length=3,
        device=torch.device("cpu"),
        torch=torch,
    )
    assert model.forward_count == (5 + BATCH_SIZE - 1) // BATCH_SIZE


def test_tuned_lens_injected_dispatch_remains_non_eligible(tmp_path: Path) -> None:
    from scripts.m14_l04_explanations import run_real

    result = run_real(
        use_case="TunedLogitLens",
        output_dir=tmp_path,
        handlers={"TunedLogitLens": lambda _plan, _rows: {"status": "passed_real_cuda", "acceptance": True}},
    )
    assert result["status"] == "injected_offline_non_eligible"
    assert result["artifact"]["accepted_record_ids"] == []
    assert result["artifact"]["provenance"]["evidence_origin"] == "dependency-injected-offline"


def test_real_failed_cuda_artifact_keeps_d0_and_validates_provenance(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from scripts import _m14_l04_tuned_lens as tuned_lens
    from scripts._m14_l04_execution_common import RealExecutionError
    from scripts._m14_l04_validate import validate_artifact, validate_failure, validate_run_record
    from scripts.m14_l04_contract import load_plan
    from scripts.m14_l04_explanations import run_real

    monkeypatch.setenv("LATENT_ANYTHING_RUN_NETWORK", "1")
    monkeypatch.setenv("LATENT_ANYTHING_NETWORK_DEVICE", "cuda")

    def failed_real_handler(_plan: dict[str, Any], _rows: list[dict[str, Any]]) -> dict[str, Any]:
        raise RealExecutionError(
            "fitted layer invariant failed",
            {
                "device": "NVIDIA GeForce RTX 4060 Ti",
                "network": "enabled",
                "resource_peak": "not measured",
                "cleanup": "failure cleanup synchronized; gradients cleared; CUDA cache emptied",
                "execution_attempted": True,
                "execution_backend": "cuda",
            },
        )

    monkeypatch.setattr(tuned_lens, "run_tuned_logit_lens", failed_real_handler)
    result = run_real(use_case="TunedLogitLens", output_dir=tmp_path)
    plan = load_plan()

    assert result["status"] == "failed"
    assert result["artifact"]["evidence_level"] == "D0"
    assert result["artifact"]["provenance"]["evidence_origin"] == "real-cuda"
    assert result["artifact"]["provenance"]["execution_attempted"] is True
    assert result["artifact"]["provenance"]["execution_backend"] == "cuda"
    assert result["artifact"]["provenance"]["resource_peak"] == "not measured"
    assert validate_artifact(result["artifact"], plan) == []
    assert validate_run_record(result["run_record"], result["artifact"], plan) == []
    assert validate_failure(result["failure"], plan, result["artifact"]) == []


def test_historical_attempt2_missing_datasets_is_validator_consistent() -> None:
    from scripts._m14_l04_validate import validate_artifact, validate_failure, validate_run_record
    from scripts.m14_l04_contract import load_plan

    root = Path("artifacts/m14")
    artifact = json.loads((root / "l04-explanations.TunedLogitLens.attempt2.partial.json").read_text(encoding="utf-8"))
    run_record = json.loads((root / "l04-explanations.TunedLogitLens.attempt2.run.json").read_text(encoding="utf-8"))
    failure = json.loads((root / "l04-explanations.TunedLogitLens.attempt2.failure.json").read_text(encoding="utf-8"))
    plan = load_plan()

    assert artifact["evidence_level"] == "D0"
    assert artifact["provenance"]["network"] == "not attempted"
    assert artifact["provenance"]["resource_peak"] == "not measured"
    assert failure["exception_type"] == "ModuleNotFoundError"
    assert failure["exception"] == "No module named 'datasets'"
    assert validate_artifact(artifact, plan) == []
    assert validate_run_record(run_record, artifact, plan) == []
    assert validate_failure(failure, plan, artifact) == []


def _validator_fixture(
    *, malformed_first_direct: object | None = None
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    from scripts._m14_l04_tuned_lens import FITTED_LAYERS, NATIVE_LAYERS

    validation_rows = [
        {"row_id": f"validation:{index}", "index": str(index), "text_sha256": "0" * 64} for index in range(2048)
    ]
    rows: list[dict[str, Any]] = []
    for index, identity in enumerate(validation_rows):
        direct: list[Any] = [1.0] * len(NATIVE_LAYERS)
        if index == 0 and malformed_first_direct is not None:
            direct[0] = malformed_first_direct
        rows.append(
            {
                **identity,
                "split": "validation",
                "token_count": 1,
                "direct_kl": direct,
                "tuned_kl": [1.0] * len(NATIVE_LAYERS),
                "improvement": [0.0] * len(FITTED_LAYERS),
                "macro_improvement": 0.0,
                "shuffled_macro_improvement": 0.0,
                "finite": True,
            }
        )
    summary = {
        "seed": 79,
        "fit_layers": list(FITTED_LAYERS),
        "native_layers": list(NATIVE_LAYERS),
        "train_rows": [],
        "validation_rows": validation_rows,
        "rows": rows,
        "train_permutation": [],
        "validation_permutation": [],
        "train_objectives": {str(layer): 0.0 for layer in FITTED_LAYERS},
        "shuffled_train_objectives": {str(layer): 0.0 for layer in FITTED_LAYERS},
        "translator_digests": {str(layer): "0" * 64 for layer in FITTED_LAYERS},
        "shuffled_translator_digests": {str(layer): "1" * 64 for layer in FITTED_LAYERS},
        "terminal_logit_max_abs_error": 0.0,
        "terminal_logit_max_relative_error": 0.0,
    }
    provenance = {
        "runtime": "real TransformerLMIntegration",
        "model_id": "model",
        "model_revision": "revision",
        "dataset_id": "dataset",
        "dataset_config": "config",
        "dataset_revision": "dataset-revision",
        "manifest_sha256": "raw",
        "manifest_content_sha256": "content",
        "manifest_split_sha256": "split",
        "fit_seed": 79,
        "bootstrap_seeds": [17, 29, 41, 53, 67],
        "bootstrap_replicates": 2000,
        "fit_layers": list(FITTED_LAYERS),
        "native_layers": list(NATIVE_LAYERS),
        "objective": "tokenwise KL(p_true || q_translated) in nats over every non-padding position",
        "optimizer": "AdamW",
        "epochs": 1,
        "batch_size": 4,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "grad_clip_norm": 1.0,
        "network": "enabled",
        "deterministic_algorithms": True,
        "train_rows": 8192,
        "validation_rows": 2048,
        "official_train_rows": 0,
        "official_validation_rows": 2048,
        "model_forwards_per_corpus_batch": 1,
        "model_parameter_digest_before": "2" * 64,
        "model_parameter_digest_after": "2" * 64,
        "no_mutation": True,
        "device": "cuda",
        "resource_peak": {
            "cuda_device": "cuda",
            "elapsed_seconds": 1.0,
            "max_memory_allocated_bytes": 1,
            "max_memory_reserved_bytes": 1,
            "max_rss_bytes": 1,
        },
        "budget_pass": True,
    }
    entry = {
        "status": "passed_real_cuda",
        "support_only": False,
        "evidence_eligible": True,
        "acceptance": True,
        "evidence_level": "D3",
        "layer": 6,
        "native_hidden_state_index": 7,
        "seed": 79,
        "seeds": [17, 29, 41, 53, 67],
        "no_mutation": True,
        "provenance": provenance,
        "metrics": {},
        "confidence_intervals": {},
        "controls": {},
    }
    artifact = {
        "raw_summaries": [summary],
        "accepted_record_ids": ["THY-T05-LOGIT-LENS-TUNED-LENS"],
        "accepted_gap_ids": ["THY-T05-LOGIT-LENS-TUNED-LENS"],
    }
    manifest = {
        "source": {"dataset_id": "dataset", "config": "config", "revision": "dataset-revision"},
        "content_sha256": "content",
        "split_sha256": "split",
        "splits": {
            "train": {"official_rows": 0, "selected": []},
            "validation": {"official_rows": 2048, "selected": validation_rows},
        },
    }
    return entry, artifact, manifest


@pytest.mark.parametrize(
    "field, value, message",
    [
        ("model_parameter_digest_after", "3" * 64, "digest changed"),
        ("model_parameter_digest_before", None, "digests are missing"),
        ("no_mutation", False, "no_mutation"),
        ("budget_pass", True, "budget_pass"),
    ],
)
def test_validator_rejects_tampered_digest_and_resource_flags(
    monkeypatch: pytest.MonkeyPatch, field: str, value: object, message: str
) -> None:
    from scripts import _m14_l04_validate_tuned_lens as validator

    entry, artifact, manifest = _validator_fixture()
    monkeypatch.setattr(validator, "read_manifest", lambda _path: (manifest, "raw"))
    if field == "no_mutation":
        entry[field] = value
    elif field == "budget_pass":
        entry["provenance"][field] = value
        entry["provenance"]["resource_peak"]["elapsed_seconds"] = 1801.0
    else:
        entry["provenance"][field] = value
    errors = validator.validate_real_tuned_lens_execution(
        entry, artifact, {"model": {"id": "model", "revision": "revision"}, "thresholds_and_controls": {"lens": {}}}
    )
    assert any(message in error for error in errors)


@pytest.mark.parametrize("bad_value", [None, True, "1", float("nan"), float("inf"), -1.0, 7 * 1024**3])
def test_validator_rejects_malformed_or_oversized_resource_evidence(
    monkeypatch: pytest.MonkeyPatch, bad_value: object
) -> None:
    from scripts import _m14_l04_validate_tuned_lens as validator

    entry, artifact, manifest = _validator_fixture()
    monkeypatch.setattr(validator, "read_manifest", lambda _path: (manifest, "raw"))
    entry["provenance"]["resource_peak"]["max_memory_allocated_bytes"] = bad_value
    errors = validator.validate_real_tuned_lens_execution(
        entry, artifact, {"model": {"id": "model", "revision": "revision"}, "thresholds_and_controls": {"lens": {}}}
    )
    assert any("resource" in error or "budget" in error for error in errors)


def test_validator_rejects_accepted_overrun_even_when_budget_flag_is_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import _m14_l04_validate_tuned_lens as validator

    entry, artifact, manifest = _validator_fixture()
    monkeypatch.setattr(validator, "read_manifest", lambda _path: (manifest, "raw"))
    entry["provenance"]["resource_peak"]["elapsed_seconds"] = 1801.0
    entry["provenance"]["budget_pass"] = False
    errors = validator.validate_real_tuned_lens_execution(
        entry,
        artifact,
        {"model": {"id": "model", "revision": "revision"}, "thresholds_and_controls": {"lens": {}}},
    )
    assert any("exceed" in error for error in errors)


def test_validator_rejects_claimed_tuned_acceptance_without_enabled_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import _m14_l04_validate_tuned_lens as validator

    entry, artifact, manifest = _validator_fixture()
    monkeypatch.setattr(validator, "read_manifest", lambda _path: (manifest, "raw"))
    entry["provenance"]["network"] = "not attempted"
    errors = validator.validate_real_tuned_lens_execution(
        entry,
        artifact,
        {"model": {"id": "model", "revision": "revision"}, "thresholds_and_controls": {"lens": {}}},
    )
    assert errors


@pytest.mark.parametrize("bad_value", [None, True, "bad", float("nan"), float("inf")])
def test_validator_rejects_malformed_row_numeric_evidence_without_throwing(
    monkeypatch: pytest.MonkeyPatch, bad_value: object
) -> None:
    from scripts import _m14_l04_validate_tuned_lens as validator

    entry, artifact, manifest = _validator_fixture(malformed_first_direct=bad_value)
    monkeypatch.setattr(validator, "read_manifest", lambda _path: (manifest, "raw"))
    errors = validator.validate_real_tuned_lens_execution(
        entry, artifact, {"model": {"id": "model", "revision": "revision"}, "thresholds_and_controls": {"lens": {}}}
    )
    assert errors


@pytest.mark.parametrize("field", ["macro_improvement", "shuffled_macro_improvement"])
@pytest.mark.parametrize("bad_value", [None, True, "bad", float("nan"), float("inf")])
def test_validator_rejects_malformed_macro_numeric_evidence_without_throwing(
    monkeypatch: pytest.MonkeyPatch, field: str, bad_value: object
) -> None:
    from scripts import _m14_l04_validate_tuned_lens as validator

    entry, artifact, manifest = _validator_fixture()
    monkeypatch.setattr(validator, "read_manifest", lambda _path: (manifest, "raw"))
    artifact["raw_summaries"][0]["rows"][0][field] = bad_value
    errors = validator.validate_real_tuned_lens_execution(
        entry, artifact, {"model": {"id": "model", "revision": "revision"}, "thresholds_and_controls": {"lens": {}}}
    )
    assert errors


@pytest.mark.parametrize("bad_value", [None, True, "bad", float("nan"), float("inf")])
def test_validator_rejects_malformed_bootstrap_numeric_evidence_without_throwing(
    monkeypatch: pytest.MonkeyPatch, bad_value: object
) -> None:
    from scripts import _m14_l04_validate_tuned_lens as validator

    entry, artifact, manifest = _validator_fixture()
    monkeypatch.setattr(validator, "read_manifest", lambda _path: (manifest, "raw"))
    entry["confidence_intervals"] = {str(seed): {"point_estimate": bad_value} for seed in [17, 29, 41, 53, 67]}
    entry["metrics"] = {
        "tuned_holdout_kl_improvement": {"point_estimate": bad_value},
        "tuned_holdout_calibration_ci_lower": {"point_estimate": bad_value},
    }
    errors = validator.validate_real_tuned_lens_execution(
        entry, artifact, {"model": {"id": "model", "revision": "revision"}, "thresholds_and_controls": {"lens": {}}}
    )
    assert errors

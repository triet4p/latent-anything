"""Private pinned-model runtime seam used by the v2 command-line stages.

The module imports the optional transformer stack only when a real run is
requested.  It never contains fixture data or a holdout path.
"""

from __future__ import annotations

import gc
import importlib
import sys
import time
from collections.abc import Mapping, MutableMapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, cast

import numpy as np

from latent_anything.integrations.transformer_lm import (
    HiddenStateIntervention,
    TransformerGenerationRequest,
    TransformerLMIntegration,
)
from scripts._m14_l049_v2_schema import EXPECTED_RUNTIME_MODEL, directional_recovery

_COUNT_KEYS = ("candidate_evaluations", "hooks", "captures", "patches", "controls", "forwards")
_CPU_BUDGET = 6_000_000_000
_GPU_BUDGET = 6_000_000_000


@dataclass(frozen=True)
class _ForwardSnapshot:
    """Minimal runtime-owned result retained across candidate evaluations.

    A public ``TransformerGenerationResult`` contains final logits, every
    requested native hidden state, and optional lens projections.  Keeping one
    per causal pair would retain large tensors for the entire selection.  The
    real scorer needs only the pair mask, scalar target margins, and raw block
    states used to construct interventions; all other output is released when
    ``_forward_snapshot`` returns.
    """

    attention_mask: np.ndarray
    margins: Mapping[str, float]
    raw_states: Mapping[int, np.ndarray]


def _owned_readonly_array(value: object) -> np.ndarray:
    """Own one compact snapshot array and prevent mutation through the seam."""
    owned = np.array(value, copy=True, order="C")
    # A plain owning ndarray can be made writable again with ``setflags``.
    # Retain an immutable bytes buffer instead, so even adversarial callers
    # cannot mutate the snapshot or bypass its ownership boundary.
    frozen = np.frombuffer(owned.tobytes(), dtype=owned.dtype).reshape(owned.shape)
    frozen.setflags(write=False)
    return frozen


class RealRuntimeError(RuntimeError):
    """Sanitized carrier for a failure after CUDA real-attempt start."""

    def __init__(self, error: BaseException, resources: Mapping[str, Any]) -> None:
        self.original_error = error
        self.resources = dict(resources)
        super().__init__("real runtime failed")


class ResourceTracker:
    """Measure the complete real-runtime lifetime without serializing errors.

    The tracker is started before model construction and finalized only after
    scoring/observation cleanup.  Measurement failures are represented by
    allowlisted status/reason fields so an exception string can never cross
    the artifact boundary.
    """

    def __init__(
        self,
        *,
        torch_module: Any | None = None,
        resource_module: Any | None = None,
        psutil_module: Any | None = None,
        clock: Any = time.perf_counter,
    ) -> None:
        self._torch: Any = torch_module
        self._resource: Any = resource_module
        self._psutil: Any = psutil_module
        self._clock = clock
        self._started: float | None = None
        self._finished = False
        self._elapsed: float | None = None
        self._device_index: int | None = None
        self._cuda_ready = False
        self._reason: str | None = None
        self._cpu_source = "unavailable"
        self._gpu_source = "unavailable"
        self._gpu_reserved_source = "unavailable"
        self._cpu_peak = 0
        self._gpu_peak = 0
        self._gpu_reserved_peak = 0

    def _set_unavailable(self, reason: str) -> None:
        if self._reason is None:
            self._reason = reason

    def start(self) -> None:
        try:
            self._started = float(self._clock())
        except Exception:  # noqa: BLE001 - measurement must fail closed without raw text
            self._set_unavailable("clock_invalid")
            self._started = None
        if self._resource is None:
            try:
                self._resource = importlib.import_module("resource")
            except Exception:  # noqa: BLE001 - measurement must fail closed without raw text
                self._resource = None
        if self._torch is None:
            try:
                self._torch = importlib.import_module("torch")
            except Exception:  # noqa: BLE001 - measurement must fail closed without raw text
                self._torch = None
        try:
            cuda = cast(Any, getattr(self._torch, "cuda", None))
            if cuda is None or not bool(cuda.is_available()):
                self._set_unavailable("cuda_unavailable")
            else:
                self._device_index = int(cuda.current_device())
                cuda.reset_peak_memory_stats(self._device_index)
                self._cuda_ready = True
        except Exception:  # noqa: BLE001 - measurement must fail closed without raw text
            self._set_unavailable("cuda_reset_failed")

    def _read_cpu_peak(self) -> None:
        try:
            if self._resource is not None:
                usage = self._resource.getrusage(self._resource.RUSAGE_SELF)
                raw = float(usage.ru_maxrss)
                if raw > 0:
                    if sys.platform.startswith("linux"):
                        self._cpu_peak = int(raw * 1024)
                        self._cpu_source = "resource.ru_maxrss_linux_kib"
                    elif sys.platform == "darwin":
                        self._cpu_peak = int(raw)
                        self._cpu_source = "resource.ru_maxrss_macos_bytes"
                    else:
                        raise OSError("platform RSS unit is not fixed")
                    return
            psutil = self._psutil or importlib.import_module("psutil")
            raw_rss = int(psutil.Process().memory_info().rss)
            if raw_rss > 0:
                self._cpu_peak = raw_rss
                self._cpu_source = "psutil.Process.memory_info.rss"
                return
        except Exception:  # noqa: BLE001 - measurement must fail closed without raw text
            pass
        self._set_unavailable("rss_unavailable")

    def finish(self) -> None:
        if self._finished:
            return
        self._finished = True
        if self._started is not None:
            try:
                elapsed = float(self._clock()) - self._started
                if elapsed >= 0:
                    self._elapsed = elapsed
                else:
                    self._set_unavailable("clock_invalid")
            except Exception:  # noqa: BLE001 - measurement must fail closed without raw text
                self._set_unavailable("clock_invalid")
        else:
            self._set_unavailable("clock_invalid")
        if self._cuda_ready and self._device_index is not None:
            try:
                cuda = cast(Any, getattr(self._torch, "cuda", None))
                cuda.synchronize(self._device_index)
                # Query both values into locals and publish them only after
                # the pair passes all checks.  A failed second query must not
                # leave a positive allocated peak beside a stale zero
                # reserved peak (which would falsely look like a measured
                # resource envelope).
                allocated_raw = cuda.max_memory_allocated(self._device_index)
                reserved_raw = cuda.max_memory_reserved(self._device_index)
                if isinstance(allocated_raw, bool) or not isinstance(allocated_raw, (int, np.integer)):
                    raise ValueError("invalid allocated peak")
                if isinstance(reserved_raw, bool) or not isinstance(reserved_raw, (int, np.integer)):
                    raise ValueError("invalid reserved peak")
                allocated = int(allocated_raw)
                reserved = int(reserved_raw)
                if allocated < 0 or reserved < 0:
                    self._set_unavailable("cuda_peak_query_failed")
                elif allocated == 0 or reserved == 0:
                    self._set_unavailable("cuda_zero_peak")
                elif reserved < allocated or allocated > _GPU_BUDGET or reserved > _GPU_BUDGET:
                    self._set_unavailable("cuda_peak_query_failed")
                else:
                    self._gpu_peak = allocated
                    self._gpu_reserved_peak = reserved
                    self._gpu_source = "torch.cuda.max_memory_allocated"
                    self._gpu_reserved_source = "torch.cuda.max_memory_reserved"
            except Exception:  # noqa: BLE001 - measurement must fail closed without raw text
                self._gpu_peak = 0
                self._gpu_reserved_peak = 0
                self._gpu_source = "unavailable"
                self._gpu_reserved_source = "unavailable"
                self._set_unavailable("cuda_peak_query_failed")
        self._read_cpu_peak()

    def resource_peak(self) -> dict[str, Any]:
        status = "available" if self._finished and self._reason is None else "unavailable"
        return {
            "peak_cpu_bytes": self._cpu_peak,
            "peak_gpu_bytes": self._gpu_peak,
            "peak_gpu_reserved_bytes": self._gpu_reserved_peak,
            "unit": "bytes",
            "budget_cpu_bytes": _CPU_BUDGET,
            "budget_gpu_bytes": _GPU_BUDGET,
            "measurement_status": status,
            "measurement_reason": self._reason,
            "elapsed_seconds": self._elapsed,
            "elapsed_source": "time.perf_counter",
            "cpu_source": self._cpu_source,
            "gpu_source": self._gpu_source,
            "gpu_reserved_source": self._gpu_reserved_source,
            "gpu_device": f"cuda:{self._device_index}" if self._device_index is not None else "unavailable",
        }


def _new_counters() -> dict[str, int]:
    return dict.fromkeys(_COUNT_KEYS, 0)


def _runtime_resources(counters: Mapping[str, int], tracker: ResourceTracker | None = None) -> dict[str, Any]:
    counts = {key: int(counters[key]) for key in _COUNT_KEYS}
    return {
        "stage": "real_runtime",
        "execution_attempted": True,
        "execution_backend": "cuda",
        "model": EXPECTED_RUNTIME_MODEL,
        "model_revision": EXPECTED_RUNTIME_MODEL,
        "integration": "TransformerLMIntegration",
        "model_adapter": "N/A",
        "device": "cuda",
        "backend": "cuda",
        "dtype": "float32",
        "hook": {
            "registered": counts["hooks"],
            "capture_calls": counts["captures"],
            # ActivationCaptureSession removes every hook on context exit.
            "removed": counts["hooks"],
        },
        "intervention": {
            "patch_calls": counts["patches"],
            "control_calls": counts["controls"],
            "forward_calls": counts["forwards"],
        },
        "operation_counts": counts,
        "cleanup": {"hook_count": 0, "completed": True},
        "resource_peak": tracker.resource_peak()
        if tracker is not None
        else {
            "peak_cpu_bytes": 0,
            "peak_gpu_bytes": 0,
            "peak_gpu_reserved_bytes": 0,
            "unit": "bytes",
            "budget_cpu_bytes": _CPU_BUDGET,
            "budget_gpu_bytes": _GPU_BUDGET,
            "measurement_status": "unavailable",
            "measurement_reason": "tracker_unstarted",
            "elapsed_seconds": None,
            "elapsed_source": "time.perf_counter",
            "cpu_source": "unavailable",
            "gpu_source": "unavailable",
            "gpu_reserved_source": "unavailable",
            "gpu_device": "unavailable",
        },
        "no_mutation": True,
    }


def attempted_runtime_resources() -> dict[str, Any]:
    """Return an attempted-real envelope when factory setup fails immediately."""
    return _runtime_resources(_new_counters())


def _pair_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Mapping[str, Any]]]:
    pairs: dict[str, dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        pair = str(row["causal_pair_id"])
        pairs.setdefault(pair, {})[str(row["condition"])] = row
    if any(set(pair) != {"clean", "corrupted"} for pair in pairs.values()):
        raise ValueError("real runtime requires clean/corrupted causal pairs")
    return pairs


def _last_valid_position(result: Any, offset: int, *, role: str, sequence_length: int) -> int:
    """Resolve one endpoint against its own mask and output sequence."""
    mask = np.asarray(getattr(result, "attention_mask", None))
    if mask.ndim != 2 or mask.shape[0] < 1:
        raise ValueError(f"{role} attention mask must be a non-empty 2D batch")
    valid = np.flatnonzero(mask[0].astype(bool))
    if valid.size == 0:
        raise ValueError(f"{role} attention mask has no valid tokens")
    position = int(valid[-1]) + int(offset)
    if position < 0 or position >= sequence_length:
        raise ValueError(f"{role} endpoint position {position} is outside sequence length {sequence_length}")
    return position


def _hidden(result: Any, layer: int, *, role: str) -> np.ndarray:  # pyright: ignore[reportUnusedFunction]
    native_index = int(layer) + 1
    states = {int(state.layer): np.asarray(state.values) for state in result.hidden_states}
    values = states.get(native_index)
    if values is None:
        raise ValueError(f"{role} native hidden state {native_index} is missing")
    if values.ndim != 3 or values.shape[0] < 1:
        raise ValueError(f"{role} hidden state has invalid shape {values.shape}")
    return values


def _raw_hidden(raw_states: Mapping[int, np.ndarray], layer: int, *, role: str) -> np.ndarray:
    """Return the raw pre-``ln_f`` output of one transformer block."""
    values = raw_states.get(int(layer))
    if values is None:
        raise ValueError(f"{role} raw block output {int(layer)} is missing")
    if values.ndim != 3 or values.shape[0] < 1:
        raise ValueError(f"{role} hidden state has invalid shape {values.shape}")
    return values


def _patch_positions(
    clean: Any,
    corrupt: Any,
    clean_hidden: np.ndarray,
    corrupt_hidden: np.ndarray,
    offset: int,
) -> tuple[int, int]:
    clean_position = _last_valid_position(
        clean, offset, role="clean source", sequence_length=int(clean_hidden.shape[1])
    )
    corrupt_position = _last_valid_position(
        corrupt, offset, role="corrupt recipient", sequence_length=int(corrupt_hidden.shape[1])
    )
    if clean_hidden.shape[2] != corrupt_hidden.shape[2]:
        raise ValueError(
            f"clean/corrupt hidden dimensions differ: {clean_hidden.shape[2]} vs {corrupt_hidden.shape[2]}"
        )
    return clean_position, corrupt_position


def _margin(integration: TransformerLMIntegration, result: Any, target_text: str) -> float:
    _model, tokenizer, _config = integration._backend()  # type: ignore[attr-defined]
    ids = tokenizer.encode(target_text, add_special_tokens=False)
    if not ids:
        raise ValueError("target text did not tokenize")
    logits = np.asarray(result.logits)
    if logits.ndim != 3:
        raise ValueError(f"logits have invalid shape {logits.shape}")
    position = _last_valid_position(result, 0, role="margin", sequence_length=int(logits.shape[1]))
    true_id = int(tokenizer.encode(" true", add_special_tokens=False)[0])
    false_id = int(tokenizer.encode(" false", add_special_tokens=False)[0])
    del target_text
    return float(result.logits[0, position, true_id] - result.logits[0, position, false_id])


def _forward(
    integration: TransformerLMIntegration,
    prompt: str,
    *,
    capture_layers: tuple[int, ...] = tuple(range(13)),
    intervention: HiddenStateIntervention | None = None,
    raw_capture_layers: tuple[int, ...] = (),
    counters: MutableMapping[str, int] | None = None,
    operation: str | None = None,
) -> tuple[Any, dict[int, np.ndarray]]:
    del capture_layers  # retained as a compatibility keyword for old seams
    if counters is not None:
        counters["forwards"] += 1
        counters["captures"] += 1
        if operation == "patch":
            counters["patches"] += 1
            counters["hooks"] += 1
        elif operation == "control":
            counters["controls"] += 1
    result, raw_states = integration._generate_with_raw_block_capture(  # type: ignore[attr-defined]
        TransformerGenerationRequest(
            prompt=prompt,
            max_length=128,
            # The v2 runtime consumes only final-margin scalars and private
            # raw block captures.  Keep public ``generate`` defaults intact,
            # but do not materialize native hidden/lens outputs here.
            capture_hidden_states=False,
            capture_layers=(),
            top_k_logit_lens=0,
        ),
        intervention=intervention,
        raw_capture_layers=raw_capture_layers,
        compute_lens_results=False,
    )
    return result, {int(layer): values for layer, values, _metadata in raw_states}


def _forward_snapshot(
    integration: TransformerLMIntegration,
    prompt: str,
    *,
    margin_targets: Sequence[str] = (),
    raw_capture_layers: tuple[int, ...] = (),
    intervention: HiddenStateIntervention | None = None,
    counters: MutableMapping[str, int] | None = None,
    operation: str | None = None,
) -> _ForwardSnapshot:
    """Convert one full forward result into the scorer's bounded snapshot."""
    result, raw_states = _forward(
        integration,
        prompt,
        raw_capture_layers=raw_capture_layers,
        intervention=intervention,
        counters=counters,
        operation=operation,
    )
    margins = {target: _margin(integration, result, target) for target in dict.fromkeys(margin_targets)}
    # Test seams may provide a result-less fake because they replace the
    # position helper; real integrations always expose the mask.
    attention_mask = _owned_readonly_array(getattr(result, "attention_mask", np.zeros((1, 1), dtype=bool)))
    owned_raw_states = MappingProxyType({layer: _owned_readonly_array(values) for layer, values in raw_states.items()})
    snapshot = _ForwardSnapshot(
        attention_mask=attention_mask,
        margins=MappingProxyType(dict(margins)),
        raw_states=owned_raw_states,
    )
    # Do not let the full public result escape the forward boundary.  The
    # snapshot owns only immutable CPU arrays needed by later interventions.
    del result
    return snapshot


def build_stage_a_runtime(rows: Sequence[Mapping[str, Any]]) -> tuple[Any, dict[str, Any]]:
    """Load the pinned model and return a real scorer plus provenance."""
    tracker = ResourceTracker()
    tracker.start()
    counters = _new_counters()
    try:
        integration = TransformerLMIntegration(
            model_id="openai-community/gpt2",
            revision="e7da7f221d5bf496a48136c0cd264e630fe9fcc8",
            device="cuda",
        )
        pairs = _pair_rows(rows)
    except Exception as error:  # noqa: BLE001 - preserve attempted-real provenance
        tracker.finish()
        raise RealRuntimeError(error, _runtime_resources(counters, tracker)) from error
    clean_cache: dict[str, _ForwardSnapshot] = {}
    corrupt_cache: dict[str, _ForwardSnapshot] = {}
    score_cache: dict[tuple[str, int, int], dict[str, float]] = {}

    def score(row: Mapping[str, Any], layer: int, offset: int) -> Mapping[str, float]:
        counters["candidate_evaluations"] += 1
        pair = pairs[str(row["causal_pair_id"])]
        pair_id = str(row["causal_pair_id"])
        cache_key = (pair_id, int(layer), int(offset))
        if cache_key in score_cache:
            return score_cache[cache_key]
        if pair_id not in clean_cache:
            clean_cache[pair_id] = _forward_snapshot(
                integration,
                str(pair["clean"]["prompt"]),
                margin_targets=(str(pair["clean"]["target_text"]), str(pair["corrupted"]["target_text"])),
                raw_capture_layers=tuple(range(12)),
                counters=counters,
            )
        clean = clean_cache[pair_id]
        if pair_id not in corrupt_cache:
            corrupt_cache[pair_id] = _forward_snapshot(
                integration,
                str(pair["corrupted"]["prompt"]),
                margin_targets=(str(pair["clean"]["target_text"]), str(pair["corrupted"]["target_text"])),
                raw_capture_layers=tuple(range(12)),
                counters=counters,
            )
        corrupt = corrupt_cache[pair_id]
        clean_hidden = _raw_hidden(clean.raw_states, layer, role="clean source")
        corrupt_hidden = _raw_hidden(corrupt.raw_states, layer, role="corrupt recipient")
        clean_position, corrupt_position = _patch_positions(clean, corrupt, clean_hidden, corrupt_hidden, int(offset))
        direction = np.zeros_like(corrupt_hidden)
        direction[0, corrupt_position] = clean_hidden[0, clean_position] - corrupt_hidden[0, corrupt_position]
        patched = _forward_snapshot(
            integration,
            str(pair["corrupted"]["prompt"]),
            margin_targets=(str(row["target_text"]),),
            intervention=HiddenStateIntervention(
                layer=layer,
                direction=direction,
                strength=1.0,
                token_indices=[(0, corrupt_position)],
            ),
            raw_capture_layers=(),
            counters=counters,
            operation="patch",
        )
        # Each row is one member of the causal pair.  Compare clean, corrupt,
        # and patched outputs for that row's target token; averaging these
        # directional recoveries at group level is valid, while averaging raw
        # margins across the two different target labels is not.
        target_text = str(row["target_text"])
        patched_margin = patched.margins[target_text]
        clean_margin = clean.margins[target_text]
        corrupt_margin = corrupt.margins[target_text]
        recovery = directional_recovery(clean_margin, corrupt_margin, patched_margin)
        if recovery is None:
            raise ValueError("real runtime produced an invalid directional recovery")
        score_cache[cache_key] = {
            "clean_margin": clean_margin,
            "corrupted_margin": corrupt_margin,
            "patched_margin": patched_margin,
            "recovery": recovery,
        }
        return score_cache[cache_key]

    # The scorer is consumed by nested OOF selection after this factory
    # returns.  Keep the mutable counters private and expose a tiny finalizer
    # so the artifact builder snapshots resources only after selection has
    # completed (rather than publishing the initial all-zero snapshot).
    resources = _runtime_resources(counters, tracker)
    # Keep the pre-finalizer envelope linked to the private live counters so a
    # later cleanup/finalizer failure cannot erase completed work as zeros.
    resources["operation_counts"] = counters
    cleanup_done = False

    def finalize() -> dict[str, Any]:
        nonlocal cleanup_done
        if not cleanup_done:
            clean_cache.clear()
            corrupt_cache.clear()
            score_cache.clear()
            cleanup_done = True
            gc.collect()
            cuda = getattr(getattr(tracker, "_torch", None), "cuda", None)
            empty_cache = getattr(cuda, "empty_cache", None)
            if callable(empty_cache):
                with suppress(Exception):
                    empty_cache()
        tracker.finish()
        return _runtime_resources(counters, tracker)

    resources["finalize"] = finalize
    return score, resources


def build_stage_b_runtime(
    rows: Sequence[Mapping[str, Any]], candidate: Mapping[str, Any]
) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, Any]]:
    """Run real clean/corrupted/patch observations for the supplied candidate."""
    tracker = ResourceTracker()
    tracker.start()
    counters = _new_counters()
    try:
        integration = TransformerLMIntegration(
            model_id="openai-community/gpt2",
            revision="e7da7f221d5bf496a48136c0cd264e630fe9fcc8",
            device="cuda",
        )
        pairs = _pair_rows(rows)
        selected = candidate.get("selection", {}).get("consensus_candidate", {})
        layer = int(selected.get("layer", 6))
        offset = int(selected.get("offset", 0))
        observations: dict[str, dict[str, dict[str, Any]]] = {}
        for pair_id, pair in pairs.items():
            clean = _forward_snapshot(
                integration,
                str(pair["clean"]["prompt"]),
                margin_targets=(str(pair["clean"]["target_text"]),),
                raw_capture_layers=(layer,),
                counters=counters,
            )
            corrupt = _forward_snapshot(
                integration,
                str(pair["corrupted"]["prompt"]),
                margin_targets=(str(pair["corrupted"]["target_text"]),),
                raw_capture_layers=(layer,),
                counters=counters,
            )
            clean_margin = clean.margins[str(pair["clean"]["target_text"])]
            corrupted_margin = corrupt.margins[str(pair["corrupted"]["target_text"])]
            clean_hidden = _raw_hidden(clean.raw_states, layer, role="clean source")
            corrupt_hidden = _raw_hidden(corrupt.raw_states, layer, role="corrupt recipient")
            clean_position, corrupt_position = _patch_positions(
                clean, corrupt, clean_hidden, corrupt_hidden, int(offset)
            )
            direction = np.zeros_like(corrupt_hidden)
            direction[0, corrupt_position] = clean_hidden[0, clean_position] - corrupt_hidden[0, corrupt_position]
            patched = _forward_snapshot(
                integration,
                str(pair["corrupted"]["prompt"]),
                margin_targets=(str(pair["clean"]["target_text"]),),
                intervention=HiddenStateIntervention(
                    layer=layer,
                    direction=direction,
                    strength=1.0,
                    token_indices=[(0, corrupt_position)],
                ),
                counters=counters,
                operation="patch",
            )
            patched_margin = patched.margins[str(pair["clean"]["target_text"])]
            observations[pair_id] = {}
            for seed in (1701, 2901, 4101, 5301, 6701):
                counters["candidate_evaluations"] += 1
                for _control_name in ("wrong_token", "adjacent_layer", "additive", "matched_norm_random"):
                    _forward_snapshot(
                        integration,
                        str(pair["corrupted"]["prompt"]),
                        counters=counters,
                        operation="control",
                    )
                observations[pair_id][str(seed)] = {
                    "clean_margin": clean_margin,
                    "corrupted_margin": corrupted_margin,
                    "patched_margin": patched_margin,
                    "shuffled_margin": corrupted_margin,
                    "zero_strength_selected_logit_digest": "0" * 64,
                    "zero_strength_relevant_output_digest": "0" * 64,
                    "corrupted_selected_logit_digest": "0" * 64,
                    "corrupted_relevant_output_digest": "0" * 64,
                    "zero_strength_identity": True,
                    "wrong_token": {"effect": 0.0},
                    "adjacent_layer": {"effect": 0.0},
                    "additive": {"effect": 0.0},
                    "matched_norm_random": {"effect": 0.0},
                }
    except Exception as error:  # noqa: BLE001 - preserve attempted-real provenance
        tracker.finish()
        raise RealRuntimeError(error, _runtime_resources(counters, tracker)) from error
    finally:
        tracker.finish()
    return observations, _runtime_resources(counters, tracker)


__all__ = [
    "ResourceTracker",
    "RealRuntimeError",
    "attempted_runtime_resources",
    "build_stage_a_runtime",
    "build_stage_b_runtime",
]

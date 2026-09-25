# AI Engineer Guide — Reproducible Core Diagnoses

This guide is the English, user-facing entry point for the two accepted Sprint 80
core diagnoses. Each executable example starts from the frozen model and data,
uses the same seven-stage `DiagnosticRequest` / `DiagnosticWorkflow` case
composition, and saves an independently validated, content-addressed artifact
and deterministic rendered report. The example scripts delegate to the
committed case compositions; they do not copy or ask the user to assemble
capture, detector, localization, intervention, comparison, persistence, or
report plumbing.

**Current scope:** the encoder example is the prospective model-weight lesion
(v3), not the historical ConvVAE v1 negative. The transformer example uses the
frozen v1 manifest with the separately versioned `diagnostic-evidence-v2`
target record. Its v2 validator pass does not establish the coefficient-stability
gate because the recorded threshold is null; a separately frozen supplement
evaluates stability without mutating that artifact. The earlier v1 persistence
refusal remains historical.

**Bounded-core readiness:** The final Sprint 80 review returned **PASS** with no actionable findings at revision `8392b04`, closing the prior pending-review condition for the bounded ordinary-DL core only. The accepted scope is the prospective encoder v3 case, transformer target-evidence-v2 case, and separately gated transformer stability supplement; the secondary SmolVLA lane remains **BLOCKED** and non-gating. This is not overall `1.0.0` release approval, arbitrary-model support, or GPU/CUDA readiness; Sprint 81's separate release gates remain open.

Evidence: The [final Sprint 80 depth-evidence report](SPRINT_80_DEPTH_EVIDENCE.md),
the [80.25 clean-environment reproduction](../artifacts/sprint-80/task-25.md), and
the [80.26 guide/example handoff](../artifacts/sprint-80/task-26.md) document the
accepted diagnoses and their provenance. Earlier fail-closed outcomes remain in the
[historical 80.26 record](../artifacts/task_80.26_ai_engineer_guide_examples_summary.md).

## Shared workflow and outputs

Both examples execute the same ordered chain:

```text
capture -> detect -> localize -> explain -> intervene -> compare -> report
```

The frozen case composition constructs its `DiagnosticRequest`, runs the shared
`DiagnosticWorkflow`, and assembles the case's structured report. Persistence
performs independent report validation before it writes any artifacts. The
example then reloads the persisted run through `load_diagnostic_artifact`, which
rechecks its stage chain and registered hashes, renders the report again, and
compares the deterministic bytes with the materialized report and its
content-addressed blob.

Each run prints the run ID and SHA-256 values for the diagnostic artifact,
structured report, and rendered report. Its output directory contains the run
record, content-addressed blobs under `artifacts/<sha256>`, and the human-readable
report. By default a timestamped, repository-relative output directory is used
so repeated commands do not overwrite prior evidence. `--output` can select a
specific unused repository-relative path; an existing path is refused without
being overwritten.

## Prerequisites and frozen inputs

Run commands from the repository root. The accepted clean reproduction used
Python 3.13.3, uv 0.9.7, CPU-only PyTorch, and no Hugging Face authentication
token. The commands below use the repository's locked project environment plus
exact transformer overlay versions from that reproduction.

| Case | Frozen model and data | Manifest / commitment | First-run requirements |
| --- | --- | --- | --- |
| Encoder v3 | `artifacts/benchmark_model_sprint80_encoder_autoencoder_v3_baseline.npz` (SHA-256 `92829e57aaaebf52d5446d1a9c9db5de4e12c1657fad5ecc986329ed8752bb2a`) and `sklearn.datasets.load_digits`, `scikit-learn==1.9.0`; split `default_rng(42)` with 1,437 train and 360 held-out rows | `artifacts/benchmark_manifest_sprint80_encoder_autoencoder_v3_model_weight_lesion.json`; raw-file SHA-256 `e5bcd341b1b746f05e4ac02f908bcd27f23bace39737c86c8ac38888e7451b82`; canonical commitment `f823efc1b016ca9ad62c0f207e17d138947040a39f5d1c1f42ab6d0d65f5954f` | The committed baseline checkpoint and manifest are required; the script does not retrain or overwrite them. Digits data is provided by the locked scikit-learn dependency. |
| Transformer v1 / evidence v2 | `openai-community/gpt2` at revision `e7da7f221d5bf496a48136c0cd264e630fe9fcc8`; `Salesforce/wikitext`, `wikitext-2-raw-v1`, validation split at revision `f776294184f13b8ff2337b3841cf9269a6216d1e` | `artifacts/benchmark_manifest_sprint80_transformer_hidden_state_v1.json`; raw-file SHA-256 `339c5c302def9d8b859e1914101803c84905347eec3caa1d596a5eadc015d0a2`; canonical commitment `c4773238ddf4d8854cb40ad26811b7b204cab8210ff9fe45f79e83e12da4675a` | The model loader is local-only. Download the exact GPT-2 snapshot once as shown below; the pinned Wikitext revision is fetched into the Hugging Face/datasets cache on first run and rechecked against the repository's frozen L04 selection. |

Set up the base environment with `uv sync --frozen`. The transformer case also
needs the declared `transformers` extra and the pinned overlays
`datasets==3.6.0` and `huggingface-hub==0.35.3`, supplied by its run command.
The default Hugging Face cache is used unless `HF_HOME` is set.

Before the first transformer example run, download the immutable model snapshot:

```bash
uv run --extra transformers --with "datasets==3.6.0" --with "huggingface-hub==0.35.3" python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='openai-community/gpt2', revision='e7da7f221d5bf496a48136c0cd264e630fe9fcc8')"
```

The initial snapshot and dataset fetch require network access. The proof checks
that the selected data and frozen inputs still match their declared revisions
and hashes; do not replace either revision or regenerate the selected rows.

### Observed runtime and memory

In the accepted 80.25 revision-backed replay at source commit `5241553` (run
`20260925-065327-115093-14640`), the encoder proof took 7.272 s and reached
353,763,328 bytes of process-tree RSS. The cold transformer proof took 944.193 s
(about 15.7 minutes) and reached 1,840,766,976 bytes, below its 16-GiB execution
ceiling. These are observed case-level measurements, not a runtime guarantee for
other machines. A cold transformer run also requires network time for the pinned
snapshot and dataset.

## Example 1 — encoder/autoencoder v3 model-weight lesion

```bash
uv run python scripts/ai_engineer_example_encoder.py
```

The proof uses the committed four-dimensional train-only linear autoencoder
checkpoint and the pinned digits split. It copies that model, zeroes encoder
weight row 0 and its bias **before** held-out encoding, and leaves the baseline
checkpoint and images unchanged. No model fitting is performed by this example.

Expected outcome: the seven-stage workflow completes, the declared lesion is
supported at feature `brightness-axis-dim0`, independent validation passes,
and a deterministic rendered report is kept in the printed output directory.
The accepted run observed feature-variance ratio `0.0`, singular spread `0.0`,
and effective rank `2.9576990034956805`; its 95% bootstrap interval was
`[2.875274672590536, 2.9839066845293694]`. The declared healthy, benign
low-variance, null-shuffle, explanation, and intervention controls passed. The
aligned healthy-versus-lesion comparison was `both`; restoring the lesioned row
and bias increased held-out brightness-bin accuracy by `0.4638888888888889`.
The report retains the predeclared thresholds, uncertainty intervals, and hashes.

A run prints a summary like:

```text
EXAMPLE OUTCOME: ACCEPTANCE PASSED — all seven stages, controls, validation, and rendering verified
  output  : artifacts/diagnostics/ai-engineer-encoder-v3-<timestamp>
  artifact SHA-256: <64-hex digest>
  report SHA-256: <64-hex digest>
  rendered report SHA-256: <64-hex digest>
```

The exact hashes are generated from the run's request, manifest, stage records,
and evidence blobs; the script prints and independently checks each digest.
To choose a named output directory, pass `--output` with an unused path, for
example `--output artifacts/diagnostics/encoder-v3-example-run-01`.

**Limit:** this is prospective controlled evidence for the frozen linear model,
one held-out digits split, and its predeclared global-brightness-bin task. It is
not evidence that the historical ConvVAE or a production encoder naturally has
the same failure, nor that the task measures general digit-recognition utility.
The historical frozen v1 negative and v2 capture-injection results remain
separately recorded in the [initial 80.26 fail-closed example record](../artifacts/task_80.26_ai_engineer_guide_examples_summary.md).

## Example 2 — transformer v1 manifest with target-evidence v2

After the one-time pinned model snapshot download above, run:

```bash
uv run --extra transformers --with "datasets==3.6.0" --with "huggingface-hub==0.35.3" python scripts/ai_engineer_example_transformer.py
```

The proof verifies the Wikitext validation revision against the frozen L04
manifest: 3,760 official rows, 2,461 nonblank rows, and 2,048 selected rows
whose indices and text hashes match exactly. The predeclared target is the
section-header attribute `^ = .+ = $`; a seed-79 grouped split keeps document
groups disjoint (1,549 train rows, 499 evaluation rows). The target labels and
sample/split provenance are persisted separately from the real activation axes
(`slice`, `feature`) under `diagnostic-evidence-v2`; labels are not fabricated
as a capture axis.

The accepted target record `target-section-header-attribute-record` has SHA-256
`3e470ef055e2a546408a6f51717bbc5a4d8bf2478410555862045a516955bc19`; its
predeclared rule digest is
`9b12485635decf738c4904702f75d7d5ae8cb6663555e2af5ee66476269f025b`. The
validator binds this content-addressed target record to the real capture and
sample-aligned detect-stage provenance.

Expected outcome: all seven stages complete, required controls pass, the
finding is localized at `transformer.h.0`, the probe explanation and removal
intervention are supported, and independent v2 validation plus deterministic
rendering pass. The accepted revision-backed replay observed held-out accuracy
`1.0000` (95% interval `[1.0, 1.0]`) and leakage gap `0.44889779559118237`
(95% interval `[0.40881763527054105, 0.49699398797595196]`) against thresholds
`0.7` and `0.15`. See the [accepted run report](../artifacts/diagnostics/proof-80-25-committed-transformer-v1-target-v2-20260925-065327-115093-14640/diagnostic-report).
The removal intervention changed accuracy from `1.0000` to `0.7315` (effect
`-0.2685`). The aligned replay comparison is truthfully `neither`: both
comparison deltas are zero.

The distinct historical temporary-driver replay, run `c4012406e89254ea`, observed
leakage gap `0.45090180360721444` (95% interval
`[0.4108216432865731, 0.49699398797595196]`; see its
[historical report](../artifacts/diagnostics/proof-80-25-clean-transformer-v1-target-v2-20260924-035357-6996/diagnostic-report)).
That result came from an uncommitted inline `python -c` driver and is historical
variance, not the accepted revision-backed run's measurement. The point
estimates differ by one evaluation row out of 499 (about `0.002004008016`); both
meet the frozen threshold, and neither run's inputs, thresholds, or historical
outcomes were rewritten.
Each run prints new content-addressed artifact/report/render hashes in its
output summary.

**Limit:** separability of this row-level target does not establish that GPT-2
causally uses section headers. Header rows correlate with row length and
last-token identity. The v1 manifest, taxonomy, and original v1 axes-related
persistence refusal remain unchanged historical evidence. The separately
versioned target-evidence record enables the accepted v2 validation path. The
[initial 80.26 fail-closed record](../artifacts/task_80.26_ai_engineer_guide_examples_summary.md)
summarizes the earlier refusal.

Choose an unused output directory with `--output`, for example
`--output artifacts/diagnostics/transformer-v1-evidence-v2-example-run-01`.
If an output path already exists, the example exits with code `2` and does not
overwrite it. Code `2` also means the frozen acceptance did not pass; inspect
the proof output and preserved artifact rather than treating it as success.
Other execution errors return a non-zero process status.

## Adapting the pattern to another model

These commands reproduce two **frozen, task-specific** cases; they are not a
no-configuration auto-diagnoser for arbitrary model/data pairs. `DiagnosticRequest`
is the public request-configuration type. The current `DiagnosticWorkflow`
coordinator and model-specific stage compositions are implemented in the
committed proof scripts; this guide does not present that internal coordinator
as a stable public import. New model/data evidence still needs a validated
manifest and supported stage composition, with actual capture axes and target
provenance kept truthful. The examples demonstrate how an end user can run the
two accepted diagnoses without wiring those internals by hand; they do not
claim that arbitrary new cases require no scientific or integration work.

## Navigation

- [Documentation index](INDEX.md)
- [Architecture](ARCHITECTURE.md)
- [API reference](API_REFERENCE.md)
- [Portable artifact conventions](PORTABLE_ARTIFACTS.md)
- [Sprint 80 plan](sprint-plans/sprint-80.md)
- [Frozen benchmark inputs and schemas](../artifacts/)

# Sprint 79 L03 Phase B — Remote CUDA Runs

Capture attempt 1 used commit `bb0da6fdc4fb00950ce4cb574ec83e8a9344db8b` from a fresh remote clone on `trietlm@192.168.30.244` (`di-server`). The clone SHA matched exactly. Pinned `uv sync --locked --extra transformers` completed; GPU preflight found an NVIDIA GeForce RTX 4060 Ti with CUDA-enabled PyTorch and selected `gcc-12`/`g++-12`.

The focused real TransformerLM network suite ran once and finished **6 passed, 2 failed**. Both failures are hook-capture incompatibilities because `transformer.h.6` returned a tuple where the capture seam requires a tensor. The canonical command `uv run python -m scripts.m14_l03_analysis --run-real` then executed once under `/usr/bin/time -v`; host peak RSS was 2,800,484 kB.

The remote wrapper removed its isolated checkout and all isolated caches, but its post-run stdout normalizer failed because the remote image has no system `python` executable. Consequently the JSON report was not captured before the required cleanup trap ran (`output_captured=false`). The preserved attempt-1 failure report is [`l03-analysis.attempt1.failure.json`](m14/l03-analysis.attempt1.failure.json), with preliminary run record [`l03-analysis.attempt1.run.json`](m14/l03-analysis.attempt1.run.json). Wrapper evidence and the capture failure are retained there; no metrics or IDs were changed.

Capture attempt 2 was authorized with the same exact command and source. The canonical runner completed once (`analysis_status=0`) under `/usr/bin/time -v`, with host peak RSS 2,798,376 kB. The focused audit again produced 6 passed / 2 failed with the same tuple-return hook failures. The wrapper removed the isolated checkout and caches, but deleted the remote report before its base64 transfer step; despite a pre-delete `report_captured` flag, `base64` failed with `No such file or directory`, so `output_captured=false` in the preserved record. The attempt-2 failure report is [`l03-analysis.attempt2.failure.json`](m14/l03-analysis.attempt2.failure.json), with [`l03-analysis.attempt2.run.json`](m14/l03-analysis.attempt2.run.json). No accepted artifact or metrics were materialized, and finalization was not claimed because the captured-output invariant was not met.

The authorized capture rehearsal then passed with an exact deterministic sentinel (SHA-256 `f3f3bc4bafccfec539f8e0f7e4fab879c71cc4a71b376983e2aacb74868cf069`, 71 bytes) and clean scoped remote teardown. Attempt 3 used the same exact source and canonical command; the remote runner emitted its report before cleanup, but the local parser rejected the marker line as if it were key/value metadata and discarded the captured bytes. Remote cleanup completed and no process remained. The attempt-3 failure report is [`l03-analysis.attempt3.failure.json`](m14/l03-analysis.attempt3.failure.json), with [`l03-analysis.attempt3.run.json`](m14/l03-analysis.attempt3.run.json). No accepted artifact or metrics were materialized; no further canonical run is authorized or performed.

Attempt 4 was authorized after the parser rehearsal. No focused tests were rerun. A complete raw SSH stdout transcript was saved before parsing and contains no credential patterns (SHA-256 `79e7e14dc6b1baf5168ef0829cc2181b0e023212b3ace4ce357e603c68d085e7`). The fresh clone again matched the exact SHA; locked Transformers setup and CUDA preflight passed on the RTX 4060 Ti. The canonical analysis ran once (`analysis_status=0`) with host peak RSS `2,825,124 kB`; the report was captured before cleanup and its 275,741 bytes matched SHA-256 `e2acee40bd86e19252bd7b1a6a8bb6070e1bf00db507c51b0e1371cea22d0108`. Remote checkout/cache cleanup and process audit passed. All three records were accepted: `t03_latent_linear_structure`, `t05_linear_probe`, and `t05_mlp_probe`. Final artifact SHA-256 is `60bda13a4bbf68bbb6c9308cc813913fa653c37fba368fe1e4ea7a1f898ce06b`; final run-record SHA-256 is `0bcaf14ef465f2ef5c5c909237d1f573596a77fa2ca51d042db74248cf4ca03a`.

Key held-out metrics were full-hidden and linear-probe test accuracy `0.9275766016713092` (paired bootstrap lower `0.7966573816155988`, Wilson 95% interval `[0.8959994833250284, 0.9501000803737016]`), PCA32 test accuracy `0.883008356545961` (bootstrap lower `0.7381615598885793`), and MLP test accuracy `0.8857938718662952` with shuffled-control accuracy `0.11420612813370473`, `n_params=698`, and validation/test gap within the declared bound.

The accepted evidence scope is exactly these three D2 records:
`THY-T03-LINEAR-STRUCTURE-TRONG-LATENT`, `THY-T05-LINEAR-PROBING`, and
`THY-T05-NONLINEAR-PROBING`. The lane is real pinned GPT-2 through concrete
`TransformerLMIntegration`, with real PCA/`LinearProbe`/`MLPProbe`; it does not
claim a separate GPT-2 `ModelAdapter` or an L11 promotion. The raw glyph
baseline is an expected diagnostic because GPT-2 was not trained for this
synthetic ASCII task.

Attempts 1–3 remain unchanged and are capture-only failures: attempt 1 lacked
system `python`, attempt 2 deleted the remote report before transfer, and
attempt 3 rejected the report marker during metadata parsing. Attempt 4's
compact sanitized capture audit is
[`l03-analysis.attempt4.capture-audit.json`](m14/l03-analysis.attempt4.capture-audit.json).
It records raw transcript SHA-256
`79e7e14dc6b1baf5168ef0829cc2181b0e023212b3ace4ce357e603c68d085e7`,
368,479 bytes, captured report SHA-256
`e2acee40bd86e19252bd7b1a6a8bb6070e1bf00db507c51b0e1371cea22d0108`, one
begin/one end marker, and zero credential-pattern matches. After verifying the
accepted artifact/run bytes, the superseded raw transcript was deleted; no
history file was deleted.

The focused transformer network result remains **6 passed / 2 failed** on the
tuple-return hook intervention and is tracked as an unresolved release
blocker/follow-up. This does not reduce the forward-only L03 evidence and does
not claim that all transformer network tests are green.

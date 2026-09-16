# Theory evidence-gap closure plan

This is the Sprint 78.38 execution plan for the non-qualifying theory rows,
updated with the Sprint 79 L02 and L03 results. The levels in
[`docs/evidence-ledger.json`](evidence-ledger.json) remain authoritative; the
immutable L02 artifact supports exactly five D2 promotions and the immutable
L03 artifact supports exactly three D2 promotions. Later row-level promotions
are recorded below without changing the unchecked Sprint 79 line 595. The
row-level machine-readable map is
[`artifacts/task_78.38_gap_map.json`](../artifacts/task_78.38_gap_map.json).

## Current gate and arithmetic

The read-only validator reports 107 capabilities, **40/63 core (63.492063%)**
and **40/65 overall (61.538462%)**. The core denominator contains 63
implementation-applicable or benchmark-only rows in T01–T09/T03B; the overall
denominator adds two applicable X01 rows. The validator reports **20 D0 rows**
and **5 D1 rows**; a qualifying row is D2 or D3. At least **20 additional core
qualifiers** are required to reach `ceil(0.95 × 63) = 60`, and at least
**19 additional overall qualifiers** are required to reach
`ceil(0.90 × 65) = 59`. The core gate is therefore the binding gate. Headline
model, causal explanation, and named integration claims target D3; ordinary
algorithm capabilities target D2 unless their row says otherwise.

### Sprint 79 L02 partial result

The committed, reproducible artifact
[`l02-geometry.json`](../artifacts/m14/l02-geometry.json) passed its artifact
and run-record validators and supports exactly four independent D2 promotions:
`THY-T03-SLERP-SPHERICAL-LINEAR-INTERPOLATION`,
`THY-T04-LERP-LINEAR-INTERPOLATION`,
`THY-T03-RIEMANNIAN-GEOMETRY-CO-BAN`, and `THY-T04-SLERP`. The artifact
self-digest is
`97d26f5fb1d12dc00658ff9cfec12a91b080bb4a1bb3cd96e7bc2ed70f9e5a58`, and the
two canonical runs were deterministic apart from their UTC run-record
timestamps.

The `THY-T01-MANIFOLD-HYPOTHESIS` record remains D1 because its held-out
ranking AUC was `0.4560546875` (threshold `0.55`) and its latent-vs-raw delta
was `-0.4124755859375` (threshold `-0.05`). Its exact-SHA rerun and failure
receipt are retained without overwriting the canonical artifact. The
`THY-T06-TRAJECTORY-SIMILARITY-METRICS` record is now D2 through the immutable
accepted rerun artifact: 128 finite trials, ranking AUC `1.0`, self-to-indexwise
ratio `0.07354340933887214` (threshold `0.95`), and self-to-unrelated ratio
`0.011170843549800243` (threshold `0.9`). These are model-induced latent
sequences from held-out sklearn digits, not physical trajectories, and make no
Fréchet claim.

### Sprint 79 L03 result

The remote CUDA run used the pinned `openai-community/gpt2` revision through
the concrete `TransformerLMIntegration` and produced the validated artifact
[`l03-analysis.json`](../artifacts/m14/l03-analysis.json) with self-digest
`60bda13a4bbf68bbb6c9308cc813913fa653c37fba368fe1e4ea7a1f898ce06b`. Its
final run record digest is
`0bcaf14ef465f2ef5c5c909237d1f573596a77fa2ca51d042db74248cf4ca03a` and the
plan digest is
`fe2a85a1691c0fe362fc5f39434898d6ea8968aeec8450a7bb61ba55fd94cfd5`.
Exactly three independent records are promoted to D2:
`THY-T03-LINEAR-STRUCTURE-TRONG-LATENT`, `THY-T05-LINEAR-PROBING`, and
`THY-T05-NONLINEAR-PROBING`.

The evidence is forward-only real pinned GPT-2 with concrete
`TransformerLMIntegration`, real PCA/`LinearProbe`/`MLPProbe`, and the
sklearn-digits glyph control. It does not claim a separate GPT-2
`ModelAdapter` or an L11 promotion; the raw-glyph baseline is an expected
diagnostic because GPT-2 was not trained for this synthetic ASCII task.
The focused network suite first returned **6 passed / 2 failed** because the
tuple-return hook intervention was incompatible with the capture seam. After
the structured-output fix in `16db80f`, the intermediate verification exposed
a separate **7 passed / 1 failed** indexing-oracle error; `9ebecfa` corrected
that test contract without remapping runtime layers. The final exact-SHA
strict-CUDA run passed **8/8**, including the native-index-7 intervention
oracle and cleanup test. The structured hook/output blocker is therefore
resolved by `16db80f` + `9ebecfa` and the retained transformer-hook
[attempt-1 failure](../artifacts/task_79_transformer_hook_remote_verification_attempt1.md)
and [attempt-2 verification](../artifacts/task_79_transformer_hook_remote_verification.md).
The forward-only L03 evidence and its D2 promotions are unchanged. The
separate native hidden-state index-12/direct-logit-lens parity follow-up is
complete as an internal semantic correction: attempt 1's missing optional
`transformers` dependency and attempt 2's exact-SHA direct-PowerShell-SSH
8/8 CUDA verification are preserved in the sanitized
[`task_79_logit_lens_remote_cuda_verification_final.json`](../artifacts/task_79_logit_lens_remote_cuda_verification_final.json)
record and digests, without an L11 promotion. Attempts 1–3 of the canonical
L03 capture workflow remain preserved capture-only failures; attempt 4 remains
represented by the sanitized capture audit.

The validator command is:

```text
uv run python scripts/validate_evidence_ledger.py --json
```

It must continue to report the honest current 38/63 and 38/65 result while
this plan is executed. No row may be deleted, relabeled, or promoted merely to
improve the percentages.

### L04 design freeze (Sprint 79 L04.1)

The five L04 records are preregistered in dependency order in
[`l04-explanations.plan.json`](../artifacts/m14/l04-explanations.plan.json):
TCAV (`THY-T05-CONCEPT-ACTIVATION-VECTORS-TCAV-KIM-ET-AL-2018`), direct plus
holdout-calibrated affine tuned lens
(`THY-T05-LOGIT-LENS-TUNED-LENS`), disentanglement
(`THY-T03-DISENTANGLEMENT`), true clean/corrupted interchange patching
(`THY-T05-ACTIVATION-PATCHING`), and additive steering
(`THY-T05-STEERING-VECTORS-ZOU-ET-AL-2023-REPRESENTATION-ENGINEERING`).
TCAV and lens depend on the completed L03 linear-probing boundary;
disentanglement, patching, and steering depend on TCAV; the resulting graph is
acyclic with no SCC. Current D0/D1 levels remain unchanged.

L04 pins `openai-community/gpt2` at
`e7da7f221d5bf496a48136c0cd264e630fe9fcc8` under MIT access and uses the
concrete `TransformerLMIntegration`; `ModelAdapter` is intentionally N/A.
The authored task/factor fixture
[`l04-prompt-factor-fixture.jsonl`](../artifacts/m14/l04-prompt-factor-fixture.jsonl)
has 24 rows/12 groups, an explicit classification task, exactly one clean and
one corrupted condition per `causal_pair_id`, group-preserving train/holdout
assignment, and frozen content/split/pair SHA-256 digests. It is a controlled
synthetic fixture with limited external validity and cannot by itself establish
D3. All seven real model/integration use cases (IG, TCAV, direct lens, tuned
lens, disentanglement, true interchange patching, additive steering) must
execute on the CUDA server through authenticated direct PowerShell `ssh.exe` at
an exact detached code SHA; Git Bash/WSL and local CPU real-model evidence are
out of scope. True interchange patching, additive steering, direct lens, and
separately fit holdout-calibrated tuned lens remain distinct executions. The
authoritative pinned WikiText-2 subset is provisioned and its content/split
digests and selection metadata are bound by the committed manifest. The first
owner-authorized exact-SHA run (`dcc76ba7f064b5b6dc2e09c20d741da4cc6e5422`)
reached real CUDA execution but failed D0 at tuned-lens aggregation. A later
setup D0 failed before model loading with `ModuleNotFoundError: datasets` because
the isolated environment omitted same-environment provisioning; it has no
semantic metrics, and its cleanup and outer SSH exit are not evidenced or
claimed. The final corrected run at exact SHA
`278a9f76f626f8b0c6a9d9c5517c9b349f08c2d5` now has a validator-clean D3
artifact with acceptance true; its sanitized audit records the real CUDA
execution, fit seed `79`, common shuffled-target mask policy, metrics,
controls, resource budget, and cleanup. Thresholds,
formulas,
aggregation units, comparator strictness, and randomized, shuffled, null,
off-target, and zero-strength controls are frozen in the machine-readable plan.
Each remote SSH invocation runs exactly one parameterized use case with raw
stdout/stderr capture and owner review before the next; a blocked tuned-lens
attempt is isolated and cannot poison unrelated records.

The final TunedLogitLens semantic/artifact evidence remains accepted at D3:
the remote start/status/cleanup markers and outer SSH exit were successful, the
full bundle was captured, and artifact, run-record, failure, and audit-linkage
validators returned no errors. The raw capture nevertheless retained
`base64: invalid input`; the decoded script bytes were not independently
hash-verified, so the audit does not claim stronger transport provenance. The
historical direct `base64 -d | bash -s --` recipe is **NOT REUSABLE** for L04.8
or later lanes. The reusable replacement is now checked in as
[`m14_l04_remote_transport.ps1`](../scripts/m14_l04_remote_transport.ps1) and
[`m14_l04_remote_payload.sh`](../scripts/m14_l04_remote_payload.sh): the
PowerShell helper normalizes and hashes exact UTF-8/no-BOM/LF bytes, launches
the native `ssh.exe` with `ProcessStartInfo`, and writes raw stdout/stderr
before parsing; its remote bootstrap decodes to a temporary file, requires
decoder exit `0`, compares the decoded SHA-256, executes, and verifies cleanup.
The separate Bash payload owns the exact detached clone, isolated caches,
same-environment preflight, one CLI invocation, bundle-before-cleanup, and
full-workdir cleanup. Offline build-only/static tests cover this contract;
another remote run remains owner-gated until the helper is committed and
executed from authenticated Windows PowerShell.

The helper's `-TransportTimeoutSeconds` defaults to 3600 and is restricted to
2400–7200 seconds. One monotonic deadline covers setup, the semantic command,
bundle creation, cleanup, asynchronous stdin and output draining, and raw
publication; the semantic protocol cap remains 1800 seconds. Expiry kills the
entire process tree, waits no more than the fixed 30-second grace, and records
`transport_termination_incomplete` plus cleanup `unknown` if termination is
not confirmed. The payload emits a sanitized absolute `L04_WORKDIR` marker
immediately after `mktemp`; it never uses wildcard cleanup.

For any future TunedLogitLens execution, the frozen plan remains immutable and
the owner-approved operational override is the reusable helper described in
the [M14 validation procedure](M14_REAL_SYSTEM_VALIDATION.md#reusable-l04-transport-authoritative):

```powershell
$RepoUrl = (git remote get-url origin).Trim()
pwsh -NoProfile -File scripts/m14_l04_remote_transport.ps1 -SshExecutable (Get-Command ssh.exe).Source -RemoteTarget trietlm@192.168.30.244 -PayloadPath scripts/m14_l04_remote_payload.sh -UseCase TunedLogitLens -CodeSha (git rev-parse HEAD).Trim() -RepoUrl $RepoUrl -RawCapturePath artifacts/m14/l04-tuned-logit-lens.raw.txt -SshConnectTimeoutSeconds 15 -TransportTimeoutSeconds 3600
```

The remote import/version/CUDA preflight must use the identical `uv` environment
and all four exact package constraints. A local PowerShell preflight using a
safe temporary `.py` file (never `python -c`) is mandatory before SSH for
dependency/import compatibility only; it does not require local CUDA. The
recorded diagnostic passed `datasets==4.8.5`, `transformers==4.57.6`,
`tokenizers==0.22.2`, and `huggingface-hub==0.35.3`. The historical wrapper
record is retained in the [M14 operational
procedure](M14_REAL_SYSTEM_VALIDATION.md#reusable-l04-transport-authoritative)
as audit evidence only. New executions must use the reusable helper/payload;
the old direct `ssh.exe target 'bash -s --' ...` recipe is not an operational
instruction. The payload creates a temporary `preflight.py` with a
single-quoted heredoc, clones and verifies the exact SHA, isolates all
UV/HF/datasets/transformers caches under one workdir, runs preflight and the
single CLI, captures and bundles before cleanup, and uses explicit markers.
Cleanup removes and verifies the full workdir, emits PASS only after `rm`
succeeds, and the PowerShell helper captures the SSH exit immediately. The
payload exports `LATENT_ANYTHING_RUN_NETWORK=1` and
`LATENT_ANYTHING_NETWORK_DEVICE=cuda` before both preflight and CLI.
The required remote marker set also includes `L04_WORKDIR`, emitted only after
the exact normal workdir path has been validated; cleanup remains a verified
marker rather than a wildcard-based assumption.
The payload routes `nvidia-smi` and the CLI's JSON diagnostics to stderr so
stdout remains limited to declared L04 markers and the bounded Base64 bundle;
the retention parser therefore remains strict on unexpected stdout and
marker-like stderr.
`python -c`, escaped `printf`, and nested remote command quoting are forbidden
because native PowerShell parsing can strip quotes/backslashes and corrupt the
script. Raw stdout/stderr is hashed before parsing. The attempt2
artifact/run/failure files and sanitized SSH audit are required committed
fixtures for the validator regression and remain setup D0 only.

The latest SHA `3273b23bc4b490114518559a994ef5e50523524a` is also setup D0:
the preflight stopped because an ad-hoc `datasets` overlay selected
`huggingface-hub==1.26.0`, incompatible with `transformers==4.57.6`. The CLI,
model, and WikiText corpus were not reached or planned; no semantic metrics,
gates, bootstrap, or resources exist. One exact `L04_CLEANUP=PASS` was emitted,
SSH and wrapper exits were both `1`, and the raw capture was sanitized and
deleted after audit verification. Its audit and one-byte exit remain retained
D0 fixtures.

The latest SHA515fe protocol-failure audit records the CRLF raw capture
(7,140 bytes; SHA-256
`3fc9865f424e4618d427a4ee8330c8cfa490fee75edaa5e8264a4a1360bff743`), exact
quote-loss `SyntaxError`, malformed cleanup marker, SSH/wrapper exit `1`, and
no CLI/model/dataset semantic execution. Cleanup remained unverified. The
sanitized audit was validated before the raw capture was deleted; only the
audit and one-byte exit record are retained, with no secrets, corpus text,
logits, hidden states, or model payload.

The first Phase B `Disentanglement` attempt at the pushed SHA
`ac371125e83ac07553c7291183b089559c4b063d` is retained as setup D0: transport
decode and SHA matching passed, `nvidia-smi` observed an RTX 4060 Ti, but the
single connection timed out while downloading the dependency environment
before the CLI. No semantic exit, bundle, metrics, acceptance ID, or remote
cleanup marker was observed; cleanup is unknown and no cleanup-only SSH or
implicit retry is allowed. Its sanitized audit is retained without the raw
capture.

The current L04.8 Disentanglement execution at exact pushed SHA
`4d3a4b6551d6091ce96c73a704e642867c2f2580` completed the real CUDA handler and
passed the strict factor-permutation, group-preserving shuffle, raw-token
baseline, seeded-repeat, and held-out-gain controls. Its retained triplet and
sanitized audit are tracked under `artifacts/m14/`; the audit records archive
SHA-256 `c222475e9591eed5fbc45f6202aff7edaf83926a8fa7a6a0e85699aacce614d0`,
all member hashes, validator PASS, and final raw status `deleted_verified`.
The theory row is therefore promoted to D2/eligible only for that SHA. The
195-byte transport timeout (`5a91d5534d5c47c20df4e176895734b379b342a55817e3f8feb61a325e50535b`)
and 266082-byte strict-retention failure
(`176a663119a70d02a440aab4f03de2ba4a2f0f85e3c29d8d53be4bc101d205b0`) remain
untracked raw captures with sanitized, non-promoting sidecars; neither is
reconstructed or promoted.

## Exhaustive row inventory

Each row appears exactly once below. Detailed prerequisites, commands, tests,
acceptance, artifact, cleanup, blocker, and dependency fields are in the JSON
map; lane-level defaults are specified in the next section.

| ID | Current | Core | Headline | Target | M14 lane | Capability / insufficiency |
|---|---:|:---:|:---:|:---:|:---:|---|
| `THY-T01-METRIC-SPACE-VA-VECTOR-SPACE` | D2 | yes | no | D2 | L01 | Existing ConvVAE/AnalysisPipeline held-out benchmark and immutable artifact verified |
| `THY-T01-MANIFOLD-HYPOTHESIS` | D1 (failed canonical + exact-SHA rerun) | yes | no | D2 | L02 | Held-out ranking remains failed: latent AUC 0.4561 vs 0.55 and raw-pixel delta -0.4125 vs -0.05; retained rerun artifact, no promotion |
| `THY-T02-VAE-HIGGINS-ET-AL-2017` | D0 | yes | no | D2 | L08 | Theory-only beta-VAE row |
| `THY-T02-VQGAN-ESSER-ET-AL-2021` | D0 | yes | no | D2 | L13 | Theory-only VQGAN row; no approved implementation |
| `THY-T03-LINEAR-STRUCTURE-TRONG-LATENT` | D2 | yes | no | D2 | L03 | Validated L03 artifact includes a real pinned GPT-2 held-out PCA structure benchmark |
| `THY-T03-DISENTANGLEMENT` | D2 | yes | yes | D2 | L04 | Exact-SHA real CUDA Disentanglement passed controls; retained triplet and deleted-verified audit are tracked |
| `THY-T03-RIEMANNIAN-GEOMETRY-CO-BAN` | D2 | yes | no | D2 | L02 | Accepted bounded density-geodesic record in the L02 artifact |
| `THY-T03-SLERP-SPHERICAL-LINEAR-INTERPOLATION` | D2 | yes | no | D2 | L02 | Accepted held-out unit-norm interpolation record in the L02 artifact |
| `THY-T03-NORMALIZING-FLOWS` | D0 | yes | no | D2 | L05 | Theory-only flow row |
| `THY-T03B-GAUSSIAN-PARAMETERS-LA-LATENT-VARIABLE` | D1 | yes | yes | D3 | L17 | Reference renderer evidence is not real 3DGS evidence |
| `THY-T03B-DYNAMIC-3DGS` | D0 | yes | yes | D3 | L17 | Theory-only dynamic-scene claim |
| `THY-T04-LERP-LINEAR-INTERPOLATION` | D2 | yes | no | D2 | L02 | Accepted held-out LERP record in the L02 artifact |
| `THY-T04-SLERP` | D2 | yes | no | D2 | L02 | Accepted independent T04 SLERP record in the L02 artifact |
| `THY-T04-DENSITY-ESTIMATION-TRONG-LATENT` | D0 | yes | no | D2 | L05 | Theory-only density row |
| `THY-T04-OPTIMAL-TRANSPORT-TRONG-LATENT` | D0 | yes | no | D2 | L05 | Theory-only transport row |
| `THY-T05-LINEAR-PROBING` | D2 | yes | yes | D2 | L03 | Validated L03 artifact includes a real pinned GPT-2 held-out LinearProbe benchmark |
| `THY-T05-NONLINEAR-PROBING` | D2 | yes | yes | D2 | L03 | Validated L03 artifact includes a real pinned GPT-2 held-out MLPProbe benchmark |
| `THY-T05-CONCEPT-ACTIVATION-VECTORS-TCAV-KIM-ET-AL-2018` | D0 (retained semantic failure) | yes | yes | D3 | L04 | Exact-SHA real CUDA TCAV artifact is validator-clean but fails Wilson lower-bound and corrected empirical-p-value gates; no promotion |
| `THY-T05-ACTIVATION-PATCHING` | D3 | yes | yes | D3 | L04 | Exact-SHA real CUDA true clean/corrupted interchange patching passed controls; retained artifact/run/failure triad is tracked |
| `THY-T05-SPARSE-AUTOENCODER-SAE-ANTHROPIC-2023` | D1 (retained stability failure) | yes | yes | D3 | L06 | Real pinned GPT-2 layer-6 SAE artifact is retained; cross-seed cosine stability remains below frozen thresholds, so no promotion |
| `THY-T05-DICTIONARY-LEARNING` | D0 | yes | no | D2 | L06 | Theory-only dictionary-learning row |
| `THY-T05-STEERING-VECTORS-ZOU-ET-AL-2023-REPRESENTATION-ENGINEERING` | D1 | yes | yes | D3 | L04 | D1 steering tests lack real causal/selective evidence |
| `THY-T05-LOGIT-LENS-TUNED-LENS` | D3 | yes | yes | D3 | L04 | Corrected exact-SHA real CUDA artifact passes validation with fit seed 79 and common source/target shuffled mask; attempt3 remains immutable historical D0 |
| `THY-T06-STOCHASTIC-TRANSITION` | D2 | yes | no | D2 | L15 | Existing seeded held-out stochastic Gaussian transition artifact is reconciled to this theory ID with explicit compact synthetic scope; no real-world or pretrained temporal-model claim |
| `THY-T06-RSSM-RECURRENT-STATE-SPACE-MODEL-DREAMER` | D0 | yes | yes | D3 | L15 | Compact RSSM is synthetic, not named Dreamer evidence |
| `THY-T06-TRAJECTORY-SIMILARITY-METRICS` | D2 | yes | no | D2 | L02 | Accepted immutable rerun: 128 finite trials; self-to-indexwise ratio 0.07354340933887214; self-to-unrelated ratio 0.011170843549800243; ranking AUC 1.0 |
| `THY-T07-MODEL-PREDICTIVE-CONTROL-MPC` | D2 | yes | no | D2 | L16 | Bounded receding-horizon MPPI evidence passes held-out recorded-trajectory baseline, replanning, bound, finite-state, and budget gates; no CEM/MPPI equivalence or real pretrained-controller claim |
| `THY-T07-POLICY-GRADIENT-TREN-IMAGINED-TRAJECTORY-DREAMER` | D0 | yes | yes | D3 | L16 | No actor-critic/policy-gradient lane |
| `THY-T07-VALUE-EQUIVALENCE-MUZERO` | D0 | yes | yes | D3 | L16 | No value-equivalence representation/training lane |
| `THY-T07-MCTS-TRONG-LATENT` | D0 | yes | no | D2 | L16 | No MCTS implementation/benchmark |
| `THY-T08-REPRESENTATION-COLLAPSE` | D2 | yes | no | D2 | L12 | Compact JEPA health and held-out prediction artifact demonstrates non-collapse and gain over a collapsed baseline; named I-JEPA/LeWM D3 remains separate |
| `THY-T08-I-JEPA-ASSRAN-ET-AL-2023` | D0 | yes | yes | D3 | L12 | Compact JEPA is not the named I-JEPA checkpoint |
| `THY-T08-V-JEPA-BARDES-ET-AL-2024` | D0 | yes | yes | D3 | L12 | No video JEPA implementation/model lane |
| `THY-T09-EMA-CODEBOOK-UPDATE` | D0 | yes | no | D2 | L13 | No dedicated EMA update evidence |
| `THY-T09-RESIDUAL-VQ-SOUNDSTREAM-ENCODEC` | D0 | yes | no | D2 | L13 | No residual-VQ implementation/evidence |
| `THY-T09-FINITE-SCALAR-QUANTIZATION-FSQ` | D0 | yes | no | D2 | L13 | No FSQ implementation/evidence |
| `THY-T09-GAIA-1-WAYVE-2023` | D0 | yes | yes | D3 | L14 | Named model survey only |
| `THY-T09-GENIE-GOOGLE-2024` | D0 | yes | yes | D3 | L14 | Named model survey only |
| `THY-X01-OPENVLA` | D0 | no | yes | D3 | L19 | Named policy survey only; non-core X01 |
| `THY-X01-LEWM-LEWORLDMODEL-2026` | D1 | no | yes | D3 | L12 | Compact JEPA is not real LeWM evidence |

## Execution contract and ordering

Execute in dependency order. First close bounded local D2 rows in L01/L02,
L03–L06, L08, L12–L16, then execute named real-model D3 lanes. A row may be
promoted only when its source, focused tests, benchmark, config, and (for D3)
artifact all exist and the validator passes.

Every lane artifact must predeclare the integer seed, exact sample and held-out
split, metric/control definitions, acceptance thresholds, backend/model/dataset
revision, license/access, environment, network policy, resource peak, cleanup,
and SHA-256. A failed run is retained and linked; metrics or evidence levels
must not be edited to hide failure. Use local isolated temp directories for
CPU lanes. Use the remote-cuda-test workflow for CUDA lanes only, with a
disposable clone and isolated cache. No model/network/CUDA work is performed
by 78.38 itself.

### Lane batches

- **L01/L02 core primitives and geometry:** use the existing sklearn-digits or
  recorded-trajectory substitutes, deterministic held-out splits, finite
  shape/dtype/no-mutation checks, endpoint/metric/DTW/geodesic controls, and
  `artifacts/m14/l01-core.json` / `l02-geometry.json`.
- **L03–L06 analysis and explanations:** pin GPT-2
  `openai-community/gpt2@e7da7f221d5bf496a48136c0cd264e630fe9fcc8` and its
  license/access; use train/holdout labels, shuffled-label/raw-input/capacity
  controls, seed confidence intervals, selectivity and intervention metrics;
  write the L03, L04, L05, and L06 artifacts. For L04, every real model or
  integration use case is a CUDA-server run transported by authenticated direct
  PowerShell `ssh.exe` with an exact-SHA isolated clone; local CPU is limited to
  offline checks. The L04 host is exactly `trietlm@192.168.30.244`. A
  real-model requirement is never satisfied by a compact fixture, and an
  authored fixture alone is never a D3 claim.
- **L08/L13–L16 bounded model/planning lanes:** preserve existing compact D2
  negative results, held-out splits, codebook/non-collapse controls, sequence
  masks, rollout drift, action bounds, fixed-zero/random-shooting controls,
  and deterministic digests. New VQGAN, EMA/residual-VQ/FSQ, MCTS, policy
  gradient, MuZero, GAIA-1, or Genie rows need owner-approved implementations
  and separate artifacts; existing nearby rows cannot be silently reused.
- **L12 temporal/JEPA:** use the compact collapse-control D2 lane first. D3
  requires the exact named checkpoint and license/access for I-JEPA or LeWM;
  V-JEPA is a separate implementation/data lane. Preserve open-loop drift and
  other negative results.
- **L17 3DGS:** remains blocked until a named checkpoint, revision,
  license/access, CUDA host, and disposable remote lane exist. The existing
  deterministic/reference renderer must not be promoted to D3.
- **L19/OpenVLA:** remains a separate named-policy D3 lane; do not count it as
  core. ACT/Diffusion/SmolVLA lanes do not prove OpenVLA.

## Explicit blockers and waivers

The 3DGS blocker is concrete: M14 L17 has no named checkpoint and therefore no
real multi-view PSNR/SSIM or intervention artifact. The SmolVLA causal claim
also remains **D2 pending a corrected pinned CUDA rerun**: the historical D3
artifact is unverified and must not be counted. Execute the corrected L21 lane
with the pinned model/dataset revisions, simulator controls, seed, thresholds,
and signed artifact before promotion.

External GitHub Actions access, missing model/data licenses, unavailable
checkpoints, CUDA/VRAM, or missing dataset access are blockers—not reasons to
lower thresholds. An owner waiver must state scope, rationale, expiry, and
whether it affects core or headline coverage; no waiver can hide an applicable
core gap or authorize a stable-release claim. Sprint 79 owns execution and
artifact signing; Sprint 80 remains stop-before-release until the validator is
at least 95% core and 90% overall and all headline D3 claims are satisfied.

### L04.8 recovery correction record

The immutable `ce4e66e` Disentanglement audit remains a truthful historical D0 failure:
the real CLI reached production and raised `KeyError('condition')`, only the
CLI-side validators were available, the exact bundle was absent after GNU tar
option-order failure, raw deletion was not verified, cleanup markers were
`PASS`, and no evidence was promoted. The recovery implementation now carries
the exact authored `clean`/`corrupted` condition through the shared reader,
rejects missing or tampered conditions, normalizes Linux peak RSS to bytes with
coherent source/unit provenance, and separates `L04_CLI_STATUS` from
`L04_BUNDLE_STATUS` while preserving the CLI-first exit policy. The NUL-safe
bundle command and exact three-member gate remain frozen. This is a local
correction only; it remains D0 and is not changed by the independently
validated current SHA `4d3a4b6551d6091ce96c73a704e642867c2f2580` closure above.

### Queue position 11 — L16 MPC reconciliation

`THY-T07-MODEL-PREDICTIVE-CONTROL-MPC` is promoted from D0 to D2 using the
target-level offline compact lane in
[`artifacts/m14/l16-mpc.json`](../artifacts/m14/l16-mpc.json). The lane uses
seed `1601`, 56 train and 16 held-out recorded sklearn-digits episodes, three
one-step receding-horizon replans, and 1,728 bounded samples. MPC return
`1.2989200818808908` exceeds the fixed-zero baseline
`0.8318605499808998` by `0.46705953189999105`; random-shooting return
`1.510070544270507` remains a reported control and is not overinterpreted.
The source/test/config/artifact/run receipt are linked in the machine-readable
ledger. The exact focused suite passed 33 tests and the ledger validator
returned `errors: []` at 39/63 core and 39/65 overall. This evidence is only
bounded compact D2; it does not establish a real pretrained controller,
CEM/MPPI equivalence, or CUDA claim. Positions 2–4 and 6–9 remain blocked and
all later queue rows and line 595 remain unchanged.

### Queue position 27 — L02 trajectory similarity reconciliation

`THY-T06-TRAJECTORY-SIMILARITY-METRICS` is promoted from D0 to D2 using the
immutable accepted rerun artifact
[`l02-geometry.rerun-90a8c64.json`](../artifacts/m14/l02-geometry.rerun-90a8c64.json)
and the target-level config/receipt
[`l02-trajectory-similarity.config.json`](../artifacts/m14/l02-trajectory-similarity.config.json).
The exact in-memory rerun at source SHA `62664b53778ba808a2fc6da96f0a250978b6eb0b`
returned 128 independent finite pair trials, ranking AUC `1.0`, median
self-to-indexwise ratio `0.07354340933887214` (maximum `0.95`), and median
self-to-unrelated ratio `0.011170843549800243` (maximum `0.9`), with unequal
length, no-self-mapping, and no-input-mutation controls passing. The focused
suite passed 143 tests and the ledger validator returned `errors: []` at
40/63 core and 40/65 overall. This bounded compact lane measures
model-induced trajectories from held-out sklearn digits; it does not claim
physical trajectories, Fréchet distance, or named-model performance.
Positions 2–4, 6–9, and 12 remain blocked; line 595 and all later queue rows
remain unchanged.

### Queue position 33 — L04 TCAV failure reconciliation

`THY-T05-CONCEPT-ACTIVATION-VECTORS-TCAV-KIM-ET-AL-2018` is the earliest
dependency-order row after queue position 27 that had its L03 prerequisite,
owner authorization, pinned GPT-2 fixture, and CUDA runtime. Positions 28 and
32 (linear and nonlinear probing) were already qualifying; positions 29
(V-JEPA), 30 (Genie), and 31 (policy-gradient) remain blocked by missing
implementation/checkpoint/access prerequisites.

The owner-reviewed exact-SHA recovery at source
`5c38b63f01d280939790e415de699ab285a228de` ran the pinned
`openai-community/gpt2@e7da7f221d5bf496a48136c0cd264e630fe9fcc8` through
`TransformerLMIntegration` at layer 6/native hidden-state index 7. Its
validator-clean artifact/run/failure triplet and sanitized remote audit are
retained at
[`l04-explanations.TCAV.attempt1.partial.json`](../artifacts/m14/l04-explanations.TCAV.attempt1.partial.json),
[`l04-explanations.TCAV.attempt1.run.json`](../artifacts/m14/l04-explanations.TCAV.attempt1.run.json),
[`l04-explanations.TCAV.attempt1.failure.json`](../artifacts/m14/l04-explanations.TCAV.attempt1.failure.json),
and
[`l04-explanations.ssh.TCAV.attempt3.audit.json`](../artifacts/m14/l04-explanations.ssh.TCAV.attempt3.audit.json).
The target-level threshold config and reconciliation receipt are
[`l04-tcav.config.json`](../artifacts/m14/l04-tcav.config.json) and
[`l04-tcav.run.json`](../artifacts/m14/l04-tcav.run.json).

The run measured held-out accuracy `0.875`, Wilson lower bound
`0.5291118177871466` against strict `> 0.55`, bootstrap lower `1.0`,
corrected empirical p `0.24` against `<= 0.05`, and intervention agreement
`1.0`; all five frozen controls passed. The two failed semantic gates retain
the row at D0, with `accepted_gap_ids=[]`, `accepted_record_ids=[]`, and no
promotion. Focused TCAV validation passed **27 tests**; the ledger validator
returned `errors: []` at **40/63 core (63.492063%)** and **40/65 overall
(61.538462%)**. The failure-preserving reconciliation summary is
[`task_79_queue33_tcav_reconciliation_summary.md`](../artifacts/task_79_queue33_tcav_reconciliation_summary.md).
No later queue row, Sprint 79 line 595, or later plan item
was changed.

### Queue position 35 — L06 SAE stability-failure reconciliation

`THY-T05-SPARSE-AUTOENCODER-SAE-ANTHROPIC-2023` is the earliest
dependency-order row after queue position 33 that was executable after the
already qualifying queue position 34 TunedLogitLens row. Positions 36
(Disentanglement), 37 (Activation Patching), and 38 (Steering) remain gated
by the failed TCAV dependency and their real-model causal contracts; position
39 (Dictionary Learning) lacks a dedicated implementation; position 40
(OpenVLA) lacks checkpoint, license/access, and adapter scope.

The retained exact-source L06 execution at
`005954c636ae7a45ac3072c69e2c118db044682b` used pinned
`openai-community/gpt2@e7da7f221d5bf496a48136c0cd264e630fe9fcc8`, the expanded
`prompt-grid-v2-1024` fixture, layer 6 hidden states, and independent seeds
`[0, 1, 2]`. The target-level config and reconciliation receipt are
[`l06-sae.config.json`](../artifacts/m14/l06-sae.config.json) and
[`l06-sae-reconciliation.run.json`](../artifacts/m14/l06-sae-reconciliation.run.json).
The genuine lane artifact, execution record, and feature atlas remain
[`l06-sae.json`](../artifacts/m14/l06-sae.json),
[`l06-sae-run.json`](../artifacts/m14/l06-sae-run.json), and
[`l06-feature-atlas.json`](../artifacts/m14/l06-feature-atlas.json).

The run captured `10,752` tokens with finite reconstruction MSE
`0.7104635182257194`, zero dead features, mean matched cosine
`0.8467253367037664` as a diagnostic, minimum matched cosine
`0.7387987235614891` against the frozen strict `> 0.85` gate, and alignment
quality `0.75` against strict `> 0.7`. The minimum-cosine gate failed, so
cross-seed stability failed despite finite reconstruction and bounded dead
features. The real pinned model evidence remains D1; no D3 promotion or
threshold change occurred.
Focused SAE validation passed **21 tests**. The ledger validator returned
`errors: []` at **40/63 core (63.492063%)** and **40/65 overall (61.538462%)**.
The failure-preserving summary is
[`task_79_queue35_sae_reconciliation_summary.md`](../artifacts/task_79_queue35_sae_reconciliation_summary.md).
Sprint 79 line 595 and all later plan rows remain unchanged.

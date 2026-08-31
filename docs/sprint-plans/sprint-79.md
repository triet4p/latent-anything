# Sprint 79 Plan

## Sprint Goal

Run the release-candidate matrix across supported Python versions, optional extras, real models, LeRobot policies, world models, plugins, tracking, and artifact compatibility.

## Atomic Tasks

Status legend: [ ] pending / [~] in progress / [x] done

- [x] Reconcile all 40 Sprint 78.38 gap records with the 24-lane M14 contract
  into a deterministic dependency queue; run bounded remote CUDA preflight and
  one representative pinned real-model smoke, retaining the GPT-2 revision
  failure and Diffusers VAE pass without evidence promotion. See
  [`task_79.1_summary.md`](../../artifacts/task_79.1_summary.md) and the
  machine-readable [`task_79.1_execution_queue.json`](../../artifacts/task_79.1_execution_queue.json).
- [x] Execute M14 L01 with the existing ConvVAE/AnalysisPipeline story on a
  deterministic sklearn-digits train/held-out split; promote only
  `THY-T01-METRIC-SPACE-VA-VECTOR-SPACE` to validator-backed D2. See
  [`l01-core.json`](../../artifacts/m14/l01-core.json) and
  [`task_79.3_summary.md`](../../artifacts/task_79.3_summary.md).
- [x] Execute M14 L02 geometry with the committed sklearn-digits held-out lane;
  promote only the four validator-backed D2 records and retain the manifold and
  trajectory-similarity failures. See [`l02-geometry.json`](../../artifacts/m14/l02-geometry.json)
  and [`task_79.4C_summary.md`](../../artifacts/task_79.4C_summary.md).
- [x] Execute M14 L03 on the pinned real GPT-2 forward-only lane; promote only
  the three validator-backed D2 records for linear structure, linear probing,
  and nonlinear probing. Preserve the initial 6-pass/2-fail tuple-return
  failure and the intermediate 7-pass/1-fail indexing-oracle failure; the
  structured hook/output cleanup blocker was then resolved by `16db80f` and
  `9ebecfa`, with exact-SHA strict-CUDA verification completing 8/8. See the
  transformer-hook attempt-1 and final attempt-2 evidence, as well as
  [`l03-analysis.json`](../../artifacts/m14/l03-analysis.json), the
  [`capture audit`](../../artifacts/m14/l03-analysis.attempt4.capture-audit.json),
  and [`task_79_l03_phase_b_summary.md`](../../artifacts/task_79_l03_phase_b_summary.md).
- [x] Resolve the separate native hidden-state index-12/direct-logit-lens
  semantics follow-up: determine whether the final hidden-state position is
  pre- or post-final-LayerNorm for direct lens parity, including the double-
  LayerNorm question. The terminal native state is post-`ln_f`; the private
  lens now skips duplicate normalization and exact offline/network parity
  tests cover all states and capture subsets. Direct PowerShell `ssh.exe`
  verification passed 8/8 on the exact committed SHA with the pinned
  `transformers` extra; attempts and sanitized digests are recorded in
  [`task_79_logit_lens_summary.md`](../../artifacts/task_79_logit_lens_summary.md).
  This remains separate from the resolved structured hook/output blocker and
  does not promote L11.
- [x] Freeze M14 L04 design before implementation: the five record IDs and
  dependency order, explicit task/clean-corrupted prompt pairs and
  content/split/pair digests, exact GPT-2 revision/license, the blocked/D0
  pinned WikiText-2 tuned-lens corpus, seven real use-case executions,
  TransformerLMIntegration boundary (`ModelAdapter` is intentionally N/A),
  direct-vs-tuned lens and interchange-vs-additive semantics, exact formulas
  and strict positive thresholds/controls, artifact/run/failure schemas, and
  the direct PowerShell `ssh.exe` CUDA workflow are recorded in
  [`l04-explanations.plan.json`](../../artifacts/m14/l04-explanations.plan.json)
  and [`task_79_l04_1_summary.md`](../../artifacts/task_79_l04_1_summary.md).
  This planning task changes no source/tests and does not promote any L04 row.
- [x] Add the side-effect-free M14 L04.2 contract checker for the frozen plan
  and authored JSONL fixture. It recomputes plan/content/split/pair digests,
  validates the five-record/seven-use-case order and fixture pair/split/label
  invariants, and checks the declared thresholds, resources, and remote
  protocol without resolving a model or tokenizer. See
  [`task_79_l04_2_summary.md`](../../artifacts/task_79_l04_2_summary.md).
- [x] Add the L04.3 fail-closed dispatch infrastructure: one-use-case partial,
  run, and retained failure envelopes with atomic caller-directory writes,
  seven execution mappings, five frozen ledger records, deterministic source
  digests, and a lazy `TransformerLMIntegration` factory identity seam. Real
  computation remains pending L04.4--L04.10; tuned lens remains blocked until
  the pinned WikiText subset is provisioned. See
  [`task_79_l04_3_summary.md`](../../artifacts/task_79_l04_3_summary.md).
- [x] Implement and execute the L04.4 Integrated Gradients support-only
  handler with the frozen 16/64-step, baseline, randomized-target,
  seeded-repeat, finite and no-mutation controls, independent group bootstrap
  summaries, and strict real-CUDA/network provenance gating. The one
  owner-reviewed exact-SHA CUDA execution is retained as a semantic failure:
  zero-baseline completeness relative error `42.8119096032` (95% CI
  `[30.1207528902, 57.0413396873]`) and batch-mean error `0.0058147719` (95% CI
  `[0.0005752487, 0.0155157174]`) exceeded the frozen `<= 0.001` gate, while
  step stability, randomized-target, seeded-repeat, and finite/no-mutation
  controls passed. The result remains D0/evidence-ineligible with empty
  accepted IDs; no promotion or coverage-count change occurred. See
  [`task_79_l04_4_summary.md`](../../artifacts/task_79_l04_4_summary.md).
- [x] Implement and close L04.5 TCAV with train-group-only concept fitting,
  held-out group/pair metrics, exact five-seed/2,000-bootstrap summaries,
  seeded null controls, genuine hidden-state interventions, and strict
  artifact/run/failure validation. The owner-reviewed attempt-3 recovery on
  the exact source SHA produced a semantic failed/D0/non-eligible result:
  accuracy `0.875` passed, Wilson lower `0.5291118178` failed `> 0.55`,
  bootstrap lower `1.0` passed, corrected empirical p `0.24` failed `<= 0.05`,
  intervention agreement `1.0` passed, and all five controls passed. Attempts
  1 and 2 remain auditable transport/capture failures; no rerun, D3 promotion,
  or coverage-count change occurred. See
  [`task_79_l04_5_summary.md`](../../artifacts/task_79_l04_5_summary.md).
- [x] Implement and close L04.6 Direct Logit Lens Phase A around the concrete
  `TransformerLMIntegration` boundary. The owner-reviewed exact-SHA remote
  execution (transport attempt 2, semantic ordinal 1, envelope attempt 1)
  captured all 13 native hidden states and passed terminal absolute/relative
  logit parity (`0.0 <= 1e-6`), held-out selectivity (`0.0009173364`, 95% CI
  `[0.0005931575, 0.0012898929]`), and all four controls on an RTX 4060 Ti in
  `33.24082692805678` s; remote disposable clone/cache cleanup passed. Direct
  lens is support-only and remains D0 with empty accepted IDs, so no ledger
  promotion or coverage-count change occurred. Attempt 1 remains an auditable
  preflight setup failure caused by raw-file-vs-canonical-plan-digest
  conflation; its exact 60/131-byte captures were hash-verified, deleted, and
  their absence recorded. See
  [`task_79_l04_6_summary.md`](../../artifacts/task_79_l04_6_summary.md) and
  the sanitized attempt audits.
- [x] Implement L04.7a as an offline-tested, network-gated WikiText-2
  acquisition and sanitized manifest pipeline. It requests only explicit
  pinned `train` and `validation` splits (never an upstream `test` split) and
  pins the Salesforce
  `wikitext-2-raw-v1` configuration/revision and license, verifies official
  split sizes, drops blank rows, selects the frozen 8192/2048 rows with seed
  79 downstream-training provenance and max-128 metadata (selection itself is
  deterministic hash/index sorting), and independently binds selected indices
  and UTF-8 text hashes through content/split digests. Raw corpus text is
  never written or committed; the frozen L04 plan remains byte-for-byte
  unchanged. The single owner-approved direct PowerShell `ssh.exe` acquisition
  from source SHA `e6b1bf71de46d6b6879ce6c57fef9e939f1d2fcc` produced the
  validator-passing manifest SHA `0908f843efd72ce93c628e34cdb27f56e37e764d196ef91be7eff6d7757b78f3`,
  content digest `bd235bad5a7643c860bca04a98ba545214f25702cd7625dd4ff591f0ea32cf7b`,
  split digest `bb2dab8721bb8e244bf38f9add6af9e5c2fc70291ce4de6cf2263f7e0f970703`,
  and official/non-blank/selected counts train `36718/23767/8192`, validation
  `3760/2461/2048`. Cleanup passed and raw captures were deleted after
  verification. The corpus blocker is provisioned/resolved, while
  `TunedLogitLens` remains D0 until translator implementation and real CUDA
  execution; no ledger promotion or rerun occurred. See
  [`task_79_l04_7a_summary.md`](../../artifacts/task_79_l04_7a_summary.md),
  [`manifest`](../../artifacts/m14/l04-wikitext-2-manifest.json),
  [`audit`](../../artifacts/m14/l04-wikitext-2-manifest.attempt1.audit.json),
  and [`exit record`](../../artifacts/m14/l04-wikitext-2-manifest.attempt1.exit.txt).
- [x] Implement L04.7 tuned-logit-lens Phase A with pinned-manifest runtime
  revalidation, streaming dense affine translators for native states 0..11,
  KL-over-every-non-padding-token training, deterministic AdamW provenance,
  row-level macro holdout improvement gates, five seed-specific bootstrap
  diagnostics, shuffled-target and terminal post-`ln_f` controls, and
    fail-closed artifact validation. Runtime uses bounded indexed scans and one
    model forward per corpus batch; validator recomputes numeric parity,
    permutations, and all pass flags from retained evidence. Offline/fake tests and the existing L04
  regression suite pass; real CUDA execution remains the required next gate.
  The first owner-authorized exact-SHA CUDA run reached the real handler but
  failed honestly at the production fitted-layer invariant
  (`tuned-lens macro metric requires exactly fitted native layers 0..11`).
  Setup and semantic failure evidence are retained. Attempt 2 is a setup D0
  (`ModuleNotFoundError: datasets`) from an isolated environment that omitted
  same-environment provisioning; its cleanup and outer SSH exit are not
  evidenced and are not claimed. The validator now accepts truthful early
  failure provenance but remains fail-closed for any claimed acceptance. The
  aggregation/provenance correction is local and awaits owner review plus a
  new authorization before any rerun. The latest SHA515fe recovery also failed
  before the CLI because PowerShell/native nested quoting stripped the
  preflight's Python quotes and backslashes; `ssh.exe` and wrapper exit were
  both `1`, with no model/dataset/CLI semantic execution or metrics. Its
  sanitized audit and exit record are retained after verified raw-capture
  deletion; cleanup remains unverified. The next run must use the owner-
  approved exact-`uv` override and the LF-normalized temp-`preflight.py`
  protocol in the M14 validation procedure. The latest SHA3273 recovery then
  failed during preflight because an ad-hoc `datasets` overlay selected
  `huggingface-hub==1.26.0`, incompatible with `transformers==4.57.6`; its
  CLI/model/dataset path was not reached or planned, one exact cleanup PASS was
  emitted, and SSH/wrapper exits were both `1`. The sanitized audit and exit
  remain D0 fixtures after verified raw deletion. L04.7 remains D0 and
  unpromoted.
- [x] Close the L04.7 post-real-run correction locally: production artifact
  assembly now preserves execution-level fit `seed=79` alongside the five
  bootstrap seeds, the shuffled-target fit/evaluation mask is explicitly
  `source_attention_mask & permuted_target_attention_mask`, and the validator
  binds that policy in provenance. A realistic successful-run regression now
  sends the actual `run_tuned_logit_lens()` payload through `build_artifact()`
  and the full validator; a variable-length regression proves padded shuffled
  target positions cannot affect fit/evaluation metrics or determinism. The
  latest retained SHA `2a6de8d` real CUDA computation recorded tuned holdout
  improvement `6.5803880806` nats, conservative lower bound `6.5399008976`
  nats, all declared controls passing, and resource budget passing on an RTX
  4060 Ti (`1180.8877603` s, `2167476736` allocated bytes,
  `2166382592` RSS bytes), but remains D0/evidence-ineligible because the
  execution artifact omitted the singular fit seed. Its cleanup marker passed;
  the outer SSH transport exited `2` after the wrapper's CRLF-sensitive numeric
  status error. The attempt3 artifact/run/failure and sanitized recovery audit
  remain byte-preserved historical payload evidence; the sanitized recovery
  audit has only its scoped failure SHA metadata corrected. The final
  owner-authorized corrected run at SHA `278a9f7` now passes the artifact,
  run-record, and failure validators as accepted D3 with seed 79 and the
  common shuffled-target mask; no retry occurred. It recorded
  `THY-T05-LOGIT-LENS-TUNED-LENS`, layer `6`, native hidden-state index `7`,
  holdout improvement `6.5803880806` nats, conservative lower bound
  `6.5399008976` nats, all controls PASS, and RTX 4060 Ti resource-budget PASS
  (`1116.8708012` s, `2065599488` allocated CUDA bytes, `2161827840` RSS
  bytes). Its sanitized audit records `L04_REMOTE_SCRIPT_START=PASS`,
  `L04_STATUS=0`, `L04_CLEANUP=PASS`, outer SSH exit `0`, and all four evidence
  validators with zero errors. The raw capture also retained
  `base64: invalid input`; decoded script bytes were not independently
  hash-verified, so semantic D3 acceptance does not claim stronger transport
  provenance. The direct base64-to-bash recipe is **NOT REUSABLE** for later
  lanes until L04.8 implements and tests remote-temp-file decode, decoder exit
  `0`, decoded-SHA comparison, execution, and cleanup. L04.7 is complete.
- [x] Implement L04.8 Phase A locally/offline around a private SRP CPU
  logistic/Brier probe, causal-group aggregation, deterministic derangement and
  factor-permutation controls, target-token-excluded raw-token baseline, and
  fail-closed artifact validation. The fake runner-to-artifact validator,
  imbalance/degenerate-bootstrap, slot-reversal, leakage, and tamper tests
  pass. Real CUDA execution, resource evidence, and the replacement remote
  temp-file transport remained the next owner-gated step at this Phase A
  checkpoint; no promotion was claimed in that earlier state. The final review pass also binds process-peak RSS provenance,
  reserved CUDA memory, canonical model digests, truthful D0 failure stages,
  factor-permutation supervision, and raw-token excluded-column digests. See
  [`task_79_l04_8_summary.md`](../../artifacts/task_79_l04_8_summary.md).
- [x] Implement the reusable L04 remote transport prerequisite locally/offline:
  [`m14_l04_remote_transport.ps1`](../../scripts/m14_l04_remote_transport.ps1)
  performs exact-byte LF/UTF-8 normalization, decoded-byte SHA verification,
  direct native `ssh.exe` capture, and sanitized build-only manifests;
  [`m14_l04_remote_payload.sh`](../../scripts/m14_l04_remote_payload.sh) owns
  the disposable detached clone, isolated caches, same-environment preflight,
  one canonical-use-case CLI invocation, bundle-before-cleanup, and full
  cleanup. Offline PowerShell/static tests pass; no remote execution or D3
  promotion is claimed until the owner runs the committed helper from
  authenticated Windows PowerShell. The helper now uses one monotonic
  transport deadline (default 3600 seconds, range 2400–7200) across setup,
  semantic execution, bundle/cleanup, and capture; timeout termination kills
  the process tree with a bounded 30-second grace. The payload emits a
  sanitized `L04_WORKDIR` marker immediately after `mktemp`. The first single
  Phase B Disentanglement attempt timed out during dependency setup before the
  CLI/bundle and remains D0 with cleanup unknown; no retry or cleanup-only SSH
  is allowed. Native OpenSSH now runs with batch mode, one connection attempt,
  and a validated 15-second connect-timeout default; active runbook examples
  derive the clone URL from the configured Git origin. The payload keeps stdout
  marker/Base64-only by routing NVIDIA-SMI and CLI diagnostics to stderr; the
  frozen M14 plan is unchanged.
- [x] Close the L04.8 recovery defects from the preserved `ce4e66e` audit:
  the shared authored-fixture reader now carries the exact `condition` into
  Disentanglement evidence and fails closed for missing/invalid values; Linux
  `ru_maxrss` is normalized to bytes and retained with `rss_unit=bytes`; and
  the remote payload uses the canonical NUL-safe tar order. The payload emits
  `L04_CLI_STATUS` immediately after the single CLI, captures tar under an
  explicit non-errexit boundary, emits `L04_BUNDLE_STATUS`, bundles exactly
  the three current-attempt artifacts even after a semantic CLI failure, and
  preserves the CLI-first/bundle-second final exit policy. Focused offline
  regressions cover the actual reader, production holdout path, RSS
  normalization/mislabelling, tar contract, and status separation. The
  `ce4e66e` audit remains byte-for-byte preserved: D0 only, with the actual
  `KeyError('condition')`, remote CLI-only validators, absent bundle, missing
  unverified raw capture deletion, cleanup PASS markers, and no promotion.
  No remote rerun or commit is authorized by this executor; see
  [`task_79_l04_8_recovery_fix_summary.md`](../../artifacts/task_79_l04_8_recovery_fix_summary.md).
- [x] Execute the single owner-authorized SHA9b36068 Disentanglement Phase B
  run through direct authenticated PowerShell `ssh.exe`. The fresh detached
  CUDA clone loaded the pinned GPT-2 through the real
  `TransformerLMIntegration`, emitted an exact three-artifact bundle, and
  passed remote cleanup/resource/model-mutation controls. The artifact's D2
  point estimates and strict gates passed for all frozen seeds, but the local
  validator initially exposed a one-to-two ULP floating-point-order mismatch
  (seed29=1; seeds17/41/67=2); the shared
  macro/gain helper now makes the retained artifact validator-clean. The run's
  historical CLI status remains `1` and must not be retroactively promoted;
  a new real execution is required for current evidence after this compatibility
 correction. See the SHA9b36068 sanitized audit.
- [x] Harden L04.8+ evidence retention with the local
  [`m14_l04_remote_postprocess.py`](../../scripts/m14_l04_remote_postprocess.py)
  boundary. The capture parser rejects duplicate/missing/inconsistent markers;
  the postprocessor verifies announced bundle/member hashes, safely inspects
  the exact three regular JSON members, runs the existing envelope validators,
  atomically retains and reopens final payloads, writes a sanitized pending
  audit, and leaves raw evidence for the separate `--finalize-delete` command,
  which reparses/rebuilds the pending audit, atomically quarantines raw,
  publishes `quarantined_pending_delete`, then deletes only after all gates
  pass. The pending audit mode is exactly
  `retained_pending_finalize` and is checked before raw movement. A
  quarantine-audit failure reverses the rename; after quarantine deletion, a
  final-audit failure restores the in-memory raw snapshot and exact pending
  audit for retry. Snapshot double failure publishes `raw_restore_failed`
  without claiming pending or deleted success.
  `--validate-only` and `--dry-run` preserve raw evidence without writes.
  Synthetic adversarial tar/marker,
  collision, rollback, idempotence, validator, audit, and reopen tests pass.
  The d9 audit remains immutable observed-but-non-closeable D2 evidence; its
  sanitized sidecar records remote semantic eligibility separately from
  `repository_promotion=false`. No historical payload reconstruction or
  promotion is allowed.
- [x] Close the current L04.8 Disentanglement evidence on exact pushed SHA
  `4d3a4b6551d6091ce96c73a704e642867c2f2580`. One direct authenticated
  PowerShell `ssh.exe` run reached the real CUDA handler, passed all frozen
  Disentanglement controls and the held-out gain gate, and produced an
  eligible D2 record. The exact triplet and sanitized audit are tracked; the
  archive/member hashes and validator reopen are PASS, and finalization records
  `deleted_verified` for the raw capture. Repository promotion is limited to
  this SHA. The prior timeout and strict-retention-failure raws remain
  untracked with sanitized `repository_promotion=false` sidecars; historical
  audits remain unchanged.
- [x] Implement L04.9 Phase A true activation patching around the concrete
  `TransformerLMIntegration`/pinned GPT-2 boundary. The private handler now
  performs clean-to-corrupted hidden-state interchange with adjacent
  layer/token, deterministic shuffled-donor, strength-grid, zero-strength,
  no-mutation, resource, and fail-closed validator controls. One later
  owner-authorized v1 real-CUDA attempt reached the pinned model but failed the
fixed true-interchange recovery gate; its single retention attempt also
failed closed on a producer failure-stage mismatch. The raw capture remains
byte-exact in the recorded pre-delete hash, then was deleted under an explicit
owner exception after size/hash verification; its sanitized non-promoting D0
sidecar records the deletion. No D3 evidence was promoted and no retry
occurred. See
  [`task_79_l04_9_summary.md`](../../artifacts/task_79_l04_9_summary.md).
- [x] Preregister L04.9 v2 offline Phase A after the failed v1 attempt. The
  immutable addendum references the frozen plan, permanently deny-lists the
  exposed v1 `g09`--`g12` holdout, changes recovery to a directional metric,
  and commits only the withheld holdout plaintext hash, 256-bit seed
  commitment, authoring digest, and power-simulation digest. The public
  train-only fixture has 36 groups; the computationally withheld holdout has
  24 groups and remains outside the repository. Six-fold train-only candidate
  selection, independent validation, synthetic Stage B, and separate v2
  single-artifact transport/retention are implemented offline. Offline Stage
  A output is an ephemeral D0 protocol fixture only; the real Stage A artifact
  is created by an authorized CUDA run after review/commit. Structured runtime
  attestations are independently recomputed at the artifact boundary; they are
  self-attestation only, and final D3 promotion additionally requires the
  transport/raw-retention audit. Stage B D3 construction is separate from the
  D2 artifact and reopens retained member envelopes; deleted raw bytes are
  accepted only through a verified pending/final audit chain. Synthetic Stage B
  remains D0/non-promoting. No real holdout evaluation or D3 claim has occurred.
- [x] Correct the v2 real-runtime shape and failure-envelope blockers found by
  the owner-reviewed Stage A diagnostic. Full-prompt output shapes now fail
  closed, clean/corrupted patch endpoints are independently aligned, and an
  incomplete real attempt emits a validator-clean D0 triad with partial
  counters. No CUDA rerun, holdout access, retention finalization, or evidence
  promotion occurred.
- [x] Separate real semantic-gate D0 outcomes from runtime-exception D0
  outcomes. Frozen `failure_kind` and `selection_complete` bindings now let
  complete candidate/no-consensus selections validate independently while
  runtime failures remain empty-selection, non-promoting triads; D1 remains
  strict.
- [x] Harden real Stage A/Stage B resource provenance. The runtime now tracks
  elapsed time from model load through cleanup, CUDA allocated/reserved peaks
  after synchronization, and Linux process RSS with explicit source/unit and
  unavailable reason fields. Independent validation rejects measured-source
  zero/negative or source/value mismatches and requires nonzero measurements
  for D1/D2. The retained SHA-66455 evidence remains a genuine semantic D0,
  but its historical zero peaks are recorded in a separate sanitized pending
  assessment sidecar; no evidence was promoted.
- [x] Close the historical SHA-66455 resource-invalid evidence under the
  explicit owner exception. The raw capture, retained triplet, and pending
  audit were each matched to their recorded pre-delete size/SHA-256 and
  deleted by literal path. The assessment sidecar now chains the prior
  digest, records `historical_resource_provenance_invalid_measured_zero_peaks`,
  preserves sanitized transport/semantic D0 metadata, and attests all five
  paths absent with `standard_finalize=false`; no promotion occurred.
- [x] Close the v2 failed-attempt evidence state locally under the explicit
  owner exception. The 6008-byte raw capture was verified at SHA-256
  `757af5cce5b4e8aa4c5b476ecc52d69ae192423179c23b3fc148510a8eafc212` and
  deleted only by literal path; the canonical sanitized Stage A sidecar now
  records `deleted_by_owner_exception`,
  `no_triad_bundle_status_66/semantic_gate_d0_validator_misclassification`,
  the prior sidecar digest, verified absence, `standard_finalize=false`, exact
  transport markers/hashes, and no triad/audit/selection or promotion. No raw
  parsing payload, sensitive body, retry, or holdout access occurred.
- [x] Correct v2 causal scoring at the model boundary. The private runtime now
  captures exact raw GPT-2 block outputs `h.0`--`h.11` (with `h.11` before
  `ln_f`) while the public native terminal hidden index remains post-`ln_f`.
  Stage A uses independently resolved clean-source/corrupt-recipient positions
  and pair-level signed directional recovery from serialized primitive clean,
  corrupted, and patched margins; validation recomputes the primitive metric.
  Tuple/list outputs, terminal-normalization parity, zero-strength identity,
  variable endpoint lengths, and hook cleanup are covered. Historical D0
  assessment sidecars and evidence bytes remain unchanged.
- [x] Harden the real post-runtime validation boundary for L04.9 v2 Stage A and
  Stage B. Validator rejection now yields an independently validated,
  non-promoting D0 triad with an allowlisted rejection code and normalized
  unavailable-resource provenance; successful semantic D0 and eligible D1/D2
  paths remain strict. The 5791-byte failed attempt was deleted under the
  explicit owner exception recorded in its sanitized assessment sidecar;
  standard finalization and promotion remain false.
- [x] Make the reusable L04 remote transport executable under both native
  Windows PowerShell 5.1/.NET Framework 4.8 and pwsh 7/.NET. Hashing now uses
  the cross-version SHA-256 API; the process seam detects `ArgumentList` and
  falls back to Windows-safe quoting, uses compatible stream/process cleanup,
  and atomically replaces an existing raw target through a unique backup.
  Native build-only and fake-process regressions cover non-zero exits, raw
  capture before parsing, existing-target replacement, and paths containing
  spaces/quotes. No network, SSH, CUDA, or holdout execution occurred.
- [x] Remediate the L049 v2 incident boundary without rerunning remote work.
  Stage A now preserves sanitized live operation counters through finalizer
  rejection, emits an allowlisted finalizer field-shape category, and keeps
  unavailable resource provenance explicit. The transport canonicalizes the
  case-insensitive UseCase and holds a true host-wide `Global\\` operating-
  system mutex keyed by canonical stage and exact lowercase source SHA from
  pre-launch through raw postprocessing; ACL-capable runtimes receive only the
  current-user/LocalSystem Synchronize+Modify rights, while modern runtimes
  explicitly clear session/user-only options. pwsh 7 and native Windows
  PowerShell 5.1 canonicalization and concurrent-launch regressions prove that
  only one child can start. Historical retention assertions bind to exact
  recorded evidence paths. The current one-completed/one-aborted-launch
  incident was initially recorded as a pending D0 assessment with promotion
  and finalization false; its raw, audit, bundle, and triad hashes remain bound
  in the sanitized incident assessment sidecar. The later owner-exception
  cleanup is recorded in the following item.
- [x] Remediate the current b295a506 evidence boundary locally. CUDA peak
  publication is now atomic across allocated/reserved queries and failed
  finalizer normalization trusts one complete live counter snapshot. The
  raw, bundle, and triad bytes/digests are unchanged. The audit was canonically
  rebuilt from 2638 bytes/SHA `c9fdfcfa5f60a010c403e143062ee6f7a9820f588308b7f13912240093456fd4`
  to 3276 bytes/SHA `914d374748e803513d3aeba04c556fa17584ebdabd3efdad4dcde6bf26e43a91`
  to distinguish generic archive member names from source-unique local paths.
  The sanitized assessment is now `deleted_by_owner_exception` D0 with one
  SSH/CLI invocation, 2592 discarded evaluations, unknown GPU subfield, and
  promotion/finalization false. Pre-delete hashes, verification, and
  post-delete absence are recorded; no remote retry or holdout access occurred.
- [x] Close the current L049 v2 incident evidence under the explicit owner
  exception. The raw capture, audit, and three triad files were each verified
  at their exact sidecar-recorded size/SHA-256, regular-file/no-reparse and
  untracked status, then deleted by literal absolute path. The canonical
  sidecar preserves all five hashes/sizes, the bundle digest metadata, one
  completed payload, two reported launches with uncertain second reachability,
  D0/unknown counters and sanitized root cause, and pre/post absence proof.
  `standard_finalize=false` and `repository_promotion=false`; this is
  irreversible owner-exception cleanup, not successful finalization or a
  promoted run.
- [x] Consolidate the L049 v2 finalizer acceptance and diagnostic paths into
  one producer-independent checker. It now distinguishes sanitized top-level,
  identity, counter, hook/intervention, cleanup, resource-peak, and
  cross-field categories while enforcing finite/nonnegative/budget and
  reserved-versus-allocated invariants. Production-shaped Stage A closure,
  live-counter, Stage B shared-envelope/idempotence, rejection-matrix, and
  late-failure no-double-finalizer regressions are covered; public validation
  remains independent.
- [x] Correct the first owner-authorized a205ca7 Stage A transport attempt:
  the remote payload now derives and validates the tracked train fixture inside
  its fresh detached clone, while Stage B accepts only sanitized absolute POSIX
  holdout/seed/candidate paths and no longer exports the unused train path.
  Native PowerShell/pwsh build-only and fake-remote regressions cover local
  path non-leakage, quoting, traversal, symlink/fixture checks, and argv/env
  separation. The raw-only a205ca7 D0 assessment was later deleted under an
  explicit owner exception after exact size/hash verification; its sidecar
  records pre-delete and post-delete absence proof. There was no
  audit/triad/bundle, promotion, or retry after the proven remote input-path
  failure, and this cleanup is not finalization.
- [ ] Build clean environments for base, each optional extra, and supported combined extras on every supported Python/platform tier.
- [ ] Run unit/property/integration tests plus strict docs, packaging, security, license, and dependency audits.
- [ ] Execute every applicable row of the 24-lane [M14 real-system matrix](../M14_REAL_SYSTEM_VALIDATION.md), with one artifact per independently verifiable capability.
- [ ] Execute the exhaustive [theory evidence-gap plan](../EVIDENCE_GAP_PLAN.md) and its row-level [machine-readable map](../../artifacts/task_78.38_gap_map.json); keep D0/D1 statuses unchanged until validator-backed D2/D3 artifacts exist, as demonstrated by the L03 promotion.
- [ ] Use the [migration guide](../MIGRATION.md) and [API reference](../API_REFERENCE.md) as the human entry points to the checked-in compatibility snapshot during RC verification.
- [ ] Execute the pinned real-model matrix: Diffusers VAE/conditional diffusion, GPT-2, I-JEPA, VQ/tokenized/world-model paths, ACT, Diffusion Policy, and SmolVLA; record the named 3DGS checkpoint or keep L17 blocked.
- [ ] Execute explanation-validity controls and confirm the theory ledger meets 95% core / 90% overall D2-or-D3 thresholds.
- [ ] Verify all 202 exports, 32 built-in registry entries, 5 entry-point groups, 12 optional profiles, CLI commands, schema migrations, negative/security cases, sync/async paths, cross-adapter composition, external plugin install/discovery, cache, streaming, and tracking backends.
- [ ] Measure performance budgets and LeRobot policy overhead against Sprint 77 gates.
- [ ] Run remote CUDA only through the remote-cuda-test skill invariants and preserve disposable-clone/cache cleanup evidence; do not use remote CUDA as a substitute for missing local tests.
- [ ] Publish an RC evidence report with failures, waivers, confidence intervals, hardware, upstream revisions, and exact reproduction commands.
- [ ] Fix only release blockers, rerun the complete affected matrix, reconcile docs/ledger conflicts, and cut the release candidate only after the external GitHub Actions account is available.

## Notes / Blockers

No percentage waiver may hide an implementation-applicable core theory gap. Any exclusion must have been classified and justified in the Sprint 27 ledger. SAM, OpenCLIP, timm, Torchvision model adapters, Open3D, trimesh, and unnamed 3DGS are not stable API claims; they remain explicit backlog/blocker rows.

Sprint 78.38 records the starting denominator and all 40 gap records; the
updated map now records 33 qualifying rows after L03. Sprint 79 owns execution
in dependency order. The historical L03 tuple-return failures are retained,
but the structured hook/output cleanup blocker is resolved by `16db80f` and
`9ebecfa` and the exact-SHA strict-CUDA 8/8 evidence. The native index-12
direct-logit-lens question is resolved as an internal semantic correction with
no public protocol/schema expansion; named 3DGS, checkpoint, and corrected
SmolVLA gaps remain explicit blockers, not evidence promotions.

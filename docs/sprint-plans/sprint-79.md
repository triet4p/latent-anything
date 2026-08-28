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
- [x] Implement the offline L04.6 Direct Logit Lens Phase A handler around the
  concrete `TransformerLMIntegration` boundary. It captures all 13 native
  hidden states, verifies terminal post-`ln_f` parity, records held-out
  target/non-target diagnostics with seeded null controls, and emits strict
  partial/run/failure envelopes. The handler is support-only and remains D0;
  no real CUDA execution or evidence promotion was performed. See
  [`task_79_l04_6_summary.md`](../../artifacts/task_79_l04_6_summary.md).
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

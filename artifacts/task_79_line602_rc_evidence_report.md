# Sprint 79 release-candidate evidence report

## Release-candidate scope and decision

This is the authoritative English evidence report for Sprint 79 line 602. It is
an evidence review for package version `0.1.0b1`, not a tag, package
publication, or stable-release approval. The review branch is
`sprint79-local-gate-remediation`. The latest fully tested source scope is
commit `88b836f799680e0cf5dfa8eb336c751c234d4452` (the line-601 evidence
commit); this report is published in a subsequent evidence-only commit. The
report commit SHA is the exact GitHub `HEAD` reported with delivery, and must
not be confused with any historical execution SHA below.

**Decision: RC evidence publication is complete, but the release is not
approved.** The release gates are candidly below threshold, several M14 rows
are partial or blocked, and lines 595, 598, and 600 remain unchecked. No waiver
changes a threshold or hides a core gap. Line 603 must not begin until this
report has been reviewed and its owners authorize release-blocker remediation.

## Evidence authority and provenance

The authoritative sources are:

- [Sprint 79 plan](../docs/sprint-plans/sprint-79.md), especially lines
  594–603.
- [M14 real-system validation contract](../docs/M14_REAL_SYSTEM_VALIDATION.md)
  and its 24-row matrix.
- [Evidence ledger](../docs/evidence-ledger.json), validated by
  `scripts/validate_evidence_ledger.py`.
- [Evidence gap plan](../docs/EVIDENCE_GAP_PLAN.md) and machine-readable
  [`task_78.38_gap_map.json`](task_78.38_gap_map.json).
- Reviewed task reports for lines 596–601 and the immutable row receipts under
  [`artifacts/m14/`](m14/).

Evidence is additive and source-bound. A historical artifact is accepted only
at the exact source SHA recorded by its receipt; a later report does not
rewrite it or silently remeasure it. D0 is documentation, D1 is implementation
plus focused tests, D2 adds a quantitative non-trivial benchmark, and D3 adds
reproducible real-model evidence. A passing focused test alone is not a D2/D3
promotion.

## Gate inventory

| Sprint 79 line | Status | Evidence and remaining condition |
|---|---|---|
| 594 M14 matrix execution/reconciliation | **CHECKED** | 13/24 accepted, 2/24 partial (L02/L04), 1/24 pending (L06), 8/24 externally/prerequisite blocked; see [`task_sprint79_m14_real_system_matrix_summary.md`](task_sprint79_m14_real_system_matrix_summary.md). |
| 595 exhaustive theory evidence-gap plan | **UNCHECKED** | All reachable rows were reconciled, but the plan is not exhausted and D0/D1 statuses remain. |
| 596 compatibility snapshot entry paths | **CHECKED** | 205 current / 202 canonical exports, 18 aliases, 32 built-ins, 5 plugin groups, 12 profiles; snapshot SHA `48d64721b73a9d0c9e73da4a41940008c70dfa7841e500bc11bc8dcd22ddf7f6`. |
| 597 pinned real-model matrix | **CHECKED** | Accepted immutable rows and explicit checkpoint/license/platform/CUDA blockers reconciled; no blocked row was promoted. |
| 598 explanation-validity controls and coverage | **UNCHECKED** | Controls and failures are recorded, but coverage is 41/63 core and 41/65 overall, below 95%/90%. |
| 599 contract/export/registry/profile verification | **CHECKED** | 202 canonical exports and all named contract categories verified; 39/39 clean profile lanes passed. |
| 600 performance and LeRobot overhead | **UNCHECKED** | 9/10 advisory budgets passed; bounded streaming is marginal and real-policy CUDA timing remains blocked. |
| 601 remote CUDA process/evidence | **CHECKED** | Exact-SHA disposable-clone runs, isolated caches, CUDA probes, focused results, and independent cleanup audits are recorded in [`task_79_line601_remote_cuda_summary.md`](task_79_line601_remote_cuda_summary.md). |
| 602 RC evidence report | **CHECKED by this publication** | This report aggregates reviewed evidence without changing outcomes or thresholds. |
| 603 fix blockers / cut candidate | **UNCHECKED and not started** | Requires explicit release-blocker plan, affected-matrix reruns, external Actions access, and review of this report. |

## Coverage gate and confidence/statistical evidence

The live validator returned `errors: []`, core `[41, 63,
0.6507936507936508]`, and overall `[41, 65, 0.6307692307692307]`.
The exact gates are `ceil(0.95 × 63) = 60` core qualifiers and
`ceil(0.90 × 65) = 59` overall qualifiers. The shortfalls are therefore **19
core** and **18 overall**. These percentages are descriptive evidence, not a
waiver. The denominator is unchanged; contextual-background rows remain
excluded only under the ledger's classification rules.

Reviewed quantitative results include:

- L03 real GPT-2 analysis: full-hidden/linear-probe accuracy `0.9275766016713092`,
  paired-bootstrap lower bound `0.7966573816155988`, Wilson 95% interval
  `[0.8959994833250284, 0.9501000803737016]`; PCA32 accuracy
  `0.883008356545961` with bootstrap lower `0.7381615598885793`; MLP accuracy
  `0.8857938718662952`, shuffled control `0.11420612813370473`,
  `n_params=698`. These are accepted D2 records for the named L03 topics.
- L04 TCAV: held-out accuracy `0.875`, exact 95% interval `[0.625, 1.0]`;
  Wilson lower `0.5291118177871466` fails the strict `>0.55` gate and corrected
  empirical `p=0.24` fails `<=0.05`. The control set passes, but semantic
  promotion remains blocked.
- L04 TunedLogitLens accepted D3: improvement `6.5803880806` nats,
  conservative lower bound `6.5399008976`, fit seed `79`, bootstrap seeds
  `[17,29,41,53,67]`; all required controls and resource caps passed at its
  exact historical SHA.
- L06 SAE: `n_tokens=10752`, reconstruction MSE `0.7104635182257194`,
  mean/min cross-seed cosine `0.8467253367037664/0.7387987235614891`;
  minimum cosine fails the predeclared `>0.85` gate while alignment `0.75`
  passes `>0.7`. L06 remains pending, not promoted.
- L02 manifold: latest held-out ranking AUC `0.4560546875` and latent-vs-raw
  delta `-0.4124755859375` fail the predeclared gates `>=0.55` and `>=-0.05`.
  Its separate DTW rerun passed, so L02 remains partial (5/6 records).
- L04 additive steering: target effect `0.05915`, selectivity `0.05432`,
  off-target `0.00483` pass their gates, but randomized-direction controls
  fail for seeds 29 (`0.32630`) and 67 (`0.10097`); D3 remains blocked.
- Line 600's repeated bounded-streaming p95 values were `3023.1`, `2456.6`,
  and `1960.9` microseconds against a 3000-microsecond advisory budget.
  Median passed in all runs; no sample or workload was removed.

## M14 row inventory

The 24 applicable authority rows reconcile as follows. “Accepted” means the
row's reviewed evidence contract, not merely a green unit test.

| Row | Outcome | Reviewed scope / blocker / owner action |
|---|---|---|
| L01 | **Accepted D2** | sklearn-digits ConvVAE/Pipeline held-out baseline; artifact `l01-core.json`; generative owner maintains bounded CPU scope. |
| L02 | **Partial D2, 5/6** | DTW accepted; manifold ranking and latent-vs-raw gates fail. Geometry owner must produce a new predeclared passing run. |
| L03 | **Accepted D2** | Real GPT-2 hidden-state analysis, three accepted records, exact historical source `bb0da6fdc4fb00950ce4cb574ec83e8a9344db8b`; current exact-HEAD CUDA integration also passed 8/8. |
| L04 | **Partial** | TunedLogitLens D3 and Disentanglement D2 accepted; IntegratedGradients/TCAV support-only or semantic failures and additive controls remain. Owner-authorized exact-SHA reruns are required before promotion. |
| L05 | **Accepted D2** | Real GPT-2 density/geodesic; mean AUROC `0.9750000000000001` and AUPRC `0.975609756097561`; geometry owner retains thresholds. |
| L06 | **Pending** | Expanded GPT-2 cross-seed capture completed, minimum-cosine gate failed; introspection owner needs a valid rerun. |
| L07 | **Accepted D2** | GPT-2/VAE intervention controls; steering effect `0.5000000000002008`, VAE patch effect `20.72073172096096`; no-mutation and reversibility passed. |
| L08 | **Accepted D2** | Held-out ConvVAE; reconstruction MSE `0.17170413275226673`, zero-baseline improvement `0.27197953282807774`; bounded CPU owner scope. |
| L09 | **Accepted D2** | `stabilityai/sd-vae-ft-mse@31f26fdeee1355a5c34592e401dd41e45d25a493`, safetensors SHA `a1d993488569e928462932c8c38a0760b874d166399b14414135bd9c42df5815`, MIT; local CPU cache-only. |
| L10 | **Blocked** | `runwayml/stable-diffusion-v1-5@39593d56b552c3a24aeb192dd11d2a1429c3102b`; high VRAM/download plus license-card/access and owner CUDA prerequisites absent. Do not substitute on 16 GB. |
| L11 | **Accepted D2 candidate** | `openai-community/gpt2@e7da7f221d5bf496a48136c0cd264e630fe9fcc8`, MIT; vocab `50257`, 13 hidden layers, hidden size `768`, parity/intervention/cleanup passed. |
| L12 | **Blocked** | `facebook/ijepa_vith14_1k@be440b1cac639542ae553e71a9c7afd925ab5fac`; high download/RAM and model-card license/access/checkpoint provisioning absent. |
| L13 | **Accepted D2** | Compact VQ-VAE/sklearn digits, seed `42`, BSD-3-Clause; perplexity `13.090496630645841`, dead-code rate `0.0`. |
| L14 | **Accepted bounded D2** | Compact VQVAE plus synthetic dynamics; horizon `8`, teacher-forced perplexity `2.684111787469027`; no named GAIA/Genie claim. |
| L15 | **Accepted D2** | Deterministic/Gaussian/RSSM compact transitions; seeded reproducibility and state carry pass; real temporal checkpoint remains a gap. |
| L16 | **Accepted D2** | CEM return `2.835069606822939`, MPPI return `2.2274987449276558`, baseline `1.4485021636139839`, 576-evaluation budget; policy/search extensions remain gaps. |
| L17 | **Blocked** | No trustworthy named 3DGS checkpoint, revision/hash, license, or access metadata. No unnamed substitution. |
| L18 | **Blocked** | `lerobot/aloha_sim_insertion_human@cc571a3c661df81b566dbfde3d5c1e85fcdf7884`; Linux LeRobot-capable dataset/license/schema capture is unavailable. |
| L19 | **Blocked / explicit non-goal** | Canonical OpenVLA BF16 checkpoint `openvla/openvla-7b-finetuned-libero-spatial@962318cec55ac10993ff0f5f43eda9a270b4c873` has `7,541,237,184` parameters and `15,082,474,368` BF16 bytes (~14.05 GiB), infeasible on the authorized RTX 4060 Ti 16 GB with runtime headroom; do not retry. |
| L20 | **Blocked; exact-HEAD smoke SKIP** | `LeTau/diffusion_aloha_insertion@6126e33` plus dataset `lerobot/aloha_sim_insertion_human_image@d93d36a`; remote CUDA preflight passed but the pinned checkpoint test skipped, with upstream access/license capture unresolved. |
| L21 | **Blocked; exact-HEAD smoke SKIP** | `lerobot/smolvla_libero@31d453f7edd78c839a8bbc39744a292686daf0de` plus `lerobot/libero@a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4`; remote CUDA preflight passed but pinned intervention skipped; model/data/license capture and corrected causal D3 evidence remain outstanding. |
| L22 | **Accepted D2** | Real isolated filesystem/runtime contract: portable envelopes, ArtifactStore, SQLite cache, rollout/stream/async, recorder and resource bounds. |
| L23 | **Accepted D1/contract** | 32 built-ins, 5 groups, plugin installation/discovery, CLI/config/serialization and 39/39 clean profile lanes. |
| L24 | **Blocked** | External GitHub Actions account/release workflow and unresolved evidence thresholds remain unavailable; no publication/tag operation. |

Totals: **13 accepted, 2 partial, 1 pending, 8 blocked**. All 24 rows are
implementation-applicable; none was silently omitted. Skips remain skips and
failures remain failures.

## Pass, failure, waiver, and exclusion accounting

### Passed and accepted

The accepted evidence set is the immutable row artifacts and reviewed summaries
for L01, L03, L05, L07–L09, L11, L13–L16, L22, and L23, plus the accepted
records within L02 and L04. Line 599's contract matrix passed, line 596's API
snapshot passed, and line 601's exact-HEAD TransformerLM CUDA integration
passed. These claims are bounded by each row's stated model, data, device,
seed, threshold, and artifact digest.

### Failures and partial results retained

The report intentionally retains the L02 manifold failure, L04 semantic/control
failures, L06 minimum-cosine failure, line-600 streaming variance, the early
L14 rollout failure, and all historical transport/setup failures. The L04
IntegratedGradients digest repair is a code/remediation success, but it does
not promote the support-only semantic record. No failure was converted to a
warning or removed from a denominator.

### Explicit blockers and exclusions (not percentage waivers)

There are **no signed release waivers** in this report. The following are
contract exclusions/blockers with owners and minimal actions, not denominator
manipulations:

1. **L10:** obtain license-card/access authorization and a GPU with declared
   headroom for the pinned SD 1.5 lane; owner: conditional-diffusion owner.
2. **L12:** provision the pinned I-JEPA checkpoint and model-card license/access;
   owner: world-model owner.
3. **L17:** obtain a named, revision-pinned, licensed 3DGS checkpoint and remote
   lane; owner: renderer owner.
4. **L18:** obtain Linux LeRobot dataset access, schema/license capture, and a
   LeRobot-capable environment; owner: dataset owner.
5. **L19/OpenVLA:** move canonical BF16 execution to an authorized >=24 GiB
   CUDA host, then implement the adapter and 500-trial LIBERO-Spatial contract;
   owner: VLA owner. The 16 GB host is not a valid waiver target.
6. **L20:** obtain upstream model/data access and license capture, then rerun
   the pinned pair through the remote Linux/CUDA workflow; owner: Diffusion
   Policy integration owner.
7. **L21:** obtain model/data/license authorization and run the corrected pinned
   simulator/intervention protocol; owner: SmolVLA/LeRobot owner.
8. **L24 and release threshold:** obtain the external GitHub Actions account and
   close the independent 95% core / 90% overall gate; owner: release owner.
9. **Theory gaps:** implement or provision the exact named rows in
   `docs/EVIDENCE_GAP_PLAN.md`; owner assignment is recorded in that plan.

No owner waiver has scope, rationale, or expiry that authorizes a stable-release
claim. No percentage waiver hides an applicable core gap.

## Hardware, software, and upstream provenance

### Local Windows evidence

The local supplemental gate ran on Windows `win32`, AMD64, Python `3.13.3`,
8 CPU threads, Torch `2.10.0`, NumPy `2.4.6`, PyArrow `24.0.0`, and no local
CUDA (`torch.cuda.is_available() == False`). Local clean-environment coverage
used Python `3.12.12`, `3.13.3`, and `3.14.0`: 39/39 base/profile lanes
passed with model/data acquisition disabled. The local release gate recorded
2209 passed, 46 skipped, and 39 warnings, plus strict Ruff, Pyright, MkDocs,
packaging, dependency, security, and license gates passing. These local results
remain separate from remote CUDA evidence.

### Remote Linux evidence

The authorized host is `trietlm@192.168.30.244` (`di-server`), Linux kernel
`6.8.0-107-generic`, x86_64, NVIDIA GeForce RTX 4060 Ti with `16,380 MiB`,
driver `580.126.20`. Remote line-601 runs used CPython `3.13.12`, Torch
`2.10.0+cu128` / CUDA runtime `12.8`, Transformers `4.57.6`, LeRobot `0.6.1`
where requested, `gcc-12`/`g++-12` `12.3.0`, and uv `0.12.6`. `nvidia-smi`
and PyTorch CUDA availability passed. `nvcc` was not installed; no test that
required a CUDA extension compiler was claimed. Remote runs used fresh clones,
`UV_CACHE_DIR`, `TORCH_EXTENSIONS_DIR`, and `CUDA_CACHE_PATH` below each temp
clone, and independent post-run absence checks passed.

### Pinned upstream licenses/access

| Upstream | Revision | License/access status |
|---|---|---|
| `openai-community/gpt2` | `e7da7f221d5bf496a48136c0cd264e630fe9fcc8` | MIT; public pinned model, cached only where receipts say so. |
| `stabilityai/sd-vae-ft-mse` | `31f26fdeee1355a5c34592e401dd41e45d25a493` | MIT; safetensors SHA recorded above; local CPU cache-only evidence. |
| `runwayml/stable-diffusion-v1-5` | `39593d56b552c3a24aeb192dd11d2a1429c3102b` | License-card/access and high-VRAM prerequisites unresolved; blocked. |
| `facebook/ijepa_vith14_1k` | `be440b1cac639542ae553e71a9c7afd925ab5fac` | Checkpoint/license/access unresolved; blocked. |
| `lerobot/aloha_sim_insertion_human` | `cc571a3c661df81b566dbfde3d5c1e85fcdf7884` | Dataset/license/schema capture unresolved; blocked. |
| `LeTau/diffusion_aloha_insertion` + image dataset | `6126e33` + `d93d36a` | Pinned IDs, but upstream access/license capture unresolved; smoke skipped. |
| `lerobot/smolvla_libero` + `lerobot/libero` | `31d453f7edd78c839a8bbc39744a292686daf0de` + `a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4` | Pinned IDs, but model/data/license and corrected causal evidence unresolved; smoke skipped. |
| OpenVLA fine-tune/base/RLDS/LIBERO | `962318cec55ac10993ff0f5f43eda9a270b4c873`, `47a0ec7fc4ec123775a391911046cf33cf9ed83f`, `6ce6aaaaabdbe590b1eef5cd29c0d33f14a08551`, `8f1084e3132a39270c3a13ebe37270a43ece2a01` | Canonical BF16 run infeasible on 16 GB; inherited Llama Community License/access and >=24 GiB host remain required. |
| Named 3DGS checkpoint | None supplied | Missing; explicit L17 blocker, no unnamed substitute. |

## Exact reproduction commands

All commands below are copied from reviewed summaries or line-601 runner
transcripts. They are not claims that a blocked command passed.

### Local and ledger gates

```text
uv run python scripts/validate_evidence_ledger.py --json
uv run pytest tests/test_m14_validation_contract.py -q
uv run pytest tests/test_m14_l04_remote_postprocess.py -q
uv run pytest tests/test_api_freeze_snapshot.py tests/test_api_compatibility.py tests/test_api_surface.py -q
uv run pytest tests/test_sprint75_streaming.py tests/test_sprint77_phase_a_benchmark.py tests/test_sprint77_phase_a_compare.py tests/test_sprint77_phase_a_profile.py -q
uv run pytest -v
uv run ruff check src tests scripts
uv run ruff format --check src tests scripts
uv run pyright
uv run --project latent-anything-theory --group dev mkdocs build --strict
uv lock --check
```

### Pinned M14 commands

```text
uv run python scripts/m14_l01_core.py
uv run python -m scripts.m14_l03_analysis --run-real
uv run --locked --extra transformers --with huggingface-hub==0.35.3 python scripts/m14_l05_density.py
LATENT_ANYTHING_RUN_NETWORK=1 uv run --locked --extra transformers --with huggingface-hub==0.35.3 python scripts/m14_l06_sae.py
LATENT_ANYTHING_RUN_NETWORK=1 uv run --locked --extra transformers --with huggingface-hub==0.35.3 python scripts/m14_l07_interventions.py
LATENT_ANYTHING_RUN_NETWORK=1 uv run --locked --extra transformers --with huggingface-hub==0.35.3 pytest tests/test_diffusers_vae.py tests/test_diffusers_vae_network.py -m network -q
LATENT_ANYTHING_RUN_NETWORK=1 uv run --locked --extra transformers --with huggingface-hub==0.35.3 python scripts/m14_l11_gpt2.py
uv run python scripts/m14_l14_tokenized.py
uv run python scripts/m14_l15_transitions.py
uv run python scripts/m14_l16_planning.py
```

### Remote CUDA line-601 workflow

Run from POSIX/Git Bash with authenticated native OpenSSH, after confirming a
clean worktree and pushing the exact source SHA:

```text
bash ./.agents/skills/remote-cuda-test/scripts/remote_cuda_test.sh \
  --remote trietlm@192.168.30.244 \
  --ref sprint79-local-gate-remediation \
  --repo-url https://github.com/triet4p/latent-anything.git \
  --extra transformers \
  --test-command 'LATENT_ANYTHING_RUN_NETWORK=0 uv run pytest tests/test_transformer_lm.py tests/test_m14_l03_analysis.py tests/test_m14_validation_contract.py -q' \
  --gpu-command 'LATENT_ANYTHING_RUN_NETWORK=1 LATENT_ANYTHING_NETWORK_DEVICE=cuda uv run pytest tests/test_transformer_lm_network.py -m network -q'
```

The exact current-HEAD run returned **8 passed, 5 deselected** and **67
broader tests passed**. The related pinned smoke commands were run in fresh
clones with `--extra lerobot-diffusion` (L20: 1 skipped; broader 8 passed/1
skipped) and `--extra lerobot-smolvla` (L21: 1 skipped; broader 11 passed/1
skipped). Their skips are blockers, not passes. See the line-601 report for
clone paths, wall durations, cleanup hashes, and host probe details.

## Cleanup and SHA provenance

The local branch was clean and pushed before each remote invocation. Every
remote invocation verified the clone `HEAD` against source SHA
`bb06db45db1fc82a72fb2d36524a707635e60d5d` before setup/testing. The report
publication itself is evidence-only and does not retroactively change those
execution SHAs. Remote temporary clones and isolated uv/Torch/CUDA caches were
removed by trap; independent `find /tmp -maxdepth 1 -type d -name
"remote-cuda-test.*" -print` audits returned no leftovers for all three runs.
No persistent server checkout was mutated. Local temporary build, package,
audit, and transcript outputs were removed or retained only in the cited
summaries.

The evidence-report commit SHA is the final pushed commit reported at delivery;
the tested source SHA remains `88b836f799680e0cf5dfa8eb336c751c234d4452` for the
line-601 evidence scope. Historical receipt SHAs are intentionally listed next
to their own artifacts and are not conflated with either value.

## Final release disposition

This report is complete and line 602 may be checked. It does **not** approve a
stable release: coverage is 41/63 core and 41/65 overall, line 595 is not
exhausted, line 598's explanation/coverage gate is not met, line 600 has an
advisory marginal lane and an unmeasured blocked real-policy lane, and M14
contains eight blocked rows plus partial/pending evidence. Line 603 must remain
untouched until owners review these blockers, obtain required access/hardware,
produce passing affected-matrix evidence, and confirm the external GitHub
Actions account.

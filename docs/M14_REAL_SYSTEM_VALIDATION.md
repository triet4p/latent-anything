# M14 — Ma trận kiểm chứng hệ thống thực và hợp đồng phát hành

Tài liệu này là hợp đồng lập kế hoạch cho Sprints 78–80. M14 chưa hoàn tất và
không được hiểu là cam kết rằng các lane dưới đây đã có bằng chứng D2/D3. Mỗi
dòng là một đơn vị kiểm chứng độc lập; trạng thái chỉ được nâng từ `planned`
sang `verified` khi có artifact bất biến, revision, lệnh tái lập và người chịu
trách nhiệm. Các tên model, revision, lệnh và tên artifact giữ bằng tiếng Anh
để có thể sao chép nguyên văn.

The exhaustive theory-gap execution plan for Sprint 79 is maintained in
[EVIDENCE_GAP_PLAN](EVIDENCE_GAP_PLAN.md), with one machine-readable row for
each current D0/D1 implementation-applicable or benchmark-only item in
[`task_78.38_gap_map.json`](../artifacts/task_78.38_gap_map.json). This plan
does not change evidence levels or supersede the validator.

The user-facing migration and frozen API references are
[MIGRATION](MIGRATION.md) and [API_REFERENCE](API_REFERENCE.md); their counts
and symbol contracts are projections of the checked-in API snapshot, not a
second source of truth.

## Quy ước và điều kiện dừng

- `D0` = tài liệu; `D1` = code + focused tests; `D2` = benchmark dữ liệu không
  tầm thường; `D3` = D2 trên model pretrained/trained thực. Bằng chứng đồ họa
  hoặc notebook một mình không đạt D2.
- Mọi real-model lane phải pin `model@revision`, dataset/backend, license/access,
  device, versions, seed, network policy, resource peak và artifact hash.
- Không tải model khi lập kế hoạch. Không dùng mock để thay real model ở lane
  tích hợp. Pure algorithm phải dùng real dataset/system substitute (ghi rõ).
- Remote CUDA phải giữ các invariant của
  `.agents/skills/remote-cuda-test/SKILL.md` (exact SHA, disposable clone,
  isolated caches, NVIDIA/CUDA check, cleanup, không sửa checkout server).
  Với L04, owner override yêu cầu transport là authenticated `ssh.exe` trực
  tiếp từ Windows PowerShell; không dùng Git Bash hoặc WSL. L04 không chạy
  remote trong giai đoạn planning và không dùng local CPU để thay real-model
  evidence.
- GitHub Actions cần tài khoản external có quyền; nếu thiếu thì là blocker,
  không đổi workflow để giả PASS. Sprint 80 dừng trước tag/publish nếu còn
  blocker, waiver chưa được owner ký, hoặc bằng chứng dưới ngưỡng 95% core /
  90% overall.

## 24 lane bắt buộc

| ID | Lane / public surface | Source → current tests/evidence | Real target / backend | Environment / command | Deterministic acceptance + artifact | Resources / network / cleanup | Status / blocker / owner |
|---|---|---|---|---|---|---|---|
| L01 | LatentSpace, LatentValue, PipelineContract, Result | `latent_space.py`, `latent_value.py`, `pipeline.py`, `adapters/conv_vae.py` → core tests and `scripts/m14_l01_core.py`, D2 | sklearn digits split; existing ConvVAE + AnalysisPipeline/PCA; NumPy/PyTorch CPU | local; `uv run python scripts/m14_l01_core.py` and `uv run pytest tests/test_m14_l01_core.py -q` | held-out reconstruction vs zero baseline, finite shapes/dtypes, stable schema/digest, no input mutation; `artifacts/m14/l01-core.json` | resource peak not measured; M14 estimate only; offline; output artifact/run record retained | verified D2; no blocker; API owner |
| L02 | geometry: covariance, projection, SO3/SE3, DTW, smoothing | `geometry.py`, `projection.py`, `pose.py`, `dtw.py`, `temporal.py`, `scripts/m14_l02_plan.py`, `scripts/m14_l02_data.py`, `scripts/m14_l02_metrics.py`, `scripts/m14_l02_envelope.py`, `scripts/m14_l02_geometry.py` → focused tests, D1/D2 synthetic; design contract [`l02-geometry.plan.json`](../artifacts/m14/l02-geometry.plan.json) | real sklearn digits held-out substitute with existing ConvVAE; model-induced latent sequences (not recorded physical trajectories); NumPy/SciPy CPU | local; side-effect-free validation: `uv run python -m scripts.m14_l02_geometry --check`; real runner: `uv run python -m scripts.m14_l02_geometry`; focused tests: `uv run pytest tests/test_m14_l02_geometry.py -q` | six independent manifold/T03-SLERP/T04-LERP/Riemannian/T04-SLERP/DTW verdicts with train-only density, 128 independent pair-path DTW trials, chance/shuffled/raw-pixel strong controls, endpoint/norm/angle tolerances; accepted artifact remains `artifacts/m14/l02-geometry.json` | offline; peak RSS not measured; retain run record and remove temporary plots/cache | 79.4C invocation fix; partial verified D2 for 4/6 records; manifold and DTW records failed honestly; no stable Fréchet or physical-trajectory claim; geometry owner |
| L03 | analysis: KMeans, probes, feature ranking | `clustering.py`, `probes.py`, `mlp_probe.py` → analysis tests, D2; concrete `TransformerLMIntegration` real forward path | GPT-2 hidden states on `openai-community/gpt2@e7da7f221d5bf496a48136c0cd264e630fe9fcc8`; sklearn digits labels | remote CUDA; `uv run pytest tests/test_clustering.py tests/test_probes.py tests/test_mlp_probe.py tests/test_m14_l03_analysis.py -q`; `uv run python -m scripts.m14_l03_analysis --run-real` | three independent accepted D2 records with seeded train/holdout metrics, baseline and memorization controls; `artifacts/m14/l03-analysis.json` self-digest `60bda13a4bbf68bbb6c9308cc813913fa653c37fba368fe1e4ea7a1f898ce06b`; run record `0bcaf14ef465f2ef5c5c909237d1f573596a77fa2ca51d042db74248cf4ca03a` | RTX 4060 Ti, CUDA 12.8, disposable clone/cache cleanup; report capture audit retained, raw transcript superseded and deleted | verified D2 forward-only; separate Sprint 79 transformer-hook verification passed 8/8 on exact SHA strict CUDA, resolving the structured hook/output cleanup blocker; no separate GPT-2 ModelAdapter or L11 claim; analysis owner |
| L04 | IntegratedGradients, TCAV, sensitivity/intervention controls | `integrated_gradients.py`, `tcav.py` → focused tests, D1; design freeze [`l04-explanations.plan.json`](../artifacts/m14/l04-explanations.plan.json) | GPT-2 `openai-community/gpt2@e7da7f221d5bf496a48136c0cd264e630fe9fcc8`; authored task/factor fixture with explicit clean/corrupted pairs and content/split/pair digests; real boundary `TransformerLMIntegration`; `ModelAdapter=N/A` by ADR | local offline contract: `uv run python -m scripts.m14_l04_explanations --check`; seven sequential real use cases (IG, TCAV, direct lens, tuned lens, disentanglement, true interchange patching, additive steering) on CUDA server through authenticated direct PowerShell `ssh.exe`; local CPU only offline checks; exact commands and cleanup are frozen in the L04 plan | completeness/sensitivity, selectivity, seeded 95% CI, shuffled/random/null/off-target/zero-strength controls, true interchange patching distinct from additive intervention, and separately fit holdout-calibrated affine tuned lens; `artifacts/m14/l04-explanations.json` plus per-use-case run/failure records | RTX 4060 Ti 8GB reference, CUDA 12.8; ~4 GB RSS/≤6 GB VRAM budget; acquire exact revision once, retain hashes/audit, remove disposable clone/cache; pinned WikiText-2 subset is provisioned and the corrected TunedLogitLens run at SHA `278a9f76f626f8b0c6a9d9c5517c9b349f08c2d5` produced validator-clean accepted D3 evidence; the historical attempt3 D0 remains preserved and unpromoted; final audit records the observed transport warning and cleanup PASS | L04.7 corrected TunedLogitLens D3 accepted; remaining L04 use cases planned; explanation owner |
| L05 | GaussianMixtureDensity, OOD, covariance/geodesic density | `density.py`, `geodesic.py` → density/geodesic tests, D1/D2 | real GPT-2 states + held-out prompts; sklearn GMM backend | local CPU; `uv run pytest tests/test_density.py tests/test_latent_anything/test_geodesic.py -q` | calibration, held-out AUROC threshold predeclared, path feasibility; `artifacts/m14/l05-density.json` | <3 GB; no secrets; remove cache | planned; threshold not yet evidenced; geometry owner |
| L06 | SAE, FeatureAtlas, cross-seed stability | `sae_evaluation.py`, `_sae_atlas.py` → SAE tests, D1/D2 | GPT-2 `openai-community/gpt2@e7da7f221d5bf496a48136c0cd264e630fe9fcc8`; real text corpus fixture | local CPU; `uv run pytest tests/test_sae_evaluation.py tests/test_sae_evaluation_network.py -m network -q` | dead-feature, reconstruction, cross-seed stability and atlas hash; `artifacts/m14/l06-sae.json` | ~4 GB; network only model; purge temporary corpus | planned; D3 gap; introspection owner |
| L07 | SubspaceProjection, steering, activation patch, LERP | `methods/activation_patch.py`, `methods/steering.py`, `projection.py`, `methods/lerp.py` → intervention tests, D1 | GPT-2 hidden states and VAE latents; PyTorch CPU | local; `uv run pytest tests/test_latent_anything/test_activation_patch.py tests/test_latent_anything/test_steering.py tests/test_latent_anything/test_projection.py tests/test_latent_anything/test_lerp.py -q` | paired control effect, reversibility/shape safety, no mutation; `artifacts/m14/l07-interventions.json` | <4 GB; offline after models; delete outputs | planned; cross-adapter evidence required; manipulation owner |
| L08 | ConvVAE adapter and manipulation pipeline | `adapters/conv_vae.py`, `manipulation_pipeline.py`, `pipeline.py` → ConvVAE held-out tests, D2 | sklearn digits 80/20; compact ConvVAE, PyTorch CPU | local; `uv run pytest tests/test_latent_anything/test_conv_vae.py tests/test_latent_anything/test_conv_vae_evidence.py -q` | ≥10% over all-zero baseline, non-degenerate latent, composition; `artifacts/m14/l08-convvae.json` | <1.5 GB, offline; no model download | planned; bounded local D2 only; generative owner |
| L09 | DiffusersAutoencoderKLAdapter | `integrations/diffusers_vae.py` → `test_diffusers_vae.py`, D2 | `stabilityai/sd-vae-ft-mse@31f26fdeee1355a5c34592e401dd41e45d25a493`, safetensors SHA `a1d993488569e928462932c8c38a0760b874d166399b14414135bd9c42df5815`, MIT | local CPU offline after cache; `uv run pytest tests/test_diffusers_vae.py tests/test_diffusers_vae_network.py -m network -q` | direct/adapter parity, mean/posterior seeded semantics, shape/dtype/finiteness; `artifacts/m14/l09-diffusers-vae.json` | ~1.5 GB RSS, 334 MB weights; no network in validation; cache manifest only | planned; license/hash portability gate; generative owner |
| L10 | conditional diffusion capture/intervention | `integrations/diffusers_conditional.py` → conditional tests, D1 | `runwayml/stable-diffusion-v1-5@39593d56b552c3a24aeb192dd11d2a1429c3102b`; Diffusers scheduler/denoiser | remote CUDA recommended; `uv run pytest tests/test_diffusers_conditional_network.py -m network -q` | paired scheduler controls, timestep trace, intervention effect and artifact hash; `artifacts/m14/l10-diffusion.json` | high VRAM/download; license card/access must be recorded; cleanup disposable cache | planned; network/VRAM/access blocker until provisioned; diffusion owner |
| L11 | TransformerLogitTarget, hidden-state adapter | `integrations/transformer_lm.py` → transformer tests, D2 candidate | `openai-community/gpt2@e7da7f221d5bf496a48136c0cd264e630fe9fcc8`, MIT; Transformers | local CPU; `uv run pytest tests/test_transformer_lm_network.py -m network -q` | vocab 50257, 13 layers, hidden 768, logits/hidden parity; `artifacts/m14/l11-gpt2.json` | ~2–4 GB; public download; remove disposable cache | direct logit-lens parity separately verified 8/8 on exact SHA strict CUDA; L11 remains planned until its complete lane artifact/capture contract; transformer owner |
| L12 | JEPAWorldModelAdapter and health metrics | `adapters/jepa.py` → JEPA tests, D2 structural | `facebook/ijepa_vith14_1k@be440b1cac639542ae553e71a9c7afd925ab5fac`, Transformers | local CPU or remote CUDA; `uv run pytest tests/test_latent_anything/test_jepa_checkpoint.py -m network -q` | finite latent prediction/health metrics, explicit decoder-free contract; `artifacts/m14/l12-ijepa.json` | high download/RAM; model-card license/access missing; retain manifest only | planned; license/access blocker; world-model owner |
| L13 | VQ-VAE and discrete latent adapter | `adapters/vq_vae.py` → VQ tests, D2 | `compact-vq-vae-v1`, sklearn digits, scikit-learn 1.9.0 BSD-3-Clause | local CPU; `uv run pytest tests/test_latent_anything/test_vq_vae.py -q` | perplexity >1, dead-code <1, encode/decode finite; `artifacts/m14/l13-vq.json` | <1 GB, offline; delete generated codebook | planned; no blocker; discrete owner |
| L14 | TokenizedWorldModel, TokenPrediction/Rollout | `tokenized_world_model.py` → tokenized tests, D2 with recorded failure | compact VQVAE + synthetic controlled dynamics; PyTorch CPU | local; `uv run pytest tests/test_latent_anything/test_tokenized_world_model.py -q` | prediction/rollout schema, bounded sequence, failure reproduced or fixed; `artifacts/m14/l14-tokenized.json` | <2 GB, offline; preserve failed artifact, clean temp | planned; early rollout failure must remain visible; world-model owner |
| L15 | Deterministic/Stochastic/RSSM transitions | `transition.py`, `rssm.py` → transition/RSSM tests, D2 synthetic | compact latent dynamics fixture; PyTorch CPU | local; `uv run pytest tests/test_latent_anything/test_transition.py -q` | one-step/rollout finite, seeded stochastic reproducibility, state carry; `artifacts/m14/l15-transitions.json` | <2 GB, offline; remove temp state | planned; real temporal model gap; runtime owner |
| L16 | reward/value, CEM, MPPI planning | `reward_value.py`, `cem.py`, `mppi.py` → planner tests, D2 synthetic | real recorded trajectory substitute + compact transition; NumPy/PyTorch CPU | local; `uv run pytest tests/test_reward_value.py tests/test_cem.py tests/test_cem_rollout.py tests/test_mppi.py tests/test_mppi_rollout.py -q` | predeclared return/regret and budget, deterministic seed, no invalid action; `artifacts/m14/l16-planning.json` | <2 GB, offline; preserve trace, clean scratch | planned; policy-grounded D3 gap; planning owner |
| L17 | Gaussian renderer/3D manipulation | `adapters/gaussian_renderer.py`, `integrations/gsplat_renderer.py` → renderer tests, D2/D3 candidate | named gsplat checkpoint required; `LATENT_ANYTHING_3DGS_CHECKPOINT`; gsplat 1.4–<2.0 | remote CUDA; `remote-cuda-test` runs `uv run pytest tests/test_latent_anything/test_gaussian_3d_renderer_network.py -m network -q` in disposable clone | render finite, multi-view PSNR/SSIM thresholds, intervention artifact; `artifacts/m14/l17-3dgs.json` | high VRAM/model download; access/license/hash required; disposable clone/cache | blocked; no named checkpoint; 3D owner |
| L18 | LeRobotDataset bridge/streaming | `integrations/lerobot_dataset.py` → dataset tests, D1/D2 | `lerobot/aloha_sim_insertion_human@cc571a3c661df81b566dbfde3d5c1e85fcdf7884`; LeRobot 0.6.1 | remote/Linux recommended; `uv run python scripts/lerobot_dataset_inspection.py lerobot/aloha_sim_insertion_human --revision cc571a3c661df81b566dbfde3d5c1e85fcdf7884 --output artifacts/m14/l18-dataset.json` | episode order, schema, bounded stream/resume, dataset license; `artifacts/m14/l18-dataset.json` | dataset download/storage; no credentials expected; delete local dataset | planned; upstream/license capture; LeRobot owner |
| L19 | ACT policy capture/intervention | `integrations/lerobot_act.py` → ACT tests, D2/D3 | `lerobot/act_aloha_sim_insertion_human@33259aa86eb45fdf85350280044a33d9d50e40c3`; same ALOHA dataset | remote CUDA/Linux; `remote-cuda-test` runs `uv run pytest tests/test_lerobot_act.py::test_pinned_public_act_checkpoint_pair_loads_through_lerobot_factories -m network -q` in disposable clone | action shape, paired intervention/control, simulator metric and revision manifest; `artifacts/m14/l19-act.json` | GPU/large download; model-card license not recorded; clean HF/cache | planned; license/access blocker; LeRobot owner |
| L20 | Diffusion Policy capture | `integrations/lerobot_diffusion.py` → Diffusion tests, D2/D3 | `LeTau/diffusion_aloha_insertion@6126e33`; dataset `lerobot/aloha_sim_insertion_human_image@d93d36a`; `aloha/AlohaInsertion-v0` | remote CUDA/Linux; `remote-cuda-test` runs `uv run pytest tests/test_lerobot_diffusion.py::test_pinned_public_diffusion_checkpoint_pair_loads_through_lerobot_factories -m network -q` in disposable clone | action distribution/causal simulation threshold, exact env/model/data IDs; `artifacts/m14/l20-diffusion-policy.json` | GPU/large download; access/license capture; disposable env/cache | planned; upstream access; LeRobot owner |
| L21 | SmolVLA capture/intervention | `integrations/lerobot_smolvla.py` → SmolVLA tests, D2 candidate | `lerobot/smolvla_libero@31d453f7edd78c839a8bbc39744a292686daf0de`; `lerobot/libero@a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4`; `libero/libero_spatial` | remote CUDA/Linux, ~16 GB GPU; `LATENT_ANYTHING_RUN_NETWORK=1 uv run pytest tests/test_lerobot_smolvla.py::test_smolvla_gpu_checkpoint_intervention_lane -m network -q` | action validity, causal intervention, simulator success threshold; `artifacts/m14/l21-smolvla.json` | ~450M bf16, large download; model/data license/access required; clean clone/cache | planned; Linux/GPU/account/access blockers; LeRobot owner |
| L22 | ArtifactStore, portable envelopes, cache, stream, recorder/tracking | `portable.py`, `artifact_store.py`, `runtime/cache.py`, `rollout_pipeline.py`, `run_record.py`, `experiment_recorder.py` → focused tests, D1/D2 | real Arrow/SQLite filesystem; MLflow/W&B optional backends | local isolated temp; `uv run pytest tests/test_portable.py tests/test_portable_results.py tests/test_artifact_store.py tests/test_latent_anything/test_cache.py tests/test_sprint75_streaming.py tests/test_run_record.py tests/test_run_record_portable.py tests/test_experiment_recorder.py tests/test_mlflow_recorder.py tests/test_wandb_recorder.py -q` | schema-v1 round trip/migration, path safety, resume/cancel, provider atomicity; `artifacts/m14/l22-runtime.json` | <1 GB, offline; tracking credentials only in opt-in lane; delete temp DB/artifacts | planned; external tracking account may block; runtime owner |
| L23 | registry/plugins/config/CLI/serialization/security | `registry.py`, `_plugin_builtins.py`, `plugin_discovery.py`, `plugin_groups.py`, `plugin_metadata.py`, `cli.py`, `config.py`, `pipeline_config.py` → registry/plugin/CLI tests, D1 | all 32 built-ins, 5 entry-point groups, hello-world separately installed plugin; base + each extra | local clean env; `uv run pytest tests/test_plugin_discovery.py tests/test_plugin_groups.py tests/test_plugin_installation.py tests/test_plugin_metadata.py tests/test_cli.py tests/test_registry_migration.py tests/test_latent_anything/test_registry.py tests/test_latent_anything/test_config.py tests/test_api_compatibility.py -q` | import isolation, discovery, install/uninstall, schema migrations, negative paths, secret/path/zip safety; `artifacts/m14/l23-contract.json` | <2 GB, no network except plugin install fixture; uninstall temp plugin/cache | planned; external GitHub Actions account blocker; API owner |
| L24 | packaging/docs/performance/release gates | `pyproject.toml`, `docs/M14_REAL_SYSTEM_VALIDATION.md`, `scripts/validate_evidence_ledger.py`, `.github/workflows/ci.yml`, `.github/workflows/optional-extras.yml`, `.github/workflows/release.yml` → full tests, MkDocs, Ruff/Pyright | clean base/all 12 profiles, supported Python/platform tiers, wheel/sdist | local then remote CI; `uv sync --locked --all-extras`; `uv run pytest -q`; `uv run mkdocs build --strict` | no diff/check errors, strict docs, type/lint/test pass, wheel import, performance budgets; `artifacts/m14/l24-rc.json` | build isolated; no secrets in artifacts; delete dist/site/temp | planned; external Actions account and unresolved evidence thresholds; release owner |

## L04.1 preregistration / design freeze

The machine-readable [L04 plan](../artifacts/m14/l04-explanations.plan.json) is
the source of truth for the five record IDs, dependency order, exact model
revision, authored fixture and digests, token/layer/seed/resource budgets,
thresholds, controls, artifact/run/failure schemas, and atomic implementation
tasks. The committed fixture is
[`l04-prompt-factor-fixture.jsonl`](../artifacts/m14/l04-prompt-factor-fixture.jsonl):
24 rows in 12 groups, with an explicit group-preserving train/holdout split and
content/split/pair SHA-256 values. It is project-authored MIT-repository content,
not an external corpus; its external validity is limited and it cannot alone
support a D3 claim.

The seven executions are sequential: IntegratedGradients (support-only), TCAV,
direct logit lens (support-only), separately holdout-calibrated affine tuned
lens, disentanglement, true clean/corrupted interchange activation patching,
and additive steering. The five ledger records are the non-support executions.
Each fixture
`causal_pair_id` has exactly one clean and one corrupted condition and remains
inside one group/split. The existing VAE
`activation_patch.py` and NumPy latent `steering.py` remain their own methods;
they are not relabeled as TransformerLM causal evidence. GPT-2 uses the concrete
`TransformerLMIntegration`; `ModelAdapter=N/A` is intentional. Direct
PowerShell `ssh.exe` with configured authentication, exact-SHA detached clone,
isolated caches, report capture before cleanup, and disposable-clone cleanup
are mandatory for every real model/integration use case in this lane. Each SSH
run receives exactly one parameterized use case and is reviewed before the next;
there is no seven-use-case loop. Tuned
lens additionally requires the pinned `Salesforce/wikitext` /
`wikitext-2-raw-v1` corpus at revision
`f776294184f13b8ff2337b3841cf9269a6216d1e` (CC BY-SA 3.0/GFDL). Its bounded
8192/2048 subset is now provisioned and bound by the committed manifest,
content/split digests, and selection metadata. The first owner-authorized
exact-SHA run (`dcc76ba7f064b5b6dc2e09c20d741da4cc6e5422`) reached real CUDA
execution and failed D0 at tuned-lens aggregation; the corrected validation
below supersedes that blocked result. Local CPU commands are contract/unit
checks only.

The next exact-SHA run (`2a6de8dbb98f824b247da23e2bc1e3cea5efea3a`) completed
the real GPT-2/WikiText computation and recorded `6.5803880806` nats point
improvement, `6.5399008976` nats conservative lower bound, passing controls,
and passing resource caps on an RTX 4060 Ti (`1180.8877603` seconds,
`2167476736` allocated CUDA bytes, `2166382592` RSS bytes). It is still
D0/evidence-ineligible: artifact validation rejected the execution entry
because artifact assembly dropped the singular fit `seed=79` while retaining
bootstrap seeds. The same historical run also exposed that the shuffled-target
control must mask `source_attention_mask & permuted_target_attention_mask`;
the local correction and validator provenance binding are now covered by
regressions. The remote wrapper emitted cleanup PASS, then its CRLF-sensitive
status encoding produced outer SSH exit `2`; this is transport evidence, not a
semantic rerun or promotion. The three attempt3 payload files are retained
byte-for-byte and are not rewritten; the sanitized recovery audit received
only the scoped failure SHA metadata correction recorded in the task summary.

The final owner-authorized corrected run at exact SHA
`278a9f76f626f8b0c6a9d9c5517c9b349f08c2d5` completed one real
GPT-2/TransformerLMIntegration TunedLogitLens execution on the pinned
WikiText-2 subset. It produced a validator-clean D3 artifact with acceptance
true, fit seed `79`, bootstrap seeds `[17, 29, 41, 53, 67]`, layer `6`, native
hidden-state index `7`, holdout improvement `6.5803880806` nats, conservative
lower bound `6.5399008976` nats, all controls passing, and resource budget
passing on an RTX 4060 Ti (`1116.8708012` seconds,
`2065599488` allocated CUDA bytes, `2161827840` RSS bytes). The shuffled
target policy is explicitly `source_attention_mask &
permuted_target_attention_mask`; artifact, run-record, and failure validators
all returned no errors. The sanitized audit records the raw capture and
bundle hashes, and retains the observed base64 decoder warning as transport
evidence; remote cleanup emitted `L04_CLEANUP=PASS` and no retry occurred.
The semantic/artifact evidence is accepted at D3, but the decoded script bytes
were not independently hash-verified because the raw capture also contained
`base64: invalid input`; this run therefore does not establish stronger
transport provenance.

#### TunedLogitLens operational override (owner-approved)

The frozen plan remains immutable. Its remote command predates the pinned
`datasets` dependency, so the following is the exact owner-approved
operational override for TunedLogitLens; it must be run in the same isolated
`uv` environment as the preflight and must not be copied into or used to
rewrite the frozen plan:

```text
uv run --locked --extra transformers --with 'datasets==4.8.5' --with 'transformers==4.57.6' --with 'tokenizers==0.22.2' --with 'huggingface-hub==0.35.3' python -m scripts.m14_l04_explanations --run-real --use-case TunedLogitLens --plan artifacts/m14/l04-explanations.plan.json --fixture artifacts/m14/l04-prompt-factor-fixture.jsonl
```

Before opening SSH, a local PowerShell preflight is mandatory. It checks
dependency resolver/import compatibility only; it deliberately does not require
local CUDA and is not real-model evidence. Write a safe temporary `.py` file
with a literal single-quoted here-string, run it with the exact resolver prefix
below, and never use `python -c` or nested command quoting. The recorded local
diagnostic passed the compatible versions `datasets==4.8.5`,
`transformers==4.57.6`, `tokenizers==0.22.2`, and
`huggingface-hub==0.35.3`:

```powershell
$localPreflightFile = (New-TemporaryFile).FullName
$localPreflightSource = @'
from importlib.metadata import version

import datasets
import huggingface_hub
import tokenizers
import transformers

expected = {
    "datasets": "4.8.5",
    "transformers": "4.57.6",
    "tokenizers": "0.22.2",
    "huggingface-hub": "0.35.3",
}
observed = {package: version(package) for package in expected}
assert observed == expected, (observed, expected)
print(observed)
'@
[IO.File]::WriteAllText(
    $localPreflightFile,
    $localPreflightSource,
    [Text.UTF8Encoding]::new($false)
)
try {
    & uv run --locked --extra transformers --with 'datasets==4.8.5' --with 'transformers==4.57.6' --with 'tokenizers==0.22.2' --with 'huggingface-hub==0.35.3' python $localPreflightFile
    $localPreflightExit = $LASTEXITCODE
    if ($localPreflightExit -ne 0) {
        throw "Local preflight failed with exit $localPreflightExit"
    }
} finally {
    Remove-Item -LiteralPath $localPreflightFile -Force -ErrorAction SilentlyContinue
}
```

The remote heredoc must additionally assert imports, versions, and CUDA
availability in that exact same environment, failing before model loading if the
assertion fails.

### Reusable L04 transport (authoritative)

All L04 real runs use the two-file boundary below. The PowerShell helper owns
only exact-byte transport and raw stdout/stderr capture; the Bash payload owns
the disposable clone, isolated caches, dependency/CUDA preflight, one CLI
invocation, bundle emission, and cleanup. This replaces the historical wrapper
below; the historical block is retained only as failure evidence and must not
be copied into a new run.

The current implementation gate is Phase A: the helper and payload are tested
locally/offline, while exact remote Bash `find`/`tar` behavior and CUDA/model
execution remain Phase B after commit/push and owner approval.

Run it directly from Windows PowerShell with the authenticated native
`ssh.exe` path. First build and inspect the sanitized manifest without opening
the connection:

```powershell
$CodeSha = (git rev-parse HEAD).Trim()
$SshPath = (Get-Command ssh.exe -ErrorAction Stop).Source
$RawCapture = Join-Path (Get-Location) "artifacts/m14/l04-disentanglement.raw.txt"
& pwsh -NoProfile -File scripts/m14_l04_remote_transport.ps1 `
  -SshExecutable $SshPath `
  -RemoteTarget "trietlm@192.168.30.244" `
  -PayloadPath (Join-Path (Get-Location) "scripts/m14_l04_remote_payload.sh") `
  -UseCase Disentanglement `
  -CodeSha $CodeSha `
  -RepoUrl "https://github.com/trietlm/latent-anything.git" `
  -RawCapturePath $RawCapture `
  -BuildOnly
```

The manifest contains only payload/bootstrap SHA-256 and byte counts, expected
markers, and redacted command arguments. The helper normalizes CRLF/CR to LF,
encodes UTF-8 without BOM, sends a single-quoted Base64 heredoc through
`System.Diagnostics.ProcessStartInfo`, and writes the raw capture before any
parsing. The remote bootstrap decodes into a collision-resistant temporary
file, emits `L04_TRANSPORT_DECODE_STATUS`, compares the decoded-byte SHA-256,
refuses mismatch or decoder failure, executes `bash <decoded-temp>`, preserves
the semantic exit, removes the file, verifies absence, and emits a distinct
`L04_TRANSPORT_CLEANUP` marker.

Raw capture publication is collision-safe: the internal lifecycle seam writes
and flushes a new same-directory temporary file, atomically replaces/moves the
requested target only after close succeeds, and reports
`raw_capture_write_succeeded=false` without hashing a stale target when
publication fails.

The Bash payload is invoked with the canonical PascalCase use case and exact
40-character detached SHA. It must emit `L04_USE_CASE`, `L04_CODE_SHA`,
`L04_STATUS`, `L04_BUNDLE_B64_BEGIN/END`, and `L04_CLEANUP`. Its dependency
preflight uses the pinned `uv` resolver and checks imports, versions, and CUDA
before the sole `scripts.m14_l04_explanations --run-real` invocation. The
artifact bundle snapshots only the three newly created partial/run/failure
artifacts for the current use case and attempt, rejects traversal, mixed
attempts, missing members, and unrelated history, then emits the bundle before
the cleanup trap removes the clone and all isolated caches. Do not run this
boundary from Git Bash, WSL, a local Bash shell, or a pre-existing remote
checkout.

The following wrapper is the historical owner-approved pattern used for the
final run. It is **NOT REUSABLE** for L04.8 or subsequent lanes: the final raw
capture contained `base64: invalid input`, and the decoded script bytes were
not independently hash-verified even though semantic execution completed.
The historical wrapper constructs the remote Bash script in PowerShell,
normalizes it to LF, and pipes it directly to `ssh.exe` with
`target 'bash -s --'`. On the remote side, create a temporary
`preflight.py` with a single-quoted heredoc, execute it, and then execute the
CLI with the same exact `uv` prefix:

```powershell
$remoteScript = @'
set -eu -o pipefail
RemoteScriptSha="${4:?missing remote script sha256}"
echo L04_REMOTE_SCRIPT_START=PASS
echo L04_REMOTE_SCRIPT_SHA256="$RemoteScriptSha"
UseCase="$1"
CodeSha="$2"
RepoUrl="$3"
workdir=""
repo_dir=""
cache_root=""
preflight_file=""
capture_file=""
bundle_file=""
cleanup_status=0
prior_exit=0

workdir="$(mktemp -d /tmp/latent-anything-l04.XXXXXX)"
case "$workdir" in /tmp/latent-anything-l04.*) ;; *) exit 90;; esac
repo_dir="$workdir/repo"
cache_root="$workdir/cache"
preflight_file="$workdir/preflight.py"
capture_file="$workdir/remote.capture"
bundle_file="$workdir/l04-capture.tgz"

cleanup() {
  prior_exit=$?
  cleanup_status=0
  trap - EXIT HUP INT TERM
  if [ -n "$workdir" ] && [ -d "$workdir" ] && rm -rf -- "$workdir"; then
    if [ -e "$workdir" ]; then cleanup_status=1; fi
  else
    cleanup_status=1
  fi
  if [ "$cleanup_status" -eq 0 ]; then
    echo L04_CLEANUP=PASS
  else
    echo L04_CLEANUP=FAIL
  fi
  if [ "$prior_exit" -ne 0 ]; then exit "$prior_exit"; fi
  exit "$cleanup_status"
}
trap cleanup EXIT HUP INT TERM

git clone --no-checkout "$RepoUrl" "$repo_dir"
git -C "$repo_dir" checkout --detach "$CodeSha"
test "$(git -C "$repo_dir" rev-parse HEAD)" = "$CodeSha"
export UV_CACHE_DIR="$cache_root/uv"
export HF_HOME="$cache_root/huggingface"
export HF_DATASETS_CACHE="$cache_root/datasets"
export TRANSFORMERS_CACHE="$cache_root/transformers"
mkdir -p "$UV_CACHE_DIR" "$HF_HOME" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE"
cd "$repo_dir"
nvidia-smi
export LATENT_ANYTHING_RUN_NETWORK=1
export LATENT_ANYTHING_NETWORK_DEVICE=cuda

cat > "$preflight_file" <<'PY'
from importlib.metadata import version

import datasets
import huggingface_hub
import tokenizers
import torch
import transformers

expected = {
    "datasets": "4.8.5",
    "transformers": "4.57.6",
    "tokenizers": "0.22.2",
    "huggingface-hub": "0.35.3",
}
observed = {package: version(package) for package in expected}
assert observed == expected
assert torch.cuda.is_available()
print(observed, "torch", torch.__version__)
PY
uv run --locked --extra transformers --with 'datasets==4.8.5' --with 'transformers==4.57.6' --with 'tokenizers==0.22.2' --with 'huggingface-hub==0.35.3' python "$preflight_file"
echo L04_USE_CASE="$UseCase"
status=0
if uv run --locked --extra transformers --with 'datasets==4.8.5' --with 'transformers==4.57.6' --with 'tokenizers==0.22.2' --with 'huggingface-hub==0.35.3' python -m scripts.m14_l04_explanations --run-real --use-case "$UseCase" --plan artifacts/m14/l04-explanations.plan.json --fixture artifacts/m14/l04-prompt-factor-fixture.jsonl 2>&1 | tee "$capture_file"; then
  status=0
else
  status=${PIPESTATUS[0]}
fi

# Capture and bundle extraction happen before the cleanup trap removes the
# disposable clone, all caches, the preflight file, and this workdir.
sha256sum "$capture_file" > "$workdir/remote.capture.sha256"
find artifacts/m14 -maxdepth 1 -type f -name 'l04-explanations*' -print > "$workdir/files.txt"
if [ -s "$workdir/files.txt" ]; then
  tar -czf "$bundle_file" -C "$repo_dir" --files-from "$workdir/files.txt" -C "$workdir" remote.capture.sha256
  echo L04_BUNDLE_B64_BEGIN
  base64 -w0 "$bundle_file"
  echo
  echo L04_BUNDLE_B64_END
else
  echo L04_BUNDLE_UNAVAILABLE
  if [ "$status" -eq 0 ]; then status=1; fi
fi
echo L04_CODE_SHA="$CodeSha"
echo L04_STATUS="$status"
exit "$status"
'@
$remoteScript = $remoteScript.Replace("`r`n", "`n").Replace("`r", "`n")
$utf8NoBom = [Text.UTF8Encoding]::new($false)
$remoteScriptBytes = $utf8NoBom.GetBytes($remoteScript)
$remoteScriptSha256 = [Convert]::ToHexString(
    [Security.Cryptography.SHA256]::HashData($remoteScriptBytes)
).ToLowerInvariant()
$remoteScriptBase64 = [Convert]::ToBase64String($remoteScriptBytes)
$remoteScriptBase64 | ssh.exe $remoteTarget 'base64 -d | bash -s --' $UseCase $CodeSha $RepoUrl $remoteScriptSha256 2>&1 | Tee-Object -FilePath $rawCapture
$sshExit = $LASTEXITCODE
```

The historical wrapper contract was intended to be complete: PowerShell computes the
SHA-256 over the exact UTF-8/no-BOM/LF bytes, encodes those bytes with
`ConvertToBase64String`, and sends the ASCII envelope through one direct
authenticated `ssh.exe` invocation. The remote `base64 -d | bash -s --`
decoder emits a start marker and the announced local script digest before it
executes the decoded bytes. The digest marker records the intended local
payload; it does not independently verify the bytes accepted by the decoder.
The raw capture must retain that marker, the local hash, and any decoder
diagnostic in its sanitized audit. This avoids native PowerShell stdin newline
and nested-quote rewrites, but the final run demonstrates that it is not a
sufficient transport-integrity check by itself.

The wrapper then clones the repository with
`--no-checkout`, checks out and verifies the exact detached SHA, and places the
UV, Hugging Face, `datasets`, and `transformers` caches under the temporary
workdir. Every trap variable is initialized before the trap is installed.
It exports `LATENT_ANYTHING_RUN_NETWORK=1` and
`LATENT_ANYTHING_NETWORK_DEVICE=cuda` before both preflight and CLI, so the
same network/device contract is inherited by each. The CLI output is captured
and hashed, and the artifact bundle is assembled and emitted before cleanup.
Cleanup removes the entire workdir (clone, all caches, preflight, capture, and
bundle), verifies the path is absent, emits PASS only when removal and absence
checks succeed, emits FAIL otherwise, and preserves a non-zero prior command
exit.

Do not use raw multiline PowerShell stdin, `python -c`, escaped `printf` marker
emitters, or nested remote command quoting for this workflow:
PowerShell/native parsing can strip quotes, backslashes, and intended newlines
before Bash receives the script. The LF-normalized UTF-8/no-BOM base64 envelope
above is historical evidence only and is **not an approved reusable transport**
until the L04.8 remote-temp-file decode/exit/SHA comparison preflight is
implemented and tested. Markers must use `echo`; the
cleanup function may emit `L04_CLEANUP=PASS` only after the
temporary file removal succeeds. Transport remains direct authenticated
`ssh.exe` from Windows PowerShell to the disposable detached clone; do not use
Git Bash or WSL. Capture `2>&1 | Tee-Object`, persist and hash raw bytes before
parsing, and capture `$LASTEXITCODE` immediately after the SSH pipeline. The
cleanup marker and outer SSH exit are required evidence fields; their absence
must be reported as unverified and cannot be inferred from a successful inner
command.

### Acceptance contract cho mỗi dòng

Executor phải điền đủ các cột trên vào JSON/YAML artifact, không ghi “TBD” cho
model/revision khi lane đã được chạy. Một artifact tối thiểu chứa `capability`,
`public_symbols`, `source`, `tests`, `evidence_tier`, `model`, `revision`,
`dataset`, `backend`, `license_access`, `environment`, `command`, `seed`,
`acceptance`, `resource_peak`, `network`, `credentials`, `cleanup`, `status`,
`blocker`, `waiver_owner`, `git_sha`, `tool_versions`, `sha256`. Failure artifact
được giữ nguyên và liên kết từ ledger; không sửa metric để che lịch sử.

## Coverage không mồ côi

### 205 top-level exports

Các nhóm sau là phân vùng **đúng theo thứ tự `latent_anything.__all__`**; mỗi
Tên xuất hiện đúng một lần, tổng 205. `E01–E08` giữ nguyên thứ tự beta; `E09`
là các tên canonical RFC0001 được thêm theo kiểu additive. Snapshot API-freeze
giữ projection canonical ổn định gồm 202 entry baseline (thay thế các tên
legacy theo RFC0001), trong khi 205 là runtime surface hiện tại. Các nhóm cần snapshot
signature/import trong L23; các lane chức năng tương ứng được chỉ ra ở L01–L24.

| Group | Symbols (exact public names) | Lane |
|---|---|---|
| E01 (1–28) | `AnalysisPipeline`, `CEMConfig`, `CEMIteration`, `CEMPlanResult`, `CEMPlanner`, `MPPIConfig`, `MPPIIteration`, `MPPIPlanResult`, `MPPIPlanner`, `MPPIRecedingHorizonResult`, `PipelineContract`, `BatchExecutor`, `CacheKey`, `CacheStats`, `DiskCacheError`, `DiskCacheStats`, `ClusterStabilityReport`, `ConceptDataset`, `ConceptDirectionResult`, `ControlBaselines`, `CovarianceConfig`, `CovarianceState`, `CrossSeedReport`, `FeatureAtlas`, `FeatureAtlasEntry`, `FeatureCrossCheck`, `FeatureRanking`, `GLOBAL_REGISTRY` | L01,L05,L06,L16,L22 |
| E02 (29–56) | `InMemoryCache`, `IntegratedGradients`, `IntegratedGradientsConfig`, `IntegratedGradientsResult`, `JEPAWorldModelAdapter`, `JEPAWorldModelConfig`, `JEPALatentHealth`, `JEPAEvaluationReport`, `JEPAPrediction`, `JEPAPredictionMetrics`, `JEPARolloutMetrics`, `KMeans`, `KMeansConfig`, `KMeansResult`, `LatentSpace`, `LatentValue`, `LinearProbe`, `LinearProbeConfig`, `LinearProbeResult`, `MLPProbe`, `MLPProbeConfig`, `MLPProbeResult`, `ManipulationPipeline`, `ManipulationPipelineSpec`, `CEMPlannerSpec`, `MPPIPlannerSpec`, `Method`, `ObjectSpec`, `OrthonormalSubspace` | L01,L03,L04,L07,L12,L16,L22 |
| E03 (57–84) | `PipelineResult`, `PipelineSpec`, `RolloutPipeline`, `RolloutPipelineSpec`, `RolloutResult`, `RewardValueEvaluationSpec`, `PoseConfig`, `PoseMetadata`, `PoseTrajectory`, `ProbeComparison`, `SubspaceProjection`, `SubspaceProjectionConfig`, `TCAV`, `TCAVConfig`, `TCAVResult`, `TCAVScore`, `TransformerLogitTarget`, `ProfileEvent`, `Registry`, `RegistryEntry`, `RuntimeProfile`, `RuntimeProfiler`, `SQLiteDiskCache`, `make_disk_cache_key`, `ArtifactStore`, `ArtifactStoreError`, `StoredArtifact`, `PortableLimits` | L01,L02,L04,L07,L11,L16,L22,L23 |
| E04 (85–112) | `PortableNodeError`, `encode_portable`, `decode_portable`, `PortableEnvelope`, `PortableResultError`, `encode_result_envelope`, `decode_result_envelope`, `ArtifactRef`, `DuplicateRunError`, `FileSystemRunRecorder`, `RunComparisonReport`, `RunRecord`, `build_comparison_report`, `compute_run_identity`, `migrate_run_record`, `SAEConfig`, `SAEEvaluationResult`, `SAEFeatureEvaluation`, `SAEFeatureMetrics`, `SAEStabilityResult`, `SE3`, `SO3`, `SensitivityReport`, `Trajectory`, `DeterministicLatentTransition`, `GaussianPrediction`, `OneStepMetrics`, `RolloutMetrics` | L02,L06,L15,L22,L23 |
| E05 (113–140) | `StochasticGaussianLatentTransition`, `StochasticOneStepMetrics`, `StochasticRollout`, `StochasticRolloutMetrics`, `LatentTransition`, `TokenPrediction`, `TokenPredictionMetrics`, `TokenRolloutMetrics`, `TokenizedEvaluationReport`, `TokenizedWorldModel`, `TokenizedWorldModelConfig`, `HoldoutEvaluation`, `LinearRewardScorer`, `MonteCarloValueEstimator`, `RewardValueDiagnostics`, `RewardValueEvaluationResult`, `RewardValueEvaluator`, `TrajectoryScoreComparison`, `ValueCalibration`, `compare_real_imagined_scores`, `compute_discounted_returns`, `compute_mppi_weights`, `RSSMLatentTransition`, `RSSMTransitionConfig`, `RSSMPrediction`, `RSSMRollout`, `RSSMOneStepMetrics`, `RSSMRolloutMetrics` | L14,L15,L16 |
| E06 (141–169) | `BoundaryMetrics`, `ChangePointResult`, `Segment`, `SegmentationConfig`, `SmoothedTrajectory`, `SmoothingConfig`, `build_feature_atlas`, `build_from_config`, `build_from_dict`, `build_manipulation_pipeline_from_config`, `build_cem_planner_from_config`, `build_mppi_planner_from_config`, `build_pipeline_from_config`, `build_reward_value_evaluator_from_config`, `build_rollout_pipeline_from_config`, `GMMConfig`, `GaussianMixtureDensity`, `DensityResult`, `DensityMetrics`, `DensityEvaluationReport`, `DensityStabilityReport`, `DTWConfig`, `DTWCostSummary`, `DTWResult`, `DensityGeodesic`, `GeodesicConfig`, `GeodesicPath`, `PathOptimizationStatus` | L02,L05,L06,L07,L16,L23 |
| E07 (170–188) | `density_cross_seed_evaluation`, `mahalanobis_baseline`, `fit_covariance_state`, `compute_dtw`, `indexwise_distance`, `detect_change_points`, `evaluate_boundaries`, `smooth_trajectory`, `smoothing_distortion`, `check_clustering_geometry`, `cluster_stability_analysis`, `compare_probes`, `compare_with_labels`, `compute_integrated_gradients`, `compute_tcav`, `coordinate_identity`, `assert_arithmetic_compatible`, `cross_check_feature`, `cross_seed_evaluation` | L02,L03,L04,L05,L06,L07 |
| E08 (189–202) | `cross_seed_sae_stability`, `evaluate_layers`, `evaluate_sae_features`, `evaluate_sensitivity`, `intervention_agreement`, `learn_linear_separator_direction`, `learn_mean_diff_direction`, `list_entries`, `load_feature_atlas`, `lookup_entry`, `nonlinear_memorization_test`, `rank_feature_examples`, `register_entry`, `save_feature_atlas` | L03,L04,L06,L07,L23 |
| E09 (203–205) | `AnalysisMethod`, `Intervention`, `InterventionPipeline` | L01,L02,L23; RFC0001 |

### Registry, plugin groups, optional profiles

| Coverage set | Exact entries | Verification |
|---|---|---|
| 32 built-in registry entries | Adapter: `conv_vae`, `gaussian_3d_renderer`, `gaussian_renderer`, `hidden_state`, `random_projection`, `vae`, `vq_vae`, `jepa_world_model`; Analysis: `pca`, `sae`, `umap`, `kmeans`, `gaussian_mixture_density`, `linear_probe`, `mlp_probe`, `tcav`, `integrated_gradients`, `sae_evaluation`; Intervention: `activation_patch`, `density_geodesic`, `lerp`, `steering`, `subspace_projection`; Runtime: `deterministic_transition`, `stochastic_transition`, `rssm_transition`, `tokenized_world_model`, `jepa_transition`, `linear_reward_scorer`, `monte_carlo_value_estimator`, `cem_planner`, `mppi_planner` | L23 snapshot, registry tests, each entry mapped to L01–L22; count must remain 32 |
| 5 plugin groups | `latent_anything.adapter`, `latent_anything.analysis`, `latent_anything.intervention`, `latent_anything.transition`, `latent_anything.planner` | L23 install/discovery/negative isolation |
| 12 optional profiles | `docs`, `diffusers`, `transformers`, `diffusers-full`, `3d`, `lerobot`, `lerobot-diffusion`, `lerobot-smolvla`, `viz`, `tracking-mlflow`, `tracking-wandb`, `tracking` | L09–L11,L17–L24 clean install matrix; no implicit import/download |

### CLI, schemas, negative/security và composition

| Surface | Exact contract and real validation |
|---|---|
| CLI | `capture-points`, `inspect-policy`, `inspect-dataset`, `replay-run`, `compare-runs`; run `--help`, malformed input, path traversal, interrupted replay, and one real artifact per applicable command. |
| Serialization | `portable-node-v1`, `result-envelope-v1` (including v0 migration), `artifact-envelope-v1`, schema-v1 run records and recorder state; round trip across clean environments and reject unknown/oversized/malformed payloads. |
| Sync/async | Execute every sync public path and its async/streaming counterpart where exposed; cancel after first/partial batch and prove bounded memory, ordering, state carry and no leaked task. |
| Composition | At minimum ConvVAE→probe→intervention, GPT-2→SAE→steering→run record, VQ→tokenized rollout→CEM/MPPI, LeRobot dataset→ACT/Diffusion/SmolVLA→recorder, and Diffusers VAE→portable artifact. Each gets a separate artifact and source-to-test trace. |
| Security/failure | Optional dependency absence, unknown registry/plugin, invalid config, duplicate run, symlink/path escape, malicious archive/member, secret redaction, network-offline assertion, cancellation, corrupted cache, schema downgrade and license metadata omission must fail closed with typed errors. |

## Explicit non-API, backlog và blockers

SAM, OpenCLIP, timm, Torchvision model adapters, Open3D, trimesh và named 3DGS
checkpoint hiện **không phải stable public API**. Không được đưa chúng vào số
liệu D3 hay release claim. Chúng là backlog (nếu người dùng phê duyệt phạm vi)
hoặc blocker của lane liên quan; L17 bị blocked cho đến khi có checkpoint có
tên, revision, license/access và artifact. Các beta artifact cũ còn nói không
có probes/planning/world-model/3D integrations; Sprint 78 phải reconcile chúng
với code hiện tại, không giữ tuyên bố stale. SmolVLA hiện có artifact D3 trong
một tài liệu nhưng ledger override là D2; ledger là nguồn sự thật cho đến khi
chạy lại và ký artifact.

## Trình tự M14 và waivers

1. Sprint 78: freeze inventory/snapshots trước, hoàn thành SRP audit toàn bộ
   `src/` (đặc biệt files lớn), refactor chỉ khi có parity snapshot; dọn xung
   đột docs; chốt migration, schema, plugin, CLI và API ADR.
2. Sprint 79: clean-environment matrix → local real lanes → remote CUDA lanes →
   performance/security/license/evidence ledger; mọi failure giữ artifact.
   Waiver chỉ do owner ký, ghi lý do, phạm vi, hạn hết hạn và không được che
   core gap.
3. Sprint 80: RC review, wheel/sdist/install, release docs và publish. Nếu còn
   bất kỳ blocker, Actions account blocker, missing credential/model license,
   threshold fail hoặc stale claim thì **stop before release**.

Artifact/command names trong bảng là normative; kết quả thực tế phải ghi SHA
code và manifest. Checklist này không tự nâng trạng thái Milestone 14.

## Trạng thái checkpoint API-freeze Sprint 78

Snapshot API và tài liệu migration hiện ghi nhận **205 export runtime** và
projection canonical ổn định **202 entry**; [MIGRATION](MIGRATION.md),
[API_REFERENCE](API_REFERENCE.md), và artifact của task 78.40 là các điểm vào
cho người dùng, còn snapshot/ledger vẫn là nguồn máy móc chuẩn. Checkpoint này
không cho phép xóa alias, bump version, tag, publish, hay tuyên bố release
readiness. Metadata vẫn là `0.1.0b1`; `0.9.0` chỉ là epoch pre-stable dự kiến và
các blocker evidence/workflow phải được giải quyết trước khi release.

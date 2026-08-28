# Task 79 L04.1 — L04 planning/design freeze

Status: complete for planning only. No source or test file was changed, and no
commit, push, model download, or remote command was performed.

## Frozen scope

L04 is the explanation lane for five records, in this order:

1. `THY-T05-CONCEPT-ACTIVATION-VECTORS-TCAV-KIM-ET-AL-2018` — TCAV, D0 → D3.
2. `THY-T05-LOGIT-LENS-TUNED-LENS` — direct lens plus separately fit affine
   tuned lens, D0 → D3.
3. `THY-T03-DISENTANGLEMENT` — factor benchmark, D0 → D2.
4. `THY-T05-ACTIVATION-PATCHING` — true clean/corrupted interchange patching,
   D1 → D3.
5. `THY-T05-STEERING-VECTORS-ZOU-ET-AL-2023-REPRESENTATION-ENGINEERING` —
   additive hidden-state steering, D1 → D3.

TCAV and lens depend on the completed L03 linear-probing boundary. The last
three depend on TCAV. Graphify found no SCC. Current ledger levels remain
unchanged; no record is promoted by this task.

## Design decisions

- Model is `openai-community/gpt2@e7da7f221d5bf496a48136c0cd264e630fe9fcc8`,
  MIT, via the concrete `TransformerLMIntegration` boundary. `ModelAdapter` is
  N/A deliberately; no fake adapter is permitted.
- The authored fixture is
  [`l04-prompt-factor-fixture.jsonl`](m14/l04-prompt-factor-fixture.jsonl),
  24 rows and 12 groups, with content SHA-256
  `f5c66f6d947c23f25d41e6aaf8982481feabc92bbff600bd929d27772fb62c0f`,
  group split SHA-256
  `7d788c18212bb1d7e345528c68af6f2bf3e0f745ca77e2d115d74ac3e964121b`, and
  causal-pair SHA-256
  `7225e73c1238b23f6521718c8401331e59653a90499f4b2d75f32dddfe6c1c9c`.
  Every row states the classification task and each pair has one clean/one
  corrupted condition in one group/split. It is project-authored
  MIT-repository content with limited external validity; it cannot alone
  support D3.
- Digest contract: content is the exact raw UTF-8/LF JSONL bytes; split is
  compact canonical JSON `{schema, rows}` with fields
  `row_id,group_id,split`, sorted `(group_id,row_id)`; pair is compact canonical
  JSON `{schema, pairs}` with fields
  `causal_pair_id,group_id,clean_row_id,corrupted_row_id,split`, sorted by
  `causal_pair_id`; both use `ensure_ascii=true`, separators `(',', ':')`, one
  LF, UTF-8/no BOM. The exact contract is in the plan.
- Tuned lens is explicitly blocked/D0 until a separately provisioned real text
  corpus exists: `Salesforce/wikitext`, config `wikitext-2-raw-v1`, revision
  `f776294184f13b8ff2337b3841cf9269a6216d1e`, CC BY-SA 3.0/GFDL. The bounded
  selection is 8192 train and 2048 validation rows; content/split digests are
  pending authorized acquisition. The authoritative source is
  <https://huggingface.co/datasets/Salesforce/wikitext/tree/f776294184f13b8ff2337b3841cf9269a6216d1e/wikitext-2-raw-v1>.
  It must never fit from the 16 authored rows.
- Layer 6 (native hidden-state index 7), last non-padding prompt position,
  target text ` true`/` false`, max length 32, seeds 17/29/41/53/67, 2,000
  bootstrap replicates, and strength grid 0/0.25/0.5/1.0 are frozen.
- True interchange patching (replace corrupted activation with captured clean
  activation) is distinct from additive `hidden + strength * direction`.
  Direct lens is distinct from tuned lens; tuned requires train-only affine fit
  and holdout-only calibration.
- The checklist contains seven separate real CUDA executions: IG and direct
  lens are support-only; TCAV, tuned lens, disentanglement, true interchange
  patching, and additive steering map to the five ledger records. All use
  `TransformerLMIntegration`; `ModelAdapter=N/A`.
- Metric contract is explicit: `task_margin = logit(target_token) -
  logit(other_class)` in logits; patching reports normalized recovery,
  steering reports paired target effect/selectivity in logits, lens reports
  NLL/KL in nats, and bootstrap aggregation is over independent groups/pairs
  rather than correlated rows. Effects use strict positive lower-CI gates;
  zero-effect gates are not accepted.
- SRP/Rule-of-Three audit: `IntegratedGradients` owns activation-space scalar
  attribution (`src/latent_anything/integrated_gradients.py:23,86,133`), TCAV
  keeps facade/model/statistics responsibilities separated
  (`src/latent_anything/tcav.py:64,569,685,747`), and
  `TransformerLMIntegration` owns the tokenized hidden-state/LM-head boundary
  (`src/latent_anything/integrations/transformer_lm.py:49,303,344,472`). The
  existing VAE `ActivationPatch` and NumPy `SteeringVector` are different
  responsibilities, so no generic L04 protocol or SRP refactor is justified.
- Thresholds and randomized, shuffled, null, off-target, matched-norm, and
  zero-strength controls are in the machine-readable plan.
- Every real model/integration use case must run on the CUDA server. Transport
  is direct authenticated `ssh.exe` to `trietlm@192.168.30.244` from Windows
  PowerShell only; Git Bash and WSL are forbidden. The exact-SHA remote
  `mktemp`/trap cleanup, capture-before-parse, isolated-cache workflow is
  frozen in the plan and was not run here. Each SSH run accepts exactly one
  parameterized use case, saves stdout/stderr and `$LASTEXITCODE`, and requires
  owner review before the next; tuned-lens blocked status is isolated.

## Review gates

- Plan: [`l04-explanations.plan.json`](m14/l04-explanations.plan.json).
- Hash normalization is frozen as: parse UTF-8 JSON; recursively sort object
  keys, preserve array order; omit top-level `plan_sha256`; serialize
  `ensure_ascii=true`, separators `(',', ':')`, one trailing LF, UTF-8 SHA-256
  with no BOM.
- Canonical unsigned plan SHA-256:
  `f3c315e356af0ee54d4196cc365ee22bd997b069d18a3e72c6b479f94e0b3e1a`.
- Sprint/global/M14/evidence-gap references updated; evidence counts and D0/D1
  statuses remain unchanged.
- Required future artifacts: `l04-explanations.json`, its run record, and a
  retained failure record for every failed attempt.
- Before implementation: owner must approve the plan hash normalization and
  exact CUDA host/authentication configuration. After implementation: run the
  evidence validator, relevant plan/contract tests, `mkdocs build --strict`,
  `git diff --check`, and Graphify update/query checkpoints.

## L04.1 gate results

- Plan/fixture structural and digest checks (raw content, canonical split,
  canonical pair, no-leak pair/group invariants): PASS.
- `uv run pytest tests/test_integrated_gradients.py tests/test_tcav.py -q`:
  71 passed, 2 skipped.
- `uv run python scripts/validate_evidence_ledger.py`: honest unchanged
  `33/63 core (52.4%)`, `33/65 overall (50.8%)`.
- `uv run mkdocs build --strict`: PASS.
- `git diff --check`: PASS (only existing LF/CRLF normalization warnings).
- `graphify update .`: PASS; final graph rebuild reported 11,549 nodes and
  22,365 edges (968 aggregated communities).

## Graphify queries used

```text
graphify query "What is the next Sprint79 L04 capability after Sprint79 L03 logit-lens closure, including target records, D0/D1/D2 evidence, dependencies, blockers, real use-case adapters and integrations?"
graphify query "L04 IntegratedGradients TCAV sensitivity activation patch steering logit lens disentanglement source tests artifacts dependencies" --budget 2500
graphify query "task 78.38 gap map L04 queue dependency cycle blocker" --budget 1800
graphify path "IntegratedGradients" "TransformerLMIntegration"
graphify path "TCAV" "TransformerLMIntegration"
graphify explain "THY-T05-CONCEPT-ACTIVATION-VECTORS-TCAV-KIM-ET-AL-2018"
graphify query "L04 design freeze five records TCAV tuned lens disentanglement true interchange additive steering TransformerLMIntegration ModelAdapter N/A" --budget 1800
graphify query "L04.1 design freeze seven real use cases tuned lens blocked D0 causal_pair_id TransformerLMIntegration" --budget 1400
```

The successful path connects the integrated-gradients network test to
`TransformerLMIntegration`; TCAV/exact theory-ID explain queries had no direct
graph node and were treated as an inventory gap rather than invented evidence.

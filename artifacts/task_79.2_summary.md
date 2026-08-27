# Sprint 79 Atomic Task 79.2 — Dependency and model-pin remediation

## Queue remediation

- The 9 prerequisite edges whose IDs are outside the 40-row gap map were
  resolved mechanically against `docs/evidence-ledger.json` effective
  statuses. All 9 are present under `overrides` with status `D2`, therefore
  all are `satisfied_qualifying`; none is missing or unsatisfied.
- The L05 cycle is now one co-scheduled SCC (`SCC-L05-01`) because both rows
  share lane `L05` and artifact `artifacts/m14/l05-density.json`. The queue
  no longer reports an ordering deadlock. The group remains blocked for
  promotion because Normalizing Flows has no stable implementation; the
  shared artifact cannot promote either claim by itself.
- Generator: [`build_sprint79_execution_queue.py`](../scripts/build_sprint79_execution_queue.py)
- Result: [`task_79.1_execution_queue.json`](task_79.1_execution_queue.json)

## GPT-2 pin remediation

The previous smoke's `gpt2@e7da7f221d5bf496a4811970ad59b19a5b3ff2a4` was a
404 because that exact revision does not exist in the canonical repository.
Authoritative Hugging Face sources identify the actual resolution target as
`openai-community/gpt2`, show the commit page at
`https://huggingface.co/openai-community/gpt2/tree/e7da7f221d5bf496a48136c0cd264e630fe9fcc8`,
and report the model-card license as MIT at
`https://huggingface.co/openai-community/gpt2`.

The source, M14 contract, evidence-gap plan, changelog, historical Sprint 39
summary, queue artifact, and regression test now use the explicit canonical
ID and full immutable revision:

`openai-community/gpt2@e7da7f221d5bf496a48136c0cd264e630fe9fcc8`

The prior failure remains preserved in `task_79.1_summary.md`; no evidence
tier was promoted by either smoke.

## Verification before commit

- Regression coverage asserts the canonical model ID and exactly 40-character
  immutable revision.
- `uv run python scripts/build_sprint79_execution_queue.py` regenerates 40
  records and 24 lanes, with 9/9 external prerequisites qualifying and one
  co-scheduled L05 SCC.
- Official source: Hugging Face model repository and pinned commit page above;
  no floating `main` revision, credential, or license acceptance was used.

## Remote CUDA retry

Both invocations were launched from authenticated external PowerShell through
Git Bash using the bundled `remote_cuda_test.sh` workflow. Each runner cloned
and verified the exact pushed source SHA, then used a disposable clone and
isolated `UV_CACHE_DIR`, `TORCH_EXTENSIONS_DIR`, and `CUDA_CACHE_PATH`.

- Source for both invocations: `https://github.com/triet4p/latent-anything.git@ecf63b5021852b6f0d1c2c11a8ba761c5913176a`.
- Server/GPU for both: `trietlm@di-server`; `NVIDIA GeForce RTX 4060 Ti`
  available; Torch `2.10.0+cu128`, CUDA `12.8` available.
- First invocation: the real model loaded and generated successfully, printing
  `cuda_logits=(1, 5, 50257) hidden_states=13`, then failed solely because the
  ad-hoc command asserted the over-specific shape `(1, 8, 50257)`. This was a
  command-level assertion failure, not repository evidence; the focused test
  did not run in that invocation. The run also emitted the unsuppressed
  Transformers warning `torch_dtype` is deprecated; use `dtype` instead. The
  runner terminated through its exit trap; no cleanup error was reported.
- Second invocation: the GPU smoke used the project contract (3D logits,
  vocabulary axis `50257`, 13 hidden states, hidden width `768`) and **PASS**ed.
  The focused network test
  `LATENT_ANYTHING_RUN_NETWORK=1 uv run pytest tests/test_transformer_lm_network.py::test_pinned_checkpoint_generates_expected_shape -m network -q`
  also **PASS**ed (`1 passed in 4.49s`). `Remote CUDA test` reported both GPU
  and broader tests as PASS; cleanup was completed by the exit trap with no
  cleanup error reported.

Neither invocation promoted an evidence tier; the first remains a recorded
command failure and the second is the sole successful retry result.

Status: **PASS-WITH-BLOCKERS — remote retry complete; awaiting owner review**.

# Sprint 79 Atomic Task 79.1 — Execution queue and CUDA preflight

## Scope

This task mechanically reconciles the 40 row-level records from Sprint 78.38
with all 24 normative M14 lanes, then performs the first bounded remote CUDA
preflight and representative real-model smoke. No evidence tier was promoted,
no model license was accepted, and no persistent server checkout was touched.

## Queue result

- Source commit: `df7d5504fd51642507dfbce1593f8758be5954a8`
- Machine-readable queue: [`task_79.1_execution_queue.json`](task_79.1_execution_queue.json)
- Generator: [`build_sprint79_execution_queue.py`](../scripts/build_sprint79_execution_queue.py)
- Gap records: 40 unique records, all retained exactly once.
- M14 lanes: all 24 (`L01`–`L24`), with 14 lanes mapped to the current gap
  records and 10 lanes having no row-level gap record in this map because they
  are covered by the separate M14 real-system matrix.
- Dependency reconciliation found 9 prerequisite edges to IDs outside the
  gap map. Task 79.2 resolved all 9 against the authoritative ledger as D2
  (`satisfied_qualifying`); they are not silently treated as gap-map rows.
- The two-record L05 cycle (`THY-T03-NORMALIZING-FLOWS` ↔
  `THY-T04-DENSITY-ESTIMATION-TRONG-LATENT`) is represented as one
  co-scheduled strongly connected group because both records share the L05
  lane and artifact. This removes ordering deadlock but retains the explicit
  blocker that Normalizing Flows has no stable implementation.
- Recommended next lane: `L01` local bounded D2, after owner review of the
  queue conflicts.

## Remote workflow provenance

The bundled `.agents/skills/remote-cuda-test/scripts/remote_cuda_test.sh` was
launched from the authenticated external PowerShell through Git Bash
(`F:\Git\bin\bash.exe`). The runner pushed and cloned the exact source SHA,
used a disposable server clone under `/tmp/remote-cuda-test.*`, isolated uv,
Torch, and CUDA caches, and installed no system packages. Its exit trap was
used for cleanup; no cleanup error was reported.

### Attempt 1 — GPT-2 pinned CUDA smoke (blocked)

- Command: `LATENT_ANYTHING_RUN_NETWORK=1 uv run python -c "import torch; from latent_anything.integrations.transformer_lm import TransformerLMIntegration, TransformerGenerationRequest; assert torch.cuda.is_available(); print(\"torch=\", torch.__version__, \"cuda=\", torch.version.cuda, \"device=\", torch.cuda.get_device_name(0)); p=TransformerLMIntegration(device=\"cuda\"); r=p.generate(TransformerGenerationRequest(prompt=\"The capital of France is\", max_length=8, seed=42, capture_hidden_states=True)); assert r.logits.shape == (1, 8, 50257); assert all(h.values.shape[-1] == 768 for h in r.hidden_states); print(\"cuda_logits=\", r.logits.shape, \"hidden_states=\", len(r.hidden_states))"`
- Test: `LATENT_ANYTHING_RUN_NETWORK=1 uv run pytest tests/test_transformer_lm_network.py::test_pinned_checkpoint_generates_expected_shape -m network -q`
- Preflight: Torch CUDA was available on `NVIDIA GeForce RTX 4060 Ti`, with
  `torch 2.10.0+cu128` and CUDA `12.8`.
- Result: **BLOCKED** before model construction. Hugging Face returned HTTP
  404 `RevisionNotFoundError` for the normative
  `gpt2@e7da7f221d5bf496a4811970ad59b19a5b3ff2a4` revision (the request was
  resolved through `openai-community/gpt2`). The pinned revision must be
  reconciled by the owner before L03/L04/L06/L11 can be promoted.
- This attempt is retained as a failure; no alternate GPT-2 revision was
  substituted.

### Attempt 2 — Diffusers VAE CUDA smoke (pass)

- Model: `stabilityai/sd-vae-ft-mse@31f26fdeee1355a5c34592e401dd41e45d25a493`
- Command: `LATENT_ANYTHING_RUN_NETWORK=1 uv run python -c "import torch, numpy as np; from latent_anything.integrations.diffusers_vae import DiffusersAutoencoderKLAdapter; assert torch.cuda.is_available(); print(\"torch=\", torch.__version__, \"cuda=\", torch.version.cuda, \"device=\", torch.cuda.get_device_name(0)); a=DiffusersAutoencoderKLAdapter(\"stabilityai/sd-vae-ft-mse\", \"31f26fdeee1355a5c34592e401dd41e45d25a493\", device=\"cuda\"); x=np.zeros((1,3,32,32), dtype=np.float32); z=a.encode(x); y=a.decode(z); assert y.shape == x.shape; assert np.isfinite(y).all(); print(\"vae_latent=\", z.shape, \"decoded=\", y.shape)"`
- Focused test: `LATENT_ANYTHING_RUN_NETWORK=1 uv run pytest tests/test_diffusers_vae_network.py::test_pinned_checkpoint_acquires_or_loads_from_cache -m network -q`
- Result: **PASS**. CUDA forward/decode returned latent `(1, 4, 4, 4)` and
  decoded `(1, 3, 32, 32)`; focused test `1 passed` in `3.01s`.
- Server: `trietlm@di-server`; GPU: `NVIDIA GeForce RTX 4060 Ti`; Torch:
  `2.10.0+cu128`; CUDA runtime: `12.8`.
- The runner reported an unauthenticated Hugging Face request warning and a
  Diffusers float-casting warning; neither was suppressed. No credentials or
  licenses were accepted. The model was used only for this smoke and no
  evidence ledger row was promoted.

## Local verification

The queue generator validates the required fields, unique IDs, exact lane
coverage, internal dependency cycles, and external prerequisites. The next
owner review must decide how to resolve the dependency conflicts and the
invalid GPT-2 revision before the next execution step.

`uv run ruff check scripts/build_sprint79_execution_queue.py`, Ruff format
check, `uv run pyright scripts/build_sprint79_execution_queue.py`, JSON
integrity assertions, and `uv run mkdocs build --strict` all passed. Graphify
was updated after the changes: 11,143 nodes, 21,270 edges, 940 communities;
its expected zero-node warnings for non-AST JSON artifacts remain visible.

Status: **PASS-WITH-BLOCKERS — awaiting owner review**.

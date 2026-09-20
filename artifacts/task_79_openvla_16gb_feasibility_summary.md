# Sprint 79 OpenVLA 16 GB Feasibility — D0 Blocker

## Decision

`THY-X01-OPENVLA` remains D0. The canonical pinned fine-tuned checkpoint must not be downloaded or constructed on the authorized RTX 4060 Ti 16 GB host: the BF16 weights floor is already larger than currently free VRAM, and an otherwise idle card would leave too little runtime headroom. No avoidable OOM was triggered.

## Pinned upstream facts

- Checkpoint: `openvla/openvla-7b-finetuned-libero-spatial@962318cec55ac10993ff0f5f43eda9a270b4c873`.
- Architecture: `OpenVLAForActionPrediction`.
- Hub API: `7,541,237,184` BF16 parameters and `15,082,474,368` safetensors bytes (`~14.05 GiB` weights).
- Official loader: `AutoModelForVision2Seq.from_pretrained(..., torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).to(cuda)` with FlashAttention 2 in the upstream utility.
- Upstream model card records 8 × A100 80 GB for LIBERO-Spatial fine-tuning; the upstream README uses A100 80 GB as the reference profile for fine-tuning and warns that Llama-2-derived pretrained weights inherit the Llama Community License.

Sources: the Hub API, pinned model card, official OpenVLA README, and official `openvla_utils.py` are listed in `artifacts/m14/l19-openvla-16gb-feasibility.json`.

## Remote measurements

Probe transport was native Windows OpenSSH `ssh.exe` invoked from `F:/Git/bin/bash.exe`, targeting `trietlm@192.168.30.244` (`di-server`). `nvidia-smi` reported:

- NVIDIA GeForce RTX 4060 Ti; driver `580.126.20`.
- `16,380 MiB` total, `12,195 MiB` free, `3,754 MiB` used.
- PID `2095304`, `/usr/local/lib/ollama/llama-server`, used `3,576 MiB`.

The project runtime was checked in a disposable clone of GitHub origin at exact SHA `74540ad4e894c68bae46673eede7bc62293016de` on branch `sprint79-local-gate-remediation`, with isolated `UV_CACHE_DIR` and `TORCH_EXTENSIONS_DIR`. `uv sync --extra transformers --no-install-project` completed; `uv run` reported `torch 2.10.0+cu128`, CUDA available, the same RTX 4060 Ti, and `torch.cuda.mem_get_info()` free/total `16,410,673,152 / 16,722,296,832` bytes. The shell exit trap removed `/tmp/openvla-feasibility.Gz4r2n` and its caches; no persistent checkout was mutated. A post-probe `nvidia-smi` showed `170 MiB` used and `15,779 MiB` free after the earlier Ollama process exited; even this idle state leaves only ~`1.53 GiB` above the BF16 weights floor by the torch runtime total.

## Arithmetic and blocker

- Current free VRAM minus BF16 weights: `-2,295,090,048` bytes (`-2.137 GiB`).
- Hypothetically idle card total minus BF16 weights: `2,093,200,512` bytes (`1.949 GiB`).
- The residual cannot defensibly cover CUDA context, vision/language activations, processor inputs, action-token generation, and allocator/workspace overhead. A 4-bit/8-bit run would be a separate, owner-approved non-canonical contract and cannot promote this BF16 row.
- Recommended unblock: a CUDA host with at least 24 GiB VRAM for a margin-bearing attempt; A100 40/80 GB matches the upstream reference profile. The production adapter/capture implementation and 500-trial paired LIBERO-Spatial intervention evaluation remain outstanding.

## Repository updates and validation

- Updated the L19 config, queue, gap map, and `docs/EVIDENCE_GAP_PLAN.md` with the measured blocker and receipt link.
- Evidence arithmetic remains `41/63` core and `41/65` overall; `uv run python scripts/validate_evidence_ledger.py` reports no errors.
- Modified JSON files parse successfully.
- No line-596+ plan item was started; no evidence tier was promoted.

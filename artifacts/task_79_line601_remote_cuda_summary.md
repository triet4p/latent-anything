# Sprint 79 line 601 — remote CUDA verification

## Scope and source of truth

This report closes only `docs/sprint-plans/sprint-79.md:601`. The source of
truth was pushed branch `sprint79-local-gate-remediation` at exact commit
`bb06db45db1fc82a72fb2d36524a707635e60d5d`; local and
`origin/sprint79-local-gate-remediation` matched and the worktree was clean
before every run. Source URL:
`https://github.com/triet4p/latent-anything.git`.

All runs used the bundled
`.agents/skills/remote-cuda-test/scripts/remote_cuda_test.sh`, launched from
native Git Bash (`F:/Git/bin/bash.exe`) with the authorized native Windows
OpenSSH executable (`C:/Windows/System32/OpenSSH/ssh.exe`). Each invocation
pushed/verified the branch, cloned to a new `/tmp/remote-cuda-test.XXXXXX`
checkout, checked clone SHA before dependency setup, used temporary `UV_CACHE_DIR`,
`TORCH_EXTENSIONS_DIR`, and `CUDA_CACHE_PATH`, and installed only in the
throwaway environment. No persistent server checkout was read or modified.

## Host and toolchain

- Server: `trietlm@192.168.30.244` (`di-server`), Linux
  `6.8.0-107-generic`, x86_64.
- GPU: NVIDIA GeForce RTX 4060 Ti, 16,380 MiB; driver `580.126.20`.
- `nvidia-smi`: PASS; PyTorch CUDA availability: PASS.
- Python: CPython `3.13.12`; PyTorch `2.10.0+cu128`; CUDA runtime `12.8`.
- Transformers lane: `4.57.6`; LeRobot lanes: `0.6.1`.
- Compiler selected by the runner: `CC=gcc-12`, `CXX=g++-12`,
  `CUDAHOSTCXX=g++-12`; host compiler `12.3.0`. `nvcc` was not installed;
  no CUDA-extension build was required by these lanes.
- Host probe command: `uname -a; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader; gcc-12 --version; g++-12 --version; uv --version` (`uv 0.12.6`).

## Executed lanes

### Focused TransformerLM / L03–L11 GPU integration

- Temporary clone: `/tmp/remote-cuda-test.TefJ0g/repo`.
- Setup: `uv sync --extra transformers`.
- GPU command:
  `LATENT_ANYTHING_RUN_NETWORK=1 LATENT_ANYTHING_NETWORK_DEVICE=cuda uv run pytest tests/test_transformer_lm_network.py -m network -q`
- Result: **PASS — 8 passed, 5 deselected in 20.88s**. This is a
  current-HEAD exact-SHA check of the pinned GPT-2 integration, numerical
  parity/intervention behavior, and hook cleanup.
- Broader supplementary command (not a replacement for local coverage):
  `LATENT_ANYTHING_RUN_NETWORK=0 uv run pytest tests/test_transformer_lm.py tests/test_m14_l03_analysis.py tests/test_m14_validation_contract.py -q`
- Broader result: **PASS — 67 passed in 11.88s**.
- Runner wall duration: **228.89s**. Clone SHA guard: PASS. Cleanup audit:
  **PASS**, no `/tmp/remote-cuda-test.*` directory remained.

### M14 L20 Diffusion Policy pinned lane

- Temporary clone: `/tmp/remote-cuda-test.7sG9t6/repo`.
- Setup: `uv sync --extra lerobot-diffusion` (LeRobot `0.6.1`, pinned
  dependency profile resolved in the disposable clone).
- GPU command:
  `LATENT_ANYTHING_RUN_NETWORK=1 LATENT_ANYTHING_NETWORK_DEVICE=cuda uv run pytest tests/test_lerobot_diffusion.py::test_pinned_public_diffusion_checkpoint_pair_loads_through_lerobot_factories -m network -q`
- Result: **SKIP — 1 skipped in 3.79s**; no checkpoint semantic execution
  occurred. This preserves the authoritative L20 blocker (upstream model/data
  access and license capture are not available); a skip is not promoted to a
  pass.
- Broader supplementary command:
  `LATENT_ANYTHING_RUN_NETWORK=0 uv run pytest tests/test_lerobot_diffusion.py -q`
- Broader result: **PASS/retained skip — 8 passed, 1 skipped in 1.66s**;
  the network checkpoint test remained skipped. No local test coverage was
  replaced.
- Runner wall duration: **199.66s**. Clone SHA guard: PASS. Cleanup audit:
  **PASS**, no `/tmp/remote-cuda-test.*` directory remained.

### M14 L21 SmolVLA pinned GPU intervention lane

- Temporary clone: `/tmp/remote-cuda-test.8jXLTe/repo`.
- Setup: `uv sync --extra lerobot-smolvla` (LeRobot `0.6.1` and the Linux
  LIBERO/MuJoCo profile resolved in the disposable clone).
- GPU command:
  `LATENT_ANYTHING_RUN_NETWORK=1 LATENT_ANYTHING_NETWORK_DEVICE=cuda uv run pytest tests/test_lerobot_smolvla.py::test_smolvla_gpu_checkpoint_intervention_lane -m network -q`
- Result: **SKIP — 1 skipped in 3.85s**; no policy checkpoint or intervention
  semantic execution occurred. The existing blocker remains: required model /
  data access and license capture are unavailable for the authoritative lane.
  The RTX 4060 Ti capacity was recorded; no workload known to exceed 16 GB was
  started.
- Broader supplementary command:
  `LATENT_ANYTHING_RUN_NETWORK=0 uv run pytest tests/test_lerobot_smolvla.py -q`
- Broader result: **PASS/retained skip — 11 passed, 1 skipped in 3.63s**;
  the network checkpoint test remained skipped. No local test coverage was
  replaced.
- Runner wall duration: **350.34s**. Clone SHA guard: PASS. Cleanup audit:
  **PASS**, no `/tmp/remote-cuda-test.*` directory remained.

## Reconciled lane inventory

| M14 lane | Line-601 disposition | Basis |
|---|---|---|
| L03 analysis | **PASS current HEAD** | TransformerLM CUDA run above; prior accepted immutable L03 artifacts remain unchanged. |
| L04 explanations | **Not rerun** | Frozen seven-use-case history and owner-authorized direct `ssh.exe` receipts are already exhausted/retained; no concrete gap in line 601 justified rerunning immutable semantic executions. L04's explicit transport exception remains authoritative. |
| L10 conditional diffusion | **Blocked** | Pinned SD 1.5 lane is high-VRAM and still lacks license-card/access prerequisites; not started on 16 GB. |
| L11 TransformerLogitTarget | **PASS current HEAD** | Same exact-HEAD TransformerLM CUDA integration run; historical accepted artifacts were not rewritten. |
| L12 I-JEPA | **Blocked** | Missing checkpoint provisioning and model-card license/access; no download started. |
| L17 3DGS | **Blocked** | No trustworthy named checkpoint, revision/hash, or license/access metadata; no unnamed substitute started. |
| L18 LeRobot dataset | **Outside CUDA scope / blocked** | Dataset inspection is a Linux/LeRobot data lane, not a CUDA lane; authoritative receipt still requires the missing owner-authorized dataset/license capture. |
| L19 OpenVLA | **Hardware-excluded historical row** | Canonical OpenVLA BF16 is not rerun on 16 GB and is removed from active CUDA/release execution; retained feasibility receipts authorize no support claim. |
| L20 Diffusion Policy | **SKIP/blocker retained** | Exact-HEAD CUDA preflight succeeded, but pinned checkpoint test skipped; no semantic pass claimed. |
| L21 SmolVLA | **SKIP/blocker retained** | Exact-HEAD CUDA preflight succeeded, but pinned intervention test skipped; no semantic pass claimed. |

Other M14 rows are local CPU/offline lanes and are intentionally kept separate
from this remote-CUDA report. Existing accepted remote artifacts (including L03,
L04, and L11 historical receipts) were not rewritten or falsely remeasured.

## Cleanup and transcript evidence

Independent post-run audits used native OpenSSH and
`find /tmp -maxdepth 1 -type d -name "remote-cuda-test.*" -print`; each returned
no path and was recorded as `CLEANUP_PASS`. The cleanup audit SHA-256 was
`ace5a03207f34ae0f28c403fbe51b01f14db91fb9ec8cbfe4030e885203e81bf` for each
one-line cleanup marker. Local quarantine transcripts (not repository source)
were retained only for report construction:

- Transformer runner: `/tmp/sprint79_line601_transformer.log`, SHA-256
  `819dd7caca1dfb51bd5df707cdb8887502c987c3c3f5672bae11f945460a6608`.
- L20 runner: `/tmp/sprint79_line601_l20.log`, SHA-256
  `6e09261a158e09576e182bf88b3c336505252ccc1c36f641c52e4987b4f7b0a4`.
- L21 runner: `/tmp/sprint79_line601_l21.log`, SHA-256
  `5569985f48dfa3e788dc34875995930472c385147d65934463909946e5c7dd7d`.
- Host probe: `/tmp/sprint79_line601_host_probe.log`, SHA-256
  `e442fe6a53f78c73496cfd00fe36e717bdde5626c2b5357dd2286f0c31026553`.

The `/tmp/remote-cuda-test.*` checkout, uv cache, Torch extension cache, CUDA
cache, and generated outputs were removed by each runner trap. No generated
artifact was copied into a persistent server checkout.

## Conclusion

Line 601's remote-CUDA process/evidence contract is satisfied for every
currently executable 16-GB-compatible pinned lane: L03/L11 passed at the exact
current SHA; L20/L21 were actually attempted and truthfully retained as skips;
all other remote candidates have concrete hardware, checkpoint, license/access,
or non-goal blockers. Remote CUDA was supplementary and did not replace missing
local tests. Line 602 **may begin** only after this report is reviewed; no line
602 work was started here.

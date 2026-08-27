# Sprint 79 Transformer Hook Remote Verification — Attempt 2

## Result

**PASS — 8 passed, 5 deselected.** The requested network selector ran once on
the exact pushed commit with strict CUDA selection. No retry or tuning was
performed.

## Source and command

- Source: `https://github.com/triet4p/latent-anything.git`
- Expected and printed clone SHA: `9ebecfaee3e6e24f4ed56fae728b43828ab614d3`
- Remote ref: `main`
- Test command: `LATENT_ANYTHING_RUN_NETWORK=1 LATENT_ANYTHING_NETWORK_DEVICE=cuda uv run pytest tests/test_transformer_lm_network.py -m network -q`
- Clone: fresh isolated `/tmp/remote-cuda-test.rCkgSr/repo`
- Runner SHA guard: PASS before dependency setup and testing

## Exact test result

```text
........                                                                 [100%]
8 passed, 5 deselected in 19.08s
```

The passing set includes the corrected intervention oracle
(`native hidden_states[6]` unchanged and `[7]` changed) and
`test_hook_cleanup_after_intervention`.

## CUDA/runtime evidence

- Host: `di-server`
- GPU: NVIDIA GeForce RTX 4060 Ti (runner status: available)
- Torch CUDA preflight: available
- Strict selector: `LATENT_ANYTHING_NETWORK_DEVICE=cuda`
- Installed runtime: CPython 3.13.12, torch 2.10.0, Transformers 4.57.6,
  CUDA 12.8 runtime packages
- The test fixtures passed `_network_device()` into
  `TransformerLMIntegration`; strict `cuda` mode requires
  `torch.cuda.is_available()`, and the integration/backend moves inputs and
  model to the selected device. The bundled runner did not emit peak-memory
  telemetry, so this record does not claim a measured allocation value.

## Cleanup and provenance

- Post-run SSH audit: PASS; no `/tmp/remote-cuda-test.*` directory remained.
- Post-run process audit: PASS; no exact `pytest` or `uv` process remained.
- Raw local transcript: `C:\Users\admin\AppData\Local\Temp\task_79_transformer_hook_remote_attempt2.log`
- Transcript bytes: 5056
- Transcript SHA-256: `19E3AE7FFD03D39EA94F485EB496BF054626FF0F7E485DACC1E7E58B0D9369D3`
- Cleanup audit transcript: `C:\Users\admin\AppData\Local\Temp\task_79_transformer_hook_remote_attempt2_cleanup.log`
- Cleanup audit bytes: 44
- Cleanup audit SHA-256: `79A5937C6D1E6D0724FE02F8B405027A56AB636A9B4945E63933AE9EF3DADA3D`
- No source, test, threshold, evidence-ledger, queue, or L04 files were
  changed during remote verification.

## Preserved attempt 1

The prior 7/1 failure remains unchanged under:

- `artifacts/task_79_transformer_hook_remote_verification_attempt1.md`
  (SHA-256 `D6AEE9CC1EF8CBB492E32F84A888B85AAAF10A561B6BBB85C5268FBD3BBF5C36`)
- `artifacts/task_79_transformer_hook_remote_verification_attempt1.json`
  (SHA-256 `BEB2DB1E96C6556C6F17C4EF67066759D538C8529810005771487CC3537209EC`)

Both attempt 1 and attempt 2 artifacts are intentionally uncommitted for
owner review.

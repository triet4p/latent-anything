# Task Summary: Sprint 79 L04.6 — DirectLogitLens closure

## Status

The DirectLogitLens real-CUDA use case is closed as successful support-only
validation. The owner-reviewed execution used the exact committed source SHA
`e9573728e4ba165735e847642cbae979dd4fdf5d` and the pinned GPT-2 revision
through the concrete `TransformerLMIntegration` boundary. It produced a
`passed_real_cuda`, evidence-eligible result, but DirectLogitLens is
support-only by the frozen L04 contract: evidence remains D0, no ledger record
was eligible for promotion, and `accepted_record_ids` /
`accepted_gap_ids` are empty.

## Semantic result

The run captured all 13 native hidden-state indices and used transformer block 6
with native hidden-state index 7. Terminal absolute and relative logit parity
were both `0.0`, passing the frozen `<= 1e-6` gates. Held-out
target/non-target selectivity was `0.0009173363650916144`, with 95% CI
`[0.0005931574705755338, 0.0012898929126095027]`, and passed its strict
positive diagnostic. Randomized-target, shuffled-label, target/non-target
finite, and terminal post-`ln_f` parity controls all passed. The run used
seeds `[17, 29, 41, 53, 67]`, an NVIDIA GeForce RTX 4060 Ti, Torch
`2.10.0`, Transformers `4.57.6`, and elapsed `33.24082692805678` s.
Remote disposable clone/cache cleanup passed.

## Remote attempts

- Attempt 1 was transport attempt 1, with no semantic command or envelope
  (`ssh_exit=1`, preflight not reached). The setup guard incorrectly
  compared the raw plan-file SHA-256 with the canonical plan object digest and
  stopped before CUDA/model execution. This is retained as a setup failure,
  not a semantic failure or model rerun.
- Attempt 2 was transport attempt 2, semantic execution ordinal 1, and
  semantic envelope attempt 1. It ran exactly one DirectLogitLens command,
  recovered exactly three envelopes, and passed the marker, exit/status,
  artifact, run, and failure linkage checks.

The attempt-1 raw stdout/stderr captures were independently checked before
deletion: stdout was exactly 60 bytes with SHA-256
`cc684fb8f2fa5c8e035b1d193087a566954308b676dd8e9c86585afcc3d69ae2`;
stderr was exactly 131 bytes with SHA-256
`19da7f84ca4593dbddb43245d993386f25a016e75bc5857a7310180adcb2a894`.
Each exact path was deleted non-recursively and verified absent. The
sanitized attempt-1 audit records the proof while omitting absolute Temp paths.
No prompts, credentials, or disposable remote paths remain in retained
DirectLogitLens evidence.

## Retained evidence

The seven sanitized DirectLogitLens evidence files under `artifacts/m14/`
are retained: attempt-1 partial/run/failure envelopes, attempt-1 setup-failure
audit/exit, and attempt-2 audit/exit. Envelope digests remain unchanged:
artifact `def8843c328e87491acf679682c73e49c52f462cedf8fb1e90032c1538b5c2a6`,
run record `3b237a3f18169f62339477203ace309f80f1ff5f60f4d4eb44a9a3d82d92d311`,
and failure `de10e253329fbfa21a355df43dd78b61daa35f2988fac294892f82f35f17a52d`.
The sanitized audit digests are attempt 1
`d85ff1c8321a516d211814e9500c2a7e8b3444e05c8c7fe24acce9a5903cf6f8` and
attempt 2 `3a47e53a6de4fcdd72bbd4023dd92136e9c626ecb8b0455136fd48bd2f6ac70e`.

## Verification

- Offline L04 contract, artifact/run/failure validators, and DirectLogitLens
  evidence sanitization: pass with zero errors.
- Focused L04 tests: 56 passed. `uv run mkdocs build --strict` and
  `git diff --check` also pass.
- `graphify update .` completed and refreshed the code graph; its routine
  zero-node warning for non-code JSON inputs did not affect the graph update.
- No source/test/frozen-plan edits, model download, CUDA/SSH rerun, release,
  tag, or version change was made during closure.

# Task Summary: L04.9 v2 D1 retention preparation

## Scope

This task records the owner-authorized real L04.9 v2 Stage A TRAIN-only run on
source SHA `76a45ea74fbb2843b7d109855c2c387ab98b3e47` and prepares its validated
train-selection candidate for a separately authorized Stage B decision. No
holdout, seed, Stage B, semantic promotion/finalization, or remote retry was
performed. The later owner-authorized official retention finalization deleted
only the raw capture; it did not promote or semantically finalize the D1 run.

## Evidence

- One native Windows PowerShell `ssh.exe` process and one remote Stage A CLI
  invocation completed with transport decode, CLI, bundle, final, and cleanup
  markers passing.
- Independent Stage A validation, bundle inspection, source-unique triad
  reopen, and audit validation passed.
- Six-fold train selection unanimously selected layer `10`, offset `0`.
  The directional OOF point estimate was `1.0281549628463458`, lower 95%
  bound `1.0240412630349276`, with 36/36 positive groups and the train gate
  passing. Evidence level is D1 and eligible for owner review only.
- The canonical source-bound candidate is
  `artifacts/m14/l04-explanations.L049V2StageA.76a45ea74fbb2843b7d109855c2c387ab98b3e47.candidate.json`:
  526043 bytes, file SHA-256
  `29bcd20ab494092abbb074bff5d99d091ec288d261a0399f97f2e2fb4f092aa2`,
  canonical artifact digest
  `9159f93401ef9f46af81aba0ef7cb7543756ba0001ac196cb87d4c1f2f7d6572`.
- The sanitized assessment is
  `artifacts/m14/l04-explanations.ssh.L049V2StageA.76a45ea74fbb2843b7d109855c2c387ab98b3e47.d1-assessment.sidecar.json`:
  7447 bytes, file SHA-256
  `735d7fca2a157aaaefdcbb2667b95ff9fd91f6445b70c78cc9cee1e82b790d66`,
  canonical sidecar digest
  `0665dcf5fdd2f02f77b92ce092e79137c05cd27a85bd73c0190306f367dd96ed`.
  Its predecessor pending sidecar digest was
  `237ba264988af961bcba793aec05cc9d8331afddaca55ec2f52204bbbc06d83e`.

## Retention dry-run

The official command is:

```text
uv run python -m scripts.m14_l04_remote_postprocess --finalize-delete --dry-run --raw-capture <source-bound-raw> --source-sha 76a45ea74fbb2843b7d109855c2c387ab98b3e47 --use-case L049V2StageA --artifact-dir artifacts/m14 --audit <source-bound-audit> --fixture artifacts/m14/l04-l049-v2-train.jsonl
```

The dry run passed against the unchanged pending audit. `finalize_delete`
quarantines and deletes only the raw capture, but rewrites the audit lifecycle
state to `deleted_verified`; the source-unique triad and candidate survive.
The dry run passed first. The owner then authorized this exact official
command, which returned `raw_status=deleted_verified` and
`quarantine.status=absent_verified`. Raw local absence is verified; the audit,
source-unique triad, and candidate survived. Remote checkout/cache absence is
not proven by this local operation.

The audit transitioned from the pending 3243-byte SHA-256
`0c81ddedac08d2747d20982f4f2e221183ed9e380504917550b6cdfd680f9d7c` to the
3397-byte `deleted_verified` audit SHA-256
`a1b60ec6804e0468716398c75c9e3508a1c982c0b312fcd8fb1c5aab737e166d`.

## Verification

`validate_stage_a` and the Stage B candidate precondition both pass using only
the repository train fixture and addendum commitments. Focused v2 and remote
postprocess tests pass (`200 passed`); the complete L04-focused suite passes
(`501 passed, 1 skipped`). Graphify was updated after the
documentation/test changes. The five current evidence files are kept
byte-exact where retained; raw is finalized/deleted by the official command,
while audit, triad, and candidate remain present. No commit was made.

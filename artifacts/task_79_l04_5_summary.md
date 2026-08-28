# Task Summary: Sprint 79 L04.5 — TCAV closure

## Status

The TCAV real-model lane is closed as an owner-reviewed semantic failure. The
validated attempt-3 recovery ran the pinned GPT-2 revision through the real
`TransformerLMIntegration` boundary, produced a failed D0/non-eligible result,
and was not rerun or promoted. `accepted_record_ids` and `accepted_gap_ids`
remain empty; no D3 claim or ledger coverage change was made.

The authoritative gap is
`THY-T05-CONCEPT-ACTIVATION-VECTORS-TCAV-KIM-ET-AL-2018`; the M14 record key
is `t05_tcav`. The authored fixture is bounded evidence only and cannot by
itself establish D3.

## Implementation

The Phase A implementation uses the frozen
`openai-community/gpt2@e7da7f221d5bf496a48136c0cd264e630fe9fcc8` revision and
the concrete `TransformerLMIntegration` boundary (`ModelAdapter=N/A`). It
fits the `tone_positive` direction on train groups only, evaluates held-out
groups/pairs at transformer block 6 / native hidden-state index 7, and records
five seeds `[17, 29, 41, 53, 67]`, 2,000 bootstrap replicates, 99 null draws,
five controls, and token IDs `true=2081`, `false=3991`. Sanitized evidence
retains no fixture prompt text, credentials, or disposable paths.

## Remote attempts and outcome

- Attempt 1 used source SHA `5c38b63f01d280939790e415de699ab285a228de`, exited
  SSH `127` with embedded TCAV status `1`, and lost the post-run bundle before
  verified extraction. Its audit explicitly records the raw-deletion policy
  violation (`raw_deleted_after_sanitized_record=false`).
- Attempt 2 used the same source SHA, exited SSH `2` before the TCAV command,
  and retained no semantic result. The raw capture was audited as exactly
  4,741 bytes with SHA-256
  `f80f4c0ded1388867c6dfa3b1c69cdf31fc355622e12202b32b991fadc709fc9` and
  then deleted. Because the raw showed no cleanup marker, both remote cleanup
  and the remote-trap result are recorded as **unverified**, not pass. The
  corrected audit records exact source, exit, size, and hash facts plus
  verified absence after deletion.
- Attempt 3 was transport attempt 3, semantic execution ordinal 2, and
  semantic envelope attempt 1. It ran exactly one TCAV command and recovered
  exactly three envelopes. The approved payload had duplicate preflight,
  transport-ordinal, semantic-ordinal, and envelope-attempt markers; each
  duplicate pair was byte-identical and accepted by owner review. Singleton
  marker and exit/status consistency checks passed; artifact, run, failure,
  and linkage validators all returned zero errors.

Attempt 3 metrics were: held-out accuracy `0.875` (pass), Wilson lower bound
`0.5291118178` against `> 0.55` (fail), bootstrap CI lower `1.0` (pass),
corrected empirical p `0.24` against `<= 0.05` (fail), intervention agreement
`1.0` (pass), and all five controls pass. The semantic result is therefore
`failed`, D0, and evidence-ineligible.

## Retained evidence

The nine sanitized TCAV evidence files are retained under `artifacts/m14/`:
the attempt-1 partial, run, and failure envelopes; attempt-1 audit and exit;
attempt-2 audit and exit; and attempt-3 audit and exit. Attempt-1 and
attempt-3 local raw captures were deleted according to their recorded audit
state. Attempt 2 was deleted only after the exact size/hash match above and
the corrected sanitized audit was complete; a post-delete existence check
confirmed absence.

## Verification

- `uv run python scripts/_m14_l04_validate_tcav.py` and the artifact/run/failure
  validators: pass with zero errors on the recovered attempt-3 envelopes.
- Focused TCAV/L04 tests: pass; no source or test files were changed during
  closure.
- `uv run mkdocs build --strict`: pass.
- `git diff --check`: pass.
- `graphify update .`: pass after final edits.
- No model/network/CUDA execution, release, tag, or version change was made
  during closure; the recorded remote result is the owner-reviewed attempt-3
  recovery described above.

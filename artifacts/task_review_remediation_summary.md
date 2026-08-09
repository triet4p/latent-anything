# Review Remediation Summary

## Scope

Closed the Milestone 11 review findings across benchmark metrics, run-record
identity and artifact safety, LeRobot adapters, dataset selection, recording
provenance, ADR reconciliation, and documentation hygiene.

## Verification

- Focused integration suite: 50 passed, 3 skipped.
- Deterministic Diffusion representation artifact regenerated with flattened
  `action_dim × horizon` metadata (`denoising_dim=8` in the fixture).
- The prior CUDA SmolVLA artifact is explicitly marked historical and
  unverified after remediation; D3 promotion remains deferred until the
  corrected pinned CUDA lane is rerun.

## Contract Changes

- Baseline bit-exactness now requires equal action counts.
- Benchmark latency counts executed model queries and records query steps.
- Run-record inputs are deeply immutable canonical snapshots; unsupported and
  non-finite values are rejected.
- Artifact references are constrained to `artifacts/<digest>` and reads are
  resolved inside the recorder artifact directory.
- Diffusion metadata/noise use flattened dimensions and policy device/dtype.
- SmolVLA intervention directions must match the expert dimension.
- Dataset iteration follows the upstream selected episode order.
- All LeRobot record helpers accept and forward environment and code version.

## Review Follow-up

- The benchmark now consumes an explicit model-execution signal on both the
  hooked and no-hook paths. A regression with `chunk_size=4` and
  `n_action_steps=2` records query steps `(0, 2, 4)` for samples, latency,
  query counts, and query-step metadata.
- Schema-v1 migration accepts only the known Windows serialization
  `artifacts\\<digest>` and canonicalizes it to `artifacts/<digest>`; all other
  paths remain rejected. Full filesystem record loading is covered on Windows.
- The causal-intervention ledger claim is restored to D2. Historical or
  unverified JSON artifacts cannot qualify a D3 entry in the validator.

# Sprint 32 Plan

## Sprint Goal

Add a safe, typed PyTorch activation-capture and intervention lifecycle that can support real transformer, diffusion, and robot-policy adapters.

## Atomic Tasks

Status legend: [ ] pending / [~] in progress / [x] done

- [x] Implement a concrete capture session for named `torch.nn.Module` locations with deterministic ordering and NumPy outputs.
- [x] Implement exception-safe hook registration/removal through a context manager and prove that repeated sessions do not leak hooks.
- [x] Represent layer, call index, batch/sequence axes, device, dtype, and source-model version in capture metadata.
- [x] Add explicit selection errors for missing, duplicate, and shape-changing modules.
- [x] Add one intervention callback path with copy/no-mutation guarantees and gradient-mode controls.
- [x] Validate on two small but structurally different PyTorch models before extracting any broader capture protocol.
- [x] Add lifecycle, device, dtype, nested-module, exception, and concurrency tests.
- [x] Update the evidence ledger, ADR/changelog/artifact, and run the strict gate.

## Notes / Blockers

This is internal infrastructure first. Public promotion waits for real model adapters to prove the vocabulary and result contract.

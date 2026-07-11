# Sprint 32 Plan

## Sprint Goal

Add a safe, typed PyTorch activation-capture and intervention lifecycle that can support real transformer, diffusion, and robot-policy adapters.

## Atomic Tasks

Status legend: [ ] pending / [~] in progress / [x] done

- [ ] Implement a concrete capture session for named `torch.nn.Module` locations with deterministic ordering and NumPy outputs.
- [ ] Implement exception-safe hook registration/removal through a context manager and prove that repeated sessions do not leak hooks.
- [ ] Represent layer, call index, batch/sequence axes, device, dtype, and source-model version in capture metadata.
- [ ] Add explicit selection errors for missing, duplicate, and shape-changing modules.
- [ ] Add one intervention callback path with copy/no-mutation guarantees and gradient-mode controls.
- [ ] Validate on two small but structurally different PyTorch models before extracting any broader capture protocol.
- [ ] Add lifecycle, device, dtype, nested-module, exception, and concurrency tests.
- [ ] Update the evidence ledger, ADR/changelog/artifact, and run the strict gate.

## Notes / Blockers

This is internal infrastructure first. Public promotion waits for real model adapters to prove the vocabulary and result contract.


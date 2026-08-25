# Sprint 34 Plan

## Sprint Goal

Prove the adapter path on a convolutional VAE trained and evaluated on a real image dataset rather than synthetic flat vectors.

## Atomic Tasks

Status legend: [ ] pending / [~] in progress / [x] done

- [x] Select a small redistributable image dataset with labels/factors, deterministic splits, and a CPU smoke subset.
- [x] Implement the concrete convolutional VAE integration with NumPy public I/O and backend-local tensor/device handling.
- [x] Train a reproducible CPU smoke checkpoint and record architecture, seed, dataset revision, and loss configuration.
- [x] Measure reconstruction quality, posterior KL, latent utilization, sample quality proxy, and interpolation validity caveat.
- [x] Route encoded values through the new latent container, two analysis methods, and one manipulation method.
- [x] Add unit tests with a tiny model and a reproducible end-to-end integration test with meaningful held-out thresholds and baseline comparison.
- [x] Produce before/after decoded artifacts plus a failure-case section, not only a best-case plot.
- [x] Apply the adapter Rule of Three check and update evidence/changelog/artifact/gates.

## Notes / Blockers

The existing flat synthetic `VAE` remains a useful test adapter. This sprint adds real-image evidence and should decide later whether both implementations remain public.

## Carryover Closure (2026-08-25)

The original implementation remains delivered, but the meaningful-integration
evidence task is being closed separately and does not touch Sprint 35.

- [x] Task C34.1: Freeze the deterministic held-out split, baseline, thresholds, provenance, and CPU budget.
- [x] Task C34.2: Implement the train-only ConvVAE benchmark, held-out composition path, artifact, and regression test.
- [x] Task C34.3: Reconcile the evidence ledger, changelog, carryover gate, and closure validation after the measured result.

Acceptance is predeclared: an 80/20 deterministic sklearn-digits split; no
held-out fitting; finite held-out reconstruction/KL/utilization; at least 10%
held-out MSE improvement over an all-zero baseline; latent utilization at
least `1e-3`; finite PCA/SAE projections and a unit steering direction on
held-out values; and a recorded CPU runtime with a 30-second advisory budget.
The train-pixel-mean baseline is retained as a stronger diagnostic and is not
required to pass, so the result cannot conceal that limitation.

# Sprint 34 Plan

## Sprint Goal

Prove the adapter path on a convolutional VAE trained and evaluated on a real image dataset rather than synthetic flat vectors.

## Atomic Tasks

Status legend: [ ] pending / [~] in progress / [x] done

- [ ] Select a small redistributable image dataset with labels/factors, deterministic splits, and a CPU smoke subset.
- [ ] Implement the concrete convolutional VAE integration with NumPy public I/O and backend-local tensor/device handling.
- [ ] Train or load a reproducible checkpoint and record architecture, seed, dataset revision, and loss configuration.
- [ ] Measure reconstruction quality, posterior KL, latent utilization, sample quality proxy, and interpolation validity.
- [ ] Route encoded values through the new latent container, at least two analysis methods, and one manipulation method.
- [ ] Add unit tests with a tiny model and a marked end-to-end integration test with quantitative thresholds.
- [ ] Produce before/after decoded artifacts plus a failure-case section, not only a best-case plot.
- [ ] Apply the adapter Rule of Three check and update evidence/changelog/artifact/gates.

## Notes / Blockers

The existing flat synthetic `VAE` remains a useful test adapter. This sprint adds real-image evidence and should decide later whether both implementations remain public.


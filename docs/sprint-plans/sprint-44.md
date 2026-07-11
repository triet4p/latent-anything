# Sprint 44 Plan

## Sprint Goal

Add Integrated Gradients attribution through the real activation/model seam with completeness and baseline-sensitivity checks.

## Atomic Tasks

Status legend: [ ] pending / [~] in progress / [x] done

- [ ] Implement Integrated Gradients for one concrete PyTorch adapter output and selected input/activation target.
- [ ] Define baseline construction explicitly and support a bounded set of integration methods/step counts.
- [ ] Preserve gradients internally while returning typed NumPy attribution results and provenance.
- [ ] Test the completeness property on analytic models and quantify approximation error.
- [ ] Evaluate baseline and step sensitivity, randomization sanity checks, and target specificity.
- [ ] Integrate with capture sessions without leaving hooks or gradients enabled after completion.
- [ ] Produce a real-model attribution artifact with positive and negative examples.
- [ ] Update evidence/ADR/changelog/artifact and gates.

## Notes / Blockers

Attribution magnitude is not causal proof. The benchmark must label it observational and compare it with intervention evidence where possible.


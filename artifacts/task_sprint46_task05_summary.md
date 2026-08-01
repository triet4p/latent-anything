# Task Summary: Sprint 46 Task 05

**Sprint:** Sprint 46
**Task:** Cross-check features with probes, concepts, and causal steering

Added `cross_check_feature` combining a linear-probe check with shuffled-label
control, concept sensitivity (mean gradient · feature direction), and causal
steering/patching agreement on the transformer seam, reusing the
`TransformerLogitTarget` scalar-target convention.

**Testing:** `TestCrossCheck` passed on a tiny linear transformer (agreement 1.0).

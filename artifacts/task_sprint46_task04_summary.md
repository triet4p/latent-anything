# Task Summary: Sprint 46 Task 04

**Sprint:** Sprint 46
**Task:** Rank feature examples and counterexamples

Added `rank_feature_examples` returning top-activating and bottom-activating
example indices with activations and optional per-example labels. Offline
tests show top examples correspond to the true source feature; the marked
GPT-2 test ranks real-token examples with decoded labels.

**Testing:** `TestFeatureRanking` passed offline; network test written (skipped).

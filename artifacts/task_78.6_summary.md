# Task Summary: TCAV statistical SRP refactor (78.6)

**Sprint:** Sprint 78  
**Task:** 78.6  
**Status:** Complete

## Scope and outcome

Extracted TCAV's domain-specific statistical responsibilities into
`src/latent_anything/_tcav_statistics.py`: score construction, bootstrap-seed
aggregation, random-concept controls, confidence intervals, binomial p-values,
Bonferroni correction, significance classification, and `TCAVResult` assembly.
The public `tcav.py` facade remains the orchestrator: it validates/learns the
initial direction, captures model gradients, and delegates assembly. No
generic analysis Protocol or lifecycle change was introduced.

## Files

- `src/latent_anything/tcav.py` — public facade/orchestration and compatibility seams.
- `src/latent_anything/_tcav_statistics.py` — focused statistical/result assembly module.
- `tests/test_tcav.py` — exact parity, property, zero-gradient, single-control, seed, p-value, and serialization coverage.
- `docs/sprint-plans/sprint-78.md` — task status.

## Metrics and responsibility map

Before 78.6: `tcav.py` was 997 LOC, 20 functions, 7 classes, and 3,470 AST
nodes; `compute_tcav` was 218 LOC. After: `tcav.py` is 884 LOC, 19 functions,
7 classes, and 2,769 AST nodes; `compute_tcav` is 111 LOC. The new statistics
module is 161 LOC, 3 functions, and 845 AST nodes; its largest function,
`assemble_tcav_result`, is 123 LOC and has one cohesive statistical concern.

The facade owns initial direction validation, per-example model-gradient
orchestration, and public API compatibility. The statistics module owns all
seed score construction, control permutations/RNG ordering, confidence
intervals, empirical and binomial p-values, multiple-comparison correction,
and immutable result assembly. It uses lazy imports of facade dataclasses and
direction learners at call time to avoid a module import cycle; no generic
Protocol was added.

## Parity and edge evidence

The exact deterministic synthetic fixture remains unchanged: aggregate score
`0.0`, aggregate CI95 `0.0`, five zero random-control scores, empirical
p-value `1.0`, `not_significant`, and intervention agreement `1.0`. Existing
public signature snapshots, error messages, registry behavior, and result
serialization tests remain green. New tests cover bounded binomial p-values,
zero-gradient scoring, one-control standard deviation (`0.0`), class-imbalance
direction data, bootstrap seed aggregation, controls, and result serialization.
The random permutation ordering and classifier seed sequence are unchanged.

## Validation

- Focused TCAV/config/registry/transformer tests: **184 passed, 2 skipped, 19 warnings**.
- Ruff scoped check: **pass**.
- Ruff format scoped check: **4 files already formatted**.
- Strict Pyright scoped check: **0 errors, 0 warnings, 0 informations**.
- Full default pytest: **1513 passed, 36 skipped, 39 warnings in 159.69s**.
- Final `git diff --check`: **pass** (only normal CRLF conversion warnings for dirty tracked files).
- Final graphify topology: **10,271 nodes / 19,895 edges / 917 communities** after the artifact update. Graphify reported the known 50 non-code JSON files with zero AST nodes; no source extraction failure occurred.

The warnings are existing registry deprecations, one sklearn convergence
warning, and UMAP random-state warnings. No model download, network
validation, remote CUDA, commit, or push was performed.

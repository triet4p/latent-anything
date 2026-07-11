# RFC 0001 — Semantic API Vocabulary

**Status:** Accepted for implementation beginning in Sprint 31

## Decision

The stable vocabulary is behavior-based:

| Current beta term | Canonical term | Meaning |
| --- | --- | --- |
| `Method` / `method_a` / Layer A | `AnalysisMethod` / `analysis` / analysis | Observes, measures, reduces, probes, attributes, clusters, or otherwise describes a representation. |
| `BMethod` / `method_b` / Layer B | `Intervention` / `intervention` / intervention | Deliberately transforms a latent or model-mediated representation; it may return data space. |
| Layer C | runtime | Executes, batches, caches, streams, profiles, or records work. |
| `ModelAdapter` / `adapter` | `ModelAdapter` / `adapter` | Bridges a concrete model/backend to framework-visible latent values. |

`AnalysisPipeline` remains a behavior-based name. `ManipulationPipeline` will
become `InterventionPipeline`, with the existing name retained as a compatibility
alias through the beta window.

## Why these names

PCA, UMAP, SAE, linear/nonlinear probes, attributors, and clusterers all answer
an analysis question even though their fitted-state and output shapes differ.
Lerp, steering, activation patching, and later causal edits all change a
representation intentionally; `Intervention` is accurate for latent-to-latent
and model-mediated data-space operations. `Transform` was rejected because it
also describes preprocessing and non-causal analysis projections. `Method` and
the A/B labels were rejected because they reveal roadmap history rather than
user intent. `Executor` was rejected as a top-level family because pipelines,
caches, profilers, and streams do not share one proven execution interface.

## Compatibility contract

| Version window | Public symbols | Registry/config kinds |
| --- | --- | --- |
| `0.1.x` | Current beta names only; snapshot protects the baseline. | `adapter`, `method_a`, `method_b`. |
| `0.2.0`–`0.8.x` | Canonical symbols are primary. `Method`, `BMethod`, and `ManipulationPipeline` remain aliases and issue one `DeprecationWarning` at construction/import boundary where practical. | `analysis` and `intervention` are canonical. Legacy kinds normalize once and emit one migration diagnostic per construction path. |
| `0.9.0` | Deprecated aliases are removed; migration reports identify repository-owned legacy configs before removal. | `method_a` and `method_b` are rejected with the exact canonical replacement. |
| `1.0.0` | Only canonical terminology is documented and supported. | No legacy kinds. |

No behavior changes are permitted while a legacy spelling is normalized. An
ambiguous spelling must fail with a clear migration error, never choose a
registry entry silently.

## Naming rules for future extensions

- Use a singular, behavior-oriented registry kind: `adapter`, `analysis`,
  `intervention`, `planner`, or `runtime`.
- Reserve the first two namespaces for the corresponding built-in families;
  external plugins use a lowercase, hyphenated provider namespace in their
  entry name, not a new generic kind.
- Result types describe their payload (`ProbeResult`, `CaptureResult`,
  `InterventionResult`), never a roadmap layer number.
- Optional integrations stay named for the upstream boundary
  (`integrations.diffusers`, `integrations.transformers`), not for a layer.
- A new family requires concrete working instances under the Rule of Three;
  `planner` and `runtime` are vocabulary reservations, not frozen protocols.

## Examples

```python
# Canonical config from Sprint 31 onward.
analysis = {"kind": "analysis", "name": "pca", "params": {"n_components": 2}}
intervention = {"kind": "intervention", "name": "steering", "params": {}}

# Plugin authors register an analysis capability, not a "method_a" capability.
registry.register("analysis", "provider-feature-probe", FeatureProbe)
```

The current runtime intentionally remains unchanged until Sprint 31 implements
the migration. The snapshot test below makes that later diff explicit.

# Sprint 28 Plan

## Sprint Goal

Choose domain vocabulary for the stable API and define a compatibility-safe migration away from roadmap-layer names such as `Method`, `BMethod`, `method_a`, and `method_b`.

## Atomic Tasks

Status legend: [ ] pending / [~] in progress / [x] done

- [x] Inventory every public class, protocol, registry kind, builder, config field, and documentation term that exposes A/B/C layer naming.
- [x] Compare candidate names against actual behavior, including reducers, probes, attributors, latent transforms, interventions, planners, and executors.
- [x] Write a naming RFC with selected canonical names, rejected alternatives, and examples for users and plugin authors.
- [x] Define alias, warning, configuration-migration, and removal windows through `0.9.0`.
- [x] Specify naming rules for future modules, registry entries, result types, and optional integrations.
- [x] Add API-name snapshot tests for the current beta surface so later migrations are deliberate.
- [x] Log the naming decision as an ADR and update the theory evidence ledger without changing runtime behavior.
- [x] Record the sprint artifact and strict documentation/test gate.

## Notes / Blockers

The likely direction is behavior-based vocabulary rather than generic layer letters, but this sprint must decide from the complete inventory. It must not rename symbols before a compatibility plan exists.

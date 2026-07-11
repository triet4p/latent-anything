# Sprint 78 Plan

## Sprint Goal

Cut `0.9.0`, freeze the intended public API, and remove only beta aliases whose deprecation window has completed.

## Atomic Tasks

Status legend: [ ] pending / [~] in progress / [x] done

- [ ] Inventory top-level exports, submodule imports, protocols, result schemas, registry/config names, extras, CLI commands, and serialization versions.
- [ ] Compare the inventory with Sprint 28 naming/deprecation policy and remove or retain aliases exactly as scheduled.
- [ ] Complete facade/module decompositions for `LatentSpace`, pipelines, and adapters where earlier evidence created stable seams.
- [ ] Add public signature, import-path, config-schema, plugin-contract, and serialized-artifact compatibility snapshots.
- [ ] Review exception taxonomy, docstrings, typing, sync/async symmetry, and optional-extra error messages.
- [ ] Run the theory ledger gate and create explicit issues/plans for every remaining D0/D1 item needed for stable.
- [ ] Publish the `0.9.0` migration guide and API reference; begin the no-unplanned-break freeze.
- [ ] Log the API-freeze ADR and update changelog/artifact/full gates.

## Notes / Blockers

After this sprint, only release-blocking corrections may change public API before `1.0.0`, and each change requires an ADR plus migration update.


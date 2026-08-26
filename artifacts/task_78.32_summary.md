# Sprint 78.32 — Exception, documentation, async, and optional-profile audit

## Verdict

**PASS-WITH-WARNINGS for the audit gates.** No public torch signature,
base-import optional leak, async blocking-I/O violation, exception-taxonomy hole,
or snapshot drift was found. A-1 and A-2 were remediated in 78.33; all public
docstring debt was closed in bounded batches 78.34–78.37. The remaining warning
is the documented owner-decision `Any` inventory.

## Scope and deterministic inventory

- Checked-in API snapshot: 205 runtime exports, 202 canonical-stable entries;
  snapshot SHA `48d64721b73a9d0c9e73da4a41940008c70dfa7841e500bc11bc8dcd22ddf7f6`
  after the 78.33 `compare_probes.linear_config` annotation narrowing.
- Source inventory: 84 non-private public `.py` modules; all 84 have module
  docstrings. Every one of the 205 top-level exports has an object docstring.
- AST public declaration scan: 972 facade/class declarations checked by the
  API inventory; the original broader public-method walk found 182
  undocumented public methods/functions across 26 modules. Atomic task 78.34
  closed the exact first batch of 42 entries in six pose/VAE/Diffusers/renderer
  modules, leaving 140 deterministic findings at that checkpoint. Atomic task
  78.35 closed the exact next batch of 74 RSSM/JEPA/tokenized/reward-value
  entries, leaving 66 deterministic findings at that checkpoint. Atomic task
  78.36 closed the exact next batch of 45 recorder/LeRobot integration entries,
  leaving 21 deterministic findings at that checkpoint. Atomic task 78.37 closed
  the final 21 entries across pipeline/planner/attribution/capture/visualization
  seams; the current public-method scan has zero missing docstrings. These were
  documentation debt, not missing top-level export documentation.
- Public signature scan: 0 signatures contain `torch`; after the 78.33
  concrete-config narrowing, the deterministic ledger records 41 `Any` token
  hits (40 typed `Any` annotations plus one literal `Any` in a default title
  string). Of the 40 typed annotations, 26 are metadata/provenance-justified,
  4 optional-backend-justified, and 10 require owner decision. The literal
  false positive is tagged separately as `signature-text false positive`; no
  internal tensor type crosses the public boundary.
- Roadmap scan found no stale `since 0.2.0`, `deprecated 0.2.0`, TODO, or
  `coming soon` claim. The only implementation note is the intentional,
  tested `MLPProbe.predict` serialization limitation cited below.

## Exception taxonomy and errors

The observed public custom taxonomy is complete against snapshot section L:
seven documented classes — `ArtifactStoreError`, `RecorderContractError`,
`PluginContractError`, `PortableNodeError`, `PortableResultError`,
`DuplicateRunError`, and `DiskCacheError`. Six are `ValueError` descendants
for invalid contracts/data and `DuplicateRunError` is a `RuntimeError` for
lifecycle duplication. No custom class is missing from the snapshot or public
module exports. Existing validation paths use actionable builtin exceptions
(`TypeError`, `KeyError`, `FileNotFoundError`, and `ImportError`) where the
failure is at the Python/input or optional-import boundary.

## Async symmetry

Snapshot section K and live introspection report **9** sync/async pairs. Every
pair has the same normalized signature and expected coroutine/async-generator
status. Analysis, manipulation, rollout, and batch execution all dispatch
blocking work through `asyncio.to_thread`; rollout streaming additionally uses
the settled worker helper so cancellation does not abandon a bounded worker.
The focused runtime/isolation suite passed **25 tests, 2 skipped**, including
async output parity, cancellation, bounded streaming, and cleanup. No direct
blocking filesystem/provider operation was found inside an async public path.

## Optional profile and isolation matrix

The machine-readable snapshot declares the 12 profiles in pyproject order:
`docs`, `diffusers`, `transformers`, `diffusers-full`, `3d`, `lerobot`,
`lerobot-diffusion`, `lerobot-smolvla`, `viz`, `tracking-mlflow`,
`tracking-wandb`, and `tracking`. `uv lock --check` passed. A fresh subprocess
with blockers for Diffusers, Transformers, tokenizers, safetensors, gsplat,
LeRobot, MLflow, W&B, Plotly, Kaleido, ipywidgets, anywidget, and IPython
imported the base package with **no leaked optional modules**. The same blocker
was applied while importing **93 public package modules**; all imported
successfully. Direct `require_optional` probes for all 12 profiles produced the
exact `ImportError` form `Optional backend '<name>' is unavailable. Install
with: uv sync --extra <profile>`.

Profile-specific boundaries are lazy and documented: Diffusers, Transformers,
LeRobot, tracking SDKs, and visualization frontends use `require_optional`;
the 3D gsplat boundary emits its own actionable `RuntimeError` naming
`uv sync --extra 3d`. Existing isolation tests also prove nested dependency
errors are preserved rather than misreported as a missing top-level package.

## Findings

### Advisory A-1 — LeRobot dataset CLI missing-extra message

`src/latent_anything/cli.py:77` previously imported
`lerobot.datasets.LeRobotDatasetMetadata` directly. In a fresh blocked-backend
process this raised raw `ModuleNotFoundError`; 78.33 now routes the root and
dataset imports through the established lazy boundary and preserves provider
dispatch. The exact actionable message is covered by a subprocess test.

### Advisory A-2 — MLPProbe.predict docstring/behavior mismatch

`src/latent_anything/mlp_probe.py:372-391` previously documented `predict` as
returning class labels, although fitted calls deliberately raise
`NotImplementedError` because model-state serialization is not implemented.
78.33 documents the fitted precondition, unsupported status, and exact error;
the behavior remains unchanged and is tested.

### Advisory A-3 — Broad public-method docstring debt — CLOSED

The original 182-entry public-method inventory was closed in four bounded
documentation-only batches (78.34–78.37), with exact source/ledger
reconciliation at zero. No behavior or signatures changed.

### Advisory A-4 — Dynamic `Any` seams

The ledger records 41 exported-signature token hits: 40 typed `Any` annotations
and one deterministic literal-text false positive. The typed set comprises 26
metadata/provenance-justified seams, 4 optional-backend-justified seams, and 10
requiring owner decision. No `torch` appears in a public signature and strict
Pyright is clean. Keep metadata mappings as-is unless a concrete schema exists;
review the 10 owner-decision seams during a future API freeze rather than
introducing speculative protocols.

## Machine-readable finding ledger

[task_78.32_findings.json](task_78.32_findings.json) is the deterministic audit
ledger. It contains **zero** missing public docstring entries (with module,
file, owner, symbol, line, kind, signature, and recommended content category)
and all **41** public-signature `Any` token hits (with import path, annotation,
location, classification, and reason). The ledger records the typed-annotation
reconciliation explicitly: 41 token hits, 40 typed annotations, and one
literal-text false positive.

- Ledger SHA-256 after the 78.37 regeneration:
  `f7aa7513b6079d628d7d35a2e637e21b3502ca610937097bd5617acb166a52b3`.
- Classification counts: metadata/provenance justified **26**;
  optional-backend justified **4**; requires owner decision **10**;
  signature-text false positive **1**.
- Exact validator output: `ledger_valid 0 41 typed_any 40 token_hits 41`.
- Exact source reconciliation: `missing_docstring_scan_exact 0 duplicates 0
  missing 0`.
- A-1 and A-2 are closed by 78.33, A-3 is closed by 78.34–78.37, and A-4 is
  the remaining classified owner-decision inventory represented by the ledger.

Task 78.34 closed 42 entries, task 78.35 closed 74 entries, task 78.36 closed
45 entries, and task 78.37 closed the final 21 entries, all without changing
signatures or runtime behavior. The public docstring inventory is now closed.

## Gates and evidence

- Focused exception/isolation/async/plugin/tracking suite: **25 passed, 2
  skipped**.
- API compatibility/snapshot/CLI suite: **17 passed** after the prior 78.31
  correction.
- Fresh full locked-viz evidence (source/tests unchanged for this audit):
  **1560 passed, 36 skipped, 39 warnings**.
- Prior final static gates remain valid: Ruff PASS, format PASS (256 files),
  strict Pyright PASS (0 errors), snapshot check PASS, and `git diff --check`
  PASS with known LF/CRLF normalization warnings.
- No source, tests, or docs were edited by this audit. Only this artifact and
  the audit-only Sprint 78 plan entry were added.
- Graphify was updated and clustered after the final 78.37 artifact: **11,065
  nodes / 21,168 edges / 943 communities**. The known 51 zero-node JSON sidecars are
  graphify extraction warnings, not source failures.

## Closure decision

The exception/async/optional isolation audit and all public-docstring remediation
are complete. A-1 and A-2 were remediated in 78.33; A-3 was closed by bounded
documentation batches 78.34–78.37; no Blocking finding was identified. A-4
remains the explicitly classified owner-decision advisory for the heterogeneous
typed-`Any` seams; no speculative protocol was introduced.

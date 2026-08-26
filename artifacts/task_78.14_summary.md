# Sprint 78 Atomic Task 78.14 — Experiment Recorder SRP Refactor

Status: complete (internal refactor and test-only snapshots; no changelog entry).

## Responsibility split

- `src/latent_anything/experiment_recorder.py` remains the public facade. It owns `ExperimentRun`/`ExperimentRecorder`, `RecorderArtifact`/`RecorderRunInfo`, stable public wrappers, local recorder/run lifecycle, identity preparation, and the existing `FileSystemRunRecorder` bridge.
- `src/latent_anything/_recorder_contract.py` owns bounded canonical JSON normalization, sensitive-value/key rejection, name/mapping/tag/metric/seed validation, safe artifact reads, reparse/symlink checks, and safe temporary artifact paths.
- `src/latent_anything/integrations/_tracking_common.py` remains the provider-neutral external-run state boundary; MLflow and W&B continue to own SDK calls and do not expose SDK values through the recorder contract.

No recorder Protocol was widened, provider SDK object was leaked, two-phase provider behavior was changed, or security/bounds relaxed. Public classes remain in `latent_anything.experiment_recorder` with stable import/module identity. Local filesystem persistence remains delegated to the existing `FileSystemRunRecorder`; no duplicate storage implementation was introduced.

## Metrics and dependency direction

Baseline `experiment_recorder.py`: 885 LOC / 5,934 AST nodes. `LocalExperimentRecorder` and `LocalExperimentRun` jointly owned persistence-facing orchestration plus contract validation.

After:

| Module | LOC | AST nodes | Main ownership |
| --- | ---: | ---: | --- |
| `experiment_recorder.py` | 658 | 4,021 | public facade, protocols, local lifecycle |
| `_recorder_contract.py` | 305 | 2,446 | canonicalization, validation, safe artifact I/O |

The public facade reduced by 227 LOC and 1,913 AST nodes. The helper has no dependency on provider adapters or SDKs; provider adapters depend only on the facade's stable wrappers and `_tracking_common` state. No new dependency cycle or generic abstraction was introduced.

## Compatibility, security, and parity evidence

- Canonical JSON ordering and recorder identity are stable across equivalent mapping orderings.
- Public protocol/dataclass module identities and `LocalExperimentRecorder.start_run` signature are snapshot-tested.
- Parent/child linkage, resume metric history, all explicit resume-identity mismatch fields, completion/failure lifecycle, immutable parameters, and non-decreasing metric steps remain covered.
- SHA-256 artifact digests, bounded bytes/memoryviews/files, traversal/absolute/Windows/colon/percent/dot path rejection, symlink/reparse checks, hostile sensitive metadata, cycles, depth/entry/string/serialized-size limits, and tamper/error paths remain covered.
- Local, fake MLflow, fake W&B, and tracking-parity flows preserve identity, metrics, child relationships, artifact bytes, and checksums. Optional real-provider lanes remain opt-in/offline and were not broadened.

Tests and gates:

- Focused recorder/run-record/artifact/provider suite: `77 passed, 2 skipped`.
- Full default pytest: `1530 passed, 36 skipped, 39 warnings`.
- Ruff check: pass.
- Ruff format: pass.
- Strict Pyright on recorder, contract, tracking-common, provider, and parity scope: `0 errors, 0 warnings, 0 informations`.
- Final `git diff --check`: pass; only normal Git LF/CRLF conversion warnings were emitted.
- Final graphify: `10,520 nodes / 20,405 edges / 923 communities`; known warning: 50 JSON files produce zero nodes and remain absent from the code graph.

## Review verdict

PASS. No model download, network access, remote CUDA, commit, or push was performed.

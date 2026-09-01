# Sprint 79 L04.9 Phase A — True Activation Patching

## Scope

Implemented the offline-tested, network-gated true clean/corrupted activation
interchange path for `TrueActivationPatching`. The implementation targets the
pinned `openai-community/gpt2` revision through the concrete
`TransformerLMIntegration` boundary; `ModelAdapter` remains `N/A` by design.

Phase A implementation was completed without model download or CUDA execution.
One later owner-authorized v1 real-CUDA attempt reached the pinned model and
produced a raw capture, but its true-interchange recovery gate failed. The
single `--retain` operation also failed closed because the producer serialized
a post-scoring semantic failure as `complete`. The raw capture was preserved
byte-exact during initial retention, then deleted under the explicit owner
exception after its size/hash were verified; the sanitized sidecar records the
deletion and remains non-promoting. No D3 evidence was promoted.

## Contract

- Clean hidden activation replaces the corrupted activation at layer 6 / native
  hidden-state index 7 and target position.
- Separate controls patch adjacent layer 5 at the target token and layer 6 at
  the independently resolved previous valid token; both clean and corrupted
  positions are resolved separately.
- The compatibility control key is `shuffled_direction`; its metadata explicitly
  identifies a deterministic split-preserving non-self shuffled donor
  activation.
- Strength diagnostics are `[0.0, 0.25, 0.5, 1.0]`; strength `1.0` is the true
  interchange acceptance point.
- Margins always use fixed GPT-2 classes `" true" - " false"`.
- Recovery aggregates causal pairs/groups before deterministic 2,000-replicate
  bootstrap. The clean/corrupted denominator is finite and strictly greater
  than `1e-12` in absolute value.
- Train and holdout shuffled donors are independently deranged within their
  split, with exact domain/range/non-self checks; layer and token off-target
  metrics are retained separately before the combined worst-case gate.
- Acceptance remains fail-closed: recovery CI lower bound `> 0.10`, combined
  worst-case held-out off-target absolute effect `<= 0.10`, zero-strength error
  `<= 1e-6`, finite controls, no mutation, and resource budget pass.

## Changed implementation

- Added private handler `scripts/_m14_l04_activation_patching.py`.
- Added fail-closed validator `scripts/_m14_l04_validate_activation_patching.py`.
- Wired dispatcher and artifact promotion for the exact record ID
  `THY-T05-ACTIVATION-PATCHING`.
- Added offline contract tests in `tests/test_m14_l04_activation_patching.py`.

The public VAE `ActivationPatch`, public transformer integration API, and frozen
L04 plan were not changed.

## v1 closure

The attempt used the exact committed source SHA and the frozen plan. It reached
real CUDA with pinned `openai-community/gpt2` through
`TransformerLMIntegration` (`ModelAdapter=N/A`), but failed the fixed positive
recovery gate while the finite off-target, shuffled-donor, zero-strength,
resource, and no-mutation controls were recorded. Retention rejected the
failure envelope's inconsistent stage; the producer now normalizes only fully
evidenced completed scoring failures to the truthful `cleanup` stage, while the
validator remains strict for `failed` plus `complete`. The sanitized sidecar records the exact
raw and bundle/member digests without sensitive execution payloads, records the
owner-exception deletion with `standard_finalize=false`, and keeps
`repository_promotion=false`, evidence level D0, and no accepted IDs. The
historical raw path is now verified absent and is not recoverable.

## v2 preregistration — frozen historical contract

The v2 addendum is immutable and references the frozen parent plan digest
`f3c315e356af0ee54d4196cc365ee22bd997b069d18a3e72c6b479f94e0b3e1a`. It
deny-lists the exposed v1 holdout groups `g09`--`g12` and all exposed v1
artifacts forever. The material metric change is directional recovery
`(patched-corrupted)/(clean-corrupted)` with a finite absolute denominator
greater than `1e-12`.

The new deterministic fixture has 36 public train groups and 24 withheld
holdout groups, two balanced clean/corrupted rows per group, disjoint prompt
families/vocabulary, and no near duplicates. The holdout plaintext and its
256-bit seed were previously outside the repository; the current owner-
authorized provisioning checkpoint copies them byte-exactly into canonical
repository-relative Stage B paths. The retained source-keyed D1 candidate is
the single candidate source of truth; no duplicate candidate copy is used.
Only the holdout plaintext hash
`295ef5f558315c629d68e2d0216567a67163e5ef4adaaf3bbc9fe8a4da96dd5f`, seed
commitment `b8e5e28908c2d2925a5bf5dcc69d852b4e31584f23f0ced2903a70f10d36b5e1`,
and authoring manifest digest
`c63059b05fb45c984bdff2ebd7ecaeee0ff0ca98dab3cc81b845bedb2e1c83c7`, plus
the exact external manifest file hash
`2849b07fd719a0a761f433892fcc031c2ab17012a538daba322dd6fa50674974`, are
recorded. The manifest digest omits only its own `manifest_sha256` field under
the addendum's explicit canonicalization rule. Experimenter authoring is
disclosed; withholding is computational.

Phase A performs six outer folds over the 36 train groups, ranks the fixed
12-layer/three-offset candidate grid on the other 30 groups, requires a
consensus winner in at least four folds, and evaluates 36 out-of-fold groups
with group bootstrap, positive-fold, and 24-positive-group gates. The frozen
model-free power sensitivity result uses 2,000 simulation draws and 2,000
bootstrap resamples per draw. Its digest is
`2102e8bb02e092f4cef5ac5b42290019e2fae3393d4c089fa8e7aa2c494ba431`, power
`0.562`, and accepted false-negative risk `0.43799999999999994`.

The offline train-only Stage A output is an ephemeral D0 protocol fixture,
never a Stage A evidence artifact. It is generated only in temporary test
locations; an actual Stage A artifact is created only by an authorized CUDA
run after review and commitment. The retained source-keyed D1 candidate,
authoring
manifest, holdout fixture, and seed now pass independent commitment and
cross-fixture validation in canonical repository-relative paths. They are
committed at source `7d1e23fdbc385909f964df05360f01027d3b6c35` (tree
`5f43b035a043faf97237cd87aa621bec61c805b1`), and a fresh local clone
revalidated the Git root, tracked status, exact hashes, and preflight before
evaluation. Its repository-derived transport
mode derives these paths only after a fresh checkout. The non-shell
`--require-tracked` preflight checks containment, regular non-symlink status,
tracked-file membership, and exact commitments before the Stage B CLI; payload
guards remain defense-in-depth, and local absolute paths are never exported.
v1's
exact-three-member retention protocol remains unchanged; v2 uses a separate
single-artifact transport/retention boundary. v2 artifacts now carry a fixed,
structurally self-attested runtime transcript bound to the fixture, candidate,
source, and addendum digests; this is self-attestation only. A final D3
promotion additionally requires the independent transport/raw-retention audit,
and synthetic Stage B remains D0/non-promoting. The `--run-real` Stage A and
Stage B paths are CUDA-only, load the pinned GPT-2 boundary, and can emit D1 or
D2 only after real capture/intervention execution and all gates pass; the Stage
B path requires explicit owner-provisioned candidate, holdout fixture, and seed
commitments. Offline protocol fixtures remain D0. The local D3 constructor
reopens retained member envelopes and accepts raw deletion only from a verified
pending/final audit chain; it does not recompute deleted bytes.

## v2 Stage B evaluation assessment

The single owner-authorized Stage B evaluation ran on exact source SHA
`6af20749b305f591d2c90d868cb09e71f623bdd0` after the repository-mode preflight
passed. One native PowerShell `ssh.exe` process launched one Stage B CLI with
the real CUDA/model/integration boundary. Marker order, bundle and triad
reopen, remote and transport cleanup markers, commitment checks, and the
independent Stage B validator all passed.

The result is a bounded D2 evaluation: 24 groups / 48 rows, five seed
summaries, 2,000 bootstrap replicates, and passing recovery and paired-
shuffled gates for every summary. The sanitized assessment records the
recomputed metric ranges, resource provenance, operation counters, and
cleanup attestation without retaining holdout content, seed values, prompts,
labels, external paths, or model free text. The evidence remains
`retained_pending_finalize`; `promotion_candidate=true` is an assessment
signal only. `evidence_eligible=false`, repository promotion, semantic and
standard finalization, and D3 remain false. The earlier provisioning sidecar
is preserved unchanged as the historical pre-run snapshot.

See the [source-bound D2 assessment](m14/l04-explanations.ssh.L049V2StageB.6af20749b305f591d2c90d868cb09e71f623bdd0.d2-assessment.sidecar.json).

## Verification

Focused regression coverage exercises the completed-scoring semantic failure
and validator-clean cleanup triad. Local gates are rerun for this closure;
the v1 real attempt is historical, non-promoting, and was not retried.

The retained SHA-66455 v2 Stage A capture is a complete semantic-gate D0:
candidate layer 10/offset 0 won all six folds, while the OOF point estimate
was `-0.06929667790730794` (lower CI95 `-0.18785552316241794`) with 13/24
positive groups, so the preregistered gate failed. Its historical resource
envelope reported measured CUDA/CPU sources with zero peaks. A separate
canonical sanitized assessment sidecar records this provenance failure as
`measured_source_with_zero_peaks`, and makes no promotion claim. After the
explicit owner exception, all five
historical raw/triplet/audit paths were verified by exact pre-delete size/SHA-256
and deleted individually; the sidecar records
`historical_resource_provenance_invalid_measured_zero_peaks`, the prior
canonical digest, and verified absence. This is not standard finalization and
no evidence was promoted.

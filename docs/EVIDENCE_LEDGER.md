# Evidence Ledger

`docs/evidence-ledger.json` is the machine-readable source of truth for the
1.0 evidence contract. `scripts/validate_evidence_ledger.py` derives one
inventory record from every bold checklist item in `docs/THEORY.md`; it does
not rewrite either file. A topic's canonical ID is
`THY-<tier>-<ASCII-normalized-topic-title>`. Renaming a theory item is an
intentional capability migration: update its ledger key in the same change.

## Evidence levels

| Level | Requirement |
| --- | --- |
| D0 | Theory/research documentation only. It never counts toward stable coverage. |
| D1 | Versioned implementation plus a focused test. |
| D2 | D1 plus a non-trivial benchmark, quantitative acceptance criterion, and reproducible configuration. |
| D3 | D2 plus a reproducible artifact from a real trained or pretrained model. |

Each evidence item is a typed record with a `role` (`source`, `test`,
`benchmark`, `config`, or `artifact`) and a local `path`. The validator checks
that every D1+ item links to versioned local evidence; D2 requires source,
test, benchmark, and config records, while D3 also requires an artifact. D1
requires source and test records.
It deliberately does not download optional models or resolve optional extras.

## Classification and denominator

Every THEORY topic is exactly one of:

- `implementation-applicable`: a framework capability that can eventually be
  implemented or benchmarked. It belongs to the denominator.
- `benchmark-only`: an evaluation claim/control rather than a standalone
  runtime feature. It belongs to the denominator, but can only qualify at D2
  or D3 through a benchmark.
- `contextual-background`: theory, historical model survey, or prerequisite
  knowledge that informs decisions without becoming a product capability. It
  is explicitly excluded only as an `{id: rationale}` ledger record.

The two release percentages are exact:

- **Core coverage**: qualifying (`D2` or `D3`) implementation-applicable and
  benchmark-only topics in T01–T09 (including T03B), divided by all topics in
  those tiers with either classification. Required: **at least 95%**.
- **Overall coverage**: qualifying implementation-applicable and benchmark-only
  topics in every tier, divided by all such topics. Required: **at least 90%**.

The current beta inventory is intentionally below these gates. D1 is useful
evidence but is not a stable-release claim.

## Contract-change evidence

Non-theory API contracts are linked here rather than misclassified as theory
capabilities. [RFC 0001](rfcs/0001-semantic-api-vocabulary.md) and
`tests/test_api_surface.py` define the Sprint 28 semantic-vocabulary baseline;
Sprint 31 will attach migration evidence to this section.

Sprint 57 adds the LeRobot dataset bridge contract. Its evidence is the
bridge source (`src/latent_anything/integrations/lerobot_dataset.py`), focused
offline alignment tests (`tests/test_lerobot_dataset_bridge.py`), the pinned
public metadata inspection (`scripts/lerobot_dataset_inspection.py` and
`artifacts/lerobot_dataset_inspection.json`), and the integration constraints
in `docs/LEROBOT_INTEGRATION.md`.

Sprint 58 extends that contract with observational ACT representation capture.
Its evidence is the ACT adapter and pinned factory loader
(`src/latent_anything/integrations/lerobot_act.py`), focused lifecycle and
lazy-import tests (`tests/test_lerobot_act.py`), the deterministic projection,
probe, trajectory, and control benchmark (`scripts/act_policy_representation_benchmark.py`
and `artifacts/act_policy_representation_benchmark.json`), and the opt-in
public checkpoint smoke marked `network`/`large_download`. This evidence does
not promote a causal-intervention claim; Sprint 61 owns environment-level
effects.

Sprint 59 extends that contract with observational LeRobot Diffusion Policy
capture. Its evidence is the Diffusion adapter and official factory loader
(`src/latent_anything/integrations/lerobot_diffusion.py`), focused offline
axis/queue/parity/analysis tests (`tests/test_lerobot_diffusion.py`), the
deterministic benchmark (`scripts/diffusion_policy_representation_benchmark.py`
and `artifacts/diffusion_policy_representation_benchmark.json`), and the
revision-pinned `lerobot-diffusion` optional profile. The marked public
checkpoint smoke is separate and does not promote causal policy or environment
effects; Sprint 61 owns those claims.

Sprint 60 extends that contract with SmolVLA capture and one bounded
action-expert intervention. Its evidence is the SmolVLA adapter and official
factory loader (`src/latent_anything/integrations/lerobot_smolvla.py`), the
revision-pinned `lerobot-smolvla` optional profile, focused offline
capture/parity/queue/intervention/measurement tests
(`tests/test_lerobot_smolvla.py`), the deterministic intervention benchmark
(`scripts/smolvla_policy_representation_benchmark.py` and
`artifacts/smolvla_policy_representation_benchmark.json`), and a marked CUDA
checkpoint lane for the pinned public pair. The intervention claim is bounded
and observational (strength-controlled additive edits on the action-expert
representation with bit-exact identity at strength zero); it does not promote
environment-level causal effects, which remain Sprint 61 scope.

Sprint 61 records the causal-intervention capability
(`THY-T05-CAUSAL-INTERVENTION-VS-OBSERVATIONAL-STUDY`) at **D2** pending a
corrected pinned CUDA rerun. Its evidence is the four-condition simulation
benchmark protocol and acceptance gate
(`src/latent_anything/integrations/lerobot_benchmark.py`), the offline fixture
suite plus the marked CUDA statistical lane (`tests/test_lerobot_benchmark.py`),
the reproducible evidence script (`scripts/smolvla_simulation_benchmark.py`),
and the frozen `SimulationBenchmarkConfig`. The prior real-model artifact
(`artifacts/smolvla_simulation_benchmark.json`) is retained as historical,
unverified context only and does not qualify as D3 evidence. ACT and Diffusion
have no intervention surface, so their claims remain observational.

Sprint 62 adds the local reproducibility contract without promoting a new
theory topic: the versioned record and atomic recorder
(`src/latent_anything/run_record.py`), LeRobot boundary helpers
(`src/latent_anything/integrations/lerobot_recording.py`), focused contract and
CLI tests (`tests/test_run_record.py` and `tests/test_cli.py`), the inspection
and comparison commands (`src/latent_anything/cli.py` and
`scripts/lerobot_run_comparison.py`), and the pinned ACT-vs-Diffusion
comparison (`artifacts/lerobot_policy_comparison.json`). The Rule-of-Three
decision freezes only the local concrete contract; external tracking backends
remain deferred to Sprint 76.

Sprint 63 promotes the first deterministic latent-transition and rollout
evidence to D2 for `THY-T06-LATENT-TRANSITION-MODEL`,
`THY-T06-ROLLOUT-LATENT-IMAGINATION`, and
`THY-T07-LATENT-IMAGINATION-HORIZON`. The concrete source is
`src/latent_anything/transition.py`, with focused offline tests in
`tests/test_latent_anything/test_transition.py` and the seeded held-out
benchmark/config/artifacts from `scripts/deterministic_transition_benchmark.py`.
This is a flat Euclidean affine-residual baseline only; stochastic and
recurrent transition variants are separately evidenced by Sprints 64–65.

Sprint 66 adds the non-theory pipeline composition contract. Its evidence is
the focused pipeline modules (`src/latent_anything/analysis_pipeline.py`,
`src/latent_anything/manipulation_pipeline.py`, and
`src/latent_anything/rollout_pipeline.py`), the shared metadata contract
(`src/latent_anything/pipeline_contract.py`), focused behavior/config/cache/
async tests (`tests/test_latent_anything/test_rollout_pipeline.py` plus the
existing pipeline/runtime regression suites), and the migration/ownership
documentation in `docs/PIPELINES.md`. This does not promote a new model or
planning claim; it makes the existing latent-transition rollout evidence
available through Pipeline #3.

Sprint 67 promotes the first reward/value evaluation evidence to D2 for
`THY-T07-REWARD-MODEL-TRONG-LATENT` and
`THY-T07-VALUE-FUNCTION-TRONG-LATENT`. The focused NumPy implementation
(`src/latent_anything/reward_value.py`) provides masked discounted returns,
terminal/padding semantics, linear reward scoring, finite-horizon Monte-Carlo
value estimation, held-out calibration, Bellman residuals, and
real-versus-imagined score bias. Its integration evidence is the configured
rollout evaluator and run-record artifact path, the deterministic tests in
`tests/test_reward_value.py`, and the controlled benchmark/config/artifact
from `scripts/reward_value_benchmark.py`. This remains synthetic D2 evidence;
it does not claim a real pretrained world-model or CUDA result.

## Quality gates for a D2/D3 promotion

- Core unit tests: all changed core behavior has deterministic focused tests;
  the full test suite must pass.
- Integration tests: a D2/D3 claim has an offline, version-pinned path; network
  acquisition is a separate marked smoke test.
- Documentation: API, constraints, exact reproduction command, and failure
  cases are linked from the capability entry.
- Explanation validity: headline explanation methods report fidelity, stability,
  a negative/selectivity control, and a causal intervention metric when the
  claim is causal.
- Compatibility: public-name/config snapshots and legacy migration tests pass
  whenever a promoted capability changes a beta surface.

## Validator contract

Run `uv run python scripts/validate_evidence_ledger.py`. It rejects missing,
duplicated, or stale IDs; invalid status/classification; absent D1+ evidence;
malformed typed records; missing level-specific roles; and evidence paths that
do not exist. CI runs the same read-only command before the Python quality gate.

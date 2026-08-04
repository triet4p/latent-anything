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

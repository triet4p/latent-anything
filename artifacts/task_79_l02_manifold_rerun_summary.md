# Sprint 79 theory-gap row — manifold hypothesis rerun

## Selected row

`THY-T01-MANIFOLD-HYPOTHESIS` is queue position 13, the earliest executable row after queue position 11 once queue position 12's named-3DGS checkpoint blocker is excluded. The exact-SHA L02 rerun was executed without overwriting the canonical artifact. Status remains D1; no promotion occurred.

## Retained evidence

The rerun used `scripts/m14_l02_geometry.py` with the frozen L02 plan and a new row-level threshold config. Its retained failure artifact and run receipt are:

- `artifacts/m14/l02-manifold-hypothesis.config.json`
- `artifacts/m14/l02-manifold-hypothesis.rerun-55aec38.json`
- `artifacts/m14/l02-manifold-hypothesis.rerun-55aec38.run.json`

The exact source SHA was `55aec384f109913340693929f55233c30572c952`. The 128 held-out pair-path trials were finite and train-only density fitting was preserved. The real-pair AUC was `0.4560546875` against the strict minimum `0.55`; latent-vs-raw AUC delta was `-0.4124755859375` against the minimum `-0.05`; shuffled-label AUC was `0.46142578125` and passed its control. The two failed target gates require D1 retention. The prior canonical failure artifact remains immutable and linked.

## Validation

```text
uv run pytest tests/test_m14_l02_geometry.py tests/test_latent_anything/test_latent_space.py tests/test_latent_anything/test_geodesic.py -q
143 passed
```

The ledger remains at `errors: []`, with coverage unchanged at 39/63 core and 39/65 overall because this row did not qualify. Positions 2–4 and 6–9 remain blocked, queue position 12 remains blocked by the absent named 3DGS checkpoint, line 595 remains unchecked, and no later plan item was changed.

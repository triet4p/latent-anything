# Sprint 14 Plan

## Sprint Goal (Completed)
Increment thứ mười một (Round 11): thêm **HiddenStateAdapter** — `ModelAdapter` instance #3, mode (ii) **no-explicit-latent**. Adapter này expose hidden-state activations như latent mà không giả định có decoder. Kết thúc: **freeze `ModelAdapter` Protocol** theo Rule of Three, nhưng không ép mọi adapter phải có `decode`.

## Why This Sprint
`ModelAdapter` 3-mode ADR vẫn là ADR pending cuối cùng. Sprint 14 dùng theory tầng 8 và tầng 14: latent có thể là activation/hidden state, không nhất thiết là bottleneck VAE có decoder.

## Atomic Tasks
Status legend: [ ] pending / [~] in progress / [x] done

- [x] Task 1: Implement `HiddenStateAdapter` in `src/latent_anything/adapters/hidden_state.py`.
- [x] Task 2: Keep it lightweight and local: fixed/random numpy MLP or deterministic feature stack; no heavyweight transformer dependency.
- [x] Task 3: `encode(data: np.ndarray) -> np.ndarray` returns hidden activations with shape `(n_samples, hidden_dim)`.
- [x] Task 4: `latent_space` returns Euclidean `LatentSpace(hidden_dim, source_model="hidden_state")` with metadata marking exposure mode `"hidden_state"`.
- [x] Task 5: Do not provide a fake learned decoder. Either omit `decode` or make unsupported decoding explicit through the frozen protocol design.
- [x] Task 6: Freeze adapter protocols in `src/latent_anything/adapters/protocols.py`: a base `ModelAdapter` for `encode` + `latent_space`, and a separate decodable surface if implementation evidence requires it.
- [x] Task 7: Migrate VAE and RandomProjection docstrings/exports to the frozen protocol surface; remove or supersede `_ModelAdapterBase`.
- [x] Task 8: Update `ActivationPatch` typing so it requires a decodable adapter contract, not every `ModelAdapter`.
- [x] Task 9: Add tests for `HiddenStateAdapter`, runtime protocol checks, and `ActivationPatch` rejecting non-decodable adapters cleanly.
- [x] Task 10: Add an end-to-end demo: data → hidden activations → PCA/UMAP visualization; no decode story.
- [x] Task 11: Run `ruff check`, `ruff format`, `pyright`, and full pytest.
- [x] Task 12: ADR check: mode (ii) confirmed; `ModelAdapter` ADR remains pending with modes i and ii confirmed; full validation waits for mode iii in Sprint 16.
- [x] Task 13: Update artifact summary, `CHANGELOG.md`, and `docs/PLAN.md`.

## Rule-of-Three Checkpoint
| Check | Status |
|---|---|
| ModelAdapter instances | VAE (#1), RandomProjection (#2), HiddenStateAdapter (#3) |
| Philosophies differ? | Yes — trained explicit latent, fixed explicit projection, no-explicit hidden activation |
| Rule branch | Instance #3 → freeze protocol and migrate |
| ADR impact | Mode (ii) confirmed; mode (iii) still pending |

## Notes / Blockers
* Do not import `transformers` or a large pretrained model in this sprint.
* The important design point is decoder optionality. A fake decoder would erase the evidence this sprint is supposed to produce.

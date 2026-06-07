# Research note conventions

## Language & audience

- Research `.md` files are written in **Vietnamese** (high-level docs language). Notebooks are English.
- Audience: someone with a **DL + math base** but new to this specific topic. Lead with intuition,
  then formalize. Don't omit the math — clarify it.

## The forbidden patterns (this is what a careful editor removes)

These slipped into early notes from raw LLM output. Never ship them:

1. **Chatbot phrasing / second person at the reader.** Delete or rewrite:
   - "Đúng vậy, bạn đã hiểu chính xác…", "Dưới đây là giải thích chi tiết…", "Bạn có thể hình dung…",
     "để bạn khắc phục", "Nếu bạn muốn…", "phương pháp … của bạn".
   - Write declaratively: "Có thể hình dung…", "Nguyên nhân gốc rễ và cách khắc phục:".
2. **Bare formulas.** Every non-trivial equation is followed by one line defining its symbols and
   stating what it means. A formula dropped in with no explanation is a defect.
3. **Typos / broken LaTeX.** Common offenders seen before: `Geosedic`→`Geodesic`, `tôi ưu`→`tối ưu`,
   `thức đo`→`thước đo`, `vài trò`→`vai trò`, `$\Sigma^-1$`→`$\Sigma^{-1}$`, `khả di biên`→`khả dĩ biên`,
   `tichs phân`→`tích phân`, `$x sinh ra`→`$x$ sinh ra`. Read the rendered math, not just the source.
4. **Non-clickable pointers.** A reference to another note must be a markdown link, not backticked text.

## Required structure

```markdown
# <Title>

> **TL;DR.** <2–3 sentences: what it is · the one key equation/idea · the main caveat.>

<intuition-first body: definition → mechanism (math) → variants → limitations → Liên hệ Latent-Anything>

## Liên quan
- [<Note A>](path) — one-line why it's related.
- ...

## Tham khảo
- Author(s), *Title* (Venue Year, arXiv:XXXX.XXXXX).
```

- **TL;DR** is mandatory and ≤ 3 sentences. It is the scannable summary; a reader who reads only it
  should grasp the gist.
- Use a **comparison table** whenever contrasting two things (explicit vs implicit, VAE vs flow, …).
- Always include a **limitations / "where it breaks"** part — completeness, not salesmanship.

## Link resolution (the part that silently breaks)

`docs_dir = latent-anything-theory`. mkdocs rewrites relative `.md` links to `.html`. Tier folders:
`01-space-representation`, `02-representation-learning`, `03-geometry-structure`, `03b-3d-representation`.

| Target | Write |
|---|---|
| Same-tier note | `[text](NN-name.md)` |
| Cross-tier note | `[text](../../<tier-folder>/research/NN-name.md)` |
| A notebook → its research | `[research/NN-name.md](../research/NN-name.md)` |
| docs/THEORY.md (outside docs_dir) | `https://github.com/triet4p/latent-anything/blob/main/docs/THEORY.md` |

- **Only link to files that already exist** — a link to a future note fails `mkdocs --strict`. Use
  plain bold text for forward references and upgrade later.
- Image refs stay `![alt](../assets/name.png)`; if you fix a misspelled alt, keep the existing
  filename so the path doesn't break.

## References

- Cite primary papers (author, title, venue, year). Add arXiv id only when verified.
- Web-search to confirm any citation you're not certain of. Niche/recent work **must** be verified.

## Verification (always)

From `latent-anything-theory/`:
```
uv run mkdocs build -f ../mkdocs.yml --strict --site-dir ../../.mkdocs-linkcheck && rm -rf ../../.mkdocs-linkcheck
```
Pass bar: exit 0, no broken-link warnings. Then keep docs/THEORY.md checkbox and mkdocs.yml nav in sync.

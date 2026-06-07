---
name: write-research-md
description: >
  Write or revise a theory research note (.md) in the latent-anything-theory project — a new topic
  from the THEORY.md roadmap, or an editorial pass on an existing note. Use whenever the user asks to
  "research <topic>", write/expand a research markdown, or clean up theory notes. Enforces web-search
  grounding (no hallucination), the TL;DR → body → Liên quan → Tham khảo structure, Vietnamese prose
  with zero chatbot phrasing, clickable cross-links that resolve under `mkdocs build --strict`, and
  keeps THEORY.md status + mkdocs.yml nav in sync. Pairs with the create-theory-notebook skill.
---

# write-research-md

Produce a research note that a newcomer **with a DL + math base** can read top-to-bottom and come
away understanding the concept, why it matters, and how it connects to the rest of Latent-Anything —
without ever doubting the facts.

## Before you start

Read the conventions and the skeleton:
→ `references/conventions.md` (style rules, link resolution, citation format, forbidden phrasing)
→ `references/template.md` (the section skeleton to fill in)

The companion skill **create-theory-notebook** turns a finished note into experiments — run it after.

---

## Step 1 — Locate and name the file

Research notes live at `latent-anything-theory/<tier-folder>/research/NN-topic-name.md` (hyphens, lowercase).

| Roadmap tier (docs/THEORY.md) | Folder |
|---|---|
| Tầng 1 — Không gian & biểu diễn | `01-space-representation` |
| Tầng 2 — Học biểu diễn | `02-representation-learning` |
| Tầng 3 — Hình học & cấu trúc | `03-geometry-structure` |
| Tầng 3B — 3D Representation | `03b-3d-representation` |

`NN` = the order within the tier. If the tier folder is new, create `<tier-folder>/research/`.

---

## Step 2 — Research with web search (mandatory)

**Never write from memory alone — this project's hard rule is no hallucination.**

- Web-search the topic before writing; pull the definition, the canonical equations, and the standard
  failure modes / limitations from primary sources.
- **Verify every niche or recent citation** (exact authors, year, venue, arXiv id) with a search.
  Well-known papers (VAE, NeRF, …) can be cited from knowledge, but confirm anything you are < 95% sure of.
- Capture the limitations / "where it breaks" — a note that only sells the idea is incomplete.

---

## Step 3 — Write the note (structure)

Follow `references/template.md`. Mandatory spine:

1. **`# Title`**
2. **`> **TL;DR.**`** — a 2–3 sentence blockquote right under the title: what it is, the one key
   equation/idea, and the main caveat. A newcomer must be able to scan only this and get the gist.
3. **Body** — definition → mechanism (with math) → variants → **limitations** → **Liên hệ với
   Latent-Anything**. Lead with intuition, then formalize. Use a comparison table when contrasting
   (e.g. explicit vs implicit, VAE vs flow).
4. **`## Liên quan`** — bullet cross-links to related notes (see Step 4).
5. **`## Tham khảo`** — references (see Step 4).

Depth bar: enough that a DL-literate reader could re-derive the core result or implement a toy
version. Don't dumb it down — clarify it.

---

## Step 4 — Cross-links and references (must resolve)

**Links are markdown links to the `.md` source; mkdocs rewrites `.md` → `.html`.** Use relative paths:

| Target | From a research note write |
|---|---|
| Same tier | `[text](NN-other.md)` |
| Other tier | `[text](../../<tier-folder>/research/NN-other.md)` |
| docs/THEORY.md (lives **outside** the mkdocs `docs_dir`) | the GitHub URL `https://github.com/triet4p/latent-anything/blob/main/docs/THEORY.md` |

**Only link to files that already exist.** A link to a not-yet-written note breaks `mkdocs --strict`.
For a forward reference, use plain bold text (e.g. **NeRF (mục tiếp theo)**) and upgrade it to a link
when that note is created.

**References format:** `Author(s), *Title* (Venue Year, arXiv:XXXX.XXXXX)`. Add the arXiv id only when
certain. Cite primary papers, not blogs.

**Annotate every formula you "drop in":** after a non-trivial equation, add one line defining each
symbol and what the result means — never leave a bare formula.

---

## Step 5 — Sync the roadmap and nav

- **docs/THEORY.md** — flip the item's checkbox: `[ ]` → `[~]` (research written, notebook pending) →
  `[x]` (research + notebook done). Convert relative dates to absolute if you add any.
- **mkdocs.yml** — add the note under its tier's nav block (create a new `- "N. Title":` block if the
  tier is new). Display titles are short Vietnamese phrases. Paths are relative to `docs_dir`
  (`latent-anything-theory/`), so omit that prefix.

---

## Step 6 — Verify

1. **Links:** from `latent-anything-theory/`, run
   `uv run mkdocs build -f ../mkdocs.yml --strict --site-dir ../../.mkdocs-linkcheck` then delete the
   output dir. **Exit 0 and zero broken-link warnings** is the pass bar. (Add `mkdocs-material` +
   `mkdocs-jupyter` to dev deps if missing.)
2. **YAML:** confirm mkdocs.yml still parses.
3. **Prose:** re-read for the forbidden patterns in `references/conventions.md` (chat phrasing,
   typos, bare formulas, non-clickable pointers).

---

## Output summary

Report: file path created/edited; web sources used for any non-obvious facts; cross-links added;
THEORY.md + mkdocs.yml changes; and the `mkdocs --strict` result.

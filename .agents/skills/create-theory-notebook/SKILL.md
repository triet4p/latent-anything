---
name: create-theory-notebook
description: >
  Create a Jupyter notebook for one or more theory research files in the latent-anything-theory
  project. Use this skill whenever the user asks to create a notebook, add visualizations for a
  theory topic, or turn research notes into interactive experiments. Also triggers when updating
  mkdocs.yml nav after adding new research files or notebooks. The skill enforces naming
  conventions, notebook structure conventions (experiment-first, visualization-heavy, every output
  explained), and keeps mkdocs.yml in sync.
---

# create-theory-notebook

Create a Jupyter notebook that brings one or more theory research files to life through
visual experiments — then wire it into mkdocs.yml.

## Before you start

Read the conventions file for naming rules and cell structure:
→ `references/conventions.md`

For a concrete starting skeleton, consult:
→ `references/notebook_template.ipynb`

---

## Step 1 — Identify source research file(s)

The user will name one or more research files, e.g.:
- `01-information-bottleneck.md` (single)
- `05-geodesic.md, 06-pullback-metric.md, 07-flatvi.md` (combined)

Locate them under `latent-anything-theory/<section>/research/`.
Read all of them in full — the notebook must be grounded in the actual theory, not a generic treatment.

**Determine section directory** from the path (e.g., `02-representation-learning`).

---

## Step 2 — Determine notebook name

Extract the number(s) from the filenames:

| Research files | Notebook name |
|---|---|
| `01-information-bottleneck.md` | `01_information_bottleneck.ipynb` |
| `02-mahalanobis-distance.md` | `02_mahalanobis_distance.ipynb` |
| `05-geodesic.md` + `06-pullback-metric.md` + `07-flatvi.md` | `05_06_07_geodesic_pullback_flatvi.ipynb` |

Rules:
- Numbers joined with `_`
- Topic words joined with `_` (underscores, not hyphens)
- Lowercase throughout
- Output path: `latent-anything-theory/<section>/notebooks/<name>.ipynb`

If a notebook already exists at that path, check whether it follows conventions before overwriting.

---

## Step 3 — Design the experiments

This is the most important step. The notebook must not just repeat the theory — it must let the
reader *see* and *feel* what the equations mean.

For each major concept in the research file(s), design one experiment:
- **What question does this experiment answer?** State it explicitly in a markdown cell.
- **What will the plot show?** Think in terms of geometry, distributions, curves, heatmaps.
- **What is the reader supposed to notice?** Write that as the explanation cell after the plot.

Aim for 3–6 experiments per research file. Combined notebooks can have more.

---

## Step 4 — Write the notebook

Follow the cell structure in `references/conventions.md` exactly.
The template in `references/notebook_template.ipynb` shows the skeleton — adapt it, don't copy blindly.

Key rules (repeated here for emphasis):
- Every code cell that produces output **must** be followed by a markdown explanation cell.
- Plots must have titles, axis labels, and legends. No unlabelled axes.
- Use `np.random.seed(...)` at the top of each experiment for reproducibility.
- All cell text (markdown and code comments) is in **English**.
- Research files are in Vietnamese — the notebook is English.
- Cell IDs must be explicit, kebab-case (e.g., `"id": "exp1-scatter-plot"`).

---

## Step 5 — Update mkdocs.yml

Open `mkdocs.yml` at the repo root. Find the `nav:` section.

**If the section already exists** in nav, add the notebook entry under its `Notebooks:` subsection.

**If the section does not exist** in nav (e.g., `02-representation-learning` is new), add the full section block following the pattern from section 1:

```yaml
- "N. Section Title":
    - "Concept 1": <section>/research/01-concept.md
    - "Concept 2": <section>/research/02-concept.md
    - Notebooks:
        - "Display Title": <section>/notebooks/NN_name.ipynb
```

The display title for notebooks should be a short Vietnamese phrase describing the topic,
matching the style of section 1 (e.g., `"Information Bottleneck"` or `"Nút thắt thông tin"`).

---

## Step 6 — Verify

After writing the notebook:
1. Run `uv run jupyter nbconvert --to notebook --execute --inplace <path>` from the section's project directory to confirm all cells execute without error.
2. If there are errors, fix them before reporting done.
3. Confirm mkdocs.yml is valid YAML (no syntax errors).

---

## Output summary

Report to the user:
- Notebook path created
- Any notebooks renamed (if existing file had wrong naming convention)
- mkdocs.yml changes made (new entries added)
- Any warnings (e.g., research file missing, section not found)

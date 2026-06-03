# Theory Notebook Conventions

## Project layout (relevant paths)

```
latent-anything-theory/
├── mkdocs.yml                            # ← update nav here (at REPO ROOT)
├── 01-space-representation/
│   ├── research/   NN-concept-name.md   # theory source (Vietnamese)
│   ├── notebooks/  NN_concept_name.ipynb # experiment notebook (English)
│   └── assets/     *.png               # images referenced by either
└── 02-representation-learning/
    ├── research/   ...
    └── notebooks/  ...
```

The `mkdocs.yml` is at the **repository root**, not inside `latent-anything-theory/`.

---

## Naming rules

### Research files
- Pattern: `NN-concept-name.md` — hyphens, lowercase
- Examples: `01-metric-space-vector-space.md`, `05-geodesic.md`

### Notebooks
- Pattern: `NN_concept_name.ipynb` — **underscores**, lowercase
- Combined: `NN_MM_PP_topic.ipynb` when one notebook covers multiple research files
- Examples:
  - `02_mahalanobis_distance.ipynb`
  - `05_06_07_geodesic_pullback_flatvi.ipynb`
  - `01_information_bottleneck.ipynb`

### Cell IDs
- Kebab-case, descriptive: `"id": "exp1-setup"`, `"id": "exp2-scatter"`
- Never auto-generated hashes — always explicit strings

---

## Notebook cell structure

A notebook follows this skeleton (not every section is mandatory, adapt to the topic):

```
[markdown] Title + learning objectives
[code]     Imports + global style setup
─── Experiment N ────────────────────────────────────
[markdown] ## Experiment N: <question this answers>
           Brief theory hook (1–3 sentences), linking to research file concept
[code]     Experiment setup (data generation, parameters)
[code]     Visualization code → produces the plot
[markdown] ### What you see
           Explain the output: what pattern, why it appears, what it means
─── ... repeat ──────────────────────────────────────
[markdown] ## Summary / Key Takeaways
           Bullet list: 1 takeaway per experiment
```

### Explanation cell rules (critical)
Every code cell that produces visible output (a plot, printed values, a table) **must** be
followed by a markdown explanation cell. This cell must answer:
1. *What does the output show?* (describe the visual/numbers)
2. *Why does it look this way?* (connect to the theory)
3. *What should the reader take away?* (the insight)

Do not write "as expected" or restate the code. Write what a thoughtful teacher would say
when pointing at the plot.

---

## Plot requirements

Every matplotlib figure must have:
- `ax.set_title(...)` — descriptive, not just the variable name
- `ax.set_xlabel(...)` and `ax.set_ylabel(...)` on every axis
- Legend if more than one series is plotted
- `fig.suptitle(...)` if using subplots
- `np.random.seed(N)` before generating random data (reproducibility)

Preferred colormaps: `RdBu_r`, `YlOrRd`, `viridis`, `plasma`, `coolwarm`.
Set `plt.rcParams` once in the imports cell for consistent style across the notebook.

---

## Language policy

| Content | Language |
|---|---|
| Research `.md` files | **Vietnamese** |
| Notebook markdown cells | **English** |
| Notebook code + comments | **English** |
| MkDocs nav display titles | **Vietnamese** (short phrase) |

---

## Imports cell template

```python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
# add domain-specific imports here (scipy, sklearn, etc.)
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
plt.rcParams.update({
    'figure.dpi': 110,
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'figure.facecolor': 'white',
    'axes.facecolor': '#f8f9fa',
    'axes.grid': True,
    'grid.alpha': 0.35,
})
print('Setup complete.')
```

---

## mkdocs.yml nav pattern

Section 1 shows the canonical pattern to follow:

```yaml
nav:
  - Trang chủ: README.md
  - "1. Biểu diễn không gian":
      - "Metric Space & Vector Space": 01-space-representation/research/01-metric-space-vector-space.md
      - "Khoảng cách Mahalanobis": 01-space-representation/research/02-mahalanobis-distance.md
      # ... more research files ...
      - Notebooks:
          - "Hiệp phương sai & Mahalanobis": 01-space-representation/notebooks/02_covariance_and_mahalanobis_exercise.ipynb
          - "Đa tạp & Lời nguyền chiều": 01-space-representation/notebooks/03_manifold_hypothesis_and_curse_of_dimensionality.ipynb
```

For a new section (e.g., `02-representation-learning`), insert after section 1:

```yaml
  - "2. Representation Learning":
      - "Information Bottleneck": 02-representation-learning/research/01-information-bottleneck.md
      - "Autoencoder": 02-representation-learning/research/02-autoencoder.md
      - Notebooks:
          - "Information Bottleneck": 02-representation-learning/notebooks/01_information_bottleneck.ipynb
```

Paths in nav are **relative to `docs_dir`** (`latent-anything-theory/`), so omit that prefix.

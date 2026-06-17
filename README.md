# latent-anything

> *Latent Understanding, Manipulation & Execution Network*

A Python framework that treats latent space as a first-class object: load latent representations from any model, inspect them, manipulate them, and execute pipelines efficiently.

## Overview

latent-anything sits as a horizontal tooling layer for anyone working with latent representations — researchers debugging models, developers building applications on pretrained models, and engineers running large-scale experiments.

**Three pillars:**

- **Introspection (A)** — Visualization, probing, clustering, sparse decomposition, trajectory analysis
- **Manipulation (B)** — Interpolation, arithmetic, steering, activation patching, composition, constrained editing
- **Runtime (C)** — Batching, caching, async execution, streaming, profiling

## Installation

> Package not yet published. Clone and install locally using [uv](https://docs.astral.sh/uv/):

```bash
git clone <repo-url>
cd latent-anything
uv sync
```

## Quick Start

```python
import latent_anything

print(latent_anything.__version__)
# 0.1.0
```

## Project Structure

```
latent-anything/
├── src/
│   └── latent_anything/   # Main framework package (src-layout)
│       └── __init__.py
├── tests/                 # Test suite mirroring src/
│   ├── conftest.py
│   └── test_latent_anything/
├── docs/                  # Architecture, theory, and design docs
│   ├── IDEA.md
│   ├── ARCHITECTURE.md
│   ├── THEORY.md
│   └── INDEX.md
├── latent-anything-theory/ # Standalone theory research sub-project
├── .agents/               # Agent rules, skills, and memory
├── .github/workflows/     # CI / deployment workflows
├── pyproject.toml         # Package config (ruff, pyright, pytest)
└── CHANGELOG.md
```

## Documentation

- [docs/IDEA.md](docs/IDEA.md) — Vision and motivation (Vietnamese)
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — Core primitives and layer design (Vietnamese)
- [docs/THEORY.md](docs/THEORY.md) — Theoretical foundations (Vietnamese)

## License

MIT — see [LICENSE](LICENSE).

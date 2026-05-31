# Lumen

**Latent Understanding, Manipulation & Execution Network**

A Python framework that treats latent space as a first-class object: load latent representations from any model, inspect them, manipulate them, and execute pipelines efficiently.

## Overview

Lumen sits as a horizontal tooling layer for anyone working with latent representations — researchers debugging models, developers building applications on pretrained models, and engineers running large-scale experiments.

**Three pillars:**

- **Introspection (A)** — Visualization, probing, clustering, sparse decomposition, trajectory analysis
- **Manipulation (B)** — Interpolation, arithmetic, steering, activation patching, composition, constrained editing
- **Runtime (C)** — Batching, caching, async execution, streaming, profiling

## Installation

> Package not yet published. Clone and install locally:

```bash
git clone <repo-url>
cd lumen
pip install -e .
```

## Quick Start

```python
import lumen

# Coming soon
```

## Project Structure

```
lumen/
├── docs/               # Architecture, theory, and design docs
│   ├── IDEA.md
│   ├── ARCHITECTURE.md
│   ├── THEORY.md
│   └── INDEX.md
├── .agents/            # Agent rules, skills, and memory
└── CHANGELOG.md
```

## Documentation

- [docs/IDEA.md](docs/IDEA.md) — Vision and motivation (Vietnamese)
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — Core primitives and layer design (Vietnamese)
- [docs/THEORY.md](docs/THEORY.md) — Theoretical foundations (Vietnamese)

## License

MIT — see [LICENSE](LICENSE).

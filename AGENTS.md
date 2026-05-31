# Lumen — Agent Instructions

## Project

**Lumen** (*Latent Understanding, Manipulation & Execution Network*) is a Python framework that treats latent space as a first-class object: load latent representations from any model, inspect them, manipulate them, and execute pipelines efficiently. Plugin-first architecture with three pillars: introspection (A), manipulation (B), runtime (C).

See [docs/IDEA.md](docs/IDEA.md) for the full vision, [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for core primitives and layer design.

## Language

- **Vietnamese** — high-level docs: IDEA.md, ARCHITECTURE.md, THEORY.md
- **English** — all source code, tests, artifacts, examples, commit messages, and output documents

See [docs/LANGUAGE.md](docs/LANGUAGE.md).

## How to explore

1. **Start here:** Read [docs/INDEX.md](docs/INDEX.md) for the document map, then follow the [read-docs skill](.agents/skills/read-docs/SKILL.md).
2. **Rules** live in `.agents/rules/` — coding conventions for Python (`uv`, PEP 723), Git (Conventional Commits, branch naming), and changelog format. Read and follow them before writing code.
3. **Skills** live in `.agents/skills/` — invoke them when their trigger applies:
   - `manage-plans` — break large work into sprint plans and atomic tasks
   - `implement-atomic-task` — execute one focused task with tests and artifact summary
   - `log-decision` / `log-lesson` — record decisions and lessons in `.agents/memory/`
4. **Memory** in `.agents/memory/` — check `decisions.md` before reversing any architectural choice; check `lessons-learned.md` before implementing anything non-trivial.
5. **Docs** in `docs/` — authoritative source of truth for project direction and architecture.


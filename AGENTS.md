# Latent-anything — Agent Instructions

## Project

**Latent Anything** (*Latent Understanding, Manipulation & Execution Network*) is a Python framework that treats latent space as a first-class object: load latent representations from any model, inspect them, manipulate them, and execute pipelines efficiently. Plugin-first architecture with three pillars: introspection (A), manipulation (B), runtime (C).

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

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

When the user types `/graphify`, use the installed graphify skill or instructions before doing anything else.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- Dirty graphify-out/ files are expected after hooks or incremental updates; dirty graph files are not a reason to skip graphify. Only skip graphify if the task is about stale or incorrect graph output, or the user explicitly says not to use it.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).

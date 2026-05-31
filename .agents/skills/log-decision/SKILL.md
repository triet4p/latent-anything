---
name: log-decision
description: Use when an architectural or technical decision has been made that future agents should not reverse without explicit instruction. Covers library choices, design patterns, API contracts, and any deliberate trade-off.
---

# Log Decision Skill

## Purpose

Record *why* a key decision was made in `.agents/memory/decisions.md` so future agents do not undo deliberate choices or re-litigate settled questions.

## When to use

Use this skill whenever:
- A library, tool, or framework is chosen over an alternative.
- An architectural pattern is adopted (e.g., immutable `Trajectory`, async-primary API).
- A deliberate trade-off is accepted (e.g., numpy public API over torch to avoid version lock-in).
- The user explicitly approves an approach after considering alternatives.

Do **not** log trivial implementation details — only decisions that would surprise or tempt a future agent to change them.

## Workflow

### 1. Identify the decision

Capture:
- What was decided (concrete and specific).
- What alternatives were on the table.
- Why this option was chosen (the actual reason, not a generic justification).
- What this decision constrains or enables going forward.

### 2. Append the entry

Open `.agents/memory/decisions.md` and append a new entry **at the bottom** of the file (below the last existing entry), using today's date.

**Entry format:**

```markdown
## [YYYY-MM-DD] <Short title of the decision>

**Decision:** What was decided.
**Alternatives considered:** What else was on the table.
**Reason:** Why this option was chosen over the alternatives.
**Consequences:** What this decision constrains or enables going forward.
```

### 3. Rules for writing entries

- **Title:** Use imperative or noun form — "Use pydantic for config, not Hydra" not "Pydantic vs Hydra".
- **Decision:** One sentence. Be concrete — name the actual library, pattern, or value.
- **Alternatives considered:** List real alternatives that were genuinely evaluated, not strawmen.
- **Reason:** This is the most important field. The reason must explain *why* in terms of this project's specific constraints — not generic best practices.
- **Consequences:** Name what becomes harder or easier as a result. This helps future agents understand the blast radius of reversing the decision.
- Do not edit or delete past entries. The log is append-only.

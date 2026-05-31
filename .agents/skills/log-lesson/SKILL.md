---
name: log-lesson
description: Use when a bug is fixed, an unexpected behavior is resolved, or an environment-specific quirk is discovered. Prevents the same mistake from happening twice across sessions.
---

# Log Lesson Skill

## Purpose

Record bugs, edge cases, and environment-specific quirks in `.agents/memory/lessons-learned.md` so future agents never repeat the same mistake.

## When to use

Use this skill whenever:
- A bug is found and fixed.
- A library or tool behaves unexpectedly (undocumented behavior, version quirk, OS-specific issue).
- A non-obvious workaround is required to make something work.
- An assumption turned out to be wrong and caused wasted effort.

Do **not** log expected errors (e.g., a typo that caused a syntax error) — only surprises that would trap a future agent who doesn't have the context.

## Workflow

### 1. Identify the lesson

Capture:
- What went wrong or behaved unexpectedly (the observable symptom).
- Why it happened (the root cause, not just "it was broken").
- What fixed it or what workaround was applied.
- The conditions that would trigger this again.

### 2. Append the entry

Open `.agents/memory/lessons-learned.md` and append a new entry **at the bottom** of the file (below the last existing entry), using today's date.

**Entry format:**

```markdown
## [YYYY-MM-DD] <Short title of the issue>

**Symptom:** What went wrong or behaved unexpectedly.
**Root cause:** Why it happened.
**Fix / workaround:** What resolved it.
**Watch out for:** Conditions that would trigger this again.
```

### 3. Rules for writing entries

- **Title:** Describe the symptom or the trap, not the solution — "UMAP fit() silently ignores n_neighbors > n_samples" not "Fixed UMAP config".
- **Symptom:** Describe what the agent or user observed — error message, wrong output, silent failure.
- **Root cause:** Explain *why* it happened. "Unknown" is acceptable if the cause is genuinely unclear, but try.
- **Fix / workaround:** Be exact — include the specific flag, config value, or code change. Vague fixes ("updated the config") are useless.
- **Watch out for:** Name the triggering condition so a future agent can recognize the situation before hitting the bug.
- Do not edit or delete past entries. The log is append-only.

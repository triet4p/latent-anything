---
name: read-docs
description: Use when reading project documents.
---

# How to read documents

When reading project documents, follow these steps:

1. Read [[INDEX]] to understand what documents exist and what they cover.
2. Read [[LANGUAGE]] to understand which language to use in each context.
3. Locate your target document in [[INDEX]] and read it. To understand the project fully, read all documents in the order listed in [[INDEX]]. There are three cases:
   - **File in `docs/`** — read it directly.
   - **Sub-folder in `docs/`** — read that folder's index file first (e.g., `docs/idea/INDEX.md`), then read its documents in order.
   - **File outside `docs/`** — read it directly, but first read all related `docs/` files, as they are the authoritative source of truth. To list files in the target folder, use any available tool (e.g., `Get-ChildItem` in PowerShell or `ls` in Linux).
4. If anything is unclear, ask for clarification.
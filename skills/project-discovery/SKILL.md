---
name: project-discovery
description: "Use when the user asks for repository overview, README summary, architecture, project structure, module responsibilities, storage layout, or implementation inventory."
metadata:
  category: mempedia
  priority: high
  tags: [mempedia, discovery, project]
---

# Project Discovery

## Goal

Produce a grounded summary of the repository based on actual files.

## When To Use

- The user asks what is in the project.
- The user asks for architecture, module layout, features, APIs, or storage structure.
- The user asks for a README-based summary or onboarding overview.

## Workflow

1. Start from the most authoritative files such as `readme.md`, `Cargo.toml`, `package.json`, schema docs, and top-level module entrypoints.
2. Confirm claims against code, not just prose.
3. Group findings into stable facts: architecture, capabilities, storage layout, entrypoints, interfaces.
4. If the final summary contains reusable project facts, queue asynchronous memory save with Layer 1 enabled.

## Tool Guidance

- Prefer `read` for authoritative files and `search` for `grep` or `glob` repository discovery.
- If existing enterprise knowledge may already cover the project, use `bash` plus `search_nodes`.
- If you deliberately persist normalized project facts, do it through `bash` with Layer 1 CLI actions, not through `read/search/edit` routing.
- Let the independent post-turn memory agent handle routine async persistence.

## Avoid

- Inferring architecture from filenames alone.
- Saving temporary conversational phrasing as knowledge.
- Inspecting or summarizing the agent's own `skills/` directory unless the user explicitly asked about skills.
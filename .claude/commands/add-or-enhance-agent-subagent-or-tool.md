---
name: add-or-enhance-agent-subagent-or-tool
description: Workflow command scaffold for add-or-enhance-agent-subagent-or-tool in mitosis-cli.
allowed_tools: ["Bash", "Read", "Write", "Grep", "Glob"]
---

# /add-or-enhance-agent-subagent-or-tool

Use this workflow when working on **add-or-enhance-agent-subagent-or-tool** in `mitosis-cli`.

## Goal

Adds a new agent subagent, tool, or enhances existing agent logic with new capabilities, types, and tests.

## Common Files

- `src/agent/subagents/*.ts`
- `src/agent/subagents/types.ts`
- `src/agent/planning/types.ts`
- `src/agent/index.ts`
- `src/agent/*.test.ts`
- `src/runtime/agent/AgentRuntime.ts`

## Suggested Sequence

1. Understand the current state and failure mode before editing.
2. Make the smallest coherent change that satisfies the workflow goal.
3. Run the most relevant verification for touched files.
4. Summarize what changed and what still needs review.

## Typical Commit Signals

- Create or update subagent/tool implementation file(s) in src/agent/subagents/ or src/agent/
- Update types in src/agent/subagents/types.ts or src/agent/planning/types.ts
- Register or expose the new subagent/tool in src/agent/index.ts
- Add or update tests in src/agent/*.test.ts
- Optionally update src/runtime/agent/AgentRuntime.ts or src/runtime/agent/Planner.ts for runtime integration

## Notes

- Treat this as a scaffold, not a hard-coded script.
- Update the command if the workflow evolves materially.
```markdown
# mitosis-cli Development Patterns

> Auto-generated skill from repository analysis

## Overview

This skill teaches you the core development conventions and workflows for contributing to the `mitosis-cli` codebase. The project is a TypeScript CLI tool using React, focused on agent-based logic, skills, and runtime planning. You'll learn how to add features, refactor components, manage skills, and write tests following the repository's established patterns.

---

## Coding Conventions

**File Naming**
- Use PascalCase for all file names.
  - Example: `AgentRuntime.ts`, `ContextBudget.ts`

**Import Style**
- Use relative imports.
  - Example:
    ```typescript
    import { AgentRuntime } from './AgentRuntime';
    ```

**Export Style**
- Use named exports.
  - Example:
    ```typescript
    export function planAgentTask() { ... }
    export type AgentType = { ... };
    ```

**Commit Messages**
- Use prefixes: `feat`, `fix`, `refactor`, `style`
- Keep commit messages concise (average ~72 characters).
  - Example: `feat: add context budget management for agent runs`

---

## Workflows

### Add or Enhance Agent Subagent or Tool
**Trigger:** When introducing a new agent subagent, tool, or extending agent planning/decision logic.  
**Command:** `/add-agent-subagent`

1. Create or update subagent/tool implementation in `src/agent/subagents/` or `src/agent/`.
2. Update types in `src/agent/subagents/types.ts` or `src/agent/planning/types.ts`.
3. Register or expose the new subagent/tool in `src/agent/index.ts`.
4. Add or update tests in `src/agent/*.test.ts`.
5. Optionally, update `src/runtime/agent/AgentRuntime.ts` or `src/runtime/agent/Planner.ts` for runtime integration.

**Example:**
```typescript
// src/agent/subagents/MySubagent.ts
export function mySubagentLogic(...) { ... }

// src/agent/subagents/types.ts
export type MySubagentType = { ... };

// src/agent/index.ts
export { mySubagentLogic } from './subagents/MySubagent';
```

---

### Feature Development with Tests and Types
**Trigger:** When developing a new feature or major enhancement in the agent or runtime.  
**Command:** `/new-feature`

1. Implement feature logic in `src/agent/` or `src/runtime/agent/`.
2. Update or add related type definitions in:
   - `src/agent/planning/types.ts`
   - `src/agent/subagents/types.ts`
   - `src/runtime/agent/types.ts`
3. Write or update automated tests in:
   - `src/agent/*.test.ts`
   - `src/runtime/agent/*.test.ts`
4. Update `src/agent/index.ts` to wire up new logic if needed.

**Example:**
```typescript
// src/agent/NewFeature.ts
export function newFeatureLogic(...) { ... }

// src/agent/NewFeature.test.ts
import { newFeatureLogic } from './NewFeature';
test('should work', () => { ... });
```

---

### Add or Enhance Skill
**Trigger:** When introducing a new skill or modifying skill parsing/management.  
**Command:** `/add-skill`

1. Create or update `SKILL.md` in the `skills/` directory.
2. Update `src/skills/router.ts` to parse or handle new skill attributes.
3. Write or update tests in `src/skills/router.test.ts`.
4. Optionally, update `src/config/projectPaths.ts` or `src/agent/index.ts` for skill loading.

**Example:**
```markdown
// skills/my-skill/SKILL.md
# My Skill
Description and usage instructions.
```
```typescript
// src/skills/router.ts
import { mySkill } from '../../skills/my-skill';
```

---

### Refactor or Remove Agent Runtime or Planner Components
**Trigger:** When refactoring, replacing, or removing agent runtime or planner logic.  
**Command:** `/refactor-runtime`

1. Delete or refactor files in `src/runtime/agent/` (e.g., `BeamSearchAgentRuntime`, `Planner`, `PathScorer`).
2. Update type definitions in `src/runtime/agent/types.ts`.
3. Update or remove related tests in:
   - `src/runtime/agent/*.test.ts`
   - `src/agent/*.test.ts`
4. Update `src/agent/index.ts` and `.env.example` to reflect changes.

---

### Improve Context Budget or Transcript Compression
**Trigger:** When optimizing context usage or transcript management for agent operations.  
**Command:** `/improve-context-budget`

1. Update or create `src/agent/contextBudget.ts` and related test files.
2. Implement or update compression logic in `src/agent/sessionCompressor.ts`.
3. Update `src/runtime/agent/AgentRuntime.ts` to integrate new logic.
4. Update or add tests in:
   - `src/agent/contextBudget.test.ts`
   - `src/agent/contextBudget.integration.test.ts`
   - `src/agent/sessionCompressor.test.ts`

**Example:**
```typescript
// src/agent/contextBudget.ts
export function manageContextBudget(...) { ... }

// src/agent/contextBudget.test.ts
import { manageContextBudget } from './contextBudget';
test('context budget is enforced', () => { ... });
```

---

## Testing Patterns

- Test files use the pattern: `*.test.ts`
- Tests are colocated with implementation files in the same directory.
- Testing framework is not explicitly specified, but follows standard TypeScript/Jest-like conventions.

**Example:**
```typescript
// src/agent/Feature.test.ts
import { featureLogic } from './Feature';
test('feature works as expected', () => {
  expect(featureLogic(...)).toBe(...);
});
```

---

## Commands

| Command                | Purpose                                                        |
|------------------------|----------------------------------------------------------------|
| /add-agent-subagent    | Add or enhance an agent subagent or tool                       |
| /new-feature           | Implement a new feature with types and tests                   |
| /add-skill             | Add or update a skill and its router logic                     |
| /refactor-runtime      | Refactor or remove agent runtime/planner components            |
| /improve-context-budget| Improve context budget or transcript compression logic         |
```

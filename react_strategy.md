# Branching ReAct Strategy for Mempedia

This guide describes the rewritten CodeCLI strategy: ReAct is treated as a functional loop, and one loop can spawn child loops when multiple real options exist.

## Core Loop

Each loop iteration must do exactly one of the following:

1. **THOUGHT → TOOL**
   Pick the single best next tool call and continue the same branch.

2. **THOUGHT → BRANCH**
   If there are multiple materially different strategies, create several child branches.
   Each child branch gets its own local objective and keeps running its own ReAct cycle.

3. **THOUGHT → FINAL**
   If the branch is done, emit a final answer for that branch.

4. **THOUGHT → ASYNC SAVE (optional)**
   If a branch has discovered durable value, call the async memory-save tool immediately.
   This should preserve only valuable information plus the raw conversation snapshot, without blocking the main loop.

Completed branches are synthesized into one user-facing answer.

## Branch Outcome Reporting

When a branch finishes (via `THOUGHT → FINAL`), it must report a structured outcome:

- **success** — the branch's goal was fully achieved (e.g. code written and tests pass).
- **partial** — some progress was made but the goal was not fully met (e.g. code written but tests not run, or context budget exhausted).
- **failed** — the goal was not met (e.g. tests fail, required resource not found).

Include a brief `outcome_reason` when the outcome is `partial` or `failed` (e.g. the first few lines of a test error). This structured information enables the **post-synthesis re-branching** mechanism described below.

## Post-Synthesis Re-Branching

After all branches complete, the synthesis stage inspects their outcomes:

1. If **all branches succeeded** (or outcomes are `unknown`), synthesize normally into a final answer.
2. If **some branches failed or are partial** and retry budget remains (`maxSynthesisRetries`, default 2):
   - Successful branch results become confirmed context for new branches.
   - Each failed/partial branch spawns a **remediation child branch** whose goal includes the failure reason.
   - The remediation branches run their own ReAct loops, then synthesis is attempted again.
3. If retries are exhausted, synthesize the best available results and finish.

This ensures that in coding tasks, a failed test in one branch does not silently drop — the system automatically retries with the error information carried forward.

## When to Branch

Branch only if the alternatives are genuinely different, for example:

- lexical search vs. graph traversal
- read top hit vs. verify version history
- inspect code path A vs. code path B
- quick answer path vs. validation path

Do **not** branch for trivial wording variations.

## When to Save Memory

Use async memory save only when at least one of these is true:

- a reusable project fact was discovered
- a stable workflow/pattern was confirmed
- a valuable branch result should be preserved before the session ends

Do **not** save every branch by default.

## Tool Order (Recommended)

1. `mempedia_search_hybrid`  
   Use for high-recall retrieval. Ask for `limit=8-12` first.

2. `mempedia_read`  
   Open the top 1-3 nodes to confirm details.

3. `mempedia_traverse`  
   Expand the graph when you need related context or dependencies.

4. `mempedia_history`  
   Validate how a fact evolved over time, or resolve conflicts.

5. `mempedia_save`  
   Persist new atomic knowledge or updates after the task.

## Search Patterns

- Broad query → `mempedia_search_hybrid(query, limit=10)`
- Disambiguate → `mempedia_read(node_id)`
- Explore neighbors → `mempedia_traverse(start_node, mode="bfs", depth_limit=1)`
- Check versions → `mempedia_history(node_id, limit=5)`

## Save Discipline

- Save only reusable, atomic knowledge.
- Avoid storing transient or conversational noise.
- Always include a clean title and concise summary in the markdown.
- Do not stop at a terse summary when richer evidence exists. Preserve facts, numbers, version details, timeline changes, viewpoints, caveats, and evidence in structured markdown sections.
- Prefer explicit sections such as `Facts`, `Data`, `History`, `Viewpoints`, `Uncertainties`, `Relations`, and `Evidence` when they help future retrieval.
- Never invent missing facts.

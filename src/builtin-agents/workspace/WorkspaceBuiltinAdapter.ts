/**
 * WorkspaceBuiltinAdapter — in-process adapter wrapping the workspace tools
 * (read / search / edit).
 *
 * This adapter delegates every tool call to the `executor` function supplied
 * at construction time.  The executor is typically a closure over the
 * PlannerToolAdapter / MainAgentAdapter instance that lives inside Agent.run().
 *
 * Phase 4 note: once the workspace implementations are extracted from
 * MainAgentAdapter this adapter will own them directly.  For now it acts
 * as a thin forwarding shim so the ecosystem registry has proper metadata.
 */

import type { SubagentAdapter, SubagentConfig } from '../../types/subagent-adapter.js';
import type { ToolDefinition, ToolExecutionContext, ToolExecutionResult } from '../../runtime/tools/types.js';
import type { SkillRecord } from '../../skills/router.js';
import { TOOLS } from '../../tools/definitions.js';

/** Function that actually executes the tool, delegating to MainAgentAdapter. */
export type BuiltinExecutorFn = (
  name: string,
  args: Record<string, unknown>,
) => Promise<string>;

const WORKSPACE_TOOL_NAMES = ['read', 'search', 'edit'] as const;

export class WorkspaceBuiltinAdapter implements SubagentAdapter {
  readonly subagentId = 'workspace-tools';

  private readonly executor: BuiltinExecutorFn;

  constructor(executor: BuiltinExecutorFn) {
    this.executor = executor;
  }

  async initialize(_config: SubagentConfig): Promise<void> {
    // No async setup required — executor is provided at construction time.
  }

  listTools(): ToolDefinition[] {
    return WORKSPACE_TOOL_NAMES.map((name) => {
      const def = TOOLS.find((t) => t.function.name === name);
      const description = def?.function.description ?? `${name} workspace tool`;
      const parameters = def?.function.parameters ?? { type: 'object', properties: {}, required: [] as string[] };
      const executor = this.executor;
      return {
        name,
        description,
        parameters,
        async execute(args: Record<string, unknown>, _ctx: ToolExecutionContext): Promise<ToolExecutionResult> {
          const start = Date.now();
          try {
            const result = await executor(name, args);
            return { success: true, result, durationMs: Date.now() - start };
          } catch (e: unknown) {
            const msg = e instanceof Error ? e.message : String(e);
            return { success: false, error: msg, durationMs: Date.now() - start };
          }
        },
      };
    });
  }

  listSkills(): SkillRecord[] {
    return [];
  }

  async executeTool(
    name: string,
    args: Record<string, unknown>,
    _ctx: ToolExecutionContext,
  ): Promise<ToolExecutionResult> {
    const start = Date.now();
    try {
      const result = await this.executor(name, args);
      return { success: true, result, durationMs: Date.now() - start };
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : String(e);
      return { success: false, error: msg, durationMs: Date.now() - start };
    }
  }

  async loadSkill(_id: string): Promise<import('../../types/subagent-adapter.js').SkillContent | null> {
    return null;
  }

  getLanguageModel(): undefined {
    return undefined;
  }

  async connectMCP?(_decl: import('../../types/subagent-manifest.js').MCPServerDecl): Promise<import('../../types/subagent-adapter.js').MCPConnection> {
    throw new Error('WorkspaceBuiltinAdapter does not support MCP connections');
  }

  async shutdown(): Promise<void> {
    // Nothing to tear down.
  }
}

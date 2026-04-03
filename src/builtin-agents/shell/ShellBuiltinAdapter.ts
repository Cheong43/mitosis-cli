/**
 * ShellBuiltinAdapter — in-process adapter wrapping the `bash` tool.
 *
 * Delegates to the executor provided at construction time.
 * Future: replace executor delegation with direct shell execution logic.
 */

import type { SubagentAdapter, SubagentConfig } from '../../types/subagent-adapter.js';
import type { ToolDefinition, ToolExecutionContext, ToolExecutionResult } from '../../runtime/tools/types.js';
import type { SkillRecord } from '../../skills/router.js';
import { TOOLS } from '../../tools/definitions.js';

export type BuiltinExecutorFn = (
  name: string,
  args: Record<string, unknown>,
) => Promise<string>;

export class ShellBuiltinAdapter implements SubagentAdapter {
  readonly subagentId = 'shell';

  private readonly executor: BuiltinExecutorFn;

  constructor(executor: BuiltinExecutorFn) {
    this.executor = executor;
  }

  async initialize(_config: SubagentConfig): Promise<void> {}

  listTools(): ToolDefinition[] {
    const def = TOOLS.find((t) => t.function.name === 'bash');
    const description = def?.function.description ?? 'Run bash shell commands';
    const parameters = def?.function.parameters ?? {
      type: 'object',
      properties: { command: { type: 'string' } },
      required: ['command'],
    };
    const executor = this.executor;
    return [
      {
        name: 'bash',
        description,
        parameters,
        async execute(args: Record<string, unknown>, _ctx: ToolExecutionContext): Promise<ToolExecutionResult> {
          const start = Date.now();
          try {
            const result = await executor('bash', args);
            return { success: true, result, durationMs: Date.now() - start };
          } catch (e: unknown) {
            const msg = e instanceof Error ? e.message : String(e);
            return { success: false, error: msg, durationMs: Date.now() - start };
          }
        },
      },
    ];
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

  async shutdown(): Promise<void> {}
}

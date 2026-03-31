import { SubagentRegistry } from './registry.js';
import type {
  SubagentInvocation,
  SubagentRunContext,
  SubagentRunResult,
} from './types.js';

export async function runSubagent(
  registry: SubagentRegistry,
  context: SubagentRunContext,
  invocation: SubagentInvocation,
): Promise<SubagentRunResult> {
  const handler = registry.get(invocation.subagent);
  if (!handler) {
    throw new Error(`Unknown subagent: ${invocation.subagent}`);
  }
  return handler.run(context, invocation as never);
}

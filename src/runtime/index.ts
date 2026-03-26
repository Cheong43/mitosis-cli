/**
 * Runtime bootstrap module.
 *
 * `createRuntime(projectRoot)` is the single composition root for the
 * governed tool / governance runtime.  Import and call this once at startup
 * to get a fully-wired `ToolRuntime`.
 *
 * Architecture:
 *
 *   Shell / UI / CLI
 *       │
 *       ▼
 *   AgentRuntime  ──────────► ToolRuntime
 *       │                         │
 *       │                         ▼
 *       │                  GovernanceRuntime
 *       │                    (policy + guards + audit)
 *       │                         │
 *       └──── (allow) ───────────►│
 *                                 ▼
 *                          ToolDefinition.execute()
 *                         (ReadFileTool, WriteFileTool, RunShellTool)
 */

import { ToolRegistry } from './tools/ToolRegistry.js';
import { ToolRuntime } from './tools/ToolRuntime.js';
import { GovernanceRuntime } from './governance/GovernanceRuntime.js';
import { InMemoryApprovalEngine } from './governance/InMemoryApprovalEngine.js';
import { ToolExecutionResult } from './tools/types.js';
import { ReadFileTool } from './tools/builtin/local/ReadFileTool.js';
import { WriteFileTool } from './tools/builtin/local/WriteFileTool.js';
import { RunShellTool } from './tools/builtin/local/RunShellTool.js';
import { Policy, ApprovalEngine } from './governance/types.js';

export { ToolRuntime } from './tools/ToolRuntime.js';
export { ToolRegistry } from './tools/ToolRegistry.js';
export { GovernanceRuntime } from './governance/GovernanceRuntime.js';
export { AuditLogger } from './governance/AuditLogger.js';
export { InMemoryApprovalEngine } from './governance/InMemoryApprovalEngine.js';
export { CallbackApprovalEngine } from './governance/CallbackApprovalEngine.js';
export type { ApprovalPrompt, ApprovalCallback } from './governance/CallbackApprovalEngine.js';
export { ExternalDirGuard } from './governance/guards/ExternalDirGuard.js';
export { DoomLoopGuard } from './governance/guards/DoomLoopGuard.js';
export { ShellSafetyGuard } from './governance/guards/ShellSafetyGuard.js';
export { ReadFileTool } from './tools/builtin/local/ReadFileTool.js';
export { WriteFileTool } from './tools/builtin/local/WriteFileTool.js';
export { RunShellTool } from './tools/builtin/local/RunShellTool.js';
export { AgentRuntime } from './agent/AgentRuntime.js';
export { SimplePlanner } from './agent/SimplePlanner.js';


export type { Policy } from './governance/types.js';
export type { ApprovalEngine } from './governance/types.js';
export type { ToolDefinition, ToolExecutionContext, ToolExecutionResult } from './tools/types.js';
export type { AgentTraceEvent, BeamPath, PathScorer, PathHistoryEntry } from './agent/types.js';

export interface RuntimeConfig {
  projectRoot: string;
  agentId?: string;
  sessionId?: string;
  /** Override policy (useful in tests). */
  policy?: Policy;
  /**
   * Default answer for the `ask` approval engine when running non-interactively.
   * Defaults to `'allow'` so that the new runtime is transparent by default
   * until a custom `ApprovalEngine` is wired in.
   * Ignored if `approvalEngine` is provided.
   */
  askDefault?: 'allow' | 'deny';
  /**
   * Custom approval engine for interactive `ask` resolution.
   * When provided, takes precedence over `askDefault`.
   */
  approvalEngine?: ApprovalEngine;
}

export interface RuntimeHandle {
  toolRuntime: ToolRuntime;
  governance: GovernanceRuntime;
  registry: ToolRegistry;
  executeTool(toolName: string, args: Record<string, unknown>): Promise<ToolExecutionResult>;
}

/**
 * Create and wire the full governed runtime.
 *
 * @param config - Runtime configuration.
 */
export function createRuntime(config: RuntimeConfig): RuntimeHandle {
  const {
    projectRoot,
    agentId = 'agent-main',
    sessionId = `session-${Date.now()}`,
    policy,
    askDefault = 'allow',
    approvalEngine: customApprovalEngine,
  } = config;

  // 1. Governance layer
  const approvalEngine = customApprovalEngine ?? new InMemoryApprovalEngine(askDefault);
  const governance = new GovernanceRuntime({ projectRoot, policy, approvalEngine });

  // 2. Tool registry
  const registry = new ToolRegistry();
  registry.registerAll([
    new ReadFileTool(),
    new WriteFileTool(),
    new RunShellTool(),
  ]);

  // 3. Tool runtime
  const toolRuntime = new ToolRuntime({
    registry,
    governance,
    context: { projectRoot, agentId, sessionId },
  });

  // 4. Convenience wrapper
  const executeTool = async (toolName: string, args: Record<string, unknown>) =>
    toolRuntime.execute(toolName, args);

  return { toolRuntime, governance, registry, executeTool };
}

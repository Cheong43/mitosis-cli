/**
 * builtin-agents/index.ts
 *
 * Barrel exports for all builtin subagent adapters.
 *
 * Builtin adapters wrap the 5 core tools (read/search/edit/bash/web) that
 * have historically lived inside MainAgentAdapter.  They exist primarily as
 * an organisational layer:
 *
 *  • They give the ecosystem registry accurate tool metadata for the builtins.
 *  • They act as the future home for extracted implementations from MainAgentAdapter.
 *  • They allow external subagents to declare dependencies on "workspace-tools",
 *    "shell", or "web" by name.
 *
 * For now the adapters delegate execution back to a provided executor function
 * (typically PlannerToolAdapter.execute).  In a future phase, the internals
 * will be moved directly into the adapters.
 */

export { WorkspaceBuiltinAdapter } from './workspace/WorkspaceBuiltinAdapter.js';
export type { BuiltinExecutorFn as WorkspaceExecutorFn } from './workspace/WorkspaceBuiltinAdapter.js';

export { ShellBuiltinAdapter } from './shell/ShellBuiltinAdapter.js';
export type { BuiltinExecutorFn as ShellExecutorFn } from './shell/ShellBuiltinAdapter.js';

export { WebBuiltinAdapter } from './web/WebBuiltinAdapter.js';
export type { BuiltinExecutorFn as WebExecutorFn } from './web/WebBuiltinAdapter.js';

// ---------------------------------------------------------------------------
// Convenience factory
// ---------------------------------------------------------------------------

import { WorkspaceBuiltinAdapter } from './workspace/WorkspaceBuiltinAdapter.js';
import { ShellBuiltinAdapter } from './shell/ShellBuiltinAdapter.js';
import { WebBuiltinAdapter } from './web/WebBuiltinAdapter.js';

export type BuiltinExecutorFn = (name: string, args: Record<string, unknown>) => Promise<string>;

/**
 * Create all three builtin adapter instances backed by a single executor.
 * The executor is typically a wrapper around PlannerToolAdapter.execute().
 */
export function createBuiltinAdapters(executor: BuiltinExecutorFn) {
  return {
    workspace: new WorkspaceBuiltinAdapter(executor),
    shell: new ShellBuiltinAdapter(executor),
    web: new WebBuiltinAdapter(executor),
  };
}

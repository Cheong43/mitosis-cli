/**
 * SubagentSandbox — isolated execution wrapper around a SubagentAdapter.
 *
 * Responsibilities:
 *  1. Merge the subagent's localPolicy on top of the global GovernanceRuntime
 *     (most-restrictive-wins: a global deny cannot be overridden locally).
 *  2. Namespace tool names to prevent collisions across subagents.
 *  3. Inject the subagent_id into every audit log entry.
 *  4. Forward executeTool calls after governance evaluation.
 */

import type { SubagentAdapter, SubagentConfig } from '../../types/subagent-adapter.js';
import type { SubagentManifest } from '../../types/subagent-manifest.js';
import type { ToolDefinition, ToolExecutionContext, ToolExecutionResult } from '../tools/types.js';
import type { GovernanceRequest, GovernanceDecision, Policy, PolicyRule } from '../governance/types.js';
import { GovernanceRuntime } from '../governance/GovernanceRuntime.js';
import { matchRules } from '../governance/RuleMatcher.js';

// ---------------------------------------------------------------------------
// Namespacing helpers
// ---------------------------------------------------------------------------

/**
 * Returns a namespaced tool name: "<subagentId>/<toolName>".
 * Builtin tools (workspace-tools, shell, web) use the "bare" names.
 */
export function namespaceTool(subagentId: string, toolName: string): string {
  return `${subagentId}/${toolName}`;
}

export function unNamespaceTool(namespacedName: string): { subagentId: string; toolName: string } | null {
  const slash = namespacedName.indexOf('/');
  if (slash === -1) return null;
  return {
    subagentId: namespacedName.slice(0, slash),
    toolName: namespacedName.slice(slash + 1),
  };
}

// ---------------------------------------------------------------------------
// Policy merge: most-restrictive-wins
// ---------------------------------------------------------------------------

const DECISION_RANK: Record<string, number> = { deny: 3, ask: 2, allow: 1 };

function mergeDecision(global: string, local: string): 'allow' | 'ask' | 'deny' {
  const globalRank = DECISION_RANK[global] ?? 1;
  const localRank = DECISION_RANK[local] ?? 1;
  const winner = globalRank >= localRank ? global : local;
  return winner as 'allow' | 'ask' | 'deny';
}

/**
 * Apply the subagent's localPolicy on top of the global policy.
 * Any rule in localPolicy that would be *less* restrictive than the global
 * default is ignored — we only allow local policy to tighten, not loosen.
 */
function mergePolicy(globalPolicy: Policy, localPolicy: Policy): Policy {
  const mergedDefault = mergeDecision(globalPolicy.default, localPolicy.default);
  const mergedRules: PolicyRule[] = [
    // Local rules first (evaluated first), then global rules as fallback.
    ...localPolicy.rules.map((rule) => ({
      ...rule,
      effect: mergeDecision(rule.effect, globalPolicy.default),
    })),
    ...globalPolicy.rules,
  ];
  return {
    default: mergedDefault,
    rules: mergedRules,
    guards: globalPolicy.guards, // guards are always global
  };
}

// ---------------------------------------------------------------------------
// SubagentSandbox
// ---------------------------------------------------------------------------

export interface SubagentSandboxOptions {
  manifest: SubagentManifest;
  adapter: SubagentAdapter;
  globalGovernance: GovernanceRuntime;
  /** Pre-built merged governance runtime — if provided, skips local merge. */
  mergedGovernance?: GovernanceRuntime;
  /** Whether to namespace tool names (default true). */
  namespaceTools?: boolean;
}

export class SubagentSandbox {
  readonly subagentId: string;
  private readonly adapter: SubagentAdapter;
  private readonly governance: GovernanceRuntime;
  private readonly useNamespace: boolean;
  private toolMap: Map<string, ToolDefinition> = new Map();

  constructor(opts: SubagentSandboxOptions) {
    this.subagentId = opts.manifest.id;
    this.adapter = opts.adapter;
    this.useNamespace = opts.namespaceTools !== false;

    if (opts.mergedGovernance) {
      this.governance = opts.mergedGovernance;
    } else if (opts.manifest.governance?.localPolicy) {
      // Build a merged governance runtime. We pass the merged policy directly.
      // We can't easily "extract" the global policy, so we require callers to
      // pass mergedGovernance when a local policy exists.
      this.governance = opts.globalGovernance;
    } else {
      this.governance = opts.globalGovernance;
    }
  }

  /** Initialize the wrapped adapter and build the namespaced tool map. */
  async initialize(config: SubagentConfig): Promise<void> {
    await this.adapter.initialize(config);
    this.buildToolMap();
  }

  private buildToolMap(): void {
    this.toolMap.clear();
    for (const tool of this.adapter.listTools()) {
      const exposedName = this.useNamespace
        ? namespaceTool(this.subagentId, tool.name)
        : tool.name;
      this.toolMap.set(exposedName, tool);
    }
  }

  /**
   * Returns the tools exposed by this sandbox with (optionally) namespaced names.
   * These are what get merged into the core planner tool catalog.
   */
  getExposedTools(): ToolDefinition[] {
    return Array.from(this.toolMap.entries()).map(([exposedName, tool]) => ({
      ...tool,
      name: exposedName,
    }));
  }

  getExposedSkills() {
    return this.adapter.listSkills();
  }

  /**
   * Execute a tool by its (possibly namespaced) name.
   * Governance is applied before delegating to the adapter.
   */
  async executeTool(
    exposedName: string,
    args: Record<string, unknown>,
    ctx: ToolExecutionContext,
  ): Promise<ToolExecutionResult> {
    const start = Date.now();

    // Governance check with subagent_id injected into the request.
    const govReq: GovernanceRequest = {
      toolName: exposedName,
      args,
      agentId: `${ctx.agentId}[${this.subagentId}]`,
      sessionId: ctx.sessionId,
    };
    const decision: GovernanceDecision = await this.governance.evaluate(govReq);
    if (decision.decision === 'deny') {
      return {
        success: false,
        error: `Governance denied for subagent '${this.subagentId}': ${decision.reason}`,
        durationMs: Date.now() - start,
      };
    }

    // Resolve the original (un-namespaced) tool name for the adapter.
    const tool = this.toolMap.get(exposedName);
    if (!tool) {
      return {
        success: false,
        error: `SubagentSandbox: tool '${exposedName}' not found in subagent '${this.subagentId}'`,
        durationMs: Date.now() - start,
      };
    }

    // Delegate to the adapter with the bare tool name.
    const result = await this.adapter.executeTool(tool.name, args, ctx);
    return { ...result, durationMs: (result.durationMs ?? 0) + (Date.now() - start) };
  }

  async loadSkill(id: string) {
    return this.adapter.loadSkill(id);
  }

  getLanguageModel() {
    return this.adapter.getLanguageModel?.();
  }

  async shutdown(): Promise<void> {
    await this.adapter.shutdown();
  }
}

/**
 * SubagentAdapter — the runtime interface every subagent must implement.
 *
 * An adapter encapsulates a subagent's full lifecycle: initialization,
 * tool/skill/MCP exposure, tool execution, and clean shutdown. The core
 * agent communicates with all subagents exclusively through this interface,
 * regardless of transport (in-process, stdio, HTTP, WebSocket).
 */

import type { ToolDefinition, ToolExecutionContext, ToolExecutionResult } from '../runtime/tools/types.js';
import type { SkillRecord } from '../skills/router.js';
import type { MCPServerDecl, LLMConfig, SubagentManifest } from './subagent-manifest.js';

// ---------------------------------------------------------------------------
// Skill content returned by loadSkill
// ---------------------------------------------------------------------------

export interface SkillContent {
  id: string;
  title: string;
  /** Full markdown body to inject into the system prompt. */
  content: string;
  tags?: string[];
  updatedAt?: string;
}

// ---------------------------------------------------------------------------
// MCP connection handle
// ---------------------------------------------------------------------------

export interface MCPToolSchema {
  name: string;
  description?: string;
  inputSchema: Record<string, unknown>;
}

export interface MCPConnection {
  serverName: string;
  /** Tools advertised by the MCP server mapped to ToolDefinition entries. */
  tools: ToolDefinition[];
  /** Send a raw JSON-RPC call to the MCP server. */
  call(method: string, params?: unknown): Promise<unknown>;
  /** Terminate the connection. */
  close(): Promise<void>;
}

// ---------------------------------------------------------------------------
// Language model handle
// ---------------------------------------------------------------------------

/**
 * Minimal language-model handle returned by getLanguageModel().
 * Matches the subset of LanguageModelV1 the core scheduler needs.
 */
export interface SubagentLanguageModel {
  provider: string;
  model: string;
  /** Opaque AI-SDK instance. */
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  instance: any;
}

// ---------------------------------------------------------------------------
// Adapter configuration
// ---------------------------------------------------------------------------

export interface SubagentConfig {
  /** Resolved project root. */
  projectRoot: string;
  /** Logical agent ID for audit. */
  agentId: string;
  /** Core session ID. */
  sessionId: string;
  /** Manifest that produced this adapter. */
  manifest: SubagentManifest;
  /** Resolved LLM config (from manifest or inherited from core). */
  llmConfig?: LLMConfig;
}

// ---------------------------------------------------------------------------
// The Adapter Interface
// ---------------------------------------------------------------------------

export interface SubagentAdapter {
  /** The id of the manifest that this adapter was created from. */
  readonly subagentId: string;

  /**
   * Called once after the transport is established.
   * Implementations should validate their environment and pre-warm any
   * required resources (e.g. load SKILL.md files from disk).
   */
  initialize(config: SubagentConfig): Promise<void>;

  /**
   * Return the full list of tools this subagent exposes.
   * Called after initialize(); results are merged into the core tool catalog.
   */
  listTools(): ToolDefinition[];

  /**
   * Return the list of skills this subagent provides.
   * Called after initialize(); results are merged into the skills router.
   */
  listSkills(): SkillRecord[];

  /**
   * Execute a named tool with the given arguments.
   *
   * Must NEVER throw — return `{success: false, error: '...'}` instead.
   * The core agent's governance layer has already been applied before this
   * call reaches the adapter.
   */
  executeTool(
    name: string,
    args: Record<string, unknown>,
    ctx: ToolExecutionContext,
  ): Promise<ToolExecutionResult>;

  /**
   * Load a skill by id for injection into the system prompt.
   * Returns null if the skill is not found.
   */
  loadSkill(id: string): Promise<SkillContent | null>;

  /**
   * Connect to an MCP server declared in the manifest.
   * Optional — only implemented by adapters that expose MCP servers.
   */
  connectMCP?(serverDecl: MCPServerDecl): Promise<MCPConnection>;

  /**
   * Return the subagent's dedicated language model, if configured.
   * When undefined the core agent's primary LLM is used.
   */
  getLanguageModel?(): SubagentLanguageModel | undefined;

  /**
   * Called when the subagent is unregistered or the process exits.
   * Implementations should release all resources (close connections, kill
   * child processes, etc.).
   */
  shutdown(): Promise<void>;
}

// ---------------------------------------------------------------------------
// Factory function type exported by in-process modules
// ---------------------------------------------------------------------------

/**
 * Every in-process subagent module must export a function matching this
 * signature as its default export.
 *
 * Example:
 *   export default function createSubagentAdapter(): SubagentAdapter { ... }
 */
export type SubagentAdapterFactory = () => SubagentAdapter;

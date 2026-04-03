/**
 * SubagentManifest — declarative descriptor for a subagent unit.
 *
 * A manifest defines everything the registry needs to discover, validate,
 * instantiate, and govern a subagent: its capabilities (tools, skills, MCP
 * servers), optional dedicated LLM config, transport type, entrypoint, and
 * the local governance policy it wants to layer on top of the global one.
 *
 * Manifests are stored as JSON files at:
 *   .mitosis/agents/<id>/manifest.json   (project-local)
 *   ~/.mitosis/agents/<id>/manifest.json  (user-global)
 */

import type { Policy } from '../runtime/governance/types.js';

// ---------------------------------------------------------------------------
// Capability declarations
// ---------------------------------------------------------------------------

/**
 * A tool capability declared in the manifest.
 * The actual schema is provided at runtime by the adapter's listTools().
 */
export interface ToolCapabilityDecl {
  /** Unique name matching what the adapter will expose via listTools(). */
  name: string;
  /** Short description used during registry capability aggregation. */
  description?: string;
}

/**
 * A skill capability declared in the manifest.
 */
export interface SkillCapabilityDecl {
  /** Skill id matching what the adapter exposes via listSkills(). */
  id: string;
  description?: string;
  category?: string;
  tags?: string[];
}

/**
 * An MCP server the subagent wants to connect to.
 * The registry will start the server process and pass the connection
 * handle to the adapter via connectMCP().
 */
export interface MCPServerDecl {
  /** Logical name for this MCP server within the subagent. */
  name: string;
  /** Command to start the server (e.g. 'npx', 'python'). */
  command: string;
  /** Arguments passed to the command. */
  args?: string[];
  /** Extra environment variables for the server process. */
  env?: Record<string, string>;
  /** Reconnect delay in ms on unexpected exit (default 5000, 0 = no reconnect). */
  reconnectDelayMs?: number;
}

// ---------------------------------------------------------------------------
// LLM configuration
// ---------------------------------------------------------------------------

export type LLMProviderKind = 'openai' | 'anthropic' | 'openai-compatible';

export interface LLMConfig {
  provider: LLMProviderKind;
  model: string;
  /** Resolved from env if omitted (e.g. OPENAI_API_KEY). */
  apiKeyEnv?: string;
  /** Base URL override for openai-compatible providers. */
  baseUrl?: string;
  /** Max tokens to request. */
  maxTokens?: number;
  /** Default temperature (0-1). */
  temperature?: number;
}

// ---------------------------------------------------------------------------
// Transport
// ---------------------------------------------------------------------------

export type TransportKind = 'in-process' | 'stdio' | 'http' | 'websocket';

export interface InProcessEntrypoint {
  transport: 'in-process';
  /**
   * Absolute or workspace-relative path to the module exporting a
   * `createSubagentAdapter(): SubagentAdapter` function.
   */
  module: string;
}

export interface StdioEntrypoint {
  transport: 'stdio';
  /** Command to launch the subagent process. */
  command: string;
  args?: string[];
  env?: Record<string, string>;
}

export interface HttpEntrypoint {
  transport: 'http';
  /** Base URL of the subagent HTTP server. */
  baseUrl: string;
  /** Optional bearer token env var name. */
  authTokenEnv?: string;
  /** Request timeout in ms. */
  timeoutMs?: number;
}

export interface WebSocketEntrypoint {
  transport: 'websocket';
  url: string;
  /** Optional bearer token env var name. */
  authTokenEnv?: string;
  /** Reconnect delay in ms (default 3000, 0 = no reconnect). */
  reconnectDelayMs?: number;
}

export type Entrypoint =
  | InProcessEntrypoint
  | StdioEntrypoint
  | HttpEntrypoint
  | WebSocketEntrypoint;

// ---------------------------------------------------------------------------
// Capabilities requested from the host
// ---------------------------------------------------------------------------

/**
 * Fine-grained capability grants that the subagent must declare before
 * receiving access to privileged host resources. The registry validates
 * these against the global CapabilityGrant policy at registration time.
 */
export type CapabilityScope =
  | 'fs:read'
  | 'fs:write'
  | 'net:fetch'
  | 'shell:exec'
  | 'mempedia:read'
  | 'mempedia:write'
  | 'llm:call'
  | 'mcp:connect';

// ---------------------------------------------------------------------------
// The Manifest
// ---------------------------------------------------------------------------

/** Semantic version string, e.g. "1.2.3". */
export type SemVer = string;

export interface SubagentManifest {
  /** Unique stable identifier for this subagent, e.g. "workspace-tools". */
  id: string;
  /** Semver version of this subagent. */
  version: SemVer;
  /** Human-readable display name. */
  name: string;
  /** Short description shown in /agents list. */
  description: string;
  /** Subagent author for marketplace attribution. */
  author?: string;
  /** Homepage / repository URL. */
  homepage?: string;

  /** Transport + entrypoint definition. */
  entrypoint: Entrypoint;

  /**
   * Capabilities this subagent exposes to the main agent.
   * Registered tools and skills are merged into the core agent's planner catalog.
   */
  capabilities: {
    /** Tools the subagent provides. */
    tools?: ToolCapabilityDecl[];
    /** Skills the subagent provides (SKILL.md equivalents). */
    skills?: SkillCapabilityDecl[];
    /** MCP servers the subagent connects to and re-exposes. */
    mcpServers?: MCPServerDecl[];
    /** Optional dedicated LLM the subagent uses instead of the core LLM. */
    llm?: LLMConfig;
  };

  /**
   * Host capabilities this subagent requires.
   * The registry validates these before completing registration.
   */
  requires?: CapabilityScope[];

  /**
   * Optional local governance policy layered on top of the global policy.
   * Only more-restrictive rules are honoured (deny > ask > allow relative to global).
   */
  governance?: {
    localPolicy?: Policy;
  };

  /**
   * Other subagent ids that must be registered before this one.
   * The registry resolves the dependency graph at load time.
   */
  dependencies?: string[];

  /** If false the subagent is loaded but not exposed to the planner. Default true. */
  enabled?: boolean;

  /** Free-form metadata for UI / marketplace filtering. */
  metadata?: Record<string, unknown>;
}

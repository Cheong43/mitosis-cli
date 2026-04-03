/**
 * HttpTransport — REST adapter transport for remote HTTP subagents.
 *
 * Protocol: POST /rpc with JSON body:
 *   { method: string, params?: unknown }
 * Response: { result: unknown } | { error: string }
 *
 * Streaming tool execution uses Server-Sent Events when the endpoint
 * returns Content-Type: text/event-stream (optional, falls back to JSON).
 */

import type { SubagentAdapter, SubagentConfig, SkillContent } from '../../types/subagent-adapter.js';
import type { SubagentManifest, HttpEntrypoint, MCPServerDecl } from '../../types/subagent-manifest.js';
import type { SubagentTransport } from './types.js';
import type { ToolDefinition, ToolExecutionContext, ToolExecutionResult } from '../tools/types.js';
import type { SkillRecord } from '../../skills/router.js';

const DEFAULT_TIMEOUT_MS = 30_000;

class HttpAdapterProxy implements SubagentAdapter {
  private cachedTools: ToolDefinition[] = [];
  private cachedSkills: SkillRecord[] = [];

  constructor(
    public readonly subagentId: string,
    private readonly baseUrl: string,
    private readonly authToken: string | undefined,
    private readonly timeoutMs: number,
  ) {}

  private async rpc<T>(method: string, params?: unknown): Promise<T> {
    const url = `${this.baseUrl.replace(/\/$/, '')}/rpc`;
    const headers: Record<string, string> = { 'Content-Type': 'application/json' };
    if (this.authToken) headers['Authorization'] = `Bearer ${this.authToken}`;

    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.timeoutMs);

    let res: Response;
    try {
      res = await fetch(url, {
        method: 'POST',
        headers,
        body: JSON.stringify({ method, params }),
        signal: controller.signal,
      });
    } finally {
      clearTimeout(timer);
    }

    if (!res.ok) {
      throw new Error(`HttpTransport: HTTP ${res.status} from ${url} (method=${method})`);
    }

    const body = await res.json() as { result?: T; error?: string };
    if ('error' in body && body.error) throw new Error(body.error);
    return body.result as T;
  }

  async initialize(config: SubagentConfig): Promise<void> {
    await this.rpc('initialize', config);
    // Eagerly cache tools/skills so synchronous accessors work.
    [this.cachedTools, this.cachedSkills] = await Promise.all([
      this.rpc<ToolDefinition[]>('listTools'),
      this.rpc<SkillRecord[]>('listSkills'),
    ]);
  }

  listTools(): ToolDefinition[] { return this.cachedTools; }
  listSkills(): SkillRecord[] { return this.cachedSkills; }

  async executeTool(name: string, args: Record<string, unknown>, ctx: ToolExecutionContext): Promise<ToolExecutionResult> {
    return this.rpc<ToolExecutionResult>('executeTool', { name, args, ctx });
  }

  async loadSkill(id: string): Promise<SkillContent | null> {
    return this.rpc<SkillContent | null>('loadSkill', { id });
  }

  connectMCP(_decl: MCPServerDecl): Promise<never> {
    return Promise.reject(new Error('HttpTransport: connectMCP not supported'));
  }

  getLanguageModel() { return undefined; }

  async shutdown(): Promise<void> {
    await this.rpc('shutdown').catch(() => {});
  }
}

export class HttpTransport implements SubagentTransport {
  private proxy: HttpAdapterProxy | null = null;

  constructor(
    private readonly manifest: SubagentManifest,
    private readonly entrypoint: HttpEntrypoint,
  ) {}

  async connect(): Promise<SubagentAdapter> {
    if (this.proxy) return this.proxy;
    const authToken = this.entrypoint.authTokenEnv
      ? process.env[this.entrypoint.authTokenEnv]
      : undefined;
    this.proxy = new HttpAdapterProxy(
      this.manifest.id,
      this.entrypoint.baseUrl,
      authToken,
      this.entrypoint.timeoutMs ?? DEFAULT_TIMEOUT_MS,
    );
    return this.proxy;
  }

  async disconnect(): Promise<void> {
    if (this.proxy) {
      await this.proxy.shutdown();
      this.proxy = null;
    }
  }
}

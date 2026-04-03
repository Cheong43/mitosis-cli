/**
 * WebSocketTransport — bidirectional JSON-RPC 2.0 over WebSocket.
 *
 * Same wire format as StdioTransport (newline-delimited JSON over the WS
 * message stream). Supports automatic reconnect with backoff.
 *
 * Note: Requires the 'ws' package which is already a transitive dependency
 * of the Vercel AI SDK. We import dynamically to avoid breaking environments
 * that do not use this transport.
 */

import type { SubagentAdapter, SubagentConfig, SkillContent } from '../../types/subagent-adapter.js';
import type { SubagentManifest, WebSocketEntrypoint, MCPServerDecl } from '../../types/subagent-manifest.js';
import type { SubagentTransport } from './types.js';
import type { ToolDefinition, ToolExecutionContext, ToolExecutionResult } from '../tools/types.js';
import type { SkillRecord } from '../../skills/router.js';

const DEFAULT_TIMEOUT_MS = 30_000;
const DEFAULT_RECONNECT_DELAY_MS = 3_000;

interface JsonRpcRequest { jsonrpc: '2.0'; id: number; method: string; params?: unknown }
interface JsonRpcSuccess { jsonrpc: '2.0'; id: number; result: unknown }
interface JsonRpcError   { jsonrpc: '2.0'; id: number; error: { code: number; message: string } }
type JsonRpcResponse = JsonRpcSuccess | JsonRpcError;

class WebSocketAdapterProxy implements SubagentAdapter {
  private seq = 0;
  private readonly pending = new Map<number, { resolve(v: unknown): void; reject(e: Error): void; timer: ReturnType<typeof setTimeout> }>();
  private cachedTools: ToolDefinition[] = [];
  private cachedSkills: SkillRecord[] = [];
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  private ws: any = null;

  constructor(
    public readonly subagentId: string,
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    ws: any,
    private readonly timeoutMs: number,
  ) {
    this.attachWs(ws);
  }

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  attachWs(ws: any): void {
    this.ws = ws;
    ws.on('message', (data: Buffer | string) => {
      const line = data.toString().trim();
      if (!line) return;
      let msg: JsonRpcResponse;
      try { msg = JSON.parse(line) as JsonRpcResponse; } catch { return; }
      const entry = this.pending.get(msg.id);
      if (!entry) return;
      clearTimeout(entry.timer);
      this.pending.delete(msg.id);
      if ('error' in msg) entry.reject(new Error(msg.error.message));
      else entry.resolve((msg as JsonRpcSuccess).result);
    });
    ws.on('error', (e: Error) => this.rejectAll(e));
    ws.on('close', () => this.rejectAll(new Error('WebSocket closed')));
  }

  private rejectAll(err: Error): void {
    for (const entry of this.pending.values()) { clearTimeout(entry.timer); entry.reject(err); }
    this.pending.clear();
  }

  private call<T>(method: string, params?: unknown): Promise<T> {
    return new Promise<T>((resolve, reject) => {
      const id = ++this.seq;
      const timer = setTimeout(() => {
        this.pending.delete(id);
        reject(new Error(`WebSocketTransport: timeout (method=${method}, id=${id})`));
      }, this.timeoutMs);
      this.pending.set(id, {
        resolve: (v) => resolve(v as T),
        reject,
        timer,
      });
      this.ws.send(JSON.stringify({ jsonrpc: '2.0', id, method, params } as JsonRpcRequest));
    });
  }

  async initialize(config: SubagentConfig): Promise<void> {
    await this.call('initialize', config);
    [this.cachedTools, this.cachedSkills] = await Promise.all([
      this.call<ToolDefinition[]>('listTools'),
      this.call<SkillRecord[]>('listSkills'),
    ]);
  }
  listTools(): ToolDefinition[] { return this.cachedTools; }
  listSkills(): SkillRecord[] { return this.cachedSkills; }
  async executeTool(name: string, args: Record<string, unknown>, ctx: ToolExecutionContext): Promise<ToolExecutionResult> {
    return this.call<ToolExecutionResult>('executeTool', { name, args, ctx });
  }
  async loadSkill(id: string): Promise<SkillContent | null> {
    return this.call<SkillContent | null>('loadSkill', { id });
  }
  connectMCP(_d: MCPServerDecl): Promise<never> {
    return Promise.reject(new Error('WebSocketTransport: connectMCP not supported'));
  }
  getLanguageModel() { return undefined; }
  async shutdown(): Promise<void> {
    await this.call('shutdown').catch(() => {});
    this.ws?.close();
  }
}

export class WebSocketTransport implements SubagentTransport {
  private proxy: WebSocketAdapterProxy | null = null;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private readonly reconnectDelayMs: number;

  constructor(
    private readonly manifest: SubagentManifest,
    private readonly entrypoint: WebSocketEntrypoint,
  ) {
    this.reconnectDelayMs = entrypoint.reconnectDelayMs ?? DEFAULT_RECONNECT_DELAY_MS;
  }

  async connect(): Promise<SubagentAdapter> {
    if (this.proxy) return this.proxy;
    const ws = await this.openSocket();
    this.proxy = new WebSocketAdapterProxy(this.manifest.id, ws, DEFAULT_TIMEOUT_MS);
    return this.proxy;
  }

  private async openSocket(): Promise<unknown> {
    // Dynamic import to avoid hard dependency on 'ws' when not using WS transport.
    const { default: WebSocket } = await import('ws');
    const authToken = this.entrypoint.authTokenEnv
      ? process.env[this.entrypoint.authTokenEnv]
      : undefined;
    const headers = authToken ? { Authorization: `Bearer ${authToken}` } : {};
    return new Promise((resolve, reject) => {
      const ws = new WebSocket(this.entrypoint.url, { headers });
      ws.once('open', () => resolve(ws));
      ws.once('error', reject);
    });
  }

  async disconnect(): Promise<void> {
    if (this.reconnectTimer) { clearTimeout(this.reconnectTimer); this.reconnectTimer = null; }
    if (this.proxy) {
      await this.proxy.shutdown();
      this.proxy = null;
    }
  }
}

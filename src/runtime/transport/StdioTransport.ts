/**
 * StdioTransport — JSON-RPC 2.0 over stdin/stdout for subprocess subagents.
 *
 * Protocol (newline-delimited JSON):
 *   → Request:  {"jsonrpc":"2.0","id":N,"method":"<m>","params":{...}}
 *   ← Response: {"jsonrpc":"2.0","id":N,"result":{...}} | {"jsonrpc":"2.0","id":N,"error":{...}}
 *
 * Methods exposed:  initialize, listTools, listSkills, executeTool, loadSkill,
 *                   getLanguageModel, shutdown
 *
 * The subprocess must write exactly one JSON response line per request.
 */

import { spawn, type ChildProcess } from 'node:child_process';
import * as readline from 'node:readline';
import type { SubagentAdapter, SubagentConfig, SkillContent, MCPConnection } from '../../types/subagent-adapter.js';
import type { SubagentManifest, StdioEntrypoint, MCPServerDecl } from '../../types/subagent-manifest.js';
import type { SubagentTransport } from './types.js';
import type { ToolDefinition, ToolExecutionContext, ToolExecutionResult } from '../tools/types.js';
import type { SkillRecord } from '../../skills/router.js';

const DEFAULT_TIMEOUT_MS = 30_000;

interface JsonRpcRequest {
  jsonrpc: '2.0';
  id: number;
  method: string;
  params?: unknown;
}

interface JsonRpcSuccess {
  jsonrpc: '2.0';
  id: number;
  result: unknown;
}

interface JsonRpcError {
  jsonrpc: '2.0';
  id: number;
  error: { code: number; message: string };
}

type JsonRpcResponse = JsonRpcSuccess | JsonRpcError;

class StdioAdapterProxy implements SubagentAdapter {
  private seq = 0;
  private readonly pending = new Map<number, { resolve(v: unknown): void; reject(e: Error): void }>();
  private rl: readline.Interface | null = null;

  constructor(
    public readonly subagentId: string,
    private readonly proc: ChildProcess,
    private readonly timeoutMs: number = DEFAULT_TIMEOUT_MS,
  ) {
    this.rl = readline.createInterface({ input: proc.stdout! });
    this.rl.on('line', (line) => this.handleLine(line));
    proc.on('error', (e) => this.rejectAll(e));
    proc.on('exit', (code) => this.rejectAll(new Error(`stdio subagent exited with code ${code}`)));
  }

  private handleLine(line: string): void {
    const trimmed = line.trim();
    if (!trimmed) return;
    let msg: JsonRpcResponse;
    try {
      msg = JSON.parse(trimmed) as JsonRpcResponse;
    } catch {
      return; // ignore malformed lines (could be debug output)
    }
    const entry = this.pending.get(msg.id);
    if (!entry) return;
    this.pending.delete(msg.id);
    if ('error' in msg) {
      entry.reject(new Error(msg.error.message));
    } else {
      entry.resolve((msg as JsonRpcSuccess).result);
    }
  }

  private rejectAll(err: Error): void {
    for (const entry of this.pending.values()) entry.reject(err);
    this.pending.clear();
  }

  private call<T>(method: string, params?: unknown): Promise<T> {
    return new Promise<T>((resolve, reject) => {
      const id = ++this.seq;
      const req: JsonRpcRequest = { jsonrpc: '2.0', id, method, params };
      const timer = setTimeout(() => {
        this.pending.delete(id);
        reject(new Error(`StdioTransport: timeout waiting for ${method} (id=${id})`));
      }, this.timeoutMs);
      this.pending.set(id, {
        resolve: (v) => { clearTimeout(timer); resolve(v as T); },
        reject: (e) => { clearTimeout(timer); reject(e); },
      });
      this.proc.stdin!.write(JSON.stringify(req) + '\n');
    });
  }

  async initialize(config: SubagentConfig): Promise<void> {
    await this.call('initialize', config);
  }
  listTools(): ToolDefinition[] {
    throw new Error('StdioAdapterProxy.listTools must be called after initialize (use async listToolsAsync)');
  }
  async listToolsAsync(): Promise<ToolDefinition[]> {
    return this.call<ToolDefinition[]>('listTools');
  }
  listSkills(): SkillRecord[] {
    throw new Error('StdioAdapterProxy.listSkills must be called after initialize (use async listSkillsAsync)');
  }
  async listSkillsAsync(): Promise<SkillRecord[]> {
    return this.call<SkillRecord[]>('listSkills');
  }
  async executeTool(name: string, args: Record<string, unknown>, ctx: ToolExecutionContext): Promise<ToolExecutionResult> {
    return this.call<ToolExecutionResult>('executeTool', { name, args, ctx });
  }
  async loadSkill(id: string): Promise<SkillContent | null> {
    return this.call<SkillContent | null>('loadSkill', { id });
  }
  async connectMCP(_decl: MCPServerDecl): Promise<MCPConnection> {
    throw new Error('StdioTransport: connectMCP delegation not supported for subprocess adapters');
  }
  getLanguageModel() { return undefined; }
  async shutdown(): Promise<void> {
    await this.call('shutdown').catch(() => {});
    this.rl?.close();
    this.proc.stdin?.destroy();
    this.proc.kill();
  }
}

export class StdioTransport implements SubagentTransport {
  private proxy: StdioAdapterProxy | null = null;

  constructor(
    private readonly manifest: SubagentManifest,
    private readonly entrypoint: StdioEntrypoint,
  ) {}

  async connect(): Promise<SubagentAdapter> {
    if (this.proxy) return this.proxy;
    const proc = spawn(this.entrypoint.command, this.entrypoint.args ?? [], {
      env: { ...process.env, ...(this.entrypoint.env ?? {}) },
      stdio: ['pipe', 'pipe', 'inherit'],
    });
    this.proxy = new StdioAdapterProxy(this.manifest.id, proc);
    return this.proxy;
  }

  async disconnect(): Promise<void> {
    if (this.proxy) {
      await this.proxy.shutdown();
      this.proxy = null;
    }
  }
}

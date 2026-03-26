import * as fs from 'node:fs';
import * as path from 'node:path';
import { execFile, spawn, type ChildProcessWithoutNullStreams } from 'node:child_process';
import * as readline from 'node:readline';

import type { ToolAction, ToolResponse } from './types.js';

type TransportMode = 'ndjson' | 'oneshot' | 'unavailable';

interface PendingRequest {
  resolve: (value: ToolResponse | Record<string, unknown>) => void;
  reject: (error: Error) => void;
  timer: NodeJS.Timeout;
}

export interface MempediaTransportStatus {
  binaryAvailable: boolean;
  binaryPath: string | null;
  transportConnected: boolean;
  memoryWriteEnabled: boolean;
  transportMode: TransportMode;
  lastError?: string;
}

const sharedTransports = new Map<string, MempediaTransport>();

function keyForTransport(projectRoot: string, binaryPath?: string): string {
  return `${path.resolve(projectRoot)}::${binaryPath?.trim() || ''}`;
}

export function resolveMempediaBinaryPath(projectRoot: string, preferredBinaryPath?: string): string | null {
  const explicit = preferredBinaryPath?.trim()
    || process.env.MEMPEDIA_BINARY?.trim()
    || process.env.MEMPEDIA_BINARY_PATH?.trim();
  if (explicit) {
    return fs.existsSync(explicit) ? explicit : null;
  }

  const candidates = [
    path.join(projectRoot, 'target', 'debug', 'mempedia'),
    path.join(projectRoot, 'target', 'release', 'mempedia'),
    path.join(projectRoot, '..', 'target', 'debug', 'mempedia'),
    path.join(projectRoot, '..', 'target', 'release', 'mempedia'),
    path.join(projectRoot, '..', '..', 'target', 'debug', 'mempedia'),
    path.join(projectRoot, '..', '..', 'target', 'release', 'mempedia'),
  ].map((candidate) => path.resolve(candidate));

  const found = [...new Set(candidates)].filter((candidate) => {
    try {
      return fs.existsSync(candidate);
    } catch {
      return false;
    }
  });

  if (found.length === 0) {
    return null;
  }

  found.sort((left, right) => {
    const leftStat = fs.statSync(left);
    const rightStat = fs.statSync(right);
    if (rightStat.mtimeMs !== leftStat.mtimeMs) {
      return rightStat.mtimeMs - leftStat.mtimeMs;
    }
    return left.includes(`${path.sep}release${path.sep}`) ? 1 : -1;
  });

  return found[0];
}

function isToolResponse(value: unknown): value is ToolResponse {
  return Boolean(value)
    && typeof value === 'object'
    && typeof (value as any).kind === 'string';
}

export class MempediaTransport {
  private readonly projectRoot: string;
  private readonly preferredBinaryPath?: string;
  private child: ChildProcessWithoutNullStreams | null = null;
  private stdoutReader: readline.Interface | null = null;
  private pending: PendingRequest[] = [];
  private mode: TransportMode = 'unavailable';
  private transportConnected = false;
  private lastError?: string;

  constructor(projectRoot: string, binaryPath?: string) {
    this.projectRoot = path.resolve(projectRoot);
    this.preferredBinaryPath = binaryPath;
  }

  getStatusSnapshot(): MempediaTransportStatus {
    const binaryPath = resolveMempediaBinaryPath(this.projectRoot, this.preferredBinaryPath);
    const binaryAvailable = Boolean(binaryPath);
    const transportMode = binaryAvailable ? this.mode : 'unavailable';
    const transportConnected = binaryAvailable && transportMode === 'ndjson' && this.transportConnected;
    const memoryWriteEnabled = binaryAvailable && (transportMode === 'ndjson' || transportMode === 'oneshot');
    return {
      binaryAvailable,
      binaryPath,
      transportConnected,
      memoryWriteEnabled,
      transportMode,
      ...(this.lastError ? { lastError: this.lastError } : {}),
    };
  }

  start(): void {
    const binaryPath = resolveMempediaBinaryPath(this.projectRoot, this.preferredBinaryPath);
    if (!binaryPath) {
      this.mode = 'unavailable';
      this.transportConnected = false;
      this.lastError = 'mempedia binary not found';
      return;
    }
    if (this.child && !this.child.killed && this.child.exitCode === null) {
      return;
    }

    try {
      const child = spawn(binaryPath, ['--project', this.projectRoot, '--serve'], {
        stdio: ['pipe', 'pipe', 'pipe'],
      });
      this.child = child;
      this.mode = 'ndjson';
      this.transportConnected = true;
      this.lastError = undefined;

      this.stdoutReader = readline.createInterface({ input: child.stdout });
      this.stdoutReader.on('line', (line) => this.handleStdoutLine(line));

      child.stderr.on('data', (chunk: Buffer | string) => {
        const text = String(chunk || '').trim();
        if (text) {
          this.lastError = text;
        }
      });

      child.on('error', (error) => {
        this.mode = 'oneshot';
        this.transportConnected = false;
        this.lastError = error.message;
        this.flushPendingWithError(`mempedia transport error: ${error.message}`);
        this.cleanupChild();
      });

      child.on('exit', (code, signal) => {
        const exitDetail = signal ? `signal ${signal}` : `code ${String(code ?? 'unknown')}`;
        this.transportConnected = false;
        this.lastError = this.lastError || `mempedia transport exited with ${exitDetail}`;
        if (this.mode === 'ndjson') {
          this.mode = 'oneshot';
        }
        this.flushPendingWithError(`mempedia transport exited with ${exitDetail}`);
        this.cleanupChild();
      });
    } catch (error: any) {
      this.mode = 'oneshot';
      this.transportConnected = false;
      this.lastError = error?.message || String(error);
    }
  }

  async send(action: ToolAction | Record<string, unknown>): Promise<ToolResponse | Record<string, unknown>> {
    const binaryPath = resolveMempediaBinaryPath(this.projectRoot, this.preferredBinaryPath);
    if (!binaryPath) {
      this.mode = 'unavailable';
      this.transportConnected = false;
      this.lastError = 'mempedia binary not found';
      return { kind: 'error', message: 'mitosis-cli: Mempedia binary not found' };
    }

    this.start();
    if (this.child && this.transportConnected && this.child.stdin.writable) {
      try {
        return await this.sendViaNdjson(action);
      } catch (error: any) {
        this.lastError = error?.message || String(error);
      }
    }

    return this.sendViaOneShot(binaryPath, action);
  }

  async probe(): Promise<MempediaTransportStatus> {
    const base = this.getStatusSnapshot();
    if (!base.binaryAvailable) {
      return base;
    }

    const response = await this.send({ action: 'read_user_preferences' });
    const snapshot = this.getStatusSnapshot();
    if (isToolResponse(response) && response.kind === 'error') {
      return {
        ...snapshot,
        memoryWriteEnabled: false,
        lastError: response.message,
      };
    }
    return snapshot;
  }

  stop(): void {
    const child = this.child;
    if (child && !child.killed && child.exitCode === null) {
      try {
        child.stdin.write(':quit\n');
      } catch {}
      child.kill();
    }
    this.mode = resolveMempediaBinaryPath(this.projectRoot, this.preferredBinaryPath) ? 'oneshot' : 'unavailable';
    this.transportConnected = false;
    this.flushPendingWithError('mempedia transport stopped');
    this.cleanupChild();
  }

  private sendViaNdjson(action: ToolAction | Record<string, unknown>): Promise<ToolResponse | Record<string, unknown>> {
    return new Promise((resolve, reject) => {
      if (!this.child || !this.child.stdin.writable) {
        reject(new Error('mempedia transport is not writable'));
        return;
      }
      const timeoutMs = Number(process.env.MEMPEDIA_TRANSPORT_TIMEOUT_MS ?? 20000);
      const timer = setTimeout(() => {
        const index = this.pending.findIndex((entry) => entry.resolve === resolve);
        if (index >= 0) {
          this.pending.splice(index, 1);
        }
        reject(new Error(`mempedia transport timeout after ${timeoutMs}ms`));
      }, Number.isFinite(timeoutMs) && timeoutMs > 0 ? timeoutMs : 20000);
      this.pending.push({ resolve, reject, timer });
      try {
        this.child.stdin.write(`${JSON.stringify(action)}\n`);
      } catch (error: any) {
        clearTimeout(timer);
        this.pending.pop();
        reject(error instanceof Error ? error : new Error(String(error)));
      }
    });
  }

  private async sendViaOneShot(binaryPath: string, action: ToolAction | Record<string, unknown>): Promise<ToolResponse | Record<string, unknown>> {
    const timeoutMs = Number(process.env.MEMPEDIA_TRANSPORT_TIMEOUT_MS ?? 20000);
    this.mode = 'oneshot';
    this.transportConnected = false;

    return await new Promise((resolve) => {
      execFile(
        binaryPath,
        ['--project', this.projectRoot, '--action', JSON.stringify(action)],
        {
          encoding: 'utf-8',
          timeout: Number.isFinite(timeoutMs) && timeoutMs > 0 ? timeoutMs : 20000,
          maxBuffer: 1024 * 1024,
        },
        (error, stdout, stderr) => {
          if (error) {
            const message = String(stderr || error.message || 'mempedia binary execution failed').trim();
            this.lastError = message;
            resolve({ kind: 'error', message });
            return;
          }
          const text = String(stdout || '').trim();
          if (!text) {
            this.lastError = 'mempedia binary produced no output';
            resolve({ kind: 'error', message: 'mempedia binary produced no output' });
            return;
          }
          try {
            const parsed = JSON.parse(text) as ToolResponse | Record<string, unknown>;
            if (isToolResponse(parsed) && parsed.kind === 'error') {
              this.lastError = parsed.message;
            } else {
              this.lastError = undefined;
            }
            resolve(parsed);
          } catch {
            this.lastError = `could not parse mempedia output: ${text.slice(0, 200)}`;
            resolve({ kind: 'error', message: this.lastError });
          }
        },
      );
    });
  }

  private handleStdoutLine(line: string): void {
    const pending = this.pending.shift();
    if (!pending) {
      return;
    }
    clearTimeout(pending.timer);
    const payload = line.trim();
    if (!payload) {
      pending.resolve({ kind: 'error', message: 'mempedia transport produced no output' });
      return;
    }
    try {
      const parsed = JSON.parse(payload) as ToolResponse | Record<string, unknown>;
      if (isToolResponse(parsed) && parsed.kind === 'error') {
        this.lastError = parsed.message;
      } else {
        this.lastError = undefined;
      }
      pending.resolve(parsed);
    } catch {
      const message = `could not parse mempedia output: ${payload.slice(0, 200)}`;
      this.lastError = message;
      pending.resolve({ kind: 'error', message });
    }
  }

  private flushPendingWithError(message: string): void {
    const pending = this.pending.splice(0);
    for (const entry of pending) {
      clearTimeout(entry.timer);
      entry.reject(new Error(message));
    }
  }

  private cleanupChild(): void {
    if (this.stdoutReader) {
      this.stdoutReader.removeAllListeners();
      this.stdoutReader.close();
      this.stdoutReader = null;
    }
    this.child = null;
  }
}

export function getSharedMempediaTransport(projectRoot: string, binaryPath?: string): MempediaTransport {
  const key = keyForTransport(projectRoot, binaryPath);
  const existing = sharedTransports.get(key);
  if (existing) {
    return existing;
  }
  const transport = new MempediaTransport(projectRoot, binaryPath);
  sharedTransports.set(key, transport);
  return transport;
}

export async function getMempediaTransportStatus(projectRoot: string, binaryPath?: string): Promise<MempediaTransportStatus> {
  const transport = getSharedMempediaTransport(projectRoot, binaryPath);
  return transport.probe();
}

export function stopAllSharedMempediaTransports(): void {
  for (const transport of sharedTransports.values()) {
    transport.stop();
  }
  sharedTransports.clear();
}

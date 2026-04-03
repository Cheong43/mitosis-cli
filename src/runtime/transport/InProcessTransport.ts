/**
 * InProcessTransport — zero-overhead in-process adapter transport.
 *
 * For builtin subagents that live in the same Node.js process, there is no
 * serialization cost. The adapter instance is held directly in memory and
 * called as ordinary async functions.
 */

import type { SubagentAdapter, SubagentAdapterFactory, SubagentConfig } from '../../types/subagent-adapter.js';
import type { SubagentManifest, InProcessEntrypoint } from '../../types/subagent-manifest.js';
import type { SubagentTransport } from './types.js';
import { createRequire } from 'node:module';
import * as path from 'node:path';
import { pathToFileURL } from 'node:url';

export class InProcessTransport implements SubagentTransport {
  private adapter: SubagentAdapter | null = null;

  constructor(
    private readonly manifest: SubagentManifest,
    private readonly entrypoint: InProcessEntrypoint,
  ) {}

  async connect(): Promise<SubagentAdapter> {
    if (this.adapter) return this.adapter;

    const modulePath = path.isAbsolute(this.entrypoint.module)
      ? this.entrypoint.module
      : path.resolve(process.cwd(), this.entrypoint.module);

    // Dynamic import — supports both ESM and CJS via the URL form.
    const moduleUrl = pathToFileURL(modulePath).href;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const mod: any = await import(moduleUrl);

    const factory: SubagentAdapterFactory = mod.default ?? mod.createSubagentAdapter;
    if (typeof factory !== 'function') {
      throw new Error(
        `InProcessTransport: module '${this.entrypoint.module}' must export a default function or 'createSubagentAdapter'.`,
      );
    }

    this.adapter = factory();
    return this.adapter;
  }

  async disconnect(): Promise<void> {
    if (this.adapter) {
      await this.adapter.shutdown();
      this.adapter = null;
    }
  }
}

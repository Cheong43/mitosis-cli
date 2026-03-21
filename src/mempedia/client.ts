import { ToolAction, ToolResponse } from './types.js';

/** Stub: MitosisCLI operates without the Mempedia Rust binary. */
export class MempediaClient {
  constructor(private projectRoot: string, private binaryPath?: string) {}

  start() {}

  async send(_action: ToolAction): Promise<ToolResponse> {
    return { kind: 'error', message: 'mitosis-cli: Mempedia binary operations are not available' } as unknown as ToolResponse;
  }

  stop() {}
}

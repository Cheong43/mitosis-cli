import { ApprovalEngine, GovernanceRequest } from './types.js';

/**
 * An `ApprovalEngine` backed by an async callback function.
 *
 * This is the interactive counterpart of `InMemoryApprovalEngine`.  Instead of
 * resolving `ask` decisions automatically, it invokes a caller-provided callback
 * that can display a confirmation prompt in the CLI or Web UI.
 *
 * The callback receives a structured `ApprovalPrompt` object and must return
 * `'allow'` or `'deny'`.
 */
export interface ApprovalPrompt {
  /** Name of the tool requesting approval. */
  toolName: string;
  /** Raw arguments the tool will be called with. */
  args: Record<string, unknown>;
  /** Human-readable reason the governance layer is asking. */
  reason: string;
}

export type ApprovalCallback = (prompt: ApprovalPrompt) => Promise<'allow' | 'deny'>;

export class CallbackApprovalEngine implements ApprovalEngine {
  private readonly callback: ApprovalCallback;

  constructor(callback: ApprovalCallback) {
    this.callback = callback;
  }

  async ask(request: GovernanceRequest, reason: string): Promise<'allow' | 'deny'> {
    return this.callback({ toolName: request.toolName, args: request.args, reason });
  }
}

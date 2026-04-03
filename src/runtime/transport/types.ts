/**
 * Shared transport types.
 */

import type { SubagentAdapter } from '../../types/subagent-adapter.js';

/**
 * Every transport implementation must produce a SubagentAdapter handle
 * on connect() and cleanly tear down on disconnect().
 */
export interface SubagentTransport {
  /** Establish the connection and return an adapter you can call. */
  connect(): Promise<SubagentAdapter>;
  /** Tear down the connection (idempotent). */
  disconnect(): Promise<void>;
}

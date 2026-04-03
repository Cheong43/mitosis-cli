/**
 * Transport factory — creates the right transport from a manifest entrypoint.
 */

export type { SubagentTransport } from './types.js';
export { InProcessTransport } from './InProcessTransport.js';
export { StdioTransport } from './StdioTransport.js';
export { HttpTransport } from './HttpTransport.js';
export { WebSocketTransport } from './WebSocketTransport.js';

import type { SubagentManifest, Entrypoint } from '../../types/subagent-manifest.js';
import type { SubagentTransport } from './types.js';
import { InProcessTransport } from './InProcessTransport.js';
import { StdioTransport } from './StdioTransport.js';
import { HttpTransport } from './HttpTransport.js';
import { WebSocketTransport } from './WebSocketTransport.js';

/**
 * Given a manifest, construct and return the appropriate transport.
 * Does NOT call connect() — the caller controls when the connection is opened.
 */
export function createTransport(manifest: SubagentManifest): SubagentTransport {
  const ep: Entrypoint = manifest.entrypoint;
  switch (ep.transport) {
    case 'in-process':
      return new InProcessTransport(manifest, ep);
    case 'stdio':
      return new StdioTransport(manifest, ep);
    case 'http':
      return new HttpTransport(manifest, ep);
    case 'websocket':
      return new WebSocketTransport(manifest, ep);
    default: {
      const _exhaustive: never = ep;
      throw new Error(`createTransport: unknown transport kind '${(_exhaustive as Entrypoint).transport}'`);
    }
  }
}

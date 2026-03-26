import { ToolAction, ToolResponse } from './types.js';
import {
  getMempediaTransportStatus,
  getSharedMempediaTransport,
  type MempediaTransportStatus,
} from './transport.js';

export class MempediaClient {
  constructor(private projectRoot: string, private binaryPath?: string) {}

  start() {
    getSharedMempediaTransport(this.projectRoot, this.binaryPath).start();
  }

  async send(action: ToolAction): Promise<ToolResponse> {
    return await getSharedMempediaTransport(this.projectRoot, this.binaryPath).send(action) as ToolResponse;
  }

  async status(): Promise<MempediaTransportStatus> {
    return getMempediaTransportStatus(this.projectRoot, this.binaryPath);
  }

  stop() {
    getSharedMempediaTransport(this.projectRoot, this.binaryPath).stop();
  }
}

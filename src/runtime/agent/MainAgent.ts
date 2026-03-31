import type { AgentStep, TranscriptMessage } from './types.js';

export interface MainAgent {
  plan(transcript: TranscriptMessage[]): Promise<AgentStep>;
}

// Backward compatibility alias
export type Planner = MainAgent;

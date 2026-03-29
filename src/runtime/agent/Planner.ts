import type { AgentStep, TranscriptMessage } from './types.js';

export interface Planner {
  plan(transcript: TranscriptMessage[]): Promise<AgentStep>;
}

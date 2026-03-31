import type { BranchOutcome } from '../runtime/agent/types.js';

export type ArtifactCategory = 'research' | 'plan' | 'impl' | 'test' | 'review' | 'doc';

export interface ArtifactMetadata {
  id: string;
  category: ArtifactCategory;
  name: string;
  path: string;
  branchId: string;
  timestamp: string;
  tags: string[];
  relatedArtifacts: string[];
}

export interface BranchWorkRecord {
  branchId: string;
  parentId: string | null;
  startTime: string;
  endTime: string;
  outcome: BranchOutcome;
  artifacts: string[];
  summary: string;
}

export interface SubagentTemplate {
  name: string;
  description: string;
  version: string;
  systemPrompt: string;
  skills: string[];
  tools: string[];
  outputCategory: ArtifactCategory;
  contextBudget: number;
}

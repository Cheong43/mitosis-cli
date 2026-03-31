import * as fs from 'fs';
import * as path from 'path';
import type { AgentBranchState } from '../runtime/agent/AgentRuntime.js';
import type { ArtifactMetadata, BranchWorkRecord, ArtifactCategory } from './types.js';

export class WorkspaceManager {
  private mitosisDir: string;
  private branchesDir: string;

  constructor(private projectRoot: string) {
    this.mitosisDir = path.join(projectRoot, '.mitosis');
    this.branchesDir = path.join(this.mitosisDir, 'branches');
  }

  private ensureDir(dir: string): void {
    if (!fs.existsSync(dir)) {
      fs.mkdirSync(dir, { recursive: true });
    }
  }

  async saveBranchWork(branchId: string, state: AgentBranchState): Promise<void> {
    const branchDir = path.join(this.branchesDir, branchId);
    this.ensureDir(branchDir);

    const record: BranchWorkRecord = {
      branchId: state.id,
      parentId: state.parentId,
      startTime: new Date().toISOString(),
      endTime: new Date().toISOString(),
      outcome: state.outcome!,
      artifacts: state.artifacts || [],
      summary: state.completionSummary || state.finalAnswer || '',
    };

    fs.writeFileSync(
      path.join(branchDir, 'metadata.json'),
      JSON.stringify(record, null, 2)
    );

    const transcriptPath = path.join(branchDir, 'transcript.jsonl');
    const transcriptLines = state.transcript.map((msg: any) => JSON.stringify(msg)).join('\n');
    fs.writeFileSync(transcriptPath, transcriptLines);

    if (state.artifacts && state.artifacts.length > 0) {
      fs.writeFileSync(
        path.join(branchDir, 'artifacts.json'),
        JSON.stringify({ artifacts: state.artifacts }, null, 2)
      );
    }
  }

  async saveArtifact(
    category: ArtifactCategory,
    name: string,
    content: string,
    branchId: string = 'main',
    tags: string[] = []
  ): Promise<string> {
    // Save to project root, not .mitosis
    const filepath = path.join(this.projectRoot, name);
    fs.writeFileSync(filepath, content);

    // Record metadata in .mitosis for tracking
    const branchDir = path.join(this.branchesDir, branchId);
    this.ensureDir(branchDir);

    const artifactRecord = {
      path: filepath,
      category,
      timestamp: new Date().toISOString(),
      tags,
    };

    const recordPath = path.join(branchDir, 'artifacts.json');
    let records = [];
    if (fs.existsSync(recordPath)) {
      records = JSON.parse(fs.readFileSync(recordPath, 'utf-8')).artifacts || [];
    }
    records.push(artifactRecord);
    fs.writeFileSync(recordPath, JSON.stringify({ artifacts: records }, null, 2));

    return filepath;
  }

  async loadRelevantWork(query: string, limit: number = 5): Promise<ArtifactMetadata[]> {
    const artifacts: ArtifactMetadata[] = [];

    // Load from branch metadata in .mitosis
    if (!fs.existsSync(this.branchesDir)) return artifacts;

    const branches = fs.readdirSync(this.branchesDir);
    for (const branchId of branches) {
      const recordPath = path.join(this.branchesDir, branchId, 'artifacts.json');
      if (!fs.existsSync(recordPath)) continue;

      const records = JSON.parse(fs.readFileSync(recordPath, 'utf-8')).artifacts || [];
      for (const record of records) {
        artifacts.push({
          id: path.basename(record.path),
          category: record.category,
          name: path.basename(record.path),
          path: record.path,
          branchId,
          timestamp: record.timestamp,
          tags: record.tags || [],
          relatedArtifacts: [],
        });
      }
    }

    return artifacts.slice(0, limit);
  }

  async listArtifacts(category?: ArtifactCategory): Promise<ArtifactMetadata[]> {
    const artifacts: ArtifactMetadata[] = [];

    if (!fs.existsSync(this.branchesDir)) return artifacts;

    const branches = fs.readdirSync(this.branchesDir);
    for (const branchId of branches) {
      const recordPath = path.join(this.branchesDir, branchId, 'artifacts.json');
      if (!fs.existsSync(recordPath)) continue;

      const records = JSON.parse(fs.readFileSync(recordPath, 'utf-8')).artifacts || [];
      for (const record of records) {
        if (category && record.category !== category) continue;

        artifacts.push({
          id: path.basename(record.path),
          category: record.category,
          name: path.basename(record.path),
          path: record.path,
          branchId,
          timestamp: record.timestamp,
          tags: record.tags || [],
          relatedArtifacts: [],
        });
      }
    }

    return artifacts;
  }
}

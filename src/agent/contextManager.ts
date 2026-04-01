/**
 * Context manager for external memory storage via Mempedia.
 *
 * Moves completed branch transcripts to persistent storage and retrieves
 * summaries on demand, reducing in-memory context usage.
 */
import type { MempediaClient } from '../mempedia/client.js';

export interface BranchArchive {
  branchId: string;
  label: string;
  goal: string;
  transcript: Array<{ role: string; content: any }>;
  outcome?: string;
  outcomeReason?: string;
  completionSummary?: string;
}

export class ContextManager {
  private archiveCache = new Map<string, string>();

  constructor(
    private mempedia: MempediaClient,
    private projectRoot: string,
  ) {}

  /**
   * Archive a completed branch's transcript to Mempedia and return a
   * compact reference string suitable for insertion into the transcript.
   */
  async archiveCompletedBranch(branch: BranchArchive): Promise<string> {
    const summary = this.buildBranchSummary(branch);
    const archiveKey = `branch_archive_${branch.branchId}_${Date.now()}`;

    try {
      await this.mempedia.send({
        action: 'ingest',
        node_id: archiveKey,
        title: `Branch archive: ${branch.label}`,
        text: JSON.stringify({
          branchId: branch.branchId,
          label: branch.label,
          goal: branch.goal,
          outcome: branch.outcome,
          outcomeReason: branch.outcomeReason,
          completionSummary: branch.completionSummary,
          transcriptLength: branch.transcript.length,
          summary,
        }),
        source: 'context_manager',
        importance: 3,
      });

      this.archiveCache.set(archiveKey, summary);
      return `[Branch "${branch.label}" archived (${archiveKey}): ${summary}]`;
    } catch {
      // Fallback to inline summary if Mempedia unavailable
      return `[Branch "${branch.label}" summary: ${summary}]`;
    }
  }

  /**
   * Retrieve and summarize an archived branch.
   */
  async retrieveBranchContext(archiveKey: string): Promise<string> {
    const cached = this.archiveCache.get(archiveKey);
    if (cached) return cached;

    try {
      const response = await this.mempedia.send({
        action: 'search_nodes',
        query: archiveKey,
        limit: 1,
      });

      const content = typeof response === 'object' && response !== null
        ? String((response as Record<string, unknown>).content || '')
        : '';

      if (content) {
        this.archiveCache.set(archiveKey, content);
        return content;
      }
    } catch {
      // Mempedia unavailable
    }

    return `[Archive ${archiveKey}: content unavailable]`;
  }

  private buildBranchSummary(branch: BranchArchive): string {
    const parts: string[] = [];
    parts.push(`goal="${branch.goal}"`);
    if (branch.outcome) parts.push(`outcome=${branch.outcome}`);
    if (branch.outcomeReason) parts.push(`reason="${branch.outcomeReason}"`);
    if (branch.completionSummary) parts.push(`summary="${branch.completionSummary}"`);
    parts.push(`steps=${branch.transcript.length}`);
    return parts.join(', ');
  }
}

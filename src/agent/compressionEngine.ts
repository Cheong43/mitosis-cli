/**
 * CompressionEngine — Four-level progressive context compression pipeline.
 *
 * Levels (ordered from cheapest to most expensive):
 *   1. Snip        — Replace unreferenced old tool results with placeholders (zero API cost).
 *   2. Microcompact — Aggressively compress individual tool results by type (zero API cost).
 *   3. Collapse    — Read-time projection: fold repeated tool-call groups (zero API cost).
 *   4. Autocompact — LLM-driven conversation summary as last resort (LLM cost, irreversible).
 *
 * The `compress()` method runs all four levels in order, stopping as soon as
 * the transcript fits within the requested `budgetChars`.
 */

import {
  extractTranscriptContentText,
  estimateTranscriptTokens,
} from './contextBudget.js';
import type { TranscriptMessage } from '../runtime/agent/types.js';

// ── Types ──────────────────────────────────────────────────────────────────

export type { TranscriptMessage };

export interface CompressionEngineOptions {
  // Snip
  snipStaleThreshold?: number;  // default 8 rounds
  snipRecentKeep?: number;      // default 6 messages

  // Microcompact
  microcompactMaxChars?: number;  // default 400

  // Collapse
  collapseThreshold?: number;      // consecutive tool-call groups to collapse, default 3
  collapsePreserveRecent?: number; // don't collapse last N messages, default 6

  // Autocompact
  autocompactMaxSummaryTokens?: number;  // default 800
  autocompactPreserveRecent?: number;    // default 4

  // LLM callback required for autocompact
  generateSummary?: (text: string, maxTokens: number) => Promise<string>;
}

export interface SnipResult {
  messages: TranscriptMessage[];
  snippedCount: number;
  freedChars: number;
}

export interface CompressionResult {
  messages: TranscriptMessage[];
  level: 'none' | 'snip' | 'microcompact' | 'collapse' | 'autocompact';
  freedChars: number;
  snippedCount: number;
  collapsedGroups: number;
  autocompacted: boolean;
}

// ── Helpers ────────────────────────────────────────────────────────────────

function totalChars(messages: TranscriptMessage[]): number {
  return messages.reduce((s, m) => s + extractTranscriptContentText(m.content).length, 0);
}

function extractActiveRefs(messages: TranscriptMessage[], recentKeep: number): Set<string> {
  const refs = new Set<string>();
  const recentMessages = messages.slice(-recentKeep);
  for (const m of recentMessages) {
    const text = extractTranscriptContentText(m.content);
    // File paths
    const filePaths = text.match(/[\w./\-_]+\.\w{1,6}/g) ?? [];
    for (const p of filePaths) refs.add(p);
    // URLs (bare hostnames + paths)
    const urls = text.match(/https?:\/\/[^\s"'<>]+/g) ?? [];
    for (const u of urls) refs.add(u);
    // JSON-like identifiers / keywords (length ≥ 4)
    const words = text.match(/\b[\w\-]{4,}\b/g) ?? [];
    for (const w of words) refs.add(w.toLowerCase());
  }
  return refs;
}

// ── Level 1: Snip ─────────────────────────────────────────────────────────

/**
 * Replace stale, unreferenced tool results with a one-line placeholder.
 * Never touches the first message or the most recent `recentKeep` messages.
 */
export function snipStaleToolResults(
  messages: TranscriptMessage[],
  options?: { staleThreshold?: number; recentKeep?: number },
): SnipResult {
  const staleThreshold = options?.staleThreshold ?? 8;
  const recentKeep = options?.recentKeep ?? 6;

  if (messages.length <= recentKeep + 1) {
    return { messages, snippedCount: 0, freedChars: 0 };
  }

  const activeRefs = extractActiveRefs(messages, recentKeep);
  const cutoffIndex = messages.length - recentKeep;

  let snippedCount = 0;
  let freedChars = 0;

  const result = messages.map((m, index) => {
    // Never snip the first message or recent messages
    if (index === 0 || index >= cutoffIndex) return m;
    // Only snip tool observations older than staleThreshold
    if (index >= messages.length - staleThreshold) return m;

    const contentText = extractTranscriptContentText(m.content);
    if (m.role !== 'user') return m;
    if (!contentText.startsWith('TOOL OBSERVATION')) return m;

    // Check if any active ref appears in this observation
    const lower = contentText.toLowerCase();
    for (const ref of activeRefs) {
      if (lower.includes(ref)) return m;
    }

    // Snip it
    const firstLine = contentText.split('\n')[0];
    const toolName = firstLine.replace('TOOL OBSERVATION for ', '').replace(':', '').split(' ')[0] || 'tool';
    const charCount = contentText.length;
    const placeholder = `[snipped: ${toolName} result, ${charCount} chars, step ~${index}]`;
    snippedCount++;
    freedChars += charCount - placeholder.length;
    return { ...m, content: placeholder };
  });

  return { messages: result, snippedCount, freedChars };
}

// ── Level 2: Microcompact ─────────────────────────────────────────────────

/**
 * Compress individual tool results more aggressively based on tool type.
 * Applied only to messages in the "middle" (not first, not recent).
 */
export function microcompactToolResult(
  toolName: string,
  result: string,
  maxChars = 400,
): string {
  if (result.length <= maxChars) return result;

  const tool = toolName.toLowerCase();

  if (tool === 'bash') {
    return microcompactBash(result, maxChars);
  }
  if (tool === 'read') {
    return microcompactRead(result, maxChars);
  }
  if (tool === 'search' || tool === 'grep') {
    return microcompactSearch(result, maxChars);
  }
  if (tool === 'web') {
    return microcompactWeb(result, maxChars);
  }
  return genericMicrocompact(result, maxChars);
}

function microcompactBash(output: string, maxChars: number): string {
  const lines = output.split('\n');
  if (lines.length <= 20) return genericMicrocompact(output, maxChars);

  const errorLines = lines.filter((l) =>
    /error|fail|exception|panic|traceback|ENOENT|EPERM|npm ERR/i.test(l),
  ).slice(0, 10);
  const tail = lines.slice(-20);

  const parts: string[] = [];
  if (errorLines.length > 0) {
    parts.push('Errors:');
    parts.push(...errorLines.map((l) => l.slice(0, 200)));
  }
  parts.push(`[${lines.length} lines total, last 20:]`);
  parts.push(...tail);

  const joined = parts.join('\n');
  return joined.length > maxChars ? joined.slice(0, maxChars) + '\n...' : joined;
}

function microcompactRead(content: string, maxChars: number): string {
  const lines = content.split('\n');
  if (lines.length <= 30) return genericMicrocompact(content, maxChars);
  const half = Math.floor(maxChars / 2 / 80);
  const head = lines.slice(0, half).join('\n');
  const tail = lines.slice(-half).join('\n');
  return `${head}\n[... ${lines.length - half * 2} lines omitted ...]\n${tail}`.slice(0, maxChars);
}

function microcompactSearch(results: string, maxChars: number): string {
  const lines = results.split('\n');
  const matchLines = lines.filter((l) => /^\S+:\d+:/.test(l) || /^(Match|Result|Found)/i.test(l));
  const kept = matchLines.slice(0, 20).join('\n');
  const suffix = matchLines.length > 20
    ? `\n[${matchLines.length - 20} more matches omitted]`
    : '';
  return (kept + suffix).slice(0, maxChars);
}

function microcompactWeb(content: string, maxChars: number): string {
  const cleaned = content
    .replace(/\n{3,}/g, '\n\n')
    .replace(/^\s*(cookie|privacy|terms|copyright|©).*$/gmi, '')
    .replace(/\[.*?(sign up|log in|subscribe).*?\]/gi, '')
    .trim();
  return genericMicrocompact(cleaned, maxChars);
}

function genericMicrocompact(text: string, maxChars: number): string {
  if (text.length <= maxChars) return text;
  const head = Math.floor(maxChars * 0.6);
  const tail = Math.floor(maxChars * 0.3);
  return `${text.slice(0, head)}\n[... ${text.length - head - tail} chars omitted ...]\n${text.slice(-tail)}`;
}

/**
 * Apply microcompact to all "middle" tool observation messages.
 */
export function microcompactTranscript(
  messages: TranscriptMessage[],
  options?: { maxChars?: number; recentKeep?: number },
): TranscriptMessage[] {
  const maxChars = options?.maxChars ?? 400;
  const recentKeep = options?.recentKeep ?? 6;
  const cutoffIndex = messages.length - recentKeep;

  return messages.map((m, index) => {
    if (index === 0 || index >= cutoffIndex) return m;
    const contentText = extractTranscriptContentText(m.content);
    if (m.role !== 'user' || !contentText.startsWith('TOOL OBSERVATION')) return m;

    // Extract tool name from header line
    const headerLine = contentText.split('\n')[0];
    const toolNameMatch = headerLine.match(/TOOL OBSERVATION for ([^\s:]+)/);
    const toolName = toolNameMatch?.[1] ?? 'tool';
    const bodyStart = contentText.indexOf('\n') + 1;
    const body = bodyStart > 0 ? contentText.slice(bodyStart) : contentText;
    const compressedBody = microcompactToolResult(toolName, body, maxChars);
    const newContent = typeof m.content === 'string'
      ? `${headerLine}\n${compressedBody}`
      : { ...m.content, text: `${headerLine}\n${compressedBody}` };
    return { ...m, content: newContent };
  });
}

// ── Level 3: Context Collapse (read-time projection) ─────────────────────

/**
 * Collapse repeated consecutive tool-call groups into a compact summary.
 * This is non-destructive: returns a new array view, original messages untouched.
 *
 * Groups N consecutive (assistant tool-decision + user observation) pairs
 * when the tool name repeats, folding them into one summary entry.
 */
export function collapseTranscript(
  messages: TranscriptMessage[],
  options?: { collapseThreshold?: number; preserveRecent?: number },
): TranscriptMessage[] {
  const threshold = options?.collapseThreshold ?? 3;
  const preserveRecent = options?.preserveRecent ?? 6;

  if (messages.length <= preserveRecent + 1) return messages;

  const protectedStart = messages.length - preserveRecent;
  const workingMessages = messages.slice(0, protectedStart);
  const recentMessages = messages.slice(protectedStart);

  // Detect runs of repeated tool pairs
  const result: TranscriptMessage[] = [];
  let i = 0;

  while (i < workingMessages.length) {
    const m = workingMessages[i];

    // Look for a run of (assistant=tool-decision, user=tool-observation) pairs
    if (i === 0) {
      result.push(m);
      i++;
      continue;
    }

    if (m.role === 'assistant') {
      const toolName = extractToolNameFromDecision(m.content);
      if (!toolName) {
        result.push(m);
        i++;
        continue;
      }

      // Count how many consecutive pairs share the same tool name
      let runLen = 0;
      let j = i;
      while (
        j + 1 < workingMessages.length &&
        workingMessages[j].role === 'assistant' &&
        extractToolNameFromDecision(workingMessages[j].content) === toolName &&
        workingMessages[j + 1].role === 'user' &&
        extractTranscriptContentText(workingMessages[j + 1].content).startsWith('TOOL OBSERVATION')
      ) {
        runLen++;
        j += 2;
      }

      if (runLen >= threshold) {
        // Collect first + last pair verbatim, summarize middle
        const firstPair = workingMessages.slice(i, i + 2);
        const lastPair = workingMessages.slice(j - 2, j);
        const middle = workingMessages.slice(i + 2, j - 2);
        const summaryLine = `[collapsed: ${runLen} consecutive "${toolName}" calls, ${middle.length} middle pairs omitted]`;
        result.push(...firstPair);
        result.push({ role: 'user', content: summaryLine });
        result.push(...lastPair);
        i = j;
      } else {
        result.push(m);
        i++;
      }
    } else {
      result.push(m);
      i++;
    }
  }

  return [...result, ...recentMessages];
}

function extractToolNameFromDecision(content: any): string | null {
  const text = extractTranscriptContentText(content);
  if (typeof text !== 'string') return null;

  // New JSON planner format
  try {
    const parsed = JSON.parse(text);
    if (parsed?.kind === 'tool' && Array.isArray(parsed.tool_calls) && parsed.tool_calls.length > 0) {
      return String(parsed.tool_calls[0]?.name ?? '').trim() || null;
    }
  } catch {
    // not JSON
  }

  // Legacy "PLANNER TOOL DECISION:\n- toolName" format
  if (text.startsWith('PLANNER TOOL DECISION:')) {
    const line = text.split('\n').find((l) => l.trim().startsWith('- '));
    const name = line?.trim().slice(2).split(/[|\s]/)[0]?.trim();
    return name || null;
  }
  return null;
}

// ── Level 4: Autocompact ──────────────────────────────────────────────────

/**
 * Summarize the entire transcript into a compact LLM-generated summary.
 * Irreversible. Only called when all prior levels fail to fit `budgetChars`.
 *
 * @param messages - Current transcript
 * @param generateSummary - Async LLM callback
 * @param options - Autocompact options
 */
export async function autocompactTranscript(
  messages: TranscriptMessage[],
  generateSummary: (text: string, maxTokens: number) => Promise<string>,
  options?: {
    maxSummaryTokens?: number;
    preserveRecent?: number;
  },
): Promise<TranscriptMessage[]> {
  const maxSummaryTokens = options?.maxSummaryTokens ?? 800;
  const preserveRecent = options?.preserveRecent ?? 4;

  if (messages.length <= preserveRecent + 1) return messages;

  const toSummarize = messages.slice(1, messages.length - preserveRecent);
  const recent = messages.slice(messages.length - preserveRecent);
  const first = messages[0];

  const summaryInput = toSummarize
    .map((m) => `[${m.role}] ${extractTranscriptContentText(m.content).slice(0, 1000)}`)
    .join('\n\n');

  let summary: string;
  try {
    summary = await generateSummary(summaryInput, maxSummaryTokens);
  } catch {
    // If LLM call fails, fall back to mechanical truncation
    summary = `[${toSummarize.length} messages — autocompact failed, mechanical summary: ${summaryInput.slice(0, 600)}...]`;
  }

  const summaryMsg: TranscriptMessage = {
    role: 'user',
    content: `[AUTOCOMPACT SUMMARY — ${toSummarize.length} messages compressed]\n${summary}`,
  };

  return [first, summaryMsg, ...recent];
}

// ── CompressionEngine ─────────────────────────────────────────────────────

/**
 * Unified four-level progressive compression engine.
 *
 * Designed to be instantiated once per Agent run and shared across all
 * phases (plan, execute, finalize, synthesis).
 */
export class CompressionEngine {
  private readonly snipStaleThreshold: number;
  private readonly snipRecentKeep: number;
  private readonly microcompactMaxChars: number;
  private readonly collapseThreshold: number;
  private readonly collapsePreserveRecent: number;
  private readonly autocompactMaxSummaryTokens: number;
  private readonly autocompactPreserveRecent: number;
  private readonly generateSummary?: (text: string, maxTokens: number) => Promise<string>;

  constructor(options?: CompressionEngineOptions) {
    this.snipStaleThreshold = options?.snipStaleThreshold ?? 8;
    this.snipRecentKeep = options?.snipRecentKeep ?? 6;
    this.microcompactMaxChars = options?.microcompactMaxChars ?? 400;
    this.collapseThreshold = options?.collapseThreshold ?? 3;
    this.collapsePreserveRecent = options?.collapsePreserveRecent ?? 6;
    this.autocompactMaxSummaryTokens = options?.autocompactMaxSummaryTokens ?? 800;
    this.autocompactPreserveRecent = options?.autocompactPreserveRecent ?? 4;
    this.generateSummary = options?.generateSummary;
  }

  /**
   * Execute the four-level pipeline in order.
   * Stops as soon as the transcript fits within `budgetChars`.
   */
  async compress(
    messages: TranscriptMessage[],
    budgetChars: number,
  ): Promise<CompressionResult> {
    const initialChars = totalChars(messages);
    if (initialChars <= budgetChars) {
      return {
        messages,
        level: 'none',
        freedChars: 0,
        snippedCount: 0,
        collapsedGroups: 0,
        autocompacted: false,
      };
    }

    // Level 1: Snip
    const { messages: sniped, snippedCount, freedChars: snipFreed } = this.snip(messages);
    if (totalChars(sniped) <= budgetChars) {
      return {
        messages: sniped,
        level: 'snip',
        freedChars: initialChars - totalChars(sniped),
        snippedCount,
        collapsedGroups: 0,
        autocompacted: false,
      };
    }

    // Level 2: Microcompact
    const microed = this.microcompact(sniped);
    if (totalChars(microed) <= budgetChars) {
      return {
        messages: microed,
        level: 'microcompact',
        freedChars: initialChars - totalChars(microed),
        snippedCount,
        collapsedGroups: 0,
        autocompacted: false,
      };
    }

    // Level 3: Collapse
    const collapsed = this.collapse(microed, budgetChars);
    const collapsedGroups = countCollapsedGroups(collapsed);
    if (totalChars(collapsed) <= budgetChars) {
      return {
        messages: collapsed,
        level: 'collapse',
        freedChars: initialChars - totalChars(collapsed),
        snippedCount,
        collapsedGroups,
        autocompacted: false,
      };
    }

    // Level 4: Autocompact (requires LLM callback)
    if (this.generateSummary) {
      const autocompacted = await this.autocompact(collapsed);
      return {
        messages: autocompacted,
        level: 'autocompact',
        freedChars: initialChars - totalChars(autocompacted),
        snippedCount,
        collapsedGroups,
        autocompacted: true,
      };
    }

    // No LLM available — return best effort (collapsed)
    return {
      messages: collapsed,
      level: 'collapse',
      freedChars: initialChars - totalChars(collapsed),
      snippedCount,
      collapsedGroups,
      autocompacted: false,
    };
  }

  /** Level 1 only */
  snip(messages: TranscriptMessage[]): SnipResult {
    return snipStaleToolResults(messages, {
      staleThreshold: this.snipStaleThreshold,
      recentKeep: this.snipRecentKeep,
    });
  }

  /** Level 2 only */
  microcompact(messages: TranscriptMessage[]): TranscriptMessage[] {
    return microcompactTranscript(messages, {
      maxChars: this.microcompactMaxChars,
      recentKeep: this.snipRecentKeep,
    });
  }

  /** Level 3 only */
  collapse(messages: TranscriptMessage[], _budgetChars?: number): TranscriptMessage[] {
    return collapseTranscript(messages, {
      collapseThreshold: this.collapseThreshold,
      preserveRecent: this.collapsePreserveRecent,
    });
  }

  /** Level 4 only */
  async autocompact(messages: TranscriptMessage[]): Promise<TranscriptMessage[]> {
    if (!this.generateSummary) return messages;
    return autocompactTranscript(messages, this.generateSummary, {
      maxSummaryTokens: this.autocompactMaxSummaryTokens,
      preserveRecent: this.autocompactPreserveRecent,
    });
  }
}

// ── Internal helpers ───────────────────────────────────────────────────────

function countCollapsedGroups(messages: TranscriptMessage[]): number {
  return messages.filter((m) => {
    const t = extractTranscriptContentText(m.content);
    return typeof t === 'string' && t.startsWith('[collapsed:');
  }).length;
}

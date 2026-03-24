/**
 * Per-session context carry-over for exhausted ReAct runs.
 *
 * When a single Agent.run() turn exhausts all steps/budget without reaching
 * a natural `kind=final` answer, the SessionCompressor compresses the run's
 * key findings into a compact carry-over block.  On the next turn for the
 * same conversationId the carry-over is injected so the agent can continue
 * from where it left off without re-exploring already-visited paths.
 *
 * Lifecycle:
 *   1. Before `run()`: retrieve carry-over via `getCarryOver(conversationId)`.
 *   2. After `run()`:  if exhausted, call `recordExhaustedRun(...)`.
 *                       if completed normally, call `clearCarryOver(...)`.
 */

import { estimateTokens } from './contextBudget.js';

// ── Types ──────────────────────────────────────────────────────────────────

export interface TraceEventLike {
  type: string;
  content: string;
  metadata?: Record<string, unknown>;
}

export interface ExhaustedRunRecord {
  /** Original user request that triggered the run. */
  userRequest: string;
  /** Compressed summary of what was explored / discovered. */
  summary: string;
  /** Tool calls that were executed (name + goal). */
  toolTrace: string[];
  /** Any partial answers or branch completion summaries. */
  partialAnswers: string[];
  /** Timestamp of the exhaustion. */
  timestamp: number;
  /** Token estimate of this record when serialized. */
  tokenEstimate: number;
}

export interface SessionCarryOver {
  /** Accumulated carry-over text ready for injection. */
  text: string;
  /** Number of exhausted runs accumulated. */
  runCount: number;
  /** Total estimated tokens in the carry-over. */
  tokenEstimate: number;
}

// ── Exhaustion detection ───────────────────────────────────────────────────

/**
 * Heuristics to determine if a run ended because it exhausted its step budget
 * rather than arriving at a natural final answer.
 *
 * Signals checked:
 *   1. Trace contains "budget reached" patterns from AgentRuntime.
 *   2. Trace contains "Forcing finalization" (the finalizeBranch fallback).
 *   3. Final answer contains "stopped because" placeholder text.
 *   4. Trace contains "Maximum depth reached" from BeamSearch.
 */
export function isRunExhausted(traces: TraceEventLike[], finalAnswer: string): boolean {
  const exhaustionPatterns = [
    /budget reached/i,
    /Forcing finalization/i,
    /Maximum (?:depth|steps?) reached/i,
  ];

  const answerExhaustionPatterns = [
    /stopped because .*(budget|step)/i,
    /Maximum (?:depth|steps?) reached without/i,
  ];

  const traceHit = traces.some((t) =>
    exhaustionPatterns.some((p) => p.test(t.content)),
  );

  const answerHit = answerExhaustionPatterns.some((p) => p.test(finalAnswer));

  return traceHit || answerHit;
}

// ── Compression ────────────────────────────────────────────────────────────

/**
 * Compress a run's traces + answer into a compact carry-over record.
 *
 * Strategy:
 *   - Extract unique tool calls (name + goal) — at most 12.
 *   - Extract any partial final answers or completion summaries.
 *   - Extract key observations (first 200 chars of each, at most 6).
 *   - Combine into a structured text block.
 */
export function compressExhaustedRun(
  userRequest: string,
  traces: TraceEventLike[],
  finalAnswer: string,
  maxChars = 3000,
): ExhaustedRunRecord {
  // 1. Collect tool calls.
  const toolTrace: string[] = [];
  const seenTools = new Set<string>();
  for (const t of traces) {
    if (t.type !== 'action') continue;
    const toolName = String(t.metadata?.toolName || '');
    const goal = t.metadata?.goal || '';
    const key = `${toolName}:${goal}`;
    if (toolName && !seenTools.has(key)) {
      seenTools.add(key);
      toolTrace.push(goal ? `${toolName} — ${goal}` : toolName);
    }
    if (toolTrace.length >= 12) break;
  }

  // 2. Collect partial answers / completion summaries from branch observations.
  const partialAnswers: string[] = [];
  for (const t of traces) {
    if (t.type === 'observation' && t.content.includes('Branch completed')) {
      const branchLabel = String(t.metadata?.branchLabel || '');
      const branchId = String(t.metadata?.branchId || '');
      partialAnswers.push(`[${branchId}/${branchLabel}] completed`);
    }
  }
  // Add the forced final answer (which may be a partial result).
  if (finalAnswer && finalAnswer.length > 20) {
    const clipped = finalAnswer.length > 600
      ? finalAnswer.slice(0, 600) + '...[truncated]'
      : finalAnswer;
    partialAnswers.push(clipped);
  }

  // 3. Collect key observations (unique tool results, skip noise).
  const keyObservations: string[] = [];
  for (const t of traces) {
    if (t.type !== 'observation') continue;
    if (t.content.length < 30) continue;
    if (/^(Recalled|Selected|Auto-queued|Context budget|Perf total|Skill router|Model .* does not support)/i.test(t.content)) continue;
    const snippet = t.content.length > 200
      ? t.content.slice(0, 200) + '...'
      : t.content;
    keyObservations.push(snippet);
    if (keyObservations.length >= 6) break;
  }

  // 4. Build summary.
  const parts: string[] = [
    `## Previous exhausted run`,
    `Request: ${userRequest.slice(0, 300)}`,
  ];

  if (toolTrace.length > 0) {
    parts.push(`Tools used: ${toolTrace.join('; ')}`);
  }

  if (keyObservations.length > 0) {
    parts.push(`Key observations:\n${keyObservations.map((o) => `- ${o}`).join('\n')}`);
  }

  if (partialAnswers.length > 0) {
    parts.push(`Partial results:\n${partialAnswers.map((a) => `- ${a}`).join('\n')}`);
  }

  parts.push(`Status: Run exhausted step budget. Continue from these findings.`);

  let summary = parts.join('\n');
  const truncSuffix = '\n...[carry-over truncated]';
  if (summary.length > maxChars) {
    summary = summary.slice(0, maxChars - truncSuffix.length) + truncSuffix;
  }

  return {
    userRequest: userRequest.slice(0, 300),
    summary,
    toolTrace,
    partialAnswers,
    timestamp: Date.now(),
    tokenEstimate: estimateTokens(summary),
  };
}

// ── SessionCompressor ──────────────────────────────────────────────────────

export class SessionCompressor {
  /**
   * Per-conversationId carry-over state.
   * Key = conversationId, Value = list of exhausted run records.
   */
  private readonly carryOverByConversation = new Map<string, ExhaustedRunRecord[]>();

  /** Maximum number of exhausted runs to accumulate before discarding oldest. */
  private readonly maxExhaustedRuns: number;

  /** Maximum total carry-over tokens per conversation. */
  private readonly maxCarryOverTokens: number;

  /** Maximum characters for a single run compression. */
  private readonly maxRunChars: number;

  constructor(options: {
    maxExhaustedRuns?: number;
    maxCarryOverTokens?: number;
    maxRunChars?: number;
  } = {}) {
    this.maxExhaustedRuns = options.maxExhaustedRuns ?? 3;
    this.maxCarryOverTokens = options.maxCarryOverTokens ?? 6000;
    this.maxRunChars = options.maxRunChars ?? 3000;
  }

  /**
   * Record an exhausted run for carry-over into the next turn.
   */
  recordExhaustedRun(
    conversationId: string,
    userRequest: string,
    traces: TraceEventLike[],
    finalAnswer: string,
  ): ExhaustedRunRecord {
    const record = compressExhaustedRun(userRequest, traces, finalAnswer, this.maxRunChars);

    const existing = this.carryOverByConversation.get(conversationId) || [];
    existing.push(record);

    // Trim to budget: drop oldest if over count or token limit.
    while (existing.length > this.maxExhaustedRuns) {
      existing.shift();
    }
    let totalTokens = existing.reduce((s, r) => s + r.tokenEstimate, 0);
    while (totalTokens > this.maxCarryOverTokens && existing.length > 1) {
      const removed = existing.shift()!;
      totalTokens -= removed.tokenEstimate;
    }

    this.carryOverByConversation.set(conversationId, existing);
    return record;
  }

  /**
   * Get the accumulated carry-over for a conversation, ready for injection
   * into the next turn's system prompt or context.
   *
   * Returns null if no carry-over exists.
   */
  getCarryOver(conversationId: string): SessionCarryOver | null {
    const records = this.carryOverByConversation.get(conversationId);
    if (!records || records.length === 0) return null;

    const text = records.map((r) => r.summary).join('\n\n---\n\n');
    const tokenEstimate = records.reduce((s, r) => s + r.tokenEstimate, 0);

    return {
      text,
      runCount: records.length,
      tokenEstimate,
    };
  }

  /**
   * Clear carry-over for a conversation (called when a run completes normally).
   */
  clearCarryOver(conversationId: string): void {
    this.carryOverByConversation.delete(conversationId);
  }

  /**
   * Check if a conversation has accumulated carry-over.
   */
  hasCarryOver(conversationId: string): boolean {
    const records = this.carryOverByConversation.get(conversationId);
    return !!records && records.length > 0;
  }
}

/**
 * Dynamic context budget calculator.
 *
 * Estimates token usage, derives optimal step/depth/width parameters, and
 * provides transcript compression for both Branching ReAct and Beam Search
 * agents based on the current session context length and model context limit.
 */

// ── Model context window registry ──────────────────────────────────────────

/** Known model context windows (tokens). */
const MODEL_CONTEXT_LIMITS: Record<string, number> = {
  // OpenAI
  'gpt-4o': 128_000,
  'gpt-4o-mini': 128_000,
  'gpt-4-turbo': 128_000,
  'gpt-4-turbo-preview': 128_000,
  'gpt-4': 8_192,
  'gpt-4-32k': 32_768,
  'gpt-3.5-turbo': 16_385,
  'o1': 200_000,
  'o1-mini': 128_000,
  'o1-preview': 128_000,
  'o3': 200_000,
  'o3-mini': 200_000,
  'o4-mini': 200_000,
  // Anthropic
  'claude-3-opus': 200_000,
  'claude-3-5-sonnet': 200_000,
  'claude-3-7-sonnet': 200_000,
  'claude-3-haiku': 200_000,
  'claude-4-sonnet': 200_000,
  'claude-opus-4': 200_000,
  // Google
  'gemini-1.5-pro': 1_000_000,
  'gemini-1.5-flash': 1_000_000,
  'gemini-2.0-flash': 1_000_000,
  // DeepSeek
  'deepseek-chat': 64_000,
  'deepseek-reasoner': 64_000,
  // Qwen
  'qwen-plus': 131_072,
  'qwen-max': 32_768,
  'qwen-turbo': 131_072,
  'qwen3-235b-a22b': 131_072,
  'qwen3-235b': 131_072,
  'qwen3-32b': 131_072,
  'qwen3': 131_072,
};

/**
 * Look up the context window for a model name.
 * Matches by prefix so "gpt-4o-2024-08-06" hits "gpt-4o".
 * Falls back to `fallback` (default 128k) if unknown.
 */
export function getModelContextLimit(model: string, fallback = 128_000): number {
  const envOverride = Number(process.env.MODEL_CONTEXT_LIMIT);
  if (Number.isFinite(envOverride) && envOverride > 0) {
    return envOverride;
  }

  const lower = model.toLowerCase();

  // Exact match first.
  if (MODEL_CONTEXT_LIMITS[lower] !== undefined) {
    return MODEL_CONTEXT_LIMITS[lower];
  }

  // Prefix match: longest matching prefix wins.
  let bestLen = 0;
  let bestLimit = fallback;
  for (const [key, limit] of Object.entries(MODEL_CONTEXT_LIMITS)) {
    if (lower.startsWith(key) && key.length > bestLen) {
      bestLen = key.length;
      bestLimit = limit;
    }
  }

  return bestLimit;
}

// ── Token estimation ───────────────────────────────────────────────────────

/** Approximate token count from a string (mixed CJK / Latin). */
export function estimateTokens(text: string): number {
  if (!text) return 0;
  // CJK characters ≈ 1–2 tokens each.  Latin ≈ 1 token per ~4 chars.
  // We use a hybrid heuristic: count CJK chars separately.
  let cjkChars = 0;
  for (let i = 0; i < text.length; i++) {
    const code = text.charCodeAt(i);
    if (
      (code >= 0x4e00 && code <= 0x9fff) || // CJK Unified Ideographs
      (code >= 0x3400 && code <= 0x4dbf) || // CJK Extension A
      (code >= 0x3000 && code <= 0x303f) || // CJK Symbols
      (code >= 0xff00 && code <= 0xffef)    // Fullwidth Forms
    ) {
      cjkChars++;
    }
  }
  const nonCjkChars = text.length - cjkChars;
  // CJK: ~1.5 tokens per char; Latin: ~0.25 tokens per char (≈4 chars/token).
  return Math.ceil(cjkChars * 1.5 + nonCjkChars * 0.25);
}

/** Estimate tokens for an array of transcript messages. */
export function estimateTranscriptTokens(messages: Array<{ role: string; content: any }>): number {
  let total = 0;
  for (const m of messages) {
    total += estimateTokens(extractTranscriptContentText(m.content)) + 4; // ~4 tokens overhead per message
  }
  return total;
}

// ── Budget computation ─────────────────────────────────────────────────────

/** Per-step token cost estimates (conservative). */
const TOKENS_PER_STEP = {
  /** Planner thought output */
  thought: 250,
  /** Average tool observation */
  observation: 600,
  /** Branch state injection */
  branchState: 120,
  /** Overhead per step total */
  get total() {
    return this.thought + this.observation + this.branchState;
  },
};

export interface ContextBudgetInput {
  /** Model name for context window lookup. */
  model: string;
  /** Pre-built system prompt text (or estimate). */
  systemPromptTokens: number;
  /** Tokens consumed by prior conversation turns injected as context. */
  conversationContextTokens: number;
  /** Tokens consumed by retrieved memory/knowledge context. */
  memoryContextTokens: number;
  /** Safety margin to reserve (tokens). Defaults to 4096. */
  safetyMargin?: number;
}

export interface ContextBudgetResult {
  /** Total model context window (tokens). */
  modelLimit: number;
  /** Tokens already committed (system + conversation + memory + margin). */
  committedTokens: number;
  /** Tokens available for the ReAct loop transcript. */
  residualBudget: number;
  /** Recommended transcript budget in characters (for buildMessages). */
  transcriptBudgetChars: number;
}

/**
 * Compute dynamic agent parameters from the current context usage and model limits.
 */
export function computeContextBudget(input: ContextBudgetInput): ContextBudgetResult {
  const modelLimit = getModelContextLimit(input.model);
  const safetyMargin = input.safetyMargin ?? 4096;
  const committed = input.systemPromptTokens + input.conversationContextTokens + input.memoryContextTokens + safetyMargin;
  const residual = Math.max(0, modelLimit - committed);

  // transcriptBudgetChars: character budget for buildMessages windowing.
  // ~4 chars per token (with mixed content assumption).
  // Raise cap for large context models (Gemini 1M, Claude 200k).
  const transcriptBudgetChars = clamp(
    Math.floor(residual * 3.5),
    4000,
    200_000,
  );

  return {
    modelLimit,
    committedTokens: committed,
    residualBudget: residual,
    transcriptBudgetChars,
  };
}

// ── Transcript compression ─────────────────────────────────────────────────

export type CompressionLevel = 'none' | 'soft' | 'medium' | 'hard';

/**
 * Determine compression level based on how much of the budget is consumed.
 */
export function getCompressionLevel(usedTokens: number, budgetTokens: number): CompressionLevel {
  if (budgetTokens <= 0) return 'hard';
  const ratio = usedTokens / budgetTokens;
  if (ratio < 0.55) return 'none';
  if (ratio < 0.72) return 'soft';
  if (ratio < 0.88) return 'medium';
  return 'hard';
}

interface TranscriptMessage {
  role: string;
  content: any;
}

function stringifyTranscriptJson(value: unknown): string {
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

function formatToolDecisionText(toolCalls: Array<{ name?: unknown; arguments?: unknown }>): string {
  const lines = toolCalls
    .map((toolCall) => {
      const name = String(toolCall?.name || '').trim() || 'tool';
      const args = toolCall?.arguments;
      const hasArgs = Boolean(
        args != null
        && (typeof args !== 'object'
          || Array.isArray(args)
          || Object.keys(args as Record<string, unknown>).length > 0),
      );
      return `- ${name}${hasArgs ? ` | arguments: ${typeof args === 'string' ? args : stringifyTranscriptJson(args)}` : ''}`;
    })
    .filter(Boolean);

  return lines.length > 0
    ? ['PLANNER TOOL DECISION:', ...lines].join('\n')
    : '';
}

function formatToolObservationText(toolName: string | undefined, body: unknown): string {
  const normalizedToolName = String(toolName || '').trim() || 'tool';
  const payload = typeof body === 'string' ? body : stringifyTranscriptJson(body);
  return `TOOL OBSERVATION for ${normalizedToolName}:\n${payload}`;
}

function extractOpenAIAssistantReplayText(message: Record<string, unknown>): string {
  const toolCalls = Array.isArray(message.tool_calls)
    ? message.tool_calls.reduce<Array<{ name?: unknown; arguments?: unknown }>>((entries, toolCall) => {
      if (!toolCall || typeof toolCall !== 'object') {
        return entries;
      }
      const record = toolCall as Record<string, unknown>;
      const functionRecord = record.function && typeof record.function === 'object'
        ? record.function as Record<string, unknown>
        : {};
      entries.push({
        name: functionRecord.name,
        arguments: typeof functionRecord.arguments === 'string'
          ? functionRecord.arguments
          : functionRecord.arguments ?? {},
      });
      return entries;
    }, [])
    : [];

  if (toolCalls.length > 0) {
    return formatToolDecisionText(toolCalls);
  }

  if (typeof message.content === 'string') {
    return message.content;
  }

  if (Array.isArray(message.content)) {
    return message.content
      .map((item) => {
        if (typeof item === 'string') {
          return item;
        }
        if (item && typeof item === 'object' && typeof (item as { text?: unknown }).text === 'string') {
          return (item as { text: string }).text;
        }
        return '';
      })
      .filter(Boolean)
      .join('\n');
  }

  return '';
}

function extractAnthropicReplayText(content: unknown[]): string {
  const toolResults = content
    .filter((item) => Boolean(item) && typeof item === 'object' && (item as { type?: unknown }).type === 'tool_result')
    .map((item) => {
      const block = item as { content?: unknown; tool_name?: unknown };
      return formatToolObservationText(
        typeof block.tool_name === 'string' ? block.tool_name : undefined,
        typeof block.content === 'string' ? block.content : stringifyTranscriptJson(block.content),
      );
    });
  if (toolResults.length > 0) {
    return toolResults.join('\n\n');
  }

  const toolCalls = content
    .filter((item) => Boolean(item) && typeof item === 'object' && (item as { type?: unknown }).type === 'tool_use')
    .map((item) => {
      const block = item as { name?: unknown; input?: unknown };
      return {
        name: block.name,
        arguments: block.input ?? {},
      };
    });
  if (toolCalls.length > 0) {
    return formatToolDecisionText(toolCalls);
  }

  return content
    .map((item) => {
      if (typeof item === 'string') {
        return item;
      }
      if (item && typeof item === 'object') {
        if (typeof (item as { text?: unknown }).text === 'string') {
          return (item as { text: string }).text;
        }
        if (typeof (item as { content?: unknown }).content === 'string') {
          return (item as { content: string }).content;
        }
      }
      return '';
    })
    .filter(Boolean)
    .join('\n');
}

export function extractTranscriptContentText(content: any): string {
  if (typeof content === 'string') {
    return content;
  }
  if (content && typeof content === 'object') {
    if ((content as { type?: unknown }).type === 'openai_tool_result') {
      const toolResult = content as { content?: unknown; tool_name?: unknown };
      return formatToolObservationText(
        typeof toolResult.tool_name === 'string' ? toolResult.tool_name : undefined,
        typeof toolResult.content === 'string' ? toolResult.content : stringifyTranscriptJson(toolResult.content),
      );
    }
    if ((content as { type?: unknown }).type === 'openai_assistant_message') {
      const message = (content as { message?: Record<string, unknown> }).message || {};
      return extractOpenAIAssistantReplayText(message);
    }
  }
  if (Array.isArray(content)) {
    return extractAnthropicReplayText(content);
  }
  if (content == null) {
    return '';
  }
  return String(content);
}

/**
 * Compress a transcript to fit within `budgetChars`.
 *
 * Strategy:
 *   - Always keep the first message (original user request).
 *   - Apply compression level:
 *     * none   — return as-is
 *     * soft   — truncate long tool observations in middle messages
 *     * medium — summarize middle observations to one-line markers
 *     * hard   — keep only first message + last N messages
 *
 * This is a synchronous, no-LLM compression.  For LLM-based summarization
 * the caller can optionally invoke an LLM on the compressed output.
 */
export function compressTranscript(
  messages: TranscriptMessage[],
  budgetChars: number,
  level?: CompressionLevel,
): TranscriptMessage[] {
  if (messages.length <= 1) return messages;

  const totalLen = messages.reduce((s, m) => s + extractTranscriptContentText(m.content).length, 0);
  const effectiveLevel = level ?? (totalLen <= budgetChars ? 'none' : undefined);

  if (effectiveLevel === 'none' && totalLen <= budgetChars) {
    return messages;
  }

  // Auto-detect level if not supplied.
  const appliedLevel = effectiveLevel ?? (
    totalLen <= budgetChars * 1.3 ? 'soft' :
    totalLen <= budgetChars * 2 ? 'medium' : 'hard'
  );

  const first = messages[0];
  const rest = messages.slice(1);

  if (appliedLevel === 'soft') {
    // Truncate long observations in middle messages (>800 chars → keep head+tail).
    const compressed = rest.map((m) => {
      const contentText = extractTranscriptContentText(m.content);
      if (m.role === 'user' && contentText.length > 800 && contentText.startsWith('TOOL OBSERVATION')) {
        const lines = contentText.split('\n');
        const header = lines.slice(0, 2).join('\n');
        const tail = lines.slice(-3).join('\n');
        return { ...m, content: `${header}\n...[observation truncated]...\n${tail}` };
      }
      return m;
    });
    return fitFromEnd([first], compressed, budgetChars);
  }

  if (appliedLevel === 'medium') {
    // Replace tool observations with one-line summaries.
    const compressed = rest.map((m) => {
      const contentText = extractTranscriptContentText(m.content);
      if (m.role === 'user' && contentText.startsWith('TOOL OBSERVATION')) {
        const firstLine = contentText.split('\n')[0];
        const charCount = contentText.length;
        return { ...m, content: `${firstLine} [${charCount} chars compressed]` };
      }
      return m;
    });
    return fitFromEnd([first], compressed, budgetChars);
  }

  // Hard: keep first + tail only.
  return fitFromEnd([first], rest, budgetChars);
}

/**
 * Keep `head` messages plus as many trailing messages from `rest` as fit
 * within `budgetChars`.
 */
function fitFromEnd(
  head: TranscriptMessage[],
  rest: TranscriptMessage[],
  budgetChars: number,
): TranscriptMessage[] {
  const headLen = head.reduce((s, m) => s + extractTranscriptContentText(m.content).length, 0);
  let budget = budgetChars - headLen;
  const kept: TranscriptMessage[] = [];
  for (let i = rest.length - 1; i >= 0 && budget > 0; i--) {
    const len = extractTranscriptContentText(rest[i].content).length;
    if (len <= budget) {
      kept.unshift(rest[i]);
      budget -= len;
    } else {
      break; // don't partially include
    }
  }
  return [...head, ...kept];
}

// ── Helpers ────────────────────────────────────────────────────────────────

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

// ── Branch-spawn transcript compression ────────────────────────────────────

const PLANNER_TOOL_DECISION_PREFIX = 'PLANNER TOOL DECISION:';
const PLANNER_BRANCH_DECISION_PREFIX = 'PLANNER BRANCH DECISION:';

function parseLegacyPlannerTranscript(content: unknown): any | null {
  if (typeof content !== 'string') {
    return null;
  }
  try {
    return JSON.parse(content);
  } catch {
    return null;
  }
}

function extractPlannerToolLines(content: unknown): string[] {
  const legacy = parseLegacyPlannerTranscript(content);
  if (legacy?.kind === 'tool' && Array.isArray(legacy.tool_calls)) {
    return legacy.tool_calls.map((tc: { name?: string; goal?: string }) => {
      const name = typeof tc?.name === 'string' ? tc.name : 'unknown';
      const goal = typeof tc?.goal === 'string' && tc.goal.trim() ? ` — ${tc.goal.trim()}` : '';
      return `${name}${goal}`;
    });
  }

  const projected = typeof content === 'string' ? content : extractTranscriptContentText(content);
  if (!projected.startsWith(PLANNER_TOOL_DECISION_PREFIX)) {
    return [];
  }

  return projected
    .split('\n')
    .slice(1)
    .map((line) => line.trim())
    .filter((line) => line.startsWith('- '))
    .map((line) => line.slice(2).trim())
    .filter(Boolean);
}

function extractPlannerToolNames(content: unknown): string[] {
  const legacy = parseLegacyPlannerTranscript(content);
  if (legacy?.kind === 'tool' && Array.isArray(legacy.tool_calls)) {
    return legacy.tool_calls
      .map((tc: { name?: string }) => (typeof tc?.name === 'string' ? tc.name.trim() : ''))
      .filter(Boolean);
  }

  return extractPlannerToolLines(content)
    .map((line) => line.match(/^([^|:]+)/)?.[1]?.trim() || line)
    .filter(Boolean);
}

function extractPlannerBranchChildCount(content: unknown): number | null {
  const legacy = parseLegacyPlannerTranscript(content);
  if (legacy?.kind === 'branch' && Array.isArray(legacy.branches)) {
    return legacy.branches.length;
  }

  const projected = typeof content === 'string' ? content : extractTranscriptContentText(content);
  if (!projected.startsWith(PLANNER_BRANCH_DECISION_PREFIX)) {
    return null;
  }

  return projected
    .split('\n')
    .slice(1)
    .map((line) => line.trim())
    .filter((line) => line.startsWith('- '))
    .length;
}

/**
 * Compress a parent branch's transcript before passing to child branches.
 * This dramatically reduces token inheritance cost, enabling deeper/wider branching.
 *
 * Strategy:
 *   - Always keep the first message (original user request).
 *   - Compress middle messages into a compact parent-activity summary.
 *   - Keep the last `tailKeep` messages verbatim for recency.
 *   - Tool observations in middle messages are reduced to one-line markers.
 */
export function compressBranchTranscript(
  messages: TranscriptMessage[],
  options: { tailKeep?: number; maxSummaryChars?: number } = {},
): TranscriptMessage[] {
  const tailKeep = options.tailKeep ?? 4;
  const maxSummaryChars = options.maxSummaryChars ?? 1500;

  if (messages.length <= tailKeep + 1) return messages;

  const first = messages[0];
  const middle = messages.slice(1, messages.length - tailKeep);
  const tail = messages.slice(messages.length - tailKeep);

  const summaryLines: string[] = [];
  let toolCallCount = 0;
  let branchCount = 0;

  for (const m of middle) {
    if (m.role === 'assistant') {
      const toolLines = extractPlannerToolLines(m.content);
      if (toolLines.length > 0) {
        for (const line of toolLines) {
          toolCallCount++;
          summaryLines.push(`> ${line}`);
        }
        continue;
      }

      const branchChildren = extractPlannerBranchChildCount(m.content);
      if (branchChildren !== null) {
        branchCount++;
        summaryLines.push(`>> branched into ${branchChildren || '?'} children`);
        continue;
      }

      const line = extractTranscriptContentText(m.content).split('\n')[0].slice(0, 120);
      if (line) summaryLines.push(`# ${line}`);
    } else {
      const contentText = extractTranscriptContentText(m.content);
      if (m.role === 'user' && contentText.startsWith('TOOL OBSERVATION')) {
        const firstLine = contentText.split('\n')[0];
        const hasError = contentText.includes('ERROR:');
        if (hasError) {
          // Preserve key test/build failure lines for downstream re-branch diagnosis.
          const errorLines = contentText
            .split('\n')
            .filter((l: string) => /error|fail|assert|panic|FAILED|expect/i.test(l))
            .slice(0, 5)
            .map((l: string) => l.slice(0, 200));
          summaryLines.push(`  ${firstLine} [ERROR] (${contentText.length}ch)`);
          if (errorLines.length > 0) {
            summaryLines.push(`    ${errorLines.join('\n    ')}`);
          }
        } else {
          summaryLines.push(`  ${firstLine} [OK] (${contentText.length}ch)`);
        }
      }
    }
  }

  let summary = summaryLines.join('\n');
  if (summary.length > maxSummaryChars) {
    summary = summary.slice(0, maxSummaryChars) + '\n...[parent history truncated]';
  }

  const parentSummary: TranscriptMessage = {
    role: 'user',
    content: `[PARENT BRANCH CONTEXT — ${toolCallCount} tool calls, ${branchCount} branching steps compressed]\n${summary}`,
  };

  return [first, parentSummary, ...tail];
}

/**
 * Apply progressive compression to a branch's transcript during execution.
 * Called after each step to keep the transcript within budget, allowing
 * more steps before hitting context limits.
 */
export function progressiveCompressBranch(
  messages: TranscriptMessage[],
  budgetChars: number,
  recentKeep: number = 6,
): TranscriptMessage[] {
  const totalLen = messages.reduce((s, m) => s + extractTranscriptContentText(m.content).length, 0);
  if (totalLen <= budgetChars) return messages;

  if (messages.length <= recentKeep + 1) {
    return compressTranscript(messages, budgetChars, 'medium');
  }

  const first = messages[0];
  const middle = messages.slice(1, messages.length - recentKeep);
  const tail = messages.slice(messages.length - recentKeep);

  // Aggressively compress middle: tool observations → one-liners,
  // assistant tool-call messages → name-only summaries.
  const compressed = middle.map((m) => {
    const contentText = extractTranscriptContentText(m.content);
    if (m.role === 'user' && contentText.startsWith('TOOL OBSERVATION')) {
      const firstLine = contentText.split('\n')[0];
      return { ...m, content: `${firstLine} [compressed]` };
    }
    if (m.role === 'assistant') {
      const toolNames = extractPlannerToolNames(m.content);
      if (toolNames.length > 0) {
        return { ...m, content: `${PLANNER_TOOL_DECISION_PREFIX} ${toolNames.join(', ')} [compressed]` };
      }
    }
    return m;
  });

  const result = [first, ...compressed, ...tail];
  const resultLen = result.reduce((s, m) => s + extractTranscriptContentText(m.content).length, 0);

  if (resultLen > budgetChars) {
    return compressTranscript(result, budgetChars, 'hard');
  }

  return result;
}

// ── Dynamic context auto-compression ───────────────────────────────────────

export interface ContextCheckResult {
  /** Transcript after potential compression. */
  transcript: Array<{ role: string; content: any }>;
  /** Whether compression was applied. */
  compressed: boolean;
  /** Whether more steps can safely be taken. */
  canContinue: boolean;
  /** Usage ratio (0–1) after any compression. */
  usageRatio: number;
  /** Estimated transcript tokens after any compression. */
  transcriptTokens: number;
}

/**
 * Check actual context utilization and auto-compress if the threshold is
 * exceeded.  Called at the top of each branch loop iteration when model
 * limits are available.
 *
 * Behaviour:
 *   1. Estimate real token count of the transcript.
 *   2. If usage < `compressThreshold` (default 92%) → no-op, continue.
 *   3. If usage ≥ threshold → progressive compress aiming at `targetRatio`
 *      (default 75%).  If still over threshold → hard compress.
 *   4. After hard compression, if usage still ≥ 95% → `canContinue = false`.
 *
 * Branch structure (depth / width) is never changed — only the transcript
 * is compacted so execution can keep going.
 */
export function checkContextAndCompress(
  transcript: Array<{ role: string; content: any }>,
  modelLimit: number,
  committedTokens: number,
  options: {
    /** Usage ratio at which compression kicks in. Default 0.85 */
    compressThreshold?: number;
    /** Target usage ratio after compression. Default 0.65 */
    targetRatio?: number;
    /** Recent messages to always keep verbatim. Default 6 */
    recentKeep?: number;
    /** Enable earlier, more aggressive compression. Default false */
    aggressiveMode?: boolean;
  } = {},
): ContextCheckResult {
  const compressThreshold = options.aggressiveMode
    ? 0.75
    : (options.compressThreshold ?? 0.85);
  const targetRatio = options.targetRatio ?? 0.65;
  const recentKeep = options.recentKeep ?? 6;

  const transcriptTokens = estimateTranscriptTokens(transcript);
  const totalUsed = committedTokens + transcriptTokens;
  const usageRatio = modelLimit > 0 ? totalUsed / modelLimit : 1;

  if (usageRatio < compressThreshold) {
    return { transcript, compressed: false, canContinue: true, usageRatio, transcriptTokens };
  }

  // Over threshold — compress to targetRatio
  const targetTranscriptTokens = Math.max(0, Math.floor(modelLimit * targetRatio) - committedTokens);
  const targetBudgetChars = Math.max(4000, targetTranscriptTokens * 4);

  // Progressive compression first
  let result = progressiveCompressBranch(transcript, targetBudgetChars, recentKeep);
  let newTokens = estimateTranscriptTokens(result);
  let newRatio = modelLimit > 0 ? (committedTokens + newTokens) / modelLimit : 1;

  if (newRatio < compressThreshold) {
    return { transcript: result, compressed: true, canContinue: true, usageRatio: newRatio, transcriptTokens: newTokens };
  }

  // Hard compression
  result = compressTranscript(result, targetBudgetChars, 'hard');
  newTokens = estimateTranscriptTokens(result);
  newRatio = modelLimit > 0 ? (committedTokens + newTokens) / modelLimit : 1;

  if (newRatio < 0.95) {
    return { transcript: result, compressed: true, canContinue: true, usageRatio: newRatio, transcriptTokens: newTokens };
  }

  // Nuclear compression: keep only first message + last 2 messages.
  // This is the last resort before giving up — ensures the branch can continue
  // with minimal context rather than being terminated.
  const nuclearKeep = 2;
  if (result.length > nuclearKeep + 1) {
    const first = result[0];
    const tail = result.slice(-nuclearKeep);
    const summaryMsg: { role: string; content: string } = {
      role: 'user',
      content: `[CONTEXT COMPRESSED — ${result.length - 1 - nuclearKeep} earlier messages were removed to free context space. Continue from the most recent state below.]`,
    };
    result = [first, summaryMsg, ...tail];
    newTokens = estimateTranscriptTokens(result);
    newRatio = modelLimit > 0 ? (committedTokens + newTokens) / modelLimit : 1;
  }

  return {
    transcript: result,
    compressed: true,
    canContinue: newRatio < 0.98,
    usageRatio: newRatio,
    transcriptTokens: newTokens,
  };
}

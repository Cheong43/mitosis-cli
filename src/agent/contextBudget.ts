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
export function estimateTranscriptTokens(messages: Array<{ role: string; content: string }>): number {
  let total = 0;
  for (const m of messages) {
    total += estimateTokens(m.content) + 4; // ~4 tokens overhead per message
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
  /** Recommended max steps per branch. */
  maxSteps: number;
  /** Recommended max branch depth. */
  maxBranchDepth: number;
  /** Recommended max branch width. */
  maxBranchWidth: number;
  /** Recommended total step budget across all branches. */
  maxTotalSteps: number;
  /** Recommended transcript budget in characters (for buildMessages). */
  transcriptBudgetChars: number;
  /** Recommended beam search depth (for BeamSearchAgentRuntime). */
  beamMaxDepth: number;
  /** Recommended beam width. */
  beamWidth: number;
  /** Recommended beam expansion factor. */
  beamExpansionFactor: number;
}

/**
 * Compute dynamic agent parameters from the current context usage and model limits.
 */
export function computeContextBudget(input: ContextBudgetInput): ContextBudgetResult {
  const modelLimit = getModelContextLimit(input.model);
  const safetyMargin = input.safetyMargin ?? 4096;
  const committed = input.systemPromptTokens + input.conversationContextTokens + input.memoryContextTokens + safetyMargin;
  const residual = Math.max(0, modelLimit - committed);

  // ── Branching ReAct parameters ─────────────────────────────────────────

  // maxSteps: how many ReAct iterations can fit in the residual budget per branch.
  const rawMaxSteps = Math.floor(residual / TOKENS_PER_STEP.total);
  const maxSteps = clamp(rawMaxSteps, 2, 20);

  // maxBranchDepth: deeper branching multiplies token usage exponentially.
  // We scale by residual budget tiers.
  const maxBranchDepth = residual >= 80_000 ? 3
    : residual >= 40_000 ? 2
    : residual >= 15_000 ? 1
    : 0;

  // maxBranchWidth: more branches = more parallel token consumption.
  const maxBranchWidth = residual >= 60_000 ? 4
    : residual >= 30_000 ? 3
    : 2;

  // maxTotalSteps: global ceiling on total work.
  const maxTotalSteps = clamp(
    Math.floor(residual / (TOKENS_PER_STEP.total * 0.6)),
    maxSteps,
    maxSteps * maxBranchWidth * (maxBranchDepth + 1),
  );

  // transcriptBudgetChars: character budget for buildMessages windowing.
  // ~4 chars per token (with mixed content assumption).
  const transcriptBudgetChars = clamp(
    Math.floor(residual * 3.5),
    4000,
    120_000,
  );

  // ── Beam Search parameters ─────────────────────────────────────────────

  // Each beam step keeps beamWidth × expansionFactor LLM calls.
  // Per step, each candidate adds ~TOKENS_PER_STEP.total to its path.
  const beamWidth = residual >= 80_000 ? 4
    : residual >= 40_000 ? 3
    : 2;

  const beamExpansionFactor = residual >= 60_000 ? 4
    : residual >= 30_000 ? 3
    : 2;

  // beamMaxDepth: each depth level each path grows by ~TOKENS_PER_STEP.total.
  const rawBeamDepth = Math.floor(residual / (TOKENS_PER_STEP.total * beamWidth));
  const beamMaxDepth = clamp(rawBeamDepth, 2, 10);

  return {
    modelLimit,
    committedTokens: committed,
    residualBudget: residual,
    maxSteps,
    maxBranchDepth,
    maxBranchWidth,
    maxTotalSteps,
    transcriptBudgetChars,
    beamMaxDepth,
    beamWidth,
    beamExpansionFactor,
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
  content: string;
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

  const totalLen = messages.reduce((s, m) => s + m.content.length, 0);
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
      if (m.role === 'user' && m.content.length > 800 && m.content.startsWith('TOOL OBSERVATION')) {
        const lines = m.content.split('\n');
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
      if (m.role === 'user' && m.content.startsWith('TOOL OBSERVATION')) {
        const firstLine = m.content.split('\n')[0];
        const charCount = m.content.length;
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
  const headLen = head.reduce((s, m) => s + m.content.length, 0);
  let budget = budgetChars - headLen;
  const kept: TranscriptMessage[] = [];
  for (let i = rest.length - 1; i >= 0 && budget > 0; i--) {
    const len = rest[i].content.length;
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

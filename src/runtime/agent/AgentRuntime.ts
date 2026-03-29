import {
  AgentTraceEvent,
  BranchKanbanCard,
  BranchKanbanSnapshot,
  BranchKanbanStatus,
  BranchDisposition,
  BranchHandoff,
  BranchOutcome,
  PlannedBranch,
  PlannedToolCall,
  SynthesisResult,
  SynthesisSharedHandoff,
  ToolObservation,
  TranscriptMessage,
} from './types.js';
import type { Planner } from './Planner.js';
import { ToolExecutionResult } from '../tools/types.js';
import {
  compressBranchTranscript,
  progressiveCompressBranch,
  checkContextAndCompress,
  estimateTranscriptTokens,
} from '../../agent/contextBudget.js';

interface AgentToolRuntime {
  execute(toolName: string, args: Record<string, unknown>): Promise<ToolExecutionResult>;
  resetSession(): void;
}

export interface AgentBranchState {
  id: string;
  parentId: string | null;
  depth: number;
  label: string;
  goal: string;
  executionGroup: number;
  dependsOn: string[];
  priority: number;
  steps: number;
  inheritedMessageCount: number;
  transcript: TranscriptMessage[];
  savedNodeIds: string[];
  completionSummary?: string;
  finalAnswer?: string;
  /** Structured outcome of this branch: success, partial, failed, or unknown. */
  outcome?: BranchOutcome;
  /** Human-readable explanation when outcome is 'partial' or 'failed'. */
  outcomeReason?: string;
  /** Retry-oriented disposition used during synthesis/remediation selection. */
  disposition?: BranchDisposition;
  /** Whether the branch ended naturally or via planner fallback finalization. */
  finalizationMode?: 'natural' | 'planner_fallback';
  /** Structured per-branch remediation handoff, when this branch is a retry child. */
  handoff?: BranchHandoff;
  /** Structured shared retry context for a remediation round. */
  sharedHandoff?: SynthesisSharedHandoff;
  /** Full branch ancestry ids, excluding this branch id. */
  ancestorBranchIds?: string[];
  /** Full branch ancestry labels, excluding this branch label. */
  ancestorBranchLabels?: string[];
}

export interface BranchPlanInput {
  branch: AgentBranchState;
  planningTranscript?: TranscriptMessage[];
  kanbanSnapshot?: BranchKanbanSnapshot;
}

export interface BranchToolExecutionInput {
  branch: AgentBranchState;
  toolCall: PlannedToolCall;
}

export interface BranchFinalizeInput {
  branch: AgentBranchState;
  reason: string;
}

export interface BranchSynthesisInput {
  userMessage: string;
  priorTranscript: TranscriptMessage[];
  branches: Array<{
    id: string;
    label: string;
    goal: string;
    savedNodeIds: string[];
    completionSummary?: string;
    finalAnswer: string;
    outcome?: BranchOutcome;
    outcomeReason?: string;
    disposition?: BranchDisposition;
    finalizationMode?: 'natural' | 'planner_fallback';
    ancestorBranchIds?: string[];
    ancestorBranchLabels?: string[];
  }>;
  /** How many synthesis-retry rounds have already occurred (0 on first call). */
  synthesisRetry: number;
  /** Maximum retries allowed — the hook can use this to decide whether to re-branch. */
  maxSynthesisRetries: number;
}

export interface BranchToolTranscriptInput {
  branch: AgentBranchState;
  toolCalls: PlannedToolCall[];
  observations: ToolObservation[];
}

export interface AgentRuntimeOptions {
  planner: Planner;
  toolRuntime: AgentToolRuntime;
  /** Maximum ReAct loop iterations before forcing a final answer. */
  maxSteps?: number;
  /** Maximum branching depth. */
  maxBranchDepth?: number;
  /** Maximum children spawned in a single branching step. */
  maxBranchWidth?: number;
  /** Max completed branches to keep before synthesis. */
  maxCompletedBranches?: number;
  /** Max number of branches allowed to run simultaneously. */
  branchConcurrency?: number;
  /** Optional override for total work budget across all branches. */
  maxTotalSteps?: number;
  /** Character budget for per-branch transcript compression. */
  transcriptBudgetChars?: number;
  /** Model context window in tokens. Enables dynamic budget mode when > 0. */
  modelLimit?: number;
  /** Tokens already committed (system prompt + conversation + memory + margin). */
  committedTokens?: number;
  /** Optional branch-aware planner override. */
  planBranch?: (input: BranchPlanInput) => Promise<import('./types.js').AgentStep>;
  /** Optional branch-aware tool execution override. */
  executeToolCall?: (input: BranchToolExecutionInput) => Promise<ToolObservation>;
  /** Optional override for how tool calls/results are written back into the transcript. */
  buildToolTranscript?: (input: BranchToolTranscriptInput) => TranscriptMessage[] | null | undefined;
  /** Optional branch finalization override when a branch hits budget. */
  finalizeBranch?: (input: BranchFinalizeInput) => Promise<string>;
  /**
   * Optional final synthesis hook when multiple branches finish.
   *
   * Return a plain `string` to finish the run, or a `SynthesisResult` to
   * either finish (`done: true`) or request re-branching (`done: false`).
   */
  synthesizeFinal?: (input: BranchSynthesisInput) => Promise<string | SynthesisResult>;
  /** Maximum post-synthesis re-branch rounds (default 2). */
  maxSynthesisRetries?: number;
  /** Trace callback invoked after each loop step. */
  onTrace?: (event: AgentTraceEvent) => void;
}

function formatPlannerToolDecisionMessage(toolCalls: PlannedToolCall[]): string {
  const lines = toolCalls.map((toolCall) => {
    const goal = toolCall.goal?.trim() ? ` | goal: ${toolCall.goal.trim()}` : '';
    const hasArguments = toolCall.arguments && Object.keys(toolCall.arguments).length > 0;
    const args = hasArguments ? ` | arguments: ${JSON.stringify(toolCall.arguments)}` : '';
    return `- ${toolCall.name}${goal}${args}`;
  });
  return ['PLANNER TOOL DECISION:', ...lines].join('\n');
}

function formatToolObservationHeader(toolCall: PlannedToolCall, fallbackToolName: string): string {
  if (toolCall.name === 'web') {
    const mode = String(toolCall.arguments?.mode || '').trim();
    if (mode === 'search') {
      const query = String(toolCall.arguments?.query || '').trim();
      if (query) {
        return `TOOL OBSERVATION for web search query=${JSON.stringify(query)}:`;
      }
    }
    if (mode === 'fetch') {
      const url = String(toolCall.arguments?.url || '').trim();
      if (url) {
        return `TOOL OBSERVATION for web fetch url=${JSON.stringify(url)}:`;
      }
    }
  }

  return `TOOL OBSERVATION for ${fallbackToolName}:`;
}

function formatPlannerBranchDecisionMessage(branches: PlannedBranch[]): string {
  const lines = branches.map((branch) => {
    const why = branch.why?.trim() ? ` | why: ${branch.why.trim()}` : '';
    const priority = typeof branch.priority === 'number' ? ` | priority: ${branch.priority}` : '';
    const executionGroup = typeof branch.executionGroup === 'number' ? ` | execution_group: ${branch.executionGroup}` : '';
    const dependsOn = branch.dependsOn?.length ? ` | depends_on: ${branch.dependsOn.join(', ')}` : '';
    return `- ${branch.label}: ${branch.goal}${why}${priority}${executionGroup}${dependsOn}`;
  });
  return ['PLANNER BRANCH DECISION:', ...lines].join('\n');
}

function normalizeExecutionGroup(value: unknown, fallback = 1): number {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    return fallback;
  }
  return Math.max(1, Math.floor(numeric));
}

function normalizeDependencyLabels(value: unknown): string[] {
  if (!Array.isArray(value)) {
    return [];
  }
  return dedupeNonEmpty(value.map((entry) => String(entry || '').trim()));
}

function renderBranchHandoffMessage(handoff?: BranchHandoff): string {
  if (!handoff) {
    return '';
  }

  const sections: string[] = [
    `canonical_task: ${handoff.canonicalTaskId}`,
  ];
  if (handoff.retryOfBranchId) {
    sections.push(`retry_of_branch: ${handoff.retryOfBranchId}`);
  }
  if (handoff.disposition) {
    sections.push(`disposition: ${handoff.disposition}`);
  }
  if (handoff.priorAttemptIds?.length) {
    sections.push(`prior_attempt_ids: ${handoff.priorAttemptIds.join(', ')}`);
  }
  if (handoff.priorIssue) {
    sections.push(`prior_issue: ${handoff.priorIssue}`);
  }
  if (handoff.priorResultSnippet) {
    sections.push(`prior_result: ${handoff.priorResultSnippet}`);
  }
  if (handoff.knownGoodFacts?.length) {
    sections.push(`known_good_facts:\n- ${handoff.knownGoodFacts.join('\n- ')}`);
  }
  if (handoff.missingFields?.length) {
    sections.push(`missing_fields:\n- ${handoff.missingFields.join('\n- ')}`);
  }
  if (handoff.blockedBy?.length) {
    sections.push(`blocked_by:\n- ${handoff.blockedBy.join('\n- ')}`);
  }
  if (handoff.exhaustedStrategies?.length) {
    sections.push(`exhausted_strategies:\n- ${handoff.exhaustedStrategies.join('\n- ')}`);
  }
  if (handoff.mustNotRepeat?.length) {
    sections.push(`must_not_repeat:\n- ${handoff.mustNotRepeat.join('\n- ')}`);
  }
  return sections.join('\n');
}

function renderSharedHandoffMessage(sharedHandoff?: SynthesisSharedHandoff): string {
  if (!sharedHandoff) {
    return '';
  }

  const lines: string[] = [`retry_round: ${sharedHandoff.retryIndex}`];
  if (sharedHandoff.successfulAttempts?.length) {
    lines.push('successful_attempts:');
    for (const attempt of sharedHandoff.successfulAttempts) {
      lines.push(`- [${attempt.branchId}] ${attempt.label} (canonical=${attempt.canonicalTaskId}): ${attempt.summary}`);
    }
  }
  if (sharedHandoff.unresolvedAttempts?.length) {
    lines.push('unresolved_attempts:');
    for (const attempt of sharedHandoff.unresolvedAttempts) {
      lines.push(`- [${attempt.retryOfBranchId || 'unknown'}] ${attempt.canonicalTaskId} (${attempt.disposition || 'unknown'})`);
    }
  }
  return lines.join('\n');
}

function previewText(value: string | undefined, maxLength = 160): string {
  const normalized = String(value || '').replace(/\s+/g, ' ').trim();
  if (!normalized) {
    return '';
  }
  return normalized.length > maxLength
    ? `${normalized.slice(0, Math.max(0, maxLength - 1)).trimEnd()}…`
    : normalized;
}

function dedupeNonEmpty(values: Array<string | undefined | null>): string[] {
  const seen = new Set<string>();
  const result: string[] = [];
  for (const value of values) {
    const normalized = previewText(String(value || ''), 240);
    if (!normalized || seen.has(normalized)) {
      continue;
    }
    seen.add(normalized);
    result.push(normalized);
  }
  return result;
}

function extractArtifactPaths(toolCalls: PlannedToolCall[]): string[] {
  const collected: string[] = [];
  for (const toolCall of toolCalls) {
    const args = toolCall.arguments || {};
    const candidates = [
      args.path,
      args.filePath,
      args.outputPath,
      args.targetPath,
      args.cwd,
    ];
    for (const candidate of candidates) {
      if (typeof candidate === 'string' && candidate.trim()) {
        collected.push(candidate.trim());
      }
    }
    if (Array.isArray(args.paths)) {
      for (const candidate of args.paths) {
        if (typeof candidate === 'string' && candidate.trim()) {
          collected.push(candidate.trim());
        }
      }
    }
  }
  return dedupeNonEmpty(collected);
}

interface WebSearchRoundSummary {
  step: number;
  queries: string[];
  explicitDomains: string[];
  resultDomains: string[];
}

function normalizeHost(value: string): string {
  return value
    .trim()
    .toLowerCase()
    .replace(/^https?:\/\//, '')
    .replace(/^www\./, '')
    .replace(/\/.*$/, '');
}

function extractHostFromUrl(value: string): string {
  try {
    return normalizeHost(new URL(value).hostname);
  } catch {
    return '';
  }
}

function extractExplicitSearchDomains(query: string): string[] {
  const matches = query.toLowerCase().matchAll(/\bsite:([a-z0-9.-]+\.[a-z]{2,})\b/g);
  return dedupeNonEmpty(Array.from(matches, (match) => normalizeHost(match[1] || '')));
}

function tokenizeSearchQuery(query: string): string[] {
  return dedupeNonEmpty(
    query
      .toLowerCase()
      .replace(/\bsite:[^\s]+/g, ' ')
      .match(/[a-z0-9]{2,}|[\u4e00-\u9fff]{2,}/g) || [],
  );
}

function querySimilarity(left: string, right: string): number {
  const normalizedLeft = previewText(left.toLowerCase(), 400);
  const normalizedRight = previewText(right.toLowerCase(), 400);
  if (!normalizedLeft || !normalizedRight) {
    return 0;
  }
  if (normalizedLeft === normalizedRight) {
    return 1;
  }
  if (normalizedLeft.includes(normalizedRight) || normalizedRight.includes(normalizedLeft)) {
    return 0.92;
  }

  const leftTokens = new Set(tokenizeSearchQuery(normalizedLeft));
  const rightTokens = new Set(tokenizeSearchQuery(normalizedRight));
  if (leftTokens.size === 0 || rightTokens.size === 0) {
    return 0;
  }

  let intersection = 0;
  for (const token of leftTokens) {
    if (rightTokens.has(token)) {
      intersection += 1;
    }
  }
  const union = new Set([...leftTokens, ...rightTokens]).size;
  return union > 0 ? intersection / union : 0;
}

function summarizeWebSearchRound(toolCalls: PlannedToolCall[], step: number): WebSearchRoundSummary | null {
  if (toolCalls.length === 0) {
    return null;
  }

  const queries: string[] = [];
  const explicitDomains: string[] = [];
  for (const toolCall of toolCalls) {
    if (toolCall.name !== 'web' || String(toolCall.arguments?.mode || '').trim() !== 'search') {
      return null;
    }
    const query = String(toolCall.arguments?.query || '').trim();
    if (!query) {
      return null;
    }
    queries.push(query);
    explicitDomains.push(...extractExplicitSearchDomains(query));
  }

  return {
    step,
    queries,
    explicitDomains: dedupeNonEmpty(explicitDomains),
    resultDomains: [],
  };
}

function enrichWebSearchRoundWithObservations(
  summary: WebSearchRoundSummary | null,
  observations: ToolObservation[],
): WebSearchRoundSummary | null {
  if (!summary) {
    return null;
  }

  const resultDomains: string[] = [];
  for (const observation of observations) {
    if (observation.toolName !== 'web') {
      continue;
    }
    try {
      const parsed = JSON.parse(String(observation.result || ''));
      if (parsed?.kind !== 'web_search' || !Array.isArray(parsed.results)) {
        continue;
      }
      for (const result of parsed.results) {
        if (typeof result?.url === 'string') {
          const host = extractHostFromUrl(result.url);
          if (host) {
            resultDomains.push(host);
          }
        }
      }
    } catch {
      // Ignore malformed observations and keep the query-only summary.
    }
  }

  return {
    ...summary,
    resultDomains: dedupeNonEmpty(resultDomains),
  };
}

function hasMaterialSearchDomainShift(
  current: WebSearchRoundSummary,
  priorRounds: WebSearchRoundSummary[],
): boolean {
  if (current.explicitDomains.length === 0) {
    return false;
  }
  const priorDomains = new Set(
    priorRounds.flatMap((round) => [...round.explicitDomains, ...round.resultDomains]).map((domain) => normalizeHost(domain)),
  );
  return current.explicitDomains.some((domain) => !priorDomains.has(normalizeHost(domain)));
}

function isNearDuplicateSearchRound(
  current: WebSearchRoundSummary,
  priorRound: WebSearchRoundSummary,
): boolean {
  if (priorRound.queries.length === 0 || current.queries.length === 0) {
    return false;
  }
  return current.queries.every((query) =>
    priorRound.queries.some((priorQuery) => querySimilarity(query, priorQuery) >= 0.6));
}

function clearPendingProviderToolReplay(branch: AgentBranchState): void {
  const candidate = branch as AgentBranchState & {
    pendingAnthropicAssistantContent?: unknown;
    pendingAnthropicToolUseIds?: unknown;
    pendingOpenAIAssistantMessage?: unknown;
  };
  if ('pendingAnthropicAssistantContent' in candidate) {
    candidate.pendingAnthropicAssistantContent = null;
  }
  if ('pendingAnthropicToolUseIds' in candidate) {
    candidate.pendingAnthropicToolUseIds = [];
  }
  if ('pendingOpenAIAssistantMessage' in candidate) {
    candidate.pendingOpenAIAssistantMessage = null;
  }
}

/**
 * AgentRuntime orchestrates the ReAct loop:
 *   thought → tool call → observation → … → final answer
 *
 * It delegates planning to a `Planner` implementation and tool execution to
 * `ToolRuntime` (which in turn enforces governance rules).
 *
 * This is intentionally decoupled from the existing `Agent` class so it can
 * be adopted incrementally — the existing `Agent` can instantiate and delegate
 * to an `AgentRuntime` for the governed code path.
 */
export class AgentRuntime {
  private static readonly PARALLEL_SAFE_TOOL_NAMES = new Set(['read', 'search', 'web']);
  private readonly planner: Planner;
  private readonly toolRuntime: AgentToolRuntime;
  private readonly maxSteps: number;
  private readonly maxBranchDepth: number;
  private readonly maxBranchWidth: number;
  private readonly maxCompletedBranches: number;
  private readonly branchConcurrency: number;
  private readonly maxTotalSteps: number | null;
  private readonly transcriptBudgetChars: number;
  private readonly modelLimit: number;
  private readonly committedTokens: number;
  private readonly planBranch?: (input: BranchPlanInput) => Promise<import('./types.js').AgentStep>;
  private readonly executeToolCall?: (input: BranchToolExecutionInput) => Promise<ToolObservation>;
  private readonly buildToolTranscript?: (input: BranchToolTranscriptInput) => TranscriptMessage[] | null | undefined;
  private readonly finalizeBranch?: (input: BranchFinalizeInput) => Promise<string>;
  private readonly synthesizeFinal?: (input: BranchSynthesisInput) => Promise<string | SynthesisResult>;
  private readonly maxSynthesisRetries: number;
  private readonly onTrace: (event: AgentTraceEvent) => void;

  constructor(options: AgentRuntimeOptions) {
    this.planner = options.planner;
    this.toolRuntime = options.toolRuntime;
    this.maxSteps = options.maxSteps ?? 200;
    this.maxBranchDepth = options.maxBranchDepth ?? 100;
    this.maxBranchWidth = options.maxBranchWidth ?? 5;
    this.maxCompletedBranches = options.maxCompletedBranches ?? 8;
    this.branchConcurrency = Number.isFinite(options.branchConcurrency)
      ? Math.max(0, Math.floor(options.branchConcurrency ?? 0))
      : 0;
    this.maxTotalSteps = options.maxTotalSteps ?? null;
    this.transcriptBudgetChars = options.transcriptBudgetChars ?? 80_000;
    this.modelLimit = options.modelLimit ?? 0;
    this.committedTokens = options.committedTokens ?? 0;
    this.planBranch = options.planBranch;
    this.executeToolCall = options.executeToolCall;
    this.buildToolTranscript = options.buildToolTranscript;
    this.finalizeBranch = options.finalizeBranch;
    this.synthesizeFinal = options.synthesizeFinal;
    this.maxSynthesisRetries = options.maxSynthesisRetries ?? 2;
    this.onTrace = options.onTrace ?? (() => undefined);
  }

  /**
   * Run the agent loop for a single user turn.
   *
   * @param userMessage - The user's input message.
   * @param priorTranscript - Previous conversation turns for context.
   * @returns The agent's final answer string.
   */
  async run(
    userMessage: string,
    priorTranscript: TranscriptMessage[] = [],
  ): Promise<string> {
    // Reset session-level governance state (doom-loop counter, etc.).
    this.toolRuntime.resetSession();

    const rootBranch: AgentBranchState = {
      id: 'B0',
      parentId: null,
      depth: 0,
      label: 'root',
      goal: 'Solve the user request end-to-end.',
      executionGroup: 1,
      dependsOn: [],
      priority: 1,
      steps: 0,
      inheritedMessageCount: 0,
      savedNodeIds: [],
      ancestorBranchIds: [],
      ancestorBranchLabels: [],
      transcript: [
        ...priorTranscript,
        { role: 'user', content: userMessage },
      ],
    };

    const queue: AgentBranchState[] = [rootBranch];
    const completed: AgentBranchState[] = [];
    const webSearchRoundHistory = new Map<string, WebSearchRoundSummary[]>();
    const siblingFamilies = new Map<string, { rootId: string; executionGroup: number }>();
    let lastTouchedBranch: AgentBranchState | null = rootBranch;
    let totalLoopSteps = 0;
    const totalLoopBudget = this.maxTotalSteps
      ?? Math.max(this.maxSteps, this.maxSteps * this.maxBranchWidth * (this.maxBranchDepth + 1));
    const kanbanCards = new Map<string, BranchKanbanCard>();

    const cloneKanbanCard = (card: BranchKanbanCard): BranchKanbanCard => ({
      ...card,
      dependsOn: [...(card.dependsOn || [])],
      blockers: [...(card.blockers || [])],
      artifacts: [...(card.artifacts || [])],
    });

    const summarizeKanban = () => {
      const cards = [...kanbanCards.values()];
      const summary = {
        total: cards.length,
        queued: cards.filter((card) => card.status === 'queued').length,
        active: cards.filter((card) => card.status === 'active').length,
        finalizing: cards.filter((card) => card.status === 'finalizing').length,
        completed: cards.filter((card) => card.status === 'completed').length,
        error: cards.filter((card) => card.status === 'error').length,
      };
      return summary;
    };

    const buildKanbanSnapshot = (): BranchKanbanSnapshot => ({
      updatedAt: Date.now(),
      summary: summarizeKanban(),
      cards: [...kanbanCards.values()]
        .map((card) => cloneKanbanCard(card))
        .sort((a, b) => (b.updatedAt - a.updatedAt) || (a.depth - b.depth) || a.branchId.localeCompare(b.branchId)),
    });

    const upsertKanbanCard = (
      branch: AgentBranchState,
      patch: Partial<BranchKanbanCard> & { status?: BranchKanbanStatus },
    ): BranchKanbanCard => {
      const current = kanbanCards.get(branch.id);
      const next: BranchKanbanCard = {
        branchId: branch.id,
        parentBranchId: branch.parentId,
        label: patch.label ?? branch.label,
        goal: patch.goal ?? branch.goal,
        executionGroup: patch.executionGroup ?? branch.executionGroup ?? current?.executionGroup,
        dependsOn: patch.dependsOn !== undefined
          ? dedupeNonEmpty(patch.dependsOn)
          : [...(branch.dependsOn || current?.dependsOn || [])],
        depth: patch.depth ?? branch.depth,
        step: patch.step ?? branch.steps,
        status: patch.status ?? current?.status ?? 'queued',
        outcome: patch.outcome ?? branch.outcome ?? current?.outcome,
        disposition: patch.disposition ?? branch.disposition ?? current?.disposition,
        summary: patch.summary ?? current?.summary,
        blockers: patch.blockers !== undefined
          ? dedupeNonEmpty(patch.blockers)
          : [...(current?.blockers || [])],
        artifacts: patch.artifacts !== undefined
          ? dedupeNonEmpty([...(current?.artifacts || []), ...patch.artifacts])
          : [...(current?.artifacts || [])],
        updatedAt: Date.now(),
      };
      kanbanCards.set(branch.id, next);
      return next;
    };

    const renderKanbanSyncMessage = (branch: AgentBranchState): string => {
      const snapshot = buildKanbanSnapshot();
      const visibleCards = snapshot.cards
        .filter((card) => card.branchId !== branch.id)
        .slice(0, 6);
      if (visibleCards.length === 0) {
        return '';
      }

      const lines = [
        'Shared branch kanban snapshot:',
        `- summary: total=${snapshot.summary.total}, active=${snapshot.summary.active}, queued=${snapshot.summary.queued}, finalizing=${snapshot.summary.finalizing}, completed=${snapshot.summary.completed}, error=${snapshot.summary.error}`,
        '- Use this to avoid duplicate work and reuse sibling results when possible.',
      ];
      for (const card of visibleCards) {
        const details = [
          `status=${card.status}`,
          `goal=${previewText(card.goal, 80)}`,
          typeof card.executionGroup === 'number' ? `execution_group=${card.executionGroup}` : '',
          card.dependsOn?.length ? `depends_on=${card.dependsOn.join(', ')}` : '',
          card.summary ? `result=${previewText(card.summary, 120)}` : '',
          card.artifacts?.length ? `artifacts=${card.artifacts.slice(0, 3).join(', ')}` : '',
          card.blockers?.length ? `blockers=${card.blockers.slice(0, 2).join(' | ')}` : '',
        ].filter(Boolean);
        lines.push(`- [${card.branchId}] ${card.label}: ${details.join(' ; ')}`);
      }
      return lines.join('\n');
    };

    const buildPlanningTranscript = (branch: AgentBranchState): TranscriptMessage[] => {
      const syncMessage = renderKanbanSyncMessage(branch);
      if (!syncMessage) {
        return branch.transcript;
      }
      if (branch.transcript.length === 0) {
        return [{ role: 'system', content: syncMessage }];
      }
      const [head, ...rest] = branch.transcript;
      if (head?.role === 'system') {
        return [
          { role: 'system', content: `${head.content}\n\n${syncMessage}` },
          ...rest,
        ];
      }
      return [
        { role: 'system', content: syncMessage },
        ...branch.transcript,
      ];
    };

    upsertKanbanCard(rootBranch, {
      status: 'queued',
      summary: 'Root branch created and waiting for scheduler.',
    });

    const traceMeta = (branch: AgentBranchState, extra: Record<string, unknown> = {}) => ({
      branchId: branch.id,
      parentBranchId: branch.parentId,
      branchLabel: branch.label,
      depth: branch.depth,
      step: branch.steps,
      kanbanCard: kanbanCards.has(branch.id) ? cloneKanbanCard(kanbanCards.get(branch.id)!) : undefined,
      kanbanSummary: summarizeKanban(),
      ...extra,
    });

    const emitBranchTrace = (
      type: AgentTraceEvent['type'],
      branch: AgentBranchState,
      content: string,
      extra: Record<string, unknown> = {},
    ) => {
      this.emit({ type, content, metadata: traceMeta(branch, extra) });
    };

    const siblingKey = (parentId: string | null, label: string) => `${parentId || '__root__'}::${label}`;

    const addCompletedBranch = (branch: AgentBranchState) => {
      if (!completed.includes(branch)) {
        completed.push(branch);
      }
    };

    const buildDefaultSynthesisAnswer = (branches: AgentBranchState[]): string => {
      if (branches.length === 0) {
        return 'Maximum steps reached without a final answer.';
      }
      if (branches.length === 1) {
        return branches[0].finalAnswer ?? 'Maximum steps reached without a final answer.';
      }
      return [...branches]
        .map((branch) => branch.finalAnswer || '')
        .find((answer) => answer.length > 0)
        ?? 'Maximum steps reached without a final answer.';
    };

    const finalizeBranchSafely = async (
      branch: AgentBranchState,
      reason: string,
      fallbackAnswer: string,
    ): Promise<string> => {
      upsertKanbanCard(branch, {
        status: 'finalizing',
        summary: previewText(reason, 160) || 'Finalizing branch output.',
      });
      if (!this.finalizeBranch) {
        return fallbackAnswer;
      }
      try {
        const finalized = await this.finalizeBranch({ branch, reason });
        return finalized?.trim() || fallbackAnswer;
      } catch (error: unknown) {
        const message = error instanceof Error ? error.message : String(error);
        emitBranchTrace('error', branch, `Finalize failed for ${branch.id}: ${message}`, { reason });
        branch.outcome = branch.outcome ?? 'failed';
        branch.outcomeReason = branch.outcomeReason ?? `finalize failed: ${message}`;
        branch.disposition = branch.disposition ?? 'planner_error';
        return fallbackAnswer;
      }
    };

    const completeErroredBranch = (branch: AgentBranchState, error: unknown, stage: string) => {
      const message = error instanceof Error ? error.message : String(error);
      upsertKanbanCard(branch, {
        status: 'error',
        summary: previewText(`${stage}: ${message}`, 160),
        blockers: [`${stage}: ${message}`],
      });
      emitBranchTrace('error', branch, `${stage}: ${message}`);
      branch.finalAnswer = branch.finalAnswer || `Branch ${branch.id} failed: ${message}`;
      branch.outcome = branch.outcome ?? 'failed';
      branch.outcomeReason = branch.outcomeReason ?? `${stage}: ${message}`;
      branch.disposition = branch.disposition ?? 'planner_error';
      addCompletedBranch(branch);
      emitBranchTrace('observation', branch, 'Branch completed with error.');
    };

    const markBranchCompleted = (
      branch: AgentBranchState,
      summaryFallback: string,
      blockers?: string[],
    ) => {
      upsertKanbanCard(branch, {
        status: 'completed',
        outcome: branch.outcome,
        disposition: branch.disposition,
        summary: previewText(
          branch.completionSummary
          || branch.finalAnswer
          || branch.outcomeReason
          || summaryFallback,
          180,
        ) || summaryFallback,
        blockers: blockers ?? (
          branch.outcome === 'failed' || branch.outcome === 'partial'
            ? dedupeNonEmpty([branch.outcomeReason])
            : []
        ),
      });
    };

    const renderObservation = (toolCall: PlannedToolCall, execResult: ToolExecutionResult): ToolObservation => ({
      toolName: toolCall.name,
      result: execResult.success
        ? JSON.stringify(execResult.result ?? '')
        : `ERROR: ${execResult.error ?? 'unknown error'}`,
      success: execResult.success,
    });

    const splitBranchTranscript = (branch: AgentBranchState) => {
      const inheritedCount = Math.max(0, Math.min(branch.inheritedMessageCount, branch.transcript.length));
      return {
        inheritedTranscript: branch.transcript.slice(0, inheritedCount),
        localTranscript: branch.transcript.slice(inheritedCount),
      };
    };

    const setBranchLocalTranscript = (
      branch: AgentBranchState,
      localTranscript: TranscriptMessage[],
      inheritedTranscript?: TranscriptMessage[],
    ) => {
      const inherited = inheritedTranscript ?? splitBranchTranscript(branch).inheritedTranscript;
      branch.inheritedMessageCount = inherited.length;
      branch.transcript = [...inherited, ...localTranscript];
    };

    // When a custom executeToolCall hook is provided, the hook is responsible
    // for emitting its own action/observation traces.  The runtime should NOT
    // double-emit — only trace when there is no hook.
    const shouldRuntimeTraceTool = () => !this.executeToolCall;

    const executeSingleToolCall = async (
      branch: AgentBranchState,
      toolCall: PlannedToolCall,
    ): Promise<ToolObservation> => {
      if (shouldRuntimeTraceTool()) {
        emitBranchTrace(
          'action',
          branch,
          `Calling ${toolCall.name}${toolCall.goal ? ` — ${toolCall.goal}` : ''}`,
          { toolName: toolCall.name, args: toolCall.arguments },
        );
      }

      if (this.executeToolCall) {
        return this.executeToolCall({ branch, toolCall });
      }

      const execResult = await this.toolRuntime.execute(toolCall.name, toolCall.arguments);
      return renderObservation(toolCall, execResult);
    };

    const emitObservationTrace = (branch: AgentBranchState, observation: ToolObservation) => {
      emitBranchTrace('observation', branch, observation.result, { toolName: observation.toolName });
    };

    const canRunToolCallsConcurrently = (toolCalls: PlannedToolCall[]) =>
      toolCalls.length > 1
      && toolCalls.every((toolCall) => AgentRuntime.PARALLEL_SAFE_TOOL_NAMES.has(toolCall.name));

    const noteExecutedToolRound = (
      branch: AgentBranchState,
      toolCalls: PlannedToolCall[],
      observations: ToolObservation[],
    ) => {
      const summary = enrichWebSearchRoundWithObservations(
        summarizeWebSearchRound(toolCalls, branch.steps),
        observations,
      );
      if (!summary) {
        webSearchRoundHistory.delete(branch.id);
        return;
      }
      const prior = webSearchRoundHistory.get(branch.id) || [];
      webSearchRoundHistory.set(branch.id, [...prior, summary].slice(-4));
    };

    const buildRepeatedSearchLoopMessage = (
      branch: AgentBranchState,
      currentRound: WebSearchRoundSummary,
      priorRounds: WebSearchRoundSummary[],
    ) => {
      const recentQueries = dedupeNonEmpty(priorRounds.flatMap((round) => round.queries)).slice(-4);
      const recentDomains = dedupeNonEmpty(priorRounds.flatMap((round) => [...round.explicitDomains, ...round.resultDomains])).slice(0, 6);
      const lines = [
        `Blocked near-duplicate web search loop in branch "${branch.label}".`,
        `You have already spent ${priorRounds.length} consecutive round(s) on near-duplicate web searches.`,
        recentQueries.length > 0 ? `Recent queries: ${recentQueries.map((query) => JSON.stringify(query)).join(', ')}` : '',
        recentDomains.length > 0 ? `Observed domains so far: ${recentDomains.join(', ')}` : '',
        `The newly proposed search queries are still too similar: ${currentRound.queries.map((query) => JSON.stringify(query)).join(', ')}`,
        'Do not continue the same search loop.',
        'Your next step must do one of these instead: fetch a promising URL you already found, materially change domains/source type/language/date scope, or finalize with the best available evidence and explicit uncertainty.',
      ].filter(Boolean);
      return lines.join('\n');
    };

    const shouldBlockRepeatedSearchLoop = (
      branch: AgentBranchState,
      toolCalls: PlannedToolCall[],
    ): string | null => {
      const currentRound = summarizeWebSearchRound(toolCalls, branch.steps);
      if (!currentRound) {
        return null;
      }
      const history = webSearchRoundHistory.get(branch.id) || [];
      if (history.length < 2) {
        return null;
      }
      const priorRounds = history.slice(-2);
      if (!priorRounds.every((round) => isNearDuplicateSearchRound(currentRound, round))) {
        return null;
      }
      if (hasMaterialSearchDomainShift(currentRound, priorRounds)) {
        return null;
      }
      return buildRepeatedSearchLoopMessage(branch, currentRound, priorRounds);
    };

    const executeToolStep = async (branch: AgentBranchState, toolCalls: PlannedToolCall[]) => {
      let observations: ToolObservation[];
      if (canRunToolCallsConcurrently(toolCalls)) {
        observations = await Promise.all(
          toolCalls.map((toolCall) => executeSingleToolCall(branch, toolCall)),
        );
      } else {
        observations = [];
        for (const toolCall of toolCalls) {
          const observation = await executeSingleToolCall(branch, toolCall);
          observations.push(observation);
        }
      }

      const customTranscript = this.buildToolTranscript?.({ branch, toolCalls, observations });
      if (customTranscript && customTranscript.length > 0) {
        branch.transcript.push(...customTranscript);
      } else {
        branch.transcript.push({
          role: 'assistant',
          content: formatPlannerToolDecisionMessage(toolCalls),
        });
        branch.transcript.push({
          role: 'user',
          content: observations
            .map((o, index) =>
              `${formatToolObservationHeader(toolCalls[index] || { name: o.toolName, arguments: {} }, o.toolName)}\n${o.result}`)
            .join('\n\n'),
        });
      }

      const observationSummary = previewText(
        observations
          .map((observation) => `${observation.toolName}: ${previewText(observation.result, 80)}`)
          .join(' | '),
        180,
      );
      upsertKanbanCard(branch, {
        status: 'active',
        summary: observationSummary || 'Tool observations recorded.',
        blockers: observations
          .filter((observation) => !observation.success)
          .map((observation) => `${observation.toolName}: ${previewText(observation.result, 120)}`),
        artifacts: extractArtifactPaths(toolCalls),
      });
      noteExecutedToolRound(branch, toolCalls, observations);

      if (shouldRuntimeTraceTool()) {
        observations.forEach((observation) => emitObservationTrace(branch, observation));
      }
    };

    const buildChildBranches = (
      branch: AgentBranchState,
      branches: PlannedBranch[],
    ): AgentBranchState[] => {
      // Compress the parent transcript before inheriting \u2014 this is the key
      // strategy that allows deeper/wider branching without blowing the budget.
      const compressedParentTranscript = compressBranchTranscript(branch.transcript, {
        tailKeep: 4,
        maxSummaryChars: 1500,
      }) as TranscriptMessage[];
      const siblingLabels = new Set(branches.map((candidate) => candidate.label));

      return branches.map((child, index) => {
        const executionGroup = normalizeExecutionGroup(child.executionGroup, 1);
        const dependsOn = normalizeDependencyLabels(child.dependsOn)
          .filter((label) => label !== child.label && siblingLabels.has(label));
        const sharedHandoffMessage = renderSharedHandoffMessage(branch.sharedHandoff);
        const branchHandoffMessage = renderBranchHandoffMessage(child.handoff);
        const kanbanSyncMessage = renderKanbanSyncMessage(branch);
        const structuredHandoffBlock = [sharedHandoffMessage, branchHandoffMessage]
          .filter((section) => section.length > 0)
          .join('\n\n');
        const transcript: TranscriptMessage[] = [
          ...compressedParentTranscript,
          {
            role: 'assistant',
            content: formatPlannerBranchDecisionMessage(branches),
          },
          {
            role: 'user',
            content: `You are now working on child branch "${child.label}".\nGoal: ${child.goal}\nWhy: ${child.why || 'Distinct strategy'}\nExecution group: ${executionGroup}\nDepends on sibling labels: ${dependsOn.length ? dependsOn.join(', ') : 'none'}\nThis branch owns only its local sub-problem.\nSibling branches in the same execution group may run in parallel. Higher execution groups wait until all lower groups under the same parent finish.\nIf depends_on is set, this branch should not start until those sibling labels have already finished.\nTreat execution_group and depends_on as binding coordination constraints, not hints.\nOnly do work that can make useful progress within this branch's scope. Do not perform downstream integration early if that integration mainly depends on sibling outputs that are still being produced elsewhere.\nRe-branch only if this branch still contains multiple genuinely independent substreams with low coordination risk. Do not branch again by default.\nDo not use multiple near-duplicate web searches as a substitute for branching.\nIf you emit multiple web searches in one response, vary the actual query terms and search dimensions, not just punctuation or word order. Change at least two of: keywords, aliases, domains, dates, language, or source type.\nYou may use tool calls, branch further if the goal truly benefits from independent sub-work, or return a final answer. Avoid repeating sibling work.${structuredHandoffBlock ? `\n\nStructured remediation handoff:\n${structuredHandoffBlock}` : ''}${kanbanSyncMessage ? `\n\n${kanbanSyncMessage}` : ''}`,
          },
        ];

        const childBranch: AgentBranchState = {
          id: `${branch.id}.${index + 1}`,
          parentId: branch.id,
          depth: branch.depth + 1,
          label: child.label,
          goal: child.goal,
          executionGroup,
          dependsOn,
          priority: Math.max(0.05, branch.priority * (child.priority ?? Math.max(0.2, 1 - index * 0.2))),
          steps: 0,
          inheritedMessageCount: transcript.length,
          savedNodeIds: branch.savedNodeIds.slice(),
          transcript,
          handoff: child.handoff,
          sharedHandoff: branch.sharedHandoff,
          ancestorBranchIds: [...(branch.ancestorBranchIds || []), branch.id],
          ancestorBranchLabels: [...(branch.ancestorBranchLabels || []), branch.label],
        };
        siblingFamilies.set(siblingKey(branch.id, child.label), {
          rootId: childBranch.id,
          executionGroup,
        });

        upsertKanbanCard(childBranch, {
          status: 'queued',
          executionGroup,
          dependsOn,
          summary: previewText(child.why || `Waiting for execution group ${executionGroup}.`, 140) || `Waiting for execution group ${executionGroup}.`,
        });
        return childBranch;
      });
    };

    const runBranch = async (branch: AgentBranchState): Promise<AgentBranchState[]> => {
      while (true) {
        // ── Context gate ───────────────────────────────────────────────
        // Each branch's ONLY hard limit is its context window. When the
        // transcript approaches the model context limit, compress it.
        // Only stop when even nuclear compression cannot free enough room.
        if (this.modelLimit > 0) {
          const { inheritedTranscript, localTranscript } = splitBranchTranscript(branch);
          const inheritedTokens = estimateTranscriptTokens(inheritedTranscript);
          const check = checkContextAndCompress(
            localTranscript,
            this.modelLimit,
            this.committedTokens + inheritedTokens,
          );
          if (check.compressed) {
            setBranchLocalTranscript(branch, check.transcript as TranscriptMessage[], inheritedTranscript);
            emitBranchTrace('observation', branch,
              `Context ${(check.usageRatio * 100).toFixed(1)}% — auto-compressed transcript.`);
          }
          if (!check.canContinue) {
            const reason = 'context window exhausted after maximum compression';
            branch.finalAnswer = await finalizeBranchSafely(
              branch,
              reason,
              `Branch ${branch.id} stopped: context window full.`,
            );
            branch.outcome = branch.outcome ?? 'partial';
            branch.outcomeReason = branch.outcomeReason ?? reason;
            branch.disposition = branch.disposition ?? 'missing_evidence';
            markBranchCompleted(branch, reason, [reason]);
            addCompletedBranch(branch);
            emitBranchTrace('observation', branch,
              `Branch completed: context exhausted (${(check.usageRatio * 100).toFixed(1)}%).`);
            return [];
          }
        } else {
          // Legacy mode (no model limit known): use transcript budget compression.
          if (branch.steps >= this.maxSteps || totalLoopSteps >= totalLoopBudget) {
            const reason = totalLoopSteps >= totalLoopBudget ? 'total budget reached' : 'step budget reached';
            branch.finalAnswer = await finalizeBranchSafely(
              branch,
              reason,
              `Branch ${branch.id} stopped because ${reason}.`,
            );
            branch.outcome = branch.outcome ?? 'partial';
            branch.outcomeReason = branch.outcomeReason ?? reason;
            branch.disposition = branch.disposition ?? 'missing_evidence';
            markBranchCompleted(branch, reason, [reason]);
            addCompletedBranch(branch);
            emitBranchTrace('observation', branch, `Branch completed after hitting ${reason}.`);
            return [];
          }
        }

        branch.steps += 1;
        totalLoopSteps += 1;
        lastTouchedBranch = branch;
        upsertKanbanCard(branch, {
          status: 'active',
          step: branch.steps,
          summary: previewText(branch.goal, 140) || 'Branch is running.',
        });

        const planningTranscript = buildPlanningTranscript(branch);
        const kanbanSnapshot = buildKanbanSnapshot();
        const agentStep = this.planBranch
          ? await this.planBranch({ branch, planningTranscript, kanbanSnapshot })
          : await this.planner.plan(planningTranscript);
        if (agentStep.thought) {
          emitBranchTrace('thought', branch, agentStep.thought);
        }

        if (agentStep.kind === 'final') {
          const needsExplicitFinalize = agentStep.finalizationMode === 'planner_fallback';
          if (needsExplicitFinalize) {
            emitBranchTrace('thought', branch, `Planner fallback reply is not treated as a real final step; explicitly finalizing ${branch.id}.`);
            branch.finalAnswer = await finalizeBranchSafely(
              branch,
              'planner format fallback after schema repair retries',
              agentStep.content,
            );
            branch.outcome = branch.outcome ?? 'partial';
            branch.outcomeReason = branch.outcomeReason ?? 'planner format fallback';
            branch.disposition = branch.disposition ?? 'planner_error';
            branch.finalizationMode = 'planner_fallback';
          } else {
            branch.finalAnswer = agentStep.content;
            branch.outcome = agentStep.outcome ?? branch.outcome ?? 'unknown';
            branch.outcomeReason = agentStep.outcomeReason ?? branch.outcomeReason;
            branch.disposition = agentStep.disposition ?? branch.disposition;
            branch.finalizationMode = agentStep.finalizationMode ?? branch.finalizationMode ?? 'natural';
          }
          branch.completionSummary = agentStep.completionSummary ?? branch.completionSummary;
          markBranchCompleted(
            branch,
            needsExplicitFinalize ? 'Planner fallback branch finalized.' : 'Branch completed.',
          );
          addCompletedBranch(branch);
          emitBranchTrace('observation', branch, needsExplicitFinalize ? 'Branch finalized after planner fallback.' : 'Branch completed.');
          return [];
        }

        if (agentStep.kind === 'tool') {
          const toolCalls = agentStep.toolCalls.slice(0, this.maxBranchWidth);
          if (toolCalls.length === 0) {
            branch.transcript.push({
              role: 'user',
              content: 'You chose a tool step but did not include any usable tool calls. Continue with at least one tool call or finish this branch.',
            });
            continue;
          }
          const repeatedSearchLoopMessage = shouldBlockRepeatedSearchLoop(branch, toolCalls);
          if (repeatedSearchLoopMessage) {
            clearPendingProviderToolReplay(branch);
            branch.transcript.push({
              role: 'user',
              content: repeatedSearchLoopMessage,
            });
            upsertKanbanCard(branch, {
              status: 'active',
              summary: 'Blocked near-duplicate web search loop; replanning required.',
              blockers: ['repeated near-duplicate web search loop'],
            });
            emitBranchTrace(
              'observation',
              branch,
              'Blocked near-duplicate web search loop; next step must fetch, materially change domains, or finalize.',
              { toolNames: toolCalls.map((toolCall) => toolCall.name) },
            );
            continue;
          }
          await executeToolStep(branch, toolCalls);

          // In dynamic mode, compression is handled by checkContextAndCompress
          // at the top of the loop.  Legacy mode: progressive compression fallback.
          if (this.modelLimit <= 0 && branch.steps >= 3) {
            const { inheritedTranscript, localTranscript } = splitBranchTranscript(branch);
            const inheritedChars = inheritedTranscript.reduce((sum, message) => sum + message.content.length, 0);
            const localBudgetChars = Math.max(4000, this.transcriptBudgetChars - inheritedChars);
            setBranchLocalTranscript(
              branch,
              progressiveCompressBranch(localTranscript, localBudgetChars, 6) as TranscriptMessage[],
              inheritedTranscript,
            );
          }
          continue;
        }

        if (branch.depth >= this.maxBranchDepth) {
          const reason = `maximum branch depth ${this.maxBranchDepth} reached`;
          branch.finalAnswer = await finalizeBranchSafely(
            branch,
            reason,
            `Branch ${branch.id} stopped because ${reason}.`,
          );
          branch.outcome = branch.outcome ?? 'partial';
          branch.outcomeReason = branch.outcomeReason ?? reason;
          branch.disposition = branch.disposition ?? 'missing_evidence';
          markBranchCompleted(branch, reason, [reason]);
          addCompletedBranch(branch);
          emitBranchTrace('observation', branch, `Branch completed after hitting ${reason}.`);
          return [];
        }

        const childPlans = agentStep.branches.slice(0, this.maxBranchWidth);
        if (childPlans.length === 0) {
          branch.transcript.push({
            role: 'user',
            content: 'Branching was rejected because no valid child branches were provided. Continue this branch without splitting.',
          });
          continue;
        }

        const childBranches = buildChildBranches(branch, childPlans);
        upsertKanbanCard(branch, {
          status: 'completed',
          summary: `Delegated to ${childPlans.length} child branch${childPlans.length === 1 ? '' : 'es'}.`,
          blockers: [],
        });
        emitBranchTrace('action', branch, `Spawning ${childPlans.length} child branches.`, { childCount: childPlans.length });
        for (const childBranch of childBranches) {
          this.emit({
            type: 'observation',
            content: `Spawned child branch ${childBranch.id}: ${childBranch.label}`,
            metadata: traceMeta(childBranch),
          });
        }
        return childBranches;
      }
    };

    const prioritySort = (a: AgentBranchState, b: AgentBranchState) =>
      (b.priority - a.priority) || (a.depth - b.depth) || (a.steps - b.steps);
    const active = new Set<Promise<void>>();
    const activeBranches = new Map<string, AgentBranchState>();
    const canLaunchMoreBranches = () =>
      (this.branchConcurrency <= 0 || active.size < this.branchConcurrency)
      && (this.modelLimit > 0 ? true : totalLoopSteps < totalLoopBudget);
    const branchBelongsToFamily = (candidate: AgentBranchState, familyRootId: string) =>
      candidate.id === familyRootId || candidate.ancestorBranchIds?.includes(familyRootId) === true;

    const isSiblingFamilyPending = (
      parentId: string | null,
      label: string,
      blockedBranches: AgentBranchState[] = [],
    ): boolean => {
      if (!parentId) {
        return false;
      }
      const family = siblingFamilies.get(siblingKey(parentId, label));
      if (!family) {
        return false;
      }
      if (queue.some((candidate) => branchBelongsToFamily(candidate, family.rootId))) {
        return true;
      }
      if (blockedBranches.some((candidate) => branchBelongsToFamily(candidate, family.rootId))) {
        return true;
      }
      return [...activeBranches.values()].some((candidate) => branchBelongsToFamily(candidate, family.rootId));
    };

    const hasEarlierSiblingExecutionGroupPending = (
      branch: AgentBranchState,
      blockedBranches: AgentBranchState[] = [],
    ): boolean => {
      if (!branch.parentId || branch.executionGroup <= 1) {
        return false;
      }

      for (const [familyKey, family] of siblingFamilies.entries()) {
        if (!familyKey.startsWith(`${branch.parentId}::`) || family.executionGroup >= branch.executionGroup) {
          continue;
        }
        const siblingLabel = familyKey.slice(branch.parentId.length + 2);
        if (isSiblingFamilyPending(branch.parentId, siblingLabel, blockedBranches)) {
          return true;
        }
      }
      return false;
    };

    const getUnmetSiblingDependencies = (
      branch: AgentBranchState,
      blockedBranches: AgentBranchState[] = [],
    ): string[] => {
      if (!branch.parentId || branch.dependsOn.length === 0) {
        return [];
      }
      return branch.dependsOn.filter((label) => isSiblingFamilyPending(branch.parentId, label, blockedBranches));
    };

    const hasUnmetSiblingDependencies = (branch: AgentBranchState): boolean =>
      getUnmetSiblingDependencies(branch).length > 0;

    const deadlockQueuedBranches = (blockedBranches: AgentBranchState[]) => {
      for (const branch of blockedBranches) {
        const blockers = [
          hasEarlierSiblingExecutionGroupPending(branch, blockedBranches)
            ? `waiting for lower execution_group siblings under ${branch.parentId}`
            : '',
          ...getUnmetSiblingDependencies(branch, blockedBranches).map((label) => `depends_on: ${label}`),
        ].filter(Boolean);
        const reason = blockers.length > 0
          ? `Scheduler deadlock: ${blockers.join('; ')}`
          : 'Scheduler deadlock: no runnable branch remained.';
        upsertKanbanCard(branch, {
          status: 'error',
          executionGroup: branch.executionGroup,
          dependsOn: branch.dependsOn,
          summary: previewText(reason, 160),
          blockers,
        });
        emitBranchTrace('error', branch, reason, { blockers });
        branch.finalAnswer = branch.finalAnswer || `Branch ${branch.id} was blocked: ${reason}`;
        branch.outcome = branch.outcome ?? 'failed';
        branch.outcomeReason = branch.outcomeReason ?? reason;
        branch.disposition = branch.disposition ?? 'planner_error';
        addCompletedBranch(branch);
      }
    };

    const launchQueuedBranches = () => {
      queue.sort(prioritySort);
      while (queue.length > 0 && canLaunchMoreBranches()) {
        const runnableIndex = queue.findIndex((branch) =>
          !hasEarlierSiblingExecutionGroupPending(branch)
          && !hasUnmetSiblingDependencies(branch));
        if (runnableIndex < 0) {
          if (active.size === 0) {
            const blockedBranches = queue.splice(0, queue.length);
            deadlockQueuedBranches(blockedBranches);
          }
          break;
        }
        const [branch] = queue.splice(runnableIndex, 1);
        upsertKanbanCard(branch, {
          status: 'active',
          executionGroup: branch.executionGroup,
          dependsOn: branch.dependsOn,
          summary: previewText(branch.goal, 140) || 'Branch picked up by scheduler.',
        });
        let task: Promise<void>;
        activeBranches.set(branch.id, branch);
        task = runBranch(branch)
          .then((childBranches) => {
            active.delete(task);
            activeBranches.delete(branch.id);
            if (canLaunchMoreBranches()) {
              childBranches.forEach((child) => queue.push(child));
            }
            launchQueuedBranches();
          })
          .catch((error: unknown) => {
            active.delete(task);
            activeBranches.delete(branch.id);
            completeErroredBranch(branch, error, 'Branch execution failed');
            launchQueuedBranches();
          });
        active.add(task);
        const dependencySuffix = branch.dependsOn.length > 0 ? `, depends_on=${branch.dependsOn.join(',')}` : '';
        emitBranchTrace('thought', branch, `[scheduler] Branch ${branch.id} started (execution_group=${branch.executionGroup}${dependencySuffix}, active=${active.size}).`);
      }
    };

    launchQueuedBranches();
    while (active.size > 0) {
      await Promise.race(active);
    }

    if (completed.length === 0 && lastTouchedBranch) {
      emitBranchTrace('thought', lastTouchedBranch, `No branch reached a natural final answer; explicitly finalizing ${lastTouchedBranch.id}.`);
      lastTouchedBranch.finalAnswer = await finalizeBranchSafely(
        lastTouchedBranch,
        'no branch reached a natural final answer',
        'No branch produced a final answer.',
      );
      emitBranchTrace('observation', lastTouchedBranch, 'Branch finalized because no branch reached a natural final answer.');
      lastTouchedBranch.outcome = lastTouchedBranch.outcome ?? 'partial';
      lastTouchedBranch.outcomeReason = lastTouchedBranch.outcomeReason ?? 'no branch reached a natural final answer';
      lastTouchedBranch.disposition = lastTouchedBranch.disposition ?? 'missing_evidence';
      markBranchCompleted(lastTouchedBranch, 'No branch reached a natural final answer.', ['no branch reached a natural final answer']);
      addCompletedBranch(lastTouchedBranch);
    }

    // ── Post-synthesis re-branch loop ──────────────────────────────────
    // After all branches complete, synthesize.  If the synthesis hook
    // reports that some branches failed and remediation branches are
    // needed, spawn them and re-enter the scheduling loop.
    let synthesisRetry = 0;
    while (true) {
      const finalBranches = completed
        .filter((branch) => typeof branch.finalAnswer === 'string' && branch.finalAnswer.length > 0);
      this.emit({
        type: 'thought',
        content: `Starting synthesis attempt ${synthesisRetry + 1}/${this.maxSynthesisRetries + 1} with ${finalBranches.length} completed branch(es).`,
      });
      let synthesisResult: SynthesisResult;
      try {
        synthesisResult = await this.synthesize(
          userMessage, priorTranscript, finalBranches, synthesisRetry,
        );
      } catch (error: unknown) {
        const message = error instanceof Error ? error.message : String(error);
        const fallbackAnswer = buildDefaultSynthesisAnswer(finalBranches);
        this.emit({ type: 'error', content: `Synthesis failed: ${message}` });
        this.emit({ type: 'observation', content: 'Falling back to default synthesis answer after synthesis failure.' });
        this.emit({ type: 'final', content: fallbackAnswer });
        return fallbackAnswer;
      }

      if (synthesisResult.done) {
        this.emit({ type: 'observation', content: 'Synthesis completed.' });
        this.emit({ type: 'final', content: synthesisResult.answer });
        return synthesisResult.answer;
      }

      if (synthesisResult.branches.length === 0 || synthesisRetry >= this.maxSynthesisRetries) {
        const reason = synthesisResult.branches.length === 0
          ? 'Synthesis requested remediation but returned zero branches; forcing completion with fallback synthesis answer.'
          : `Synthesis requested remediation beyond retry limit (${this.maxSynthesisRetries}); forcing completion with fallback synthesis answer.`;
        const fallbackAnswer = buildDefaultSynthesisAnswer(finalBranches);
        this.emit({ type: 'error', content: reason });
        this.emit({ type: 'final', content: fallbackAnswer });
        return fallbackAnswer;
      }

      // Re-branch: synthesis requested remediation branches.
      synthesisRetry++;
      this.emit({
        type: 'thought',
        content: `Synthesis requested ${synthesisResult.branches.length} remediation branch(es) (retry ${synthesisRetry}/${this.maxSynthesisRetries}).`,
      });

      // Build a synthetic parent branch that carries the completed context.
      const contextSummary = synthesisResult.context;
      const remedyParent: AgentBranchState = {
        id: `R${synthesisRetry}`,
        parentId: null,
        depth: 0,
        label: `synthesis-retry-${synthesisRetry}`,
        goal: 'Remediate failed branches from previous round.',
        executionGroup: 1,
        dependsOn: [],
        priority: 1,
        steps: 0,
        inheritedMessageCount: 0,
        savedNodeIds: completed.flatMap((b) => b.savedNodeIds),
        ancestorBranchIds: [],
        ancestorBranchLabels: [],
        transcript: [
          ...priorTranscript,
          { role: 'user', content: userMessage },
          {
            role: 'assistant',
            content: `Previous round summary:\n${contextSummary}`,
          },
        ],
        sharedHandoff: synthesisResult.handoff,
      };

      const remedyChildren = buildChildBranches(remedyParent, synthesisResult.branches);

      // Reset scheduling state for the new round.
      queue.length = 0;
      // Keep completed branches from previous rounds (their results feed the next synthesis).
      for (const child of remedyChildren) {
        queue.push(child);
      }

      launchQueuedBranches();
      while (active.size > 0) {
        await Promise.race(active);
      }
      // Loop back to synthesize with the updated completed set.
    }
  }

  private async synthesize(
    userMessage: string,
    priorTranscript: TranscriptMessage[],
    branches: AgentBranchState[],
    synthesisRetry: number,
  ): Promise<SynthesisResult> {
    if (branches.length === 0) {
      return { done: true, answer: 'Maximum steps reached without a final answer.' };
    }

    if (this.synthesizeFinal) {
      const hookResult = await this.synthesizeFinal({
        userMessage,
        priorTranscript,
        branches: branches.map((branch) => ({
          id: branch.id,
          label: branch.label,
          goal: branch.goal,
          savedNodeIds: branch.savedNodeIds,
          completionSummary: branch.completionSummary,
          finalAnswer: branch.finalAnswer ?? '',
          outcome: branch.outcome,
          outcomeReason: branch.outcomeReason,
          disposition: branch.disposition,
          finalizationMode: branch.finalizationMode,
          ancestorBranchIds: branch.ancestorBranchIds,
          ancestorBranchLabels: branch.ancestorBranchLabels,
        })),
        synthesisRetry,
        maxSynthesisRetries: this.maxSynthesisRetries,
      });

      // Backward compatible: if the hook returns a plain string, wrap it.
      if (typeof hookResult === 'string') {
        return { done: true, answer: hookResult };
      }
      return hookResult;
    }

    // Default synthesis (no hook): always done.
    if (branches.length === 1) {
      return { done: true, answer: branches[0].finalAnswer ?? 'Maximum steps reached without a final answer.' };
    }

    const answer = [...branches]
      .map((branch) => branch.finalAnswer || '')
      .find((a) => a.length > 0)
      ?? 'Maximum steps reached without a final answer.';
    return { done: true, answer };
  }

  private emit(event: AgentTraceEvent): void {
    try {
      this.onTrace(event);
    } catch {
      // Never let trace callbacks crash the loop.
    }
  }
}

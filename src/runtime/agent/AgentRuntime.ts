import {
  AgentStep,
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
import { WorkspaceManager } from '../../persistence/WorkspaceManager.js';

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
  /** Canonical plan version last seen by this branch. */
  planVersionSeen?: number;
  /** Branch-local excerpt extracted from the canonical plan. */
  planExcerpt?: string;
  /** Alignment checks that the branch should satisfy while executing. */
  alignmentChecks?: string[];
  /** Full branch ancestry ids, excluding this branch id. */
  ancestorBranchIds?: string[];
  /** Full branch ancestry labels, excluding this branch label. */
  ancestorBranchLabels?: string[];
  /** File paths created by this branch. */
  artifacts?: string[];
  /** Consecutive tool execution failures count. */
  consecutiveFailures?: number;
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
  fallbackAnswer: string;
}

export interface BranchSynthesisInput {
  userMessage: string;
  priorTranscript: TranscriptMessage[];
  archivedBranchSummary?: string;
  canonicalPlanVersion?: number;
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
    planVersionSeen?: number;
    planExcerpt?: string;
    alignmentChecks?: string[];
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
  /**
   * Additional token overhead per planner call (system prompt envelope,
   * metadata message, output reserve) that sits on top of the branch
   * transcript.  Used by the context gate so branches stop before the
   * planner input can overflow the model context window.
   */
  plannerEnvelopeTokens?: number;
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
  /** Optional workspace manager for persisting branch work. */
  workspaceManager?: WorkspaceManager;
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

interface ChildPlanValidationResult {
  valid: boolean;
  normalizedBranches: PlannedBranch[];
  errors: string[];
}

function validateChildPlanSet(branches: PlannedBranch[], maxBranchWidth: number): ChildPlanValidationResult {
  const errors: string[] = [];
  const trimmedBranches = branches.slice(0, maxBranchWidth);
  if (branches.length > maxBranchWidth) {
    errors.push(`planner returned ${branches.length} branches, exceeding maxBranchWidth=${maxBranchWidth}`);
  }

  const labelToIndex = new Map<string, number>();
  const normalizedBranches = trimmedBranches.map((branch) => ({
    ...branch,
    executionGroup: normalizeExecutionGroup(branch.executionGroup, 1),
    dependsOn: normalizeDependencyLabels(branch.dependsOn),
  }));

  normalizedBranches.forEach((branch, index) => {
    const normalizedLabel = String(branch.label || '').trim();
    if (!normalizedLabel) {
      errors.push(`branch[${index}] is missing a label`);
      return;
    }
    if (labelToIndex.has(normalizedLabel)) {
      errors.push(`duplicate sibling branch label: ${normalizedLabel}`);
      return;
    }
    labelToIndex.set(normalizedLabel, index);
  });

  const outgoing = Array.from({ length: normalizedBranches.length }, () => [] as number[]);
  const indegree = Array.from({ length: normalizedBranches.length }, () => 0);

  normalizedBranches.forEach((branch, index) => {
    const dependencies = normalizeDependencyLabels(branch.dependsOn);
    dependencies.forEach((dependencyLabel) => {
      if (dependencyLabel === branch.label) {
        errors.push(`branch "${branch.label}" cannot depend on itself`);
        return;
      }
      const dependencyIndex = labelToIndex.get(dependencyLabel);
      if (dependencyIndex === undefined) {
        errors.push(`branch "${branch.label}" depends_on unknown sibling "${dependencyLabel}"`);
        return;
      }
      const dependencyGroup = normalizedBranches[dependencyIndex]?.executionGroup ?? 1;
      const branchGroup = branch.executionGroup ?? 1;
      if (dependencyGroup > branchGroup) {
        errors.push(`branch "${branch.label}" depends_on later execution_group sibling "${dependencyLabel}"`);
        return;
      }
      outgoing[dependencyIndex].push(index);
      indegree[index] += 1;
    });
  });

  const ready: number[] = [];
  indegree.forEach((count, index) => {
    if (count === 0) {
      ready.push(index);
    }
  });
  let processed = 0;
  while (ready.length > 0) {
    const current = ready.shift()!;
    processed += 1;
    outgoing[current].forEach((target) => {
      indegree[target] -= 1;
      if (indegree[target] === 0) {
        ready.push(target);
      }
    });
  }

  if (processed < normalizedBranches.length) {
    errors.push('planner returned cyclic sibling depends_on graph');
  }

  return {
    valid: errors.length === 0,
    normalizedBranches,
    errors: dedupeNonEmpty(errors),
  };
}

function isMutatingToolCall(toolCall: PlannedToolCall): boolean {
  return toolCall.name === 'edit' || toolCall.name === 'bash';
}

function buildArchivedBranchSummary(branches: AgentBranchState[]): string {
  const lines = branches
    .map((branch) => {
      const summary = previewText(
        branch.completionSummary
        || branch.finalAnswer
        || branch.outcomeReason
        || branch.goal,
        220,
      );
      return [
        `[${branch.id}] ${branch.label}`,
        branch.outcome ? `outcome=${branch.outcome}` : '',
        branch.disposition ? `disposition=${branch.disposition}` : '',
        summary ? `summary=${summary}` : '',
      ].filter(Boolean).join(' | ');
    })
    .filter(Boolean);
  return lines.join('\n');
}

function selectBranchesForSynthesis(
  branches: AgentBranchState[],
  maxCompletedBranches: number,
): {
  selectedBranches: AgentBranchState[];
  archivedSummary: string;
} {
  if (branches.length <= maxCompletedBranches) {
    return {
      selectedBranches: branches,
      archivedSummary: '',
    };
  }

  const frontier = branches.filter((branch) => !branches.some((other) =>
    other !== branch
    && Array.isArray(other.ancestorBranchIds)
    && other.ancestorBranchIds.includes(branch.id),
  ));

  const selected = new Map<string, AgentBranchState>();
  if (frontier.length >= maxCompletedBranches) {
    const prioritizedFrontier = [...frontier].sort((left, right) => {
      const leftHealthy = left.outcome !== 'failed'
        && left.outcome !== 'partial'
        && left.disposition !== 'planner_error'
        && left.finalizationMode !== 'planner_fallback'
        && typeof left.finalAnswer === 'string'
        && left.finalAnswer.length > 0;
      const rightHealthy = right.outcome !== 'failed'
        && right.outcome !== 'partial'
        && right.disposition !== 'planner_error'
        && right.finalizationMode !== 'planner_fallback'
        && typeof right.finalAnswer === 'string'
        && right.finalAnswer.length > 0;
      if (leftHealthy !== rightHealthy) {
        return leftHealthy ? -1 : 1;
      }
      return branches.indexOf(right) - branches.indexOf(left);
    });
    const trimmedFrontier = prioritizedFrontier.slice(0, maxCompletedBranches);
    trimmedFrontier.forEach((branch) => {
      selected.set(branch.id, branch);
    });
    const archived = branches.filter((branch) => !selected.has(branch.id));
    return {
      selectedBranches: branches.filter((branch) => selected.has(branch.id)),
      archivedSummary: archived.length > 0 ? buildArchivedBranchSummary(archived) : '',
    };
  }
  frontier.forEach((branch) => {
    selected.set(branch.id, branch);
  });

  const successfulAncestors = [...branches]
    .reverse()
    .filter((branch) =>
      !selected.has(branch.id)
      && branch.outcome === 'success'
      && frontier.some((candidate) => candidate.ancestorBranchIds?.includes(branch.id)),
    );

  for (const branch of successfulAncestors) {
    if (selected.size >= maxCompletedBranches) {
      break;
    }
    selected.set(branch.id, branch);
  }

  const recentResolved = [...branches]
    .reverse()
    .filter((branch) =>
      !selected.has(branch.id)
      && (branch.outcome === 'success' || branch.disposition === 'resolved'),
    );

  for (const branch of recentResolved) {
    if (selected.size >= maxCompletedBranches) {
      break;
    }
    selected.set(branch.id, branch);
  }

  const orderedSelected = branches.filter((branch) => selected.has(branch.id));
  const archived = branches.filter((branch) => !selected.has(branch.id));
  return {
    selectedBranches: orderedSelected,
    archivedSummary: archived.length > 0 ? buildArchivedBranchSummary(archived) : '',
  };
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
  private readonly plannerEnvelopeTokens: number;
  private readonly planBranch?: (input: BranchPlanInput) => Promise<import('./types.js').AgentStep>;
  private readonly executeToolCall?: (input: BranchToolExecutionInput) => Promise<ToolObservation>;
  private readonly buildToolTranscript?: (input: BranchToolTranscriptInput) => TranscriptMessage[] | null | undefined;
  private readonly finalizeBranch?: (input: BranchFinalizeInput) => Promise<string>;
  private readonly synthesizeFinal?: (input: BranchSynthesisInput) => Promise<string | SynthesisResult>;
  private readonly maxSynthesisRetries: number;
  private readonly onTrace: (event: AgentTraceEvent) => void;
  private readonly workspaceManager?: WorkspaceManager;

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
    this.plannerEnvelopeTokens = options.plannerEnvelopeTokens ?? 0;
    this.planBranch = options.planBranch;
    this.executeToolCall = options.executeToolCall;
    this.buildToolTranscript = options.buildToolTranscript;
    this.finalizeBranch = options.finalizeBranch;
    this.synthesizeFinal = options.synthesizeFinal;
    this.maxSynthesisRetries = options.maxSynthesisRetries ?? 2;
    this.onTrace = options.onTrace ?? (() => undefined);
    this.workspaceManager = options.workspaceManager;
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
    const siblingFamilies = new Map<string, { rootId: string; executionGroup: number }>();
    let lastTouchedBranch: AgentBranchState | null = rootBranch;
    let totalLoopSteps = 0;
    let activeMutatingBranchId: string | null = null;
    const mutationWaitQueue: Array<() => void> = [];
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
      // Merge the kanban sync into the leading system message so there is always
      // exactly ONE system-role entry in the transcript.  Models such as MiniMax
      // reject requests that contain more than one system role (error 2013).
      const [first, ...rest] = branch.transcript;
      if (first?.role === 'system') {
        return [
          { role: 'system', content: `${syncMessage}\n\n${String(first.content)}` },
          ...rest,
        ];
      }
      // No leading system message — create one so the sync always lands at index 0.
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
        // Persist branch work if workspace manager is available
        if (this.workspaceManager) {
          this.workspaceManager.saveBranchWork(branch.id, branch).catch((err) => {
            console.error(`Failed to persist branch ${branch.id}:`, err);
          });
        }
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
        const finalized = await this.finalizeBranch({ branch, reason, fallbackAnswer });
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
      const terminalStatus: BranchKanbanStatus = (
        branch.disposition === 'planner_error' || branch.finalizationMode === 'planner_fallback'
      )
        ? 'error'
        : 'completed';
      upsertKanbanCard(branch, {
        status: terminalStatus,
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

    const acquireMutatingBranchLock = async (branch: AgentBranchState) => {
      if (activeMutatingBranchId === null || activeMutatingBranchId === branch.id) {
        activeMutatingBranchId = branch.id;
        return () => {
          if (activeMutatingBranchId === branch.id) {
            activeMutatingBranchId = null;
          }
          const next = mutationWaitQueue.shift();
          next?.();
        };
      }

      emitBranchTrace(
        'thought',
        branch,
        `Waiting for mutating branch lock; ${activeMutatingBranchId} is currently executing workspace-changing tools.`,
      );
      await new Promise<void>((resolve) => {
        mutationWaitQueue.push(resolve);
      });
      activeMutatingBranchId = branch.id;
      emitBranchTrace('thought', branch, 'Acquired mutating branch lock.');
      return () => {
        if (activeMutatingBranchId === branch.id) {
          activeMutatingBranchId = null;
        }
        const next = mutationWaitQueue.shift();
        next?.();
      };
    };

    const emitObservationTrace = (branch: AgentBranchState, observation: ToolObservation) => {
      emitBranchTrace('observation', branch, observation.result, { toolName: observation.toolName });
    };

    const canRunToolCallsConcurrently = (toolCalls: PlannedToolCall[]) =>
      toolCalls.length > 1
      && toolCalls.every((toolCall) => AgentRuntime.PARALLEL_SAFE_TOOL_NAMES.has(toolCall.name));

    const executeToolStep = async (branch: AgentBranchState, toolCalls: PlannedToolCall[]) => {
      const needsMutatingLock = toolCalls.some((toolCall) => isMutatingToolCall(toolCall));
      const releaseMutatingLock = needsMutatingLock
        ? await acquireMutatingBranchLock(branch)
        : null;

      let observations: ToolObservation[];
      try {
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
      } finally {
        releaseMutatingLock?.();
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

      if (shouldRuntimeTraceTool()) {
        observations.forEach((observation) => emitObservationTrace(branch, observation));
      }

      // Track consecutive failures
      const allFailed = observations.length > 0 && observations.every((o) => !o.success);
      if (allFailed) {
        branch.consecutiveFailures = (branch.consecutiveFailures || 0) + 1;
      } else if (observations.some((o) => o.success)) {
        branch.consecutiveFailures = 0;
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
            content: `You are now working on child branch "${child.label}".\nGoal: ${child.goal}\nWhy: ${child.why || 'Distinct strategy'}\nExecution group: ${executionGroup}\nDepends on sibling labels: ${dependsOn.length ? dependsOn.join(', ') : 'none'}\nCanonical plan version: ${child.planVersion ?? branch.planVersionSeen ?? 0}\nBranch plan excerpt:\n${child.planExcerpt || branch.planExcerpt || '(none)'}\nAlignment checks: ${child.alignmentChecks?.length ? child.alignmentChecks.join(' | ') : '(none)'}\nThis branch owns only its local sub-problem.\nSibling branches in the same execution group may run in parallel. Higher execution groups wait until all lower groups under the same parent finish.\nIf depends_on is set, this branch should not start until those sibling labels have already finished.\nTreat execution_group and depends_on as binding coordination constraints, not hints.\nOnly do work that can make useful progress within this branch's scope. Do not perform downstream integration early if that integration mainly depends on sibling outputs that are still being produced elsewhere.\nRe-branch only if this branch still contains multiple genuinely independent substreams with low coordination risk. Do not branch again by default.\nUse web only in mode=fetch and only when you already have a trustworthy URL.\nYou may use tool calls, branch further if the goal truly benefits from independent sub-work, or return a final answer. Avoid repeating sibling work.${structuredHandoffBlock ? `\n\nStructured remediation handoff:\n${structuredHandoffBlock}` : ''}${kanbanSyncMessage ? `\n\n${kanbanSyncMessage}` : ''}`,
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
          planVersionSeen: child.planVersion ?? branch.planVersionSeen,
          planExcerpt: child.planExcerpt ?? branch.planExcerpt,
          alignmentChecks: child.alignmentChecks ?? [],
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
      const MAX_CONSECUTIVE_FAILURES = 5;

      try {
        while (true) {
        // Check consecutive failures limit
        if ((branch.consecutiveFailures || 0) >= MAX_CONSECUTIVE_FAILURES) {
          const reason = `${MAX_CONSECUTIVE_FAILURES} consecutive tool failures`;
          branch.finalAnswer = await finalizeBranchSafely(
            branch,
            reason,
            `Branch ${branch.id} stopped: all tool calls failed ${MAX_CONSECUTIVE_FAILURES} times in a row.`,
          );
          branch.outcome = 'failed';
          branch.outcomeReason = reason;
          branch.disposition = 'tool_failure';
          markBranchCompleted(branch, reason, [reason]);
          addCompletedBranch(branch);
          emitBranchTrace('observation', branch, `Branch stopped after ${MAX_CONSECUTIVE_FAILURES} consecutive failures.`);
          return [];
        }

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
            this.committedTokens + this.plannerEnvelopeTokens + inheritedTokens,
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
        let agentStep: AgentStep;
        try {
          agentStep = this.planBranch
            ? await this.planBranch({ branch, planningTranscript, kanbanSnapshot })
            : await this.planner.plan(planningTranscript);
        } catch (error: unknown) {
          const errorMessage = error instanceof Error ? error.message : String(error);
          emitBranchTrace('error', branch, `Planner failed: ${errorMessage}`);
          branch.finalAnswer = await finalizeBranchSafely(branch, 'planner error', `Planner failed: ${errorMessage}`);
          branch.outcome = 'failed';
          branch.outcomeReason = `planner error: ${errorMessage}`;
          branch.disposition = 'planner_error';
          markBranchCompleted(branch, 'Planner error.', [errorMessage]);
          addCompletedBranch(branch);
          return [];
        }
        if (agentStep.thought) {
          emitBranchTrace('thought', branch, agentStep.thought);
        }

        if (agentStep.kind === 'final') {
          const needsExplicitFinalize = agentStep.finalizationMode === 'planner_fallback';
          if (needsExplicitFinalize) {
            const fallbackFinalizeReason = agentStep.outcomeReason
              || agentStep.completionSummary
              || 'planner format fallback after schema repair retries';
            emitBranchTrace('thought', branch, `Planner fallback reply is not treated as a real final step; explicitly finalizing ${branch.id}.`);
            branch.finalAnswer = await finalizeBranchSafely(
              branch,
              fallbackFinalizeReason,
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

        const validatedChildPlanSet = validateChildPlanSet(childPlans, this.maxBranchWidth);
        if (!validatedChildPlanSet.valid) {
          branch.transcript.push({
            role: 'user',
            content: `Branching was rejected because the child coordination graph was invalid:\n- ${validatedChildPlanSet.errors.join('\n- ')}\nContinue this branch without splitting until you can produce a valid sibling graph.`,
          });
          upsertKanbanCard(branch, {
            status: 'active',
            summary: 'Rejected invalid child branch graph; replanning required.',
            blockers: validatedChildPlanSet.errors,
          });
          emitBranchTrace(
            'observation',
            branch,
            `Rejected invalid child branch graph: ${validatedChildPlanSet.errors.join(' | ')}`,
          );
          continue;
        }

        const childBranches = buildChildBranches(branch, validatedChildPlanSet.normalizedBranches);
        markBranchCompleted(
          branch,
          `Delegated to ${validatedChildPlanSet.normalizedBranches.length} child branch${validatedChildPlanSet.normalizedBranches.length === 1 ? '' : 'es'}.`,
        );
        addCompletedBranch(branch);
        emitBranchTrace('action', branch, `Spawning ${validatedChildPlanSet.normalizedBranches.length} child branches.`, { childCount: validatedChildPlanSet.normalizedBranches.length });
        for (const childBranch of childBranches) {
          this.emit({
            type: 'observation',
            content: `Spawned child branch ${childBranch.id}: ${childBranch.label}`,
            metadata: traceMeta(childBranch),
          });
        }
        return childBranches;
      }
      } catch (error: unknown) {
        // Catch-all to prevent branch Promise from never resolving
        const errorMessage = error instanceof Error ? error.message : String(error);
        emitBranchTrace('error', branch, `Branch execution failed with unhandled error: ${errorMessage}`);
        branch.finalAnswer = await finalizeBranchSafely(
          branch,
          'unhandled error',
          `Branch ${branch.id} failed: ${errorMessage}`,
        );
        branch.outcome = 'failed';
        branch.outcomeReason = `unhandled error: ${errorMessage}`;
        branch.disposition = 'planner_error';
        markBranchCompleted(branch, 'Unhandled error.', [errorMessage]);
        addCompletedBranch(branch);
        return [];
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

        // Wrap with timeout protection to prevent deadlock
        const BRANCH_TIMEOUT_MS = 600000; // 10 minutes
        const timeoutPromise = new Promise<never>((_, reject) => {
          setTimeout(() => reject(new Error(`Branch ${branch.id} timeout after ${BRANCH_TIMEOUT_MS}ms`)), BRANCH_TIMEOUT_MS);
        });

        task = Promise.race([runBranch(branch), timeoutPromise])
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

    // Global timeout protection to prevent infinite deadlock
    const GLOBAL_TIMEOUT_MS = 1800000; // 30 minutes
    const globalTimeoutPromise = new Promise<void>((resolve) => {
      setTimeout(() => {
        if (active.size > 0) {
          this.emit({
            type: 'error',
            content: `Global timeout: ${active.size} branch(es) still active after ${GLOBAL_TIMEOUT_MS}ms. Forcing completion.`,
          });
          // Force complete all stuck branches
          for (const branch of activeBranches.values()) {
            completeErroredBranch(branch, new Error('Global timeout'), 'Branch forced to complete due to global timeout');
          }
          active.clear();
          activeBranches.clear();
        }
        resolve();
      }, GLOBAL_TIMEOUT_MS);
    });

    await Promise.race([
      (async () => {
        while (active.size > 0) {
          await Promise.race(active);
        }
      })(),
      globalTimeoutPromise,
    ]);

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
      const completedForSynthesis = completed
        .filter((branch) => typeof branch.finalAnswer === 'string' && branch.finalAnswer.length > 0);
      const {
        selectedBranches: finalBranches,
        archivedSummary: archivedBranchSummary,
      } = selectBranchesForSynthesis(completedForSynthesis, this.maxCompletedBranches);
      this.emit({
        type: 'thought',
        content: `Starting synthesis attempt ${synthesisRetry + 1}/${this.maxSynthesisRetries + 1} with ${finalBranches.length} selected branch(es) out of ${completedForSynthesis.length} completed branch(es).`,
      });
      if (archivedBranchSummary) {
        this.emit({
          type: 'observation',
          content: `Archived ${completedForSynthesis.length - finalBranches.length} completed branch(es) into compact synthesis summary to stay within maxCompletedBranches=${this.maxCompletedBranches}.`,
        });
      }
      let synthesisResult: SynthesisResult;
      try {
        synthesisResult = await this.synthesize(
          userMessage, priorTranscript, finalBranches, synthesisRetry, archivedBranchSummary,
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
    archivedBranchSummary = '',
  ): Promise<SynthesisResult> {
    if (branches.length === 0) {
      return { done: true, answer: 'Maximum steps reached without a final answer.' };
    }

    if (this.synthesizeFinal) {
      const hookResult = await this.synthesizeFinal({
        userMessage,
        priorTranscript,
        archivedBranchSummary,
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
          planVersionSeen: branch.planVersionSeen,
          planExcerpt: branch.planExcerpt,
          alignmentChecks: branch.alignmentChecks,
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

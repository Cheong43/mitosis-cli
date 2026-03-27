import { AgentTraceEvent, PlannedBranch, PlannedToolCall, ToolObservation, TranscriptMessage } from './types.js';
import { Planner } from './SimplePlanner.js';
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
  priority: number;
  steps: number;
  inheritedMessageCount: number;
  transcript: TranscriptMessage[];
  savedNodeIds: string[];
  completionSummary?: string;
  finalAnswer?: string;
}

export interface BranchPlanInput {
  branch: AgentBranchState;
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
  }>;
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
  /** Optional final synthesis hook when multiple branches finish. */
  synthesizeFinal?: (input: BranchSynthesisInput) => Promise<string>;
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

function formatPlannerBranchDecisionMessage(branches: PlannedBranch[]): string {
  const lines = branches.map((branch) => {
    const why = branch.why?.trim() ? ` | why: ${branch.why.trim()}` : '';
    const priority = typeof branch.priority === 'number' ? ` | priority: ${branch.priority}` : '';
    return `- ${branch.label}: ${branch.goal}${why}${priority}`;
  });
  return ['PLANNER BRANCH DECISION:', ...lines].join('\n');
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
  private readonly synthesizeFinal?: (input: BranchSynthesisInput) => Promise<string>;
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
      priority: 1,
      steps: 0,
      inheritedMessageCount: 0,
      savedNodeIds: [],
      transcript: [
        ...priorTranscript,
        { role: 'user', content: userMessage },
      ],
    };

    const queue: AgentBranchState[] = [rootBranch];
    const completed: AgentBranchState[] = [];
    let lastTouchedBranch: AgentBranchState | null = rootBranch;
    let totalLoopSteps = 0;
    const totalLoopBudget = this.maxTotalSteps
      ?? Math.max(this.maxSteps, this.maxSteps * this.maxBranchWidth * (this.maxBranchDepth + 1));

    const traceMeta = (branch: AgentBranchState, extra: Record<string, unknown> = {}) => ({
      branchId: branch.id,
      parentBranchId: branch.parentId,
      branchLabel: branch.label,
      depth: branch.depth,
      step: branch.steps,
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

    const executeToolStep = async (branch: AgentBranchState, toolCalls: PlannedToolCall[]) => {
      let observations: ToolObservation[];
      if (canRunToolCallsConcurrently(toolCalls)) {
        observations = await Promise.all(
          toolCalls.map((toolCall) => executeSingleToolCall(branch, toolCall)),
        );
        if (shouldRuntimeTraceTool()) {
          observations.forEach((observation) => emitObservationTrace(branch, observation));
        }
      } else {
        observations = [];
        for (const toolCall of toolCalls) {
          const observation = await executeSingleToolCall(branch, toolCall);
          observations.push(observation);
          if (shouldRuntimeTraceTool()) {
            emitObservationTrace(branch, observation);
          }
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
            .map((o) => `TOOL OBSERVATION for ${o.toolName}:\n${o.result}`)
            .join('\n\n'),
        });
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

      return branches.map((child, index) => {
        const transcript: TranscriptMessage[] = [
          ...compressedParentTranscript,
          {
            role: 'assistant',
            content: formatPlannerBranchDecisionMessage(branches),
          },
          {
            role: 'user',
            content: `You are now working on child branch "${child.label}".\nGoal: ${child.goal}\nWhy: ${child.why || 'Distinct strategy'}\nIf this child goal can still be split into 2 or more genuinely independent evidence streams or workstreams, prefer planner_branch again before issuing direct work tools.\nDo not use multiple near-duplicate web searches as a substitute for branching.\nYou may use tool calls, branch further if the goal benefits from parallel exploration, or return a final answer. Avoid repeating sibling work.`,
          },
        ];

        return {
          id: `${branch.id}.${index + 1}`,
          parentId: branch.id,
          depth: branch.depth + 1,
          label: child.label,
          goal: child.goal,
          priority: Math.max(0.05, branch.priority * (child.priority ?? Math.max(0.2, 1 - index * 0.2))),
          steps: 0,
          inheritedMessageCount: transcript.length,
          savedNodeIds: branch.savedNodeIds.slice(),
          transcript,
        };
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
            if (this.finalizeBranch) {
              branch.finalAnswer = await this.finalizeBranch({ branch, reason });
            }
            branch.finalAnswer = branch.finalAnswer || `Branch ${branch.id} stopped: context window full.`;
            completed.push(branch);
            emitBranchTrace('observation', branch,
              `Branch completed: context exhausted (${(check.usageRatio * 100).toFixed(1)}%).`);
            return [];
          }
        } else {
          // Legacy mode (no model limit known): use transcript budget compression.
          if (branch.steps >= this.maxSteps || totalLoopSteps >= totalLoopBudget) {
            const reason = totalLoopSteps >= totalLoopBudget ? 'total budget reached' : 'step budget reached';
            if (this.finalizeBranch) {
              branch.finalAnswer = await this.finalizeBranch({ branch, reason });
            }
            branch.finalAnswer = branch.finalAnswer || `Branch ${branch.id} stopped because ${reason}.`;
            completed.push(branch);
            emitBranchTrace('observation', branch, `Branch completed after hitting ${reason}.`);
            return [];
          }
        }

        branch.steps += 1;
        totalLoopSteps += 1;
        lastTouchedBranch = branch;

        const agentStep = this.planBranch
          ? await this.planBranch({ branch })
          : await this.planner.plan(branch.transcript);
        if (agentStep.thought) {
          emitBranchTrace('thought', branch, agentStep.thought);
        }

        if (agentStep.kind === 'final') {
          const needsExplicitFinalize = agentStep.finalizationMode === 'planner_fallback';
          if (needsExplicitFinalize) {
            emitBranchTrace('thought', branch, `Planner fallback reply is not treated as a real final step; explicitly finalizing ${branch.id}.`);
            if (this.finalizeBranch) {
              branch.finalAnswer = await this.finalizeBranch({
                branch,
                reason: 'planner format fallback after schema repair retries',
              });
            }
            branch.finalAnswer = branch.finalAnswer || agentStep.content;
          } else {
            branch.finalAnswer = agentStep.content;
          }
          branch.completionSummary = agentStep.completionSummary ?? branch.completionSummary;
          completed.push(branch);
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
          if (this.finalizeBranch) {
            branch.finalAnswer = await this.finalizeBranch({ branch, reason });
          }
          branch.finalAnswer = branch.finalAnswer || `Branch ${branch.id} stopped because ${reason}.`;
          completed.push(branch);
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

        emitBranchTrace('action', branch, `Spawning ${childPlans.length} child branches.`, { childCount: childPlans.length });
        const childBranches = buildChildBranches(branch, childPlans);
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
    const canLaunchMoreBranches = () =>
      (this.branchConcurrency <= 0 || active.size < this.branchConcurrency)
      && (this.modelLimit > 0 ? true : totalLoopSteps < totalLoopBudget);

    const launchQueuedBranches = () => {
      queue.sort(prioritySort);
      while (queue.length > 0 && canLaunchMoreBranches()) {
        const branch = queue.shift()!;
        let task: Promise<void>;
        task = runBranch(branch)
          .then((childBranches) => {
            active.delete(task);
            if (canLaunchMoreBranches()) {
              childBranches.forEach((child) => queue.push(child));
            }
            launchQueuedBranches();
          })
          .catch((error: unknown) => {
            active.delete(task);
            emitBranchTrace('error', branch, error instanceof Error ? error.message : String(error));
            launchQueuedBranches();
          });
        active.add(task);
        emitBranchTrace('thought', branch, `[scheduler] Branch ${branch.id} started (RPM-governed queue, active=${active.size}).`);
      }
    };

    launchQueuedBranches();
    while (active.size > 0) {
      await Promise.race(active);
    }

    if (completed.length === 0 && lastTouchedBranch) {
      if (this.finalizeBranch) {
        emitBranchTrace('thought', lastTouchedBranch, `No branch reached a natural final answer; explicitly finalizing ${lastTouchedBranch.id}.`);
        lastTouchedBranch.finalAnswer = await this.finalizeBranch({
          branch: lastTouchedBranch,
          reason: 'no branch reached a natural final answer',
        });
        emitBranchTrace('observation', lastTouchedBranch, 'Branch finalized because no branch reached a natural final answer.');
      }
      lastTouchedBranch.finalAnswer = lastTouchedBranch.finalAnswer || 'No branch produced a final answer.';
      completed.push(lastTouchedBranch);
    }

    const finalBranches = completed
      .filter((branch) => typeof branch.finalAnswer === 'string' && branch.finalAnswer.length > 0);
    const synthesized = await this.synthesize(userMessage, priorTranscript, finalBranches);
    this.emit({ type: 'final', content: synthesized });
    return synthesized;
  }

  private async synthesize(
    userMessage: string,
    priorTranscript: TranscriptMessage[],
    branches: AgentBranchState[],
  ): Promise<string> {
    if (branches.length === 0) {
      return 'Maximum steps reached without a final answer.';
    }

    if (this.synthesizeFinal) {
      return this.synthesizeFinal({
        userMessage,
        priorTranscript,
        branches: branches.map((branch) => ({
          id: branch.id,
          label: branch.label,
          goal: branch.goal,
          savedNodeIds: branch.savedNodeIds,
          completionSummary: branch.completionSummary,
          finalAnswer: branch.finalAnswer ?? '',
        })),
      });
    }

    if (branches.length === 1) {
      return branches[0].finalAnswer ?? 'Maximum steps reached without a final answer.';
    }

    return [...branches]
      .map((branch) => branch.finalAnswer || '')
      .find((answer) => answer.length > 0)
      ?? 'Maximum steps reached without a final answer.';
  }

  private emit(event: AgentTraceEvent): void {
    try {
      this.onTrace(event);
    } catch {
      // Never let trace callbacks crash the loop.
    }
  }
}

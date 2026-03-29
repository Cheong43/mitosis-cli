/**
 * Shared types for the agent runtime layer.
 */

/** A single step in the agent's ReAct loop. */
export type AgentStepKind = 'tool' | 'branch' | 'final';

/** Structured outcome of a completed branch. */
export type BranchOutcome = 'success' | 'partial' | 'failed' | 'unknown';

/**
 * Retry-oriented interpretation of a branch result.
 *
 * This is intentionally finer-grained than `BranchOutcome`, so synthesis can
 * distinguish between:
 * - missing evidence that is worth one more focused remediation attempt
 * - external blocking conditions that should not trigger repeated retries
 * - exhausted search loops that should only continue if a genuinely new
 *   strategy is available
 */
export type BranchDisposition =
  | 'resolved'
  | 'missing_evidence'
  | 'blocked_external'
  | 'exhausted_search'
  | 'planner_error'
  | 'superseded'
  | 'unknown';

/** Structured remediation context attached to a retry branch. */
export interface BranchHandoff {
  canonicalTaskId: string;
  retryOfBranchId?: string;
  priorAttemptIds?: string[];
  disposition?: BranchDisposition;
  priorIssue?: string;
  priorResultSnippet?: string;
  knownGoodFacts?: string[];
  missingFields?: string[];
  exhaustedStrategies?: string[];
  blockedBy?: string[];
  mustNotRepeat?: string[];
}

/** Structured round-level context shared by every remediation child branch. */
export interface SynthesisSharedHandoff {
  retryIndex: number;
  successfulAttempts?: Array<{
    branchId: string;
    label: string;
    canonicalTaskId: string;
    summary: string;
  }>;
  unresolvedAttempts?: BranchHandoff[];
}

export type BranchKanbanStatus = 'queued' | 'active' | 'finalizing' | 'completed' | 'error';

export interface BranchKanbanCard {
  branchId: string;
  parentBranchId: string | null;
  label: string;
  goal: string;
  executionGroup?: number;
  dependsOn?: string[];
  depth: number;
  step: number;
  status: BranchKanbanStatus;
  outcome?: BranchOutcome;
  disposition?: BranchDisposition;
  summary?: string;
  blockers?: string[];
  artifacts?: string[];
  updatedAt: number;
}

export interface BranchKanbanSnapshot {
  updatedAt: number;
  summary: {
    total: number;
    queued: number;
    active: number;
    finalizing: number;
    completed: number;
    error: number;
  };
  cards: BranchKanbanCard[];
}

interface AgentStepBase {
  /** Optional planner thought for tracing/debugging. */
  thought?: string;
}

/** A tool call planned by the agent. */
export interface PlannedToolCall {
  name: string;
  arguments: Record<string, unknown>;
  /** Optional hint about the call's purpose. */
  goal?: string;
}

/** A child branch planned by the agent. */
export interface PlannedBranch {
  label: string;
  goal: string;
  why?: string;
  priority?: number;
  /**
   * Sibling branches in the same execution group may run in parallel.
   * Higher-numbered groups wait until all lower groups under the same parent
   * complete before they are eligible to start.
   */
  executionGroup?: number;
  /**
   * Optional sibling branch labels that must finish before this branch may
   * start. Labels must come from the same planner_branch call.
   */
  dependsOn?: string[];
  handoff?: BranchHandoff;
}

/** A finalized answer produced by the agent. */
export interface FinalAnswer extends AgentStepBase {
  kind: 'final';
  content: string;
  completionSummary?: string;
  /**
   * Structured outcome reported by the planner.  When the planner omits this
   * the runtime infers the value from context (e.g. forced finalization → 'partial').
   */
  outcome?: BranchOutcome;
  /** Human-readable explanation when outcome is 'partial' or 'failed'. */
  outcomeReason?: string;
  /** Retry-oriented disposition for synthesis/remediation. */
  disposition?: BranchDisposition;
  /**
   * `planner_fallback` means the planner never produced a usable branch-final
   * step, so the runtime should explicitly finalize the branch instead of
   * counting this as a natural completion.
   */
  finalizationMode?: 'natural' | 'planner_fallback';
}

/** A tool-execution step. */
export interface ToolStep extends AgentStepBase {
  kind: 'tool';
  toolCalls: PlannedToolCall[];
}

/** A branch-spawning step. */
export interface BranchStep extends AgentStepBase {
  kind: 'branch';
  branches: PlannedBranch[];
}

export type AgentStep = ToolStep | BranchStep | FinalAnswer;

/**
 * Result of the post-branch synthesis stage.
 *
 * - `done: true`  → final answer ready; the run ends.
 * - `done: false` → one or more branches need remediation; the runtime
 *   spawns new child branches and re-enters the scheduling loop.
 */
export type SynthesisResult =
  | { done: true; answer: string }
  | { done: false; branches: PlannedBranch[]; context: string; handoff?: SynthesisSharedHandoff };

/** Observation returned after executing a tool call. */
export interface ToolObservation {
  toolName: string;
  result: string;
  success: boolean;
}

export type TranscriptMessageContent = any;

/** A single turn in the agent's running transcript. */
export interface TranscriptMessage {
  role: 'system' | 'user' | 'assistant';
  content: TranscriptMessageContent;
}

/** Trace event emitted by the agent runtime for UI/logging. */
export interface AgentTraceEvent {
  type: 'thought' | 'action' | 'observation' | 'error' | 'final';
  content: string;
  metadata?: Record<string, unknown>;
}

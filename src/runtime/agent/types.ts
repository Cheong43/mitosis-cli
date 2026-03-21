/**
 * Shared types for the agent runtime layer.
 */

/** A single step in the agent's ReAct loop. */
export type AgentStepKind = 'tool' | 'branch' | 'final';

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
}

/** A finalized answer produced by the agent. */
export interface FinalAnswer extends AgentStepBase {
  kind: 'final';
  content: string;
  completionSummary?: string;
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

/** Observation returned after executing a tool call. */
export interface ToolObservation {
  toolName: string;
  result: string;
  success: boolean;
}

/** A single turn in the agent's running transcript. */
export interface TranscriptMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

/** Trace event emitted by the agent runtime for UI/logging. */
export interface AgentTraceEvent {
  type: 'thought' | 'action' | 'observation' | 'error' | 'final';
  content: string;
  metadata?: Record<string, unknown>;
}

// ─── Beam Search Types ───────────────────────────────────────────────────────

/** A single step recorded in a beam search path's history. */
export interface PathHistoryEntry {
  thought: string;
  action: PlannedToolCall;
  observation: ToolObservation;
}

/**
 * A candidate path in beam search.
 *
 * Each path records the full reasoning trajectory (transcript + structured
 * history) so that the scorer and pruner can evaluate quality without
 * re-running the LLM.
 */
export interface BeamPath {
  /** Unique path identifier (e.g. "P0", "P0.1"). */
  id: string;
  /** Parent path id, or null for the initial path. */
  parentId: string | null;
  /** Current depth (number of expand steps applied). */
  depth: number;
  /** Running transcript used as LLM context. */
  transcript: TranscriptMessage[];
  /** Structured history of (thought, action, observation) triples. */
  history: PathHistoryEntry[];
  /** Scorer-assigned value (higher is better). */
  score: number;
  /** Whether this path has produced a final answer. */
  isComplete: boolean;
  /** Final answer text, set when the path terminates. */
  finalAnswer?: string;
  /** Optional completion summary. */
  completionSummary?: string;
}

/**
 * Scoring interface for beam search paths.
 *
 * Implementations evaluate how promising a path is, returning a numeric
 * score (higher ⇒ better).  The scorer is the core heuristic that guides
 * beam pruning.
 */
export interface PathScorer {
  score(path: BeamPath): number;
}

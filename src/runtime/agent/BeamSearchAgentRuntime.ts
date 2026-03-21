import type {
  AgentTraceEvent,
  BeamPath,
  PathHistoryEntry,
  PathScorer,
  PlannedToolCall,
  TranscriptMessage,
  ToolObservation,
} from './types.js';
import type { Planner } from './SimplePlanner.js';
import type { ToolExecutionResult } from '../tools/types.js';
import { DefaultPathScorer } from './DefaultPathScorer.js';

// ─── Helper interfaces ──────────────────────────────────────────────────────

interface BeamToolRuntime {
  execute(toolName: string, args: Record<string, unknown>): Promise<ToolExecutionResult>;
  resetSession(): void;
}

// ─── Options ─────────────────────────────────────────────────────────────────

export interface BeamSearchAgentRuntimeOptions {
  planner: Planner;
  toolRuntime: BeamToolRuntime;
  /** Number of candidate paths to keep after each pruning step (k). */
  beamWidth?: number;
  /** Maximum expansion depth before forcing termination. */
  maxDepth?: number;
  /**
   * Number of candidate next-steps to generate per path in each expansion.
   * This maps to asking the LLM for multiple alternatives or calling it
   * multiple times with different temperatures.
   */
  expansionFactor?: number;
  /** Path scorer implementation.  Defaults to `DefaultPathScorer`. */
  scorer?: PathScorer;
  /** Trace callback invoked after each loop step. */
  onTrace?: (event: AgentTraceEvent) => void;
  /**
   * Optional hook: generate multiple (thought, toolCall) candidates for a
   * given transcript.  When provided it replaces the default
   * single-LLM-call-per-candidate approach.
   */
  generateCandidates?: (
    transcript: TranscriptMessage[],
    n: number,
  ) => Promise<Array<{ thought: string; toolCall: PlannedToolCall }>>;
  /** Optional hook: synthesise a single answer from the best path. */
  synthesizeFinal?: (path: BeamPath) => Promise<string>;
}

// ─── Runtime ─────────────────────────────────────────────────────────────────

/**
 * BeamSearchAgentRuntime combines Beam Search with the ReAct pattern.
 *
 * Instead of greedily following a single reasoning chain the runtime
 * maintains **k** candidate paths (the *beam*) and at every step:
 *
 *  1. **Expands** each active path into multiple candidate next-steps.
 *  2. **Scores** every candidate.
 *  3. **Prunes** down to the top-k candidates.
 *  4. Repeats until a path terminates or the depth limit is hit.
 *
 * This overcomes the "single greedy chain" limitation of vanilla ReAct
 * by exploring diverse strategies in parallel.
 */
export class BeamSearchAgentRuntime {
  private static readonly PARALLEL_SAFE_TOOL_NAMES = new Set(['read', 'search', 'web']);

  private readonly planner: Planner;
  private readonly toolRuntime: BeamToolRuntime;
  private readonly beamWidth: number;
  private readonly maxDepth: number;
  private readonly expansionFactor: number;
  private readonly scorer: PathScorer;
  private readonly onTrace: (event: AgentTraceEvent) => void;
  private readonly generateCandidates?: BeamSearchAgentRuntimeOptions['generateCandidates'];
  private readonly synthesizeFinal?: BeamSearchAgentRuntimeOptions['synthesizeFinal'];

  constructor(options: BeamSearchAgentRuntimeOptions) {
    this.planner = options.planner;
    this.toolRuntime = options.toolRuntime;
    this.beamWidth = options.beamWidth ?? 3;
    this.maxDepth = options.maxDepth ?? 5;
    this.expansionFactor = options.expansionFactor ?? 3;
    this.scorer = options.scorer ?? new DefaultPathScorer();
    this.onTrace = options.onTrace ?? (() => undefined);
    this.generateCandidates = options.generateCandidates;
    this.synthesizeFinal = options.synthesizeFinal;
  }

  // ── public entry ─────────────────────────────────────────────────────────

  /**
   * Run the beam-search ReAct loop.
   *
   * @param userMessage      The user's input.
   * @param priorTranscript  Previous conversation turns for context.
   * @returns The agent's final answer string.
   */
  async run(
    userMessage: string,
    priorTranscript: TranscriptMessage[] = [],
  ): Promise<string> {
    this.toolRuntime.resetSession();

    // Initialise the beam with a single seed path.
    const initialPath: BeamPath = {
      id: 'P0',
      parentId: null,
      depth: 0,
      transcript: [
        ...priorTranscript,
        { role: 'user', content: userMessage },
      ],
      history: [],
      score: 0,
      isComplete: false,
    };

    let beam: BeamPath[] = [initialPath];

    for (let step = 0; step < this.maxDepth; step++) {
      const candidates: BeamPath[] = [];

      // Expand each path in the current beam.
      for (const path of beam) {
        if (this.terminate(path)) {
          candidates.push(path);
          continue;
        }

        const expanded = await this.expand(path, step);
        candidates.push(...expanded);
      }

      // Early exit: all paths terminated.
      if (candidates.length > 0 && candidates.every((p) => this.terminate(p))) {
        return this.selectBest(candidates);
      }

      // Score & prune.
      beam = this.prune(candidates);

      this.emit({
        type: 'observation',
        content: `[beam] Step ${step + 1}/${this.maxDepth}: ${beam.length} paths in beam (${candidates.length} candidates evaluated).`,
        metadata: {
          step: step + 1,
          beamSize: beam.length,
          candidatesEvaluated: candidates.length,
          topScore: beam[0]?.score ?? 0,
        },
      });
    }

    // Depth limit reached – return the best path found.
    return this.selectBest(beam);
  }

  // ── expand ───────────────────────────────────────────────────────────────

  /**
   * Expand a single path into `expansionFactor` candidate next-paths.
   *
   * Each candidate consists of a (thought, action, observation) triple
   * appended to the parent path's history and transcript.
   */
  private async expand(path: BeamPath, step: number): Promise<BeamPath[]> {
    const candidates: Array<{ thought: string; toolCall: PlannedToolCall }> =
      this.generateCandidates
        ? await this.generateCandidates(path.transcript, this.expansionFactor)
        : await this.defaultGenerateCandidates(path.transcript, this.expansionFactor);

    // Execute each candidate's tool call and build the new path.
    const expanded: BeamPath[] = [];
    for (let i = 0; i < candidates.length; i++) {
      const { thought, toolCall } = candidates[i];

      this.emit({
        type: 'thought',
        content: thought,
        metadata: { pathId: path.id, candidateIndex: i, step },
      });

      this.emit({
        type: 'action',
        content: `Calling ${toolCall.name}${toolCall.goal ? ` — ${toolCall.goal}` : ''}`,
        metadata: { pathId: path.id, toolName: toolCall.name, args: toolCall.arguments },
      });

      const observation = await this.executeTool(toolCall);

      this.emit({
        type: 'observation',
        content: observation.result,
        metadata: { pathId: path.id, toolName: observation.toolName },
      });

      const historyEntry: PathHistoryEntry = { thought, action: toolCall, observation };

      const newTranscript: TranscriptMessage[] = [
        ...path.transcript,
        {
          role: 'assistant',
          content: JSON.stringify({ thought, tool_call: toolCall }),
        },
        {
          role: 'user',
          content: `TOOL OBSERVATION for ${observation.toolName}:\n${observation.result}`,
        },
      ];

      // Check if the planner considers this path finished.
      const { isComplete, finalAnswer, completionSummary } =
        await this.checkCompletion(newTranscript);

      const newPath: BeamPath = {
        id: `${path.id}.${i}`,
        parentId: path.id,
        depth: path.depth + 1,
        transcript: newTranscript,
        history: [...path.history, historyEntry],
        score: 0, // will be set by scorer
        isComplete,
        finalAnswer,
        completionSummary,
      };

      newPath.score = this.scorer.score(newPath);
      expanded.push(newPath);
    }

    return expanded;
  }

  // ── candidate generation ─────────────────────────────────────────────────

  /**
   * Default strategy: call the planner N times and collect tool-call steps.
   *
   * If the planner returns a `final` step on any call the result is marked
   * as complete and no tool call is generated for that candidate.
   */
  private async defaultGenerateCandidates(
    transcript: TranscriptMessage[],
    n: number,
  ): Promise<Array<{ thought: string; toolCall: PlannedToolCall }>> {
    const results: Array<{ thought: string; toolCall: PlannedToolCall }> = [];

    for (let i = 0; i < n; i++) {
      const step = await this.planner.plan(transcript);
      if (step.kind === 'tool' && step.toolCalls.length > 0) {
        results.push({
          thought: step.thought ?? '',
          toolCall: step.toolCalls[0],
        });
      } else if (step.kind === 'final') {
        // Build a pseudo tool-call so the expansion still produces a path
        // that will be immediately recognised as complete.
        results.push({
          thought: step.thought ?? step.content,
          toolCall: {
            name: '__final__',
            arguments: { answer: step.content },
            goal: 'Provide final answer',
          },
        });
      }
      // branch steps are ignored in beam search – they are replaced by the
      // beam mechanism itself.
    }

    return results;
  }

  // ── tool execution ───────────────────────────────────────────────────────

  private async executeTool(toolCall: PlannedToolCall): Promise<ToolObservation> {
    // Handle the synthetic __final__ marker.
    if (toolCall.name === '__final__') {
      return {
        toolName: '__final__',
        result: String(toolCall.arguments.answer ?? ''),
        success: true,
      };
    }

    const execResult = await this.toolRuntime.execute(toolCall.name, toolCall.arguments);
    return {
      toolName: toolCall.name,
      result: execResult.success
        ? JSON.stringify(execResult.result ?? '')
        : `ERROR: ${execResult.error ?? 'unknown error'}`,
      success: execResult.success,
    };
  }

  // ── completion check ─────────────────────────────────────────────────────

  /**
   * Ask the planner once more whether, given the new transcript, the task
   * is done.  This is a lightweight probe — if the planner returns `final`
   * the path is marked complete.
   */
  private async checkCompletion(
    transcript: TranscriptMessage[],
  ): Promise<{ isComplete: boolean; finalAnswer?: string; completionSummary?: string }> {
    try {
      const probe = await this.planner.plan(transcript);
      if (probe.kind === 'final') {
        return {
          isComplete: true,
          finalAnswer: probe.content,
          completionSummary: probe.completionSummary,
        };
      }
    } catch {
      // Non-fatal – treat as "not yet complete".
    }
    return { isComplete: false };
  }

  // ── scoring ──────────────────────────────────────────────────────────────

  // Scoring is delegated to `this.scorer` – see `PathScorer` interface.

  // ── pruning ──────────────────────────────────────────────────────────────

  /**
   * Prune the candidate set down to `beamWidth` paths.
   *
   * 1. Deduplicate by state hash (keep higher-scored duplicate).
   * 2. Ensure completed paths are always retained.
   * 3. Fill remaining slots by score.
   */
  private prune(candidates: BeamPath[]): BeamPath[] {
    // Deduplicate.
    const unique = new Map<string, BeamPath>();
    for (const p of candidates) {
      const key = this.stateHash(p);
      const existing = unique.get(key);
      if (!existing || p.score > existing.score) {
        unique.set(key, p);
      }
    }

    const deduped = [...unique.values()];
    deduped.sort((a, b) => b.score - a.score);

    const completed = deduped.filter((p) => p.isComplete);
    const active = deduped.filter((p) => !p.isComplete);

    // Always keep completed paths (up to beamWidth), then fill with active.
    const kept = completed.slice(0, this.beamWidth);
    const remaining = this.beamWidth - kept.length;
    kept.push(...active.slice(0, remaining));

    return kept;
  }

  // ── termination ──────────────────────────────────────────────────────────

  /**
   * A path terminates when it has a final answer **or** has reached the
   * maximum depth.
   */
  private terminate(path: BeamPath): boolean {
    return path.isComplete || path.depth >= this.maxDepth;
  }

  // ── selection ────────────────────────────────────────────────────────────

  private async selectBest(paths: BeamPath[]): Promise<string> {
    if (paths.length === 0) {
      return 'Maximum depth reached without a final answer.';
    }

    // Prefer completed paths; otherwise pick highest score.
    const sorted = [...paths].sort((a, b) => {
      if (a.isComplete !== b.isComplete) return a.isComplete ? -1 : 1;
      return b.score - a.score;
    });

    const best = sorted[0];

    if (this.synthesizeFinal) {
      const answer = await this.synthesizeFinal(best);
      this.emit({ type: 'final', content: answer });
      return answer;
    }

    const answer = best.finalAnswer ?? 'Maximum depth reached without a final answer.';
    this.emit({ type: 'final', content: answer });
    return answer;
  }

  // ── utilities ────────────────────────────────────────────────────────────

  /** Lightweight hash of a path's observable state for deduplication. */
  private stateHash(path: BeamPath): string {
    // Use the last few transcript entries as a fingerprint.
    const tail = path.transcript.slice(-4);
    return tail.map((m) => `${m.role}:${m.content}`).join('|');
  }

  private emit(event: AgentTraceEvent): void {
    try {
      this.onTrace(event);
    } catch {
      // Never let trace callbacks crash the loop.
    }
  }
}

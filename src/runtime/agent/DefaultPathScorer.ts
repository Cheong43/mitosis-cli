import type { BeamPath, PathScorer } from './types.js';

/**
 * Default heuristic scorer for beam search paths.
 *
 * Signals:
 *  1. **Tool success rate** – fraction of observations that succeeded.
 *  2. **Completion bonus** – completed paths get a bonus.
 *  3. **Depth penalty** – longer paths are mildly penalised to prefer concise
 *     solutions.
 *
 * All weights can be overridden via the constructor.
 */
export class DefaultPathScorer implements PathScorer {
  private readonly successWeight: number;
  private readonly completionBonus: number;
  private readonly depthPenalty: number;

  constructor(options: {
    successWeight?: number;
    completionBonus?: number;
    depthPenalty?: number;
  } = {}) {
    this.successWeight = options.successWeight ?? 0.6;
    this.completionBonus = options.completionBonus ?? 0.3;
    this.depthPenalty = options.depthPenalty ?? 0.01;
  }

  score(path: BeamPath): number {
    const entries = path.history;

    // 1. Tool success rate (0–1).
    const successRate =
      entries.length > 0
        ? entries.filter((e) => e.observation.success).length / entries.length
        : 0.5; // neutral when no history

    // 2. Completion bonus.
    const completion = path.isComplete ? this.completionBonus : 0;

    // 3. Depth penalty.
    const penalty = this.depthPenalty * path.depth;

    return successRate * this.successWeight + completion - penalty;
  }
}

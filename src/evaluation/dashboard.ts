/**
 * Metrics visualization and monitoring dashboard.
 *
 * Tracks metrics over time, generates reports, and alerts on regressions.
 */
import { MetricsCollector, type EvaluationMetrics } from './metrics.js';

export interface DashboardSnapshot {
  timestamp: string;
  metrics: EvaluationMetrics;
  alerts: string[];
}

export class Dashboard {
  private history: DashboardSnapshot[] = [];
  private alertCallbacks: Array<(alerts: string[]) => void> = [];

  constructor(private collector: MetricsCollector) {}

  /**
   * Take a snapshot of the current metrics and check for regressions.
   */
  snapshot(): DashboardSnapshot {
    const metrics = this.collector.export();
    const alerts = this.collector.checkRegressions();

    const snap: DashboardSnapshot = {
      timestamp: new Date().toISOString(),
      metrics,
      alerts,
    };

    this.history.push(snap);

    if (alerts.length > 0) {
      for (const cb of this.alertCallbacks) {
        cb(alerts);
      }
    }

    return snap;
  }

  /**
   * Register a callback to be invoked when regressions are detected.
   */
  onAlert(callback: (alerts: string[]) => void): void {
    this.alertCallbacks.push(callback);
  }

  /**
   * Get the full history of snapshots.
   */
  getHistory(): DashboardSnapshot[] {
    return [...this.history];
  }

  /**
   * Compare the latest snapshot against a baseline.
   */
  compareToBaseline(baseline: EvaluationMetrics): Record<string, { baseline: number; current: number; delta: number }> {
    const current = this.collector.export();
    const keys = Object.keys(baseline) as Array<keyof EvaluationMetrics>;

    const comparison: Record<string, { baseline: number; current: number; delta: number }> = {};
    for (const key of keys) {
      const bVal = baseline[key];
      const cVal = current[key];
      if (typeof bVal === 'number' && typeof cVal === 'number') {
        comparison[key] = {
          baseline: bVal,
          current: cVal,
          delta: cVal - bVal,
        };
      }
    }

    return comparison;
  }

  /**
   * Generate a human-readable report of the latest snapshot.
   */
  generateReport(): string {
    const snap = this.history.length > 0
      ? this.history[this.history.length - 1]
      : this.snapshot();

    const m = snap.metrics;
    const lines: string[] = [
      '═══ LLM Output Quality Report ═══',
      `Timestamp: ${snap.timestamp}`,
      '',
      '── Format Correctness ──',
      `  JSON Parse Success Rate:    ${(m.jsonParseSuccessRate * 100).toFixed(1)}%`,
      `  Schema Validation Rate:     ${(m.schemaValidationRate * 100).toFixed(1)}%`,
      `  Avg Repair Attempts:        ${m.repairAttemptsAvg.toFixed(2)}`,
      '',
      '── Tool Call Accuracy ──',
      `  Valid Tool Call Rate:        ${(m.validToolCallRate * 100).toFixed(1)}%`,
      `  Parallel Violations:        ${m.parallelToolCallViolations}`,
      `  Unknown Tool Call Rate:      ${(m.unknownToolCallRate * 100).toFixed(1)}%`,
      '',
      '── Context Efficiency ──',
      `  Avg Tokens per Step:         ${m.avgTokensPerStep.toFixed(0)}`,
      `  Compression Trigger Rate:   ${(m.compressionTriggerRate * 100).toFixed(1)}%`,
      `  Context Exhaustion Rate:    ${(m.contextExhaustionRate * 100).toFixed(1)}%`,
      '',
      '── Performance ──',
      `  Avg Latency:                ${m.avgLatencyMs.toFixed(0)} ms`,
      `  Total Cost:                 $${m.totalCost.toFixed(4)}`,
      `  Total Runs:                 ${m.totalRuns}`,
    ];

    if (snap.alerts.length > 0) {
      lines.push('');
      lines.push('── ALERTS ──');
      for (const alert of snap.alerts) {
        lines.push(`  ⚠ ${alert}`);
      }
    }

    lines.push('');
    lines.push('═══════════════════════════════');

    return lines.join('\n');
  }

  /**
   * Persist all history to a file.
   */
  async persistHistory(filepath: string): Promise<void> {
    const fs = await import('fs/promises');
    const dir = filepath.substring(0, filepath.lastIndexOf('/'));
    await fs.mkdir(dir, { recursive: true });
    await fs.writeFile(filepath, JSON.stringify(this.history, null, 2), 'utf-8');
  }
}

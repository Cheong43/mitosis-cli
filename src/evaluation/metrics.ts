/**
 * Evaluation metrics collection and tracking.
 *
 * Records LLM output quality metrics across format correctness,
 * tool call accuracy, hallucination detection, context efficiency,
 * and performance.
 */

export interface EvaluationMetrics {
  // Format correctness
  jsonParseSuccessRate: number;
  schemaValidationRate: number;
  repairAttemptsAvg: number;

  // Tool call accuracy
  validToolCallRate: number;
  parallelToolCallViolations: number;
  unknownToolCallRate: number;

  // Hallucination detection
  invalidFilePathRate: number;
  nonexistentToolRate: number;

  // Context efficiency
  avgTokensPerStep: number;
  compressionTriggerRate: number;
  contextExhaustionRate: number;

  // Performance
  avgLatencyMs: number;
  totalCost: number;
  totalRuns: number;
}

interface MetricSample {
  timestamp: number;
  category: string;
  key: string;
  value: number;
}

export class MetricsCollector {
  private samples: MetricSample[] = [];
  private counters = {
    jsonParseAttempts: 0,
    jsonParseSuccesses: 0,
    schemaValidationAttempts: 0,
    schemaValidationSuccesses: 0,
    totalRepairAttempts: 0,
    toolCallAttempts: 0,
    validToolCalls: 0,
    parallelViolations: 0,
    unknownToolCalls: 0,
    invalidFilePaths: 0,
    nonexistentTools: 0,
    totalTokens: 0,
    totalSteps: 0,
    compressionTriggers: 0,
    compressionChecks: 0,
    contextExhaustions: 0,
    totalLatencyMs: 0,
    totalCost: 0,
    totalRuns: 0,
  };

  recordJsonParse(success: boolean, attempts: number): void {
    this.counters.jsonParseAttempts++;
    if (success) this.counters.jsonParseSuccesses++;
    this.counters.totalRepairAttempts += attempts;
    this.addSample('format', 'json_parse', success ? 1 : 0);
  }

  recordSchemaValidation(success: boolean): void {
    this.counters.schemaValidationAttempts++;
    if (success) this.counters.schemaValidationSuccesses++;
    this.addSample('format', 'schema_validation', success ? 1 : 0);
  }

  recordToolCall(valid: boolean, toolName: string, isParallel: boolean): void {
    this.counters.toolCallAttempts++;
    if (valid) {
      this.counters.validToolCalls++;
    }
    if (isParallel) {
      this.counters.parallelViolations++;
    }
    this.addSample('tool', toolName, valid ? 1 : 0);
  }

  recordUnknownToolCall(toolName: string): void {
    this.counters.unknownToolCalls++;
    this.addSample('hallucination', 'unknown_tool', 1);
  }

  recordInvalidFilePath(path: string): void {
    this.counters.invalidFilePaths++;
    this.addSample('hallucination', 'invalid_path', 1);
  }

  recordContextUsage(tokens: number, compressed: boolean, exhausted: boolean): void {
    this.counters.totalTokens += tokens;
    this.counters.totalSteps++;
    this.counters.compressionChecks++;
    if (compressed) this.counters.compressionTriggers++;
    if (exhausted) this.counters.contextExhaustions++;
    this.addSample('context', 'tokens', tokens);
  }

  recordLatency(ms: number, cost: number): void {
    this.counters.totalLatencyMs += ms;
    this.counters.totalCost += cost;
    this.counters.totalRuns++;
    this.addSample('performance', 'latency_ms', ms);
  }

  export(): EvaluationMetrics {
    const safeDiv = (num: number, den: number) => den > 0 ? num / den : 0;

    return {
      jsonParseSuccessRate: safeDiv(this.counters.jsonParseSuccesses, this.counters.jsonParseAttempts),
      schemaValidationRate: safeDiv(this.counters.schemaValidationSuccesses, this.counters.schemaValidationAttempts),
      repairAttemptsAvg: safeDiv(this.counters.totalRepairAttempts, this.counters.jsonParseAttempts),
      validToolCallRate: safeDiv(this.counters.validToolCalls, this.counters.toolCallAttempts),
      parallelToolCallViolations: this.counters.parallelViolations,
      unknownToolCallRate: safeDiv(this.counters.unknownToolCalls, this.counters.toolCallAttempts),
      invalidFilePathRate: safeDiv(this.counters.invalidFilePaths, this.counters.totalSteps),
      nonexistentToolRate: safeDiv(this.counters.nonexistentTools, this.counters.toolCallAttempts),
      avgTokensPerStep: safeDiv(this.counters.totalTokens, this.counters.totalSteps),
      compressionTriggerRate: safeDiv(this.counters.compressionTriggers, this.counters.compressionChecks),
      contextExhaustionRate: safeDiv(this.counters.contextExhaustions, this.counters.compressionChecks),
      avgLatencyMs: safeDiv(this.counters.totalLatencyMs, this.counters.totalRuns),
      totalCost: this.counters.totalCost,
      totalRuns: this.counters.totalRuns,
    };
  }

  getSamples(): MetricSample[] {
    return [...this.samples];
  }

  reset(): void {
    this.samples = [];
    for (const key of Object.keys(this.counters) as Array<keyof typeof this.counters>) {
      this.counters[key] = 0;
    }
  }

  async persist(filepath: string): Promise<void> {
    const fs = await import('fs/promises');
    const metrics = this.export();
    const report = {
      timestamp: new Date().toISOString(),
      metrics,
      sampleCount: this.samples.length,
    };
    await fs.mkdir(filepath.substring(0, filepath.lastIndexOf('/')), { recursive: true });
    await fs.writeFile(filepath, JSON.stringify(report, null, 2), 'utf-8');
  }

  /**
   * Check if any metric has regressed below its threshold.
   */
  checkRegressions(thresholds?: Partial<EvaluationMetrics>): string[] {
    const defaults: Partial<EvaluationMetrics> = {
      jsonParseSuccessRate: 0.95,
      schemaValidationRate: 0.95,
      validToolCallRate: 0.90,
      contextExhaustionRate: 0.10,
    };
    const effective = { ...defaults, ...thresholds };
    const metrics = this.export();
    const alerts: string[] = [];

    if (effective.jsonParseSuccessRate !== undefined && metrics.jsonParseSuccessRate < effective.jsonParseSuccessRate) {
      alerts.push(`JSON parse success rate ${(metrics.jsonParseSuccessRate * 100).toFixed(1)}% < ${(effective.jsonParseSuccessRate * 100).toFixed(1)}% threshold`);
    }
    if (effective.schemaValidationRate !== undefined && metrics.schemaValidationRate < effective.schemaValidationRate) {
      alerts.push(`Schema validation rate ${(metrics.schemaValidationRate * 100).toFixed(1)}% < ${(effective.schemaValidationRate * 100).toFixed(1)}% threshold`);
    }
    if (effective.validToolCallRate !== undefined && metrics.validToolCallRate < effective.validToolCallRate) {
      alerts.push(`Valid tool call rate ${(metrics.validToolCallRate * 100).toFixed(1)}% < ${(effective.validToolCallRate * 100).toFixed(1)}% threshold`);
    }
    if (metrics.parallelToolCallViolations > 0) {
      alerts.push(`${metrics.parallelToolCallViolations} parallel tool call violation(s) detected`);
    }
    if (effective.contextExhaustionRate !== undefined && metrics.contextExhaustionRate > effective.contextExhaustionRate) {
      alerts.push(`Context exhaustion rate ${(metrics.contextExhaustionRate * 100).toFixed(1)}% > ${(effective.contextExhaustionRate * 100).toFixed(1)}% threshold`);
    }

    return alerts;
  }

  private addSample(category: string, key: string, value: number): void {
    this.samples.push({ timestamp: Date.now(), category, key, value });
  }
}

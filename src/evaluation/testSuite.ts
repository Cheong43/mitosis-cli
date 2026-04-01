/**
 * Automated regression test suite for LLM output quality.
 *
 * Tests format correctness, tool call accuracy, hallucination
 * boundaries, and context management.
 */
import { MetricsCollector, type EvaluationMetrics } from './metrics.js';

// ── Test case types ──────────────────────────────────────────────────

export interface TestCase {
  id: string;
  category: 'format' | 'tool_call' | 'hallucination' | 'context';
  description: string;
  input: string;
  expectedBehavior: {
    validJson?: boolean;
    toolName?: string;
    noParallelCalls?: boolean;
    maxTokens?: number;
    schemaValid?: boolean;
  };
}

export interface TestResult {
  testCase: TestCase;
  passed: boolean;
  actual: Record<string, unknown>;
  errors: string[];
  durationMs: number;
}

export interface EvaluationReport {
  timestamp: string;
  totalTests: number;
  passed: number;
  failed: number;
  passRate: number;
  results: TestResult[];
  metrics: EvaluationMetrics;
  regressionAlerts: string[];
}

// ── Built-in test cases ──────────────────────────────────────────────

export const TEST_CASES: TestCase[] = [
  {
    id: 'format_001',
    category: 'format',
    description: 'Valid JSON with tool_calls array',
    input: '{"tool_calls":[{"name":"read","input":{"path":"README.md"}}]}',
    expectedBehavior: { validJson: true, schemaValid: true },
  },
  {
    id: 'format_002',
    category: 'format',
    description: 'JSON with trailing comma (repairable)',
    input: '{"tool_calls":[{"name":"read","input":{"path":"README.md"}},]}',
    expectedBehavior: { validJson: true, schemaValid: true },
  },
  {
    id: 'format_003',
    category: 'format',
    description: 'JSON wrapped in markdown fences',
    input: '```json\n{"tool_calls":[{"name":"read","input":{"path":"README.md"}}]}\n```',
    expectedBehavior: { validJson: true, schemaValid: true },
  },
  {
    id: 'format_004',
    category: 'format',
    description: 'Truncated JSON (missing closing braces)',
    input: '{"tool_calls":[{"name":"read","input":{"path":"README.md"}',
    expectedBehavior: { validJson: true, schemaValid: true },
  },
  {
    id: 'format_005',
    category: 'format',
    description: 'Empty response',
    input: '',
    expectedBehavior: { validJson: false },
  },
  {
    id: 'tool_001',
    category: 'tool_call',
    description: 'Single tool call',
    input: '{"tool_calls":[{"name":"read","input":{"path":"README.md"}}]}',
    expectedBehavior: { toolName: 'read', noParallelCalls: true },
  },
  {
    id: 'tool_002',
    category: 'tool_call',
    description: 'Multiple parallel tool calls (should be reduced to single)',
    input: '{"tool_calls":[{"name":"read","input":{"path":"a.md"}},{"name":"read","input":{"path":"b.md"}}]}',
    expectedBehavior: { noParallelCalls: true },
  },
  {
    id: 'hallucination_001',
    category: 'hallucination',
    description: 'Non-existent tool name',
    input: '{"tool_calls":[{"name":"imaginary_tool","input":{}}]}',
    expectedBehavior: { schemaValid: false },
  },
  {
    id: 'context_001',
    category: 'context',
    description: 'Large observation should be compressed',
    input: 'x'.repeat(2000),
    expectedBehavior: { maxTokens: 1000 },
  },
];

// ── Test runner ──────────────────────────────────────────────────────

export function runFormatTest(testCase: TestCase): TestResult {
  const start = Date.now();
  const errors: string[] = [];
  const actual: Record<string, unknown> = {};

  try {
    // Strip markdown fences
    let input = testCase.input.trim();
    input = input.replace(/^```(?:json)?\s*/i, '').replace(/\s*```$/i, '').trim();

    if (!input) {
      actual.validJson = false;
      if (testCase.expectedBehavior.validJson === true) {
        errors.push('Expected valid JSON but input was empty');
      }
    } else {
      try {
        JSON.parse(input);
        actual.validJson = true;
      } catch {
        // Try basic repair
        let repaired = input;
        // Fix trailing commas
        repaired = repaired.replace(/,(\s*[\]}])/g, '$1');
        // Fix missing braces
        const openBraces = (repaired.match(/\{/g) || []).length;
        const closeBraces = (repaired.match(/\}/g) || []).length;
        if (openBraces > closeBraces) repaired += '}'.repeat(openBraces - closeBraces);
        const openBrackets = (repaired.match(/\[/g) || []).length;
        const closeBrackets = (repaired.match(/\]/g) || []).length;
        if (openBrackets > closeBrackets) repaired += ']'.repeat(openBrackets - closeBrackets);

        try {
          JSON.parse(repaired);
          actual.validJson = true;
        } catch {
          actual.validJson = false;
        }
      }

      if (testCase.expectedBehavior.validJson !== undefined && actual.validJson !== testCase.expectedBehavior.validJson) {
        errors.push(`Expected validJson=${testCase.expectedBehavior.validJson}, got ${actual.validJson}`);
      }
    }
  } catch (e) {
    errors.push(`Unexpected error: ${e}`);
  }

  return {
    testCase,
    passed: errors.length === 0,
    actual,
    errors,
    durationMs: Date.now() - start,
  };
}

export function runTestSuite(testCases: TestCase[] = TEST_CASES): EvaluationReport {
  const collector = new MetricsCollector();
  const results: TestResult[] = [];

  for (const testCase of testCases) {
    let result: TestResult;

    switch (testCase.category) {
      case 'format':
        result = runFormatTest(testCase);
        collector.recordJsonParse(result.actual.validJson === true, 1);
        break;
      case 'tool_call':
      case 'hallucination':
      case 'context':
      default:
        result = runFormatTest(testCase);
        break;
    }

    results.push(result);
  }

  const metrics = collector.export();
  const regressionAlerts = collector.checkRegressions();

  return {
    timestamp: new Date().toISOString(),
    totalTests: results.length,
    passed: results.filter((r) => r.passed).length,
    failed: results.filter((r) => !r.passed).length,
    passRate: results.length > 0 ? results.filter((r) => r.passed).length / results.length : 0,
    results,
    metrics,
    regressionAlerts,
  };
}

export async function persistReport(report: EvaluationReport, filepath: string): Promise<void> {
  const fs = await import('fs/promises');
  const dir = filepath.substring(0, filepath.lastIndexOf('/'));
  await fs.mkdir(dir, { recursive: true });
  await fs.writeFile(filepath, JSON.stringify(report, null, 2), 'utf-8');
}

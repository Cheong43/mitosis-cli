/**
 * Multi-stage JSON repair pipeline with structured error feedback.
 *
 * Centralizes JSON syntax repair, schema validation, and semantic checks
 * into a single pipeline whose error output is LLM-friendly.
 */
import { z } from 'zod';
import { fixArrayFormatting, fixUnterminatedStrings } from './llm.js';

// ── Public types ──────────────────────────────────────────────────────

export interface RepairError {
  path: string;
  message: string;
  severity: 'error' | 'warning';
}

export interface RepairResult<T> {
  success: boolean;
  data?: T;
  errors: RepairError[];
  repairAttempts: number;
}

export type SemanticValidator<T> = (data: T) => RepairError[];

// ── Pipeline ──────────────────────────────────────────────────────────

/**
 * Repair raw text, validate against a Zod schema, and run optional
 * semantic validators.  Returns a structured result that can feed back
 * into an LLM retry prompt.
 */
export function repairAndValidate<T>(
  rawText: string,
  schema: z.ZodSchema<T>,
  options?: {
    maxAttempts?: number;
    semanticValidators?: SemanticValidator<T>[];
  },
): RepairResult<T> {
  const maxAttempts = options?.maxAttempts ?? 3;
  const errors: RepairError[] = [];
  let repairAttempts = 0;

  // Stage 1 – Syntax repair
  let json = rawText.trim();
  if (!json) {
    return { success: false, errors: [{ path: '$', message: 'Empty response body', severity: 'error' }], repairAttempts: 0 };
  }

  // Strip <think> blocks
  json = json.replace(/<think>[\s\S]*?<\/think>/gi, '').trim();

  // Strip markdown fences
  json = json.replace(/^```(?:json)?\s*/i, '').replace(/\s*```$/i, '').trim();

  // Try progressive repair levels
  const repairStages: Array<(input: string) => string> = [
    (s) => s, // Raw parse
    (s) => fixArrayFormatting(s),
    (s) => fixUnterminatedStrings(fixArrayFormatting(s)),
    (s) => extractAndRepair(s),
  ];

  let parsed: unknown = undefined;
  for (const stage of repairStages.slice(0, maxAttempts + 1)) {
    repairAttempts++;
    try {
      parsed = JSON.parse(stage(json));
      break;
    } catch {
      // Continue to next repair stage
    }
  }

  if (parsed === undefined) {
    errors.push({
      path: '$',
      message: 'Failed to parse JSON after all repair attempts',
      severity: 'error',
    });
    return { success: false, errors, repairAttempts };
  }

  // Stage 2 – Schema validation
  const schemaResult = schema.safeParse(parsed);
  if (!schemaResult.success) {
    for (const issue of schemaResult.error.issues) {
      errors.push({
        path: issue.path.length > 0 ? issue.path.join('.') : '$',
        message: issue.message,
        severity: 'error',
      });
    }
    return { success: false, data: undefined, errors, repairAttempts };
  }

  // Stage 3 – Semantic validation
  const semanticErrors: RepairError[] = [];
  if (options?.semanticValidators) {
    for (const validator of options.semanticValidators) {
      semanticErrors.push(...validator(schemaResult.data));
    }
  }

  const hasFatalErrors = semanticErrors.some((e) => e.severity === 'error');
  if (hasFatalErrors) {
    return { success: false, data: schemaResult.data, errors: [...errors, ...semanticErrors], repairAttempts };
  }

  return {
    success: true,
    data: schemaResult.data,
    errors: [...errors, ...semanticErrors], // May include warnings
    repairAttempts,
  };
}

// ── LLM-friendly error formatting ────────────────────────────────────

/**
 * Convert repair errors into a compact message suitable for injecting
 * into an LLM retry prompt.
 */
export function buildStructuredErrorFeedback(errors: RepairError[]): string {
  if (errors.length === 0) return '';

  const errorLines = errors
    .filter((e) => e.severity === 'error')
    .map((e) => `  • ${e.path}: ${e.message}`);

  const warningLines = errors
    .filter((e) => e.severity === 'warning')
    .map((e) => `  ⚠ ${e.path}: ${e.message}`);

  const sections: string[] = [];
  if (errorLines.length > 0) {
    sections.push('Validation errors:', ...errorLines);
  }
  if (warningLines.length > 0) {
    sections.push('Warnings:', ...warningLines);
  }
  sections.push('Fix these specific issues in your response.');

  return sections.join('\n');
}

// ── Internal helpers ─────────────────────────────────────────────────

/**
 * Extract the first JSON object from a larger string and attempt deep
 * repair including fixing trailing commas and unterminated strings.
 */
function extractAndRepair(text: string): string {
  const firstBrace = text.indexOf('{');
  const lastBrace = text.lastIndexOf('}');

  if (firstBrace < 0) {
    throw new SyntaxError('No JSON object found');
  }

  let extracted = lastBrace > firstBrace
    ? text.slice(firstBrace, lastBrace + 1)
    : text.slice(firstBrace);

  // Apply all repair strategies
  extracted = fixArrayFormatting(extracted);
  extracted = fixUnterminatedStrings(extracted);

  // Remove trailing commas before closing braces/brackets
  extracted = extracted.replace(/,\s*([\]}])/g, '$1');

  // Remove single-line comments (// ...)
  extracted = extracted.replace(/\/\/[^\n]*/g, '');

  return extracted;
}

// ── Semantic validators (reusable) ───────────────────────────────────

/** Ensure all branch labels in a tool-call plan are unique. */
export function validateUniqueBranchLabels(data: { branches?: Array<{ label?: string }> }): RepairError[] {
  if (!data.branches || !Array.isArray(data.branches)) return [];
  const seen = new Set<string>();
  const errors: RepairError[] = [];
  for (let i = 0; i < data.branches.length; i++) {
    const label = data.branches[i]?.label;
    if (label && seen.has(label)) {
      errors.push({
        path: `branches[${i}].label`,
        message: `Duplicate branch label "${label}"`,
        severity: 'error',
      });
    }
    if (label) seen.add(label);
  }
  return errors;
}

/** Ensure tool_calls array is non-empty. */
export function validateNonEmptyToolCalls(data: { tool_calls?: unknown[] }): RepairError[] {
  if (!data.tool_calls || data.tool_calls.length === 0) {
    return [{
      path: 'tool_calls',
      message: 'At least one tool call is required',
      severity: 'error',
    }];
  }
  return [];
}

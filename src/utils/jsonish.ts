import JSON5 from 'json5';

function clip(text: string, maxChars = 240): string {
  const normalized = text.replace(/\s+/g, ' ').trim();
  if (normalized.length <= maxChars) {
    return normalized;
  }
  return `${normalized.slice(0, Math.max(0, maxChars - 3))}...`;
}

function stripFences(raw: string): string {
  return raw
    .replace(/^```json\s*/i, '')
    .replace(/^```\s*/i, '')
    .replace(/\s*```\s*$/i, '')
    .replace(/^`+|`+$/g, '')
    .trim();
}

function normalizeQuotes(raw: string): string {
  return raw
    .replace(/[\u201C\u201D]/g, '"')
    .replace(/[\u2018\u2019]/g, "'")
    .trim();
}

function extractBalancedSegmentAt(input: string, start: number, openChar: '{' | '[', closeChar: '}' | ']'): string | null {
  if (start < 0 || input[start] !== openChar) {
    return null;
  }
  let depth = 0;
  let inString = false;
  let stringQuote = '';
  let escaped = false;
  for (let i = start; i < input.length; i += 1) {
    const ch = input[i];
    if (inString) {
      if (escaped) {
        escaped = false;
        continue;
      }
      if (ch === '\\') {
        escaped = true;
        continue;
      }
      if (ch === stringQuote) {
        inString = false;
        stringQuote = '';
      }
      continue;
    }
    if (ch === '"' || ch === '\'') {
      inString = true;
      stringQuote = ch;
      continue;
    }
    if (ch === openChar) {
      depth += 1;
      continue;
    }
    if (ch === closeChar) {
      depth -= 1;
      if (depth === 0) {
        return input.slice(start, i + 1);
      }
    }
  }
  return null;
}

function collectBalancedSegments(input: string): string[] {
  const segments: string[] = [];
  const seen = new Set<string>();
  for (let i = 0; i < input.length; i += 1) {
    const ch = input[i];
    if (ch !== '{' && ch !== '[') {
      continue;
    }
    const segment = ch === '{'
      ? extractBalancedSegmentAt(input, i, '{', '}')
      : extractBalancedSegmentAt(input, i, '[', ']');
    if (!segment) {
      continue;
    }
    const normalized = normalizeQuotes(segment);
    if (!seen.has(normalized)) {
      seen.add(normalized);
      segments.push(normalized);
    }
  }
  return segments;
}

function collectJsonishCandidates(raw: string): string[] {
  const stripped = normalizeQuotes(stripFences(raw).replace(/^\uFEFF/, '').trim());
  if (!stripped) {
    return [''];
  }

  const candidates: string[] = [];
  const seen = new Set<string>();
  const push = (value: string) => {
    const normalized = normalizeQuotes(value);
    if (!seen.has(normalized)) {
      seen.add(normalized);
      candidates.push(normalized);
    }
  };

  push(stripped);
  for (const segment of collectBalancedSegments(stripped)) {
    push(segment);
  }

  return candidates;
}

export function extractJsonishCandidates(raw: string): string[] {
  return collectJsonishCandidates(raw);
}

export function extractJsonishText(raw: string): string {
  return collectJsonishCandidates(raw)[0] || '';
}

export function parseJsonish(raw: string): unknown {
  const candidates = collectJsonishCandidates(raw);
  const errors: string[] = [];

  for (const candidate of candidates) {
    try {
      return JSON.parse(candidate);
    } catch (jsonError: any) {
      try {
        return JSON5.parse(candidate);
      } catch (json5Error: any) {
        const detail = json5Error?.message || jsonError?.message || 'unknown parse error';
        errors.push(`${detail} :: ${clip(candidate, 120)}`);
      }
    }
  }

  throw new Error(`Failed to parse model JSON output. Attempts: ${errors.slice(0, 3).join(' | ')}. Raw preview: ${clip(normalizeQuotes(raw), 240)}`);
}
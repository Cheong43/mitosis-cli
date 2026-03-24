import test from 'node:test';
import assert from 'node:assert/strict';
import {
  estimateTokens,
  estimateTranscriptTokens,
  getModelContextLimit,
  computeContextBudget,
  compressTranscript,
  getCompressionLevel,
} from './contextBudget.js';

// ── estimateTokens ─────────────────────────────────────────────────────────

test('estimateTokens: empty string returns 0', () => {
  assert.equal(estimateTokens(''), 0);
});

test('estimateTokens: pure Latin text', () => {
  const text = 'Hello world, this is a test.';
  const tokens = estimateTokens(text);
  // ~27 chars of Latin → ~7 tokens (0.25 per char)
  assert.ok(tokens >= 5 && tokens <= 15, `Expected 5-15 tokens, got ${tokens}`);
});

test('estimateTokens: CJK text yields higher token count per char', () => {
  const latin = 'abcdefghijklmnop'; // 16 chars
  const cjk = '你好世界这是一个测试用例共十六'; // 14 CJK chars
  const latinTokens = estimateTokens(latin);
  const cjkTokens = estimateTokens(cjk);
  assert.ok(cjkTokens > latinTokens, `CJK tokens (${cjkTokens}) should exceed Latin tokens (${latinTokens})`);
});

test('estimateTokens: mixed CJK and Latin', () => {
  const mixed = 'Hello你好World世界';
  const tokens = estimateTokens(mixed);
  assert.ok(tokens > 0, 'Should produce positive token count');
});

// ── estimateTranscriptTokens ───────────────────────────────────────────────

test('estimateTranscriptTokens: includes message overhead', () => {
  const messages = [
    { role: 'user', content: 'Hi' },
    { role: 'assistant', content: 'Hello' },
  ];
  const tokens = estimateTranscriptTokens(messages);
  // Each message: content tokens + 4 overhead
  assert.ok(tokens >= 8, `Expected at least 8 tokens (overhead), got ${tokens}`);
});

// ── getModelContextLimit ───────────────────────────────────────────────────

test('getModelContextLimit: exact match', () => {
  assert.equal(getModelContextLimit('gpt-4o'), 128_000);
});

test('getModelContextLimit: prefix match (versioned model)', () => {
  assert.equal(getModelContextLimit('gpt-4o-2024-08-06'), 128_000);
});

test('getModelContextLimit: claude models', () => {
  assert.equal(getModelContextLimit('claude-3-7-sonnet'), 200_000);
});

test('getModelContextLimit: unknown model returns fallback', () => {
  assert.equal(getModelContextLimit('my-custom-model'), 128_000);
  assert.equal(getModelContextLimit('my-custom-model', 32_000), 32_000);
});

test('getModelContextLimit: respects MODEL_CONTEXT_LIMIT env override', () => {
  const orig = process.env.MODEL_CONTEXT_LIMIT;
  try {
    process.env.MODEL_CONTEXT_LIMIT = '16000';
    assert.equal(getModelContextLimit('gpt-4o'), 16_000);
  } finally {
    if (orig === undefined) {
      delete process.env.MODEL_CONTEXT_LIMIT;
    } else {
      process.env.MODEL_CONTEXT_LIMIT = orig;
    }
  }
});

// ── computeContextBudget ───────────────────────────────────────────────────

test('computeContextBudget: basic computation for gpt-4o', () => {
  const result = computeContextBudget({
    model: 'gpt-4o',
    systemPromptTokens: 3000,
    conversationContextTokens: 1000,
    memoryContextTokens: 500,
  });

  assert.equal(result.modelLimit, 128_000);
  // committed = 3000 + 1000 + 500 + 4096 = 8596
  assert.equal(result.committedTokens, 8596);
  // residual = 128000 - 8596 = 119404
  assert.equal(result.residualBudget, 119_404);

  // With ~119k residual tokens:
  assert.ok(result.maxSteps >= 2 && result.maxSteps <= 20, `maxSteps should be 2-20, got ${result.maxSteps}`);
  assert.ok(result.maxBranchDepth >= 2, `large budget should allow depth >= 2, got ${result.maxBranchDepth}`);
  assert.ok(result.maxBranchWidth >= 3, `large budget should allow width >= 3, got ${result.maxBranchWidth}`);
  assert.ok(result.transcriptBudgetChars >= 4000, `transcriptBudgetChars should be >= 4000, got ${result.transcriptBudgetChars}`);
  assert.ok(result.beamMaxDepth >= 2, `beamMaxDepth should be >= 2, got ${result.beamMaxDepth}`);
  assert.ok(result.beamWidth >= 2, `beamWidth should be >= 2, got ${result.beamWidth}`);
});

test('computeContextBudget: small model context constrains parameters', () => {
  const result = computeContextBudget({
    model: 'gpt-4',
    systemPromptTokens: 3000,
    conversationContextTokens: 2000,
    memoryContextTokens: 1000,
  });

  assert.equal(result.modelLimit, 8_192);
  // committed = 3000 + 2000 + 1000 + 4096 = 10096 > 8192 → residual = 0
  assert.equal(result.residualBudget, 0);
  // With 0 residual: all params should be at minimum
  assert.equal(result.maxSteps, 2);
  assert.equal(result.maxBranchDepth, 0);
  assert.equal(result.maxBranchWidth, 2);
  assert.equal(result.transcriptBudgetChars, 4000);
  assert.equal(result.beamMaxDepth, 2);
  assert.equal(result.beamWidth, 2);
});

test('computeContextBudget: medium residual gives moderate params', () => {
  // Simulate: 128k model, 90k committed → ~38k residual
  const result = computeContextBudget({
    model: 'gpt-4o',
    systemPromptTokens: 40_000,
    conversationContextTokens: 30_000,
    memoryContextTokens: 16_000,
    safetyMargin: 4000,
  });

  // committed = 40000 + 30000 + 16000 + 4000 = 90000
  // residual = 38000
  assert.equal(result.residualBudget, 38_000);
  assert.ok(result.maxBranchDepth <= 2, `38k residual should give depth <= 2, got ${result.maxBranchDepth}`);
  assert.ok(result.maxSteps >= 2 && result.maxSteps <= 20, `maxSteps in valid range, got ${result.maxSteps}`);
});

test('computeContextBudget: very large model (gemini) gives generous params', () => {
  const result = computeContextBudget({
    model: 'gemini-2.0-flash',
    systemPromptTokens: 3000,
    conversationContextTokens: 1000,
    memoryContextTokens: 500,
  });

  assert.equal(result.modelLimit, 1_000_000);
  assert.ok(result.residualBudget > 900_000);
  assert.ok(result.maxSteps === 20, `1M context should max out steps at 20, got ${result.maxSteps}`);
  assert.ok(result.maxBranchDepth === 3, `1M context should allow depth=3, got ${result.maxBranchDepth}`);
});

// ── getCompressionLevel ────────────────────────────────────────────────────

test('getCompressionLevel: low usage → none', () => {
  assert.equal(getCompressionLevel(5000, 20_000), 'none');
});

test('getCompressionLevel: 60% usage → soft', () => {
  assert.equal(getCompressionLevel(12_000, 20_000), 'soft');
});

test('getCompressionLevel: 80% usage → medium', () => {
  assert.equal(getCompressionLevel(16_000, 20_000), 'medium');
});

test('getCompressionLevel: 95% usage → hard', () => {
  assert.equal(getCompressionLevel(19_000, 20_000), 'hard');
});

test('getCompressionLevel: zero budget → hard', () => {
  assert.equal(getCompressionLevel(100, 0), 'hard');
});

// ── compressTranscript ─────────────────────────────────────────────────────

test('compressTranscript: short transcript returned as-is', () => {
  const messages = [
    { role: 'user', content: 'Hello' },
    { role: 'assistant', content: 'Hi there' },
  ];
  const result = compressTranscript(messages, 10_000);
  assert.equal(result.length, 2);
  assert.equal(result[0].content, 'Hello');
  assert.equal(result[1].content, 'Hi there');
});

test('compressTranscript: soft compression truncates long observations', () => {
  const longObs = 'TOOL OBSERVATION for search:\n' + 'x'.repeat(2000);
  const messages = [
    { role: 'user', content: 'Find files' },
    { role: 'assistant', content: '{"kind":"tool"}' },
    { role: 'user', content: longObs },
    { role: 'assistant', content: 'Found results' },
  ];
  // Budget smaller than total but big enough to trigger soft
  const totalLen = messages.reduce((s, m) => s + m.content.length, 0);
  const result = compressTranscript(messages, Math.floor(totalLen * 0.8), 'soft');
  // The observation should be truncated
  const obsMessage = result.find((m) => m.content.includes('TOOL OBSERVATION'));
  if (obsMessage) {
    assert.ok(obsMessage.content.includes('[observation truncated]'), 'Should contain truncation marker');
    assert.ok(obsMessage.content.length < longObs.length, 'Should be shorter than original');
  }
});

test('compressTranscript: medium compression replaces observations with summaries', () => {
  const longObs = 'TOOL OBSERVATION for read:\n' + 'line\n'.repeat(500);
  const messages = [
    { role: 'user', content: 'Read the file' },
    { role: 'assistant', content: '{"kind":"tool"}' },
    { role: 'user', content: longObs },
    { role: 'assistant', content: 'Here is the content' },
  ];
  const result = compressTranscript(messages, 500, 'medium');
  const obsMessage = result.find((m) => m.content.includes('TOOL OBSERVATION'));
  if (obsMessage) {
    assert.ok(obsMessage.content.includes('chars compressed'), 'Should contain compression marker');
  }
});

test('compressTranscript: hard compression keeps first + tail', () => {
  const messages: Array<{ role: string; content: string }> = [
    { role: 'user', content: 'Original user request with lots of detail' },
  ];
  for (let i = 0; i < 20; i++) {
    messages.push({ role: 'assistant', content: `Step ${i} thought: ${'x'.repeat(100)}` });
    messages.push({ role: 'user', content: `Step ${i} observation: ${'y'.repeat(100)}` });
  }
  const result = compressTranscript(messages, 600, 'hard');
  // First message should always be present
  assert.equal(result[0].content, messages[0].content);
  // Should be much shorter than original
  assert.ok(result.length < messages.length, `Hard compression should reduce message count from ${messages.length} to less`);
});

test('compressTranscript: single message returned as-is', () => {
  const messages = [{ role: 'user', content: 'Just one message' }];
  const result = compressTranscript(messages, 10);
  assert.equal(result.length, 1);
  assert.equal(result[0].content, 'Just one message');
});

import test from 'node:test';
import assert from 'node:assert/strict';
import {
  SessionCompressor,
  isRunExhausted,
  compressExhaustedRun,
  type TraceEventLike,
} from './sessionCompressor.js';

// ── isRunExhausted ─────────────────────────────────────────────────────────

test('isRunExhausted: detects "budget reached" in traces', () => {
  const traces: TraceEventLike[] = [
    { type: 'thought', content: 'Planning next step...' },
    { type: 'observation', content: 'Branch completed after hitting step budget reached.' },
  ];
  assert.ok(isRunExhausted(traces, 'Here is my answer'));
});

test('isRunExhausted: detects "total budget reached" in traces', () => {
  const traces: TraceEventLike[] = [
    { type: 'observation', content: 'Branch completed after hitting total budget reached.' },
  ];
  assert.ok(isRunExhausted(traces, 'partial result'));
});

test('isRunExhausted: detects "Forcing finalization" in traces', () => {
  const traces: TraceEventLike[] = [
    { type: 'thought', content: 'Forcing finalization for B0: step budget reached' },
  ];
  assert.ok(isRunExhausted(traces, 'forced answer'));
});

test('isRunExhausted: detects exhaustion in final answer', () => {
  const traces: TraceEventLike[] = [
    { type: 'thought', content: 'normal thought' },
  ];
  assert.ok(isRunExhausted(traces, 'Branch B0 stopped because step budget reached.'));
});

test('isRunExhausted: detects beam search depth exhaustion', () => {
  const traces: TraceEventLike[] = [
    { type: 'observation', content: 'Maximum depth reached without a final answer.' },
  ];
  assert.ok(isRunExhausted(traces, 'Maximum depth reached without a final answer.'));
});

test('isRunExhausted: returns false for normal completion', () => {
  const traces: TraceEventLike[] = [
    { type: 'thought', content: 'Found the answer' },
    { type: 'observation', content: 'Branch completed.' },
  ];
  assert.ok(!isRunExhausted(traces, '1+1等于2。'));
});

test('isRunExhausted: returns false for empty traces', () => {
  assert.ok(!isRunExhausted([], 'normal answer'));
});

// ── compressExhaustedRun ───────────────────────────────────────────────────

test('compressExhaustedRun: extracts tool trace', () => {
  const traces: TraceEventLike[] = [
    { type: 'action', content: 'Calling search — find config files', metadata: { toolName: 'search', goal: 'find config files' } },
    { type: 'observation', content: 'Found 3 files...' },
    { type: 'action', content: 'Calling read — read package.json', metadata: { toolName: 'read', goal: 'read package.json' } },
    { type: 'observation', content: '{"name": "test"}' },
    { type: 'observation', content: 'Branch completed after hitting step budget reached.' },
  ];

  const record = compressExhaustedRun('What is this project?', traces, 'Branch B0 stopped because step budget.');
  assert.ok(record.toolTrace.length === 2, `Expected 2 tools, got ${record.toolTrace.length}`);
  assert.ok(record.toolTrace[0].includes('search'), 'Should include search');
  assert.ok(record.toolTrace[1].includes('read'), 'Should include read');
  assert.ok(record.summary.includes('Previous exhausted run'), 'Should have carry-over header');
  assert.ok(record.summary.includes('Tools used'), 'Should mention tools');
  assert.ok(record.tokenEstimate > 0, 'Token estimate should be positive');
});

test('compressExhaustedRun: limits tool trace to 12', () => {
  const traces: TraceEventLike[] = [];
  for (let i = 0; i < 20; i++) {
    traces.push({
      type: 'action',
      content: `Calling tool_${i}`,
      metadata: { toolName: `tool_${i}`, goal: `goal ${i}` },
    });
  }
  const record = compressExhaustedRun('test', traces, 'exhausted');
  assert.ok(record.toolTrace.length <= 12, `Tool trace should be <= 12, got ${record.toolTrace.length}`);
});

test('compressExhaustedRun: includes partial answer', () => {
  const longAnswer = 'This is a partial answer with some useful findings. '.repeat(5);
  const record = compressExhaustedRun('test query', [], longAnswer);
  assert.ok(record.partialAnswers.length > 0, 'Should include partial answer');
  assert.ok(record.summary.includes('Partial results'), 'Summary should mention partial results');
});

test('compressExhaustedRun: respects maxChars', () => {
  const traces: TraceEventLike[] = [];
  for (let i = 0; i < 50; i++) {
    traces.push({ type: 'observation', content: 'x'.repeat(500) });
  }
  const record = compressExhaustedRun('test', traces, 'x'.repeat(2000), 1500);
  assert.ok(record.summary.length <= 1500, `Summary should be <= 1500 chars, got ${record.summary.length}`);
});

test('compressExhaustedRun: deduplicates tool calls', () => {
  const traces: TraceEventLike[] = [
    { type: 'action', content: 'Calling search', metadata: { toolName: 'search', goal: 'find files' } },
    { type: 'action', content: 'Calling search', metadata: { toolName: 'search', goal: 'find files' } },
    { type: 'action', content: 'Calling search', metadata: { toolName: 'search', goal: 'different query' } },
  ];
  const record = compressExhaustedRun('test', traces, 'answer');
  assert.equal(record.toolTrace.length, 2, 'Should deduplicate identical tool calls');
});

// ── SessionCompressor ──────────────────────────────────────────────────────

test('SessionCompressor: no carry-over initially', () => {
  const sc = new SessionCompressor();
  assert.equal(sc.getCarryOver('conv-1'), null);
  assert.ok(!sc.hasCarryOver('conv-1'));
});

test('SessionCompressor: record and retrieve carry-over', () => {
  const sc = new SessionCompressor();
  const traces: TraceEventLike[] = [
    { type: 'action', content: 'Calling search', metadata: { toolName: 'search' } },
    { type: 'observation', content: 'Branch completed after hitting step budget reached.' },
  ];

  sc.recordExhaustedRun('conv-1', 'What is this project?', traces, 'Branch B0 stopped.');

  assert.ok(sc.hasCarryOver('conv-1'));
  const carry = sc.getCarryOver('conv-1');
  assert.ok(carry !== null);
  assert.equal(carry!.runCount, 1);
  assert.ok(carry!.text.includes('Previous exhausted run'));
  assert.ok(carry!.tokenEstimate > 0);
});

test('SessionCompressor: accumulates multiple exhausted runs', () => {
  const sc = new SessionCompressor();
  const traces: TraceEventLike[] = [
    { type: 'observation', content: 'budget reached' },
  ];

  sc.recordExhaustedRun('conv-1', 'Query 1', traces, 'partial answer 1');
  sc.recordExhaustedRun('conv-1', 'Query 2', traces, 'partial answer 2');

  const carry = sc.getCarryOver('conv-1');
  assert.ok(carry !== null);
  assert.equal(carry!.runCount, 2);
  assert.ok(carry!.text.includes('Query 1'));
  assert.ok(carry!.text.includes('Query 2'));
});

test('SessionCompressor: limits to maxExhaustedRuns', () => {
  const sc = new SessionCompressor({ maxExhaustedRuns: 2 });
  const traces: TraceEventLike[] = [];

  sc.recordExhaustedRun('conv-1', 'Query 1', traces, 'a1');
  sc.recordExhaustedRun('conv-1', 'Query 2', traces, 'a2');
  sc.recordExhaustedRun('conv-1', 'Query 3', traces, 'a3');

  const carry = sc.getCarryOver('conv-1');
  assert.ok(carry !== null);
  assert.equal(carry!.runCount, 2);
  // Oldest (Query 1) should be dropped.
  assert.ok(!carry!.text.includes('Query 1'), 'Oldest run should be dropped');
  assert.ok(carry!.text.includes('Query 2'));
  assert.ok(carry!.text.includes('Query 3'));
});

test('SessionCompressor: clearCarryOver removes state', () => {
  const sc = new SessionCompressor();
  sc.recordExhaustedRun('conv-1', 'Q', [], 'A');
  assert.ok(sc.hasCarryOver('conv-1'));

  sc.clearCarryOver('conv-1');
  assert.ok(!sc.hasCarryOver('conv-1'));
  assert.equal(sc.getCarryOver('conv-1'), null);
});

test('SessionCompressor: conversations are isolated', () => {
  const sc = new SessionCompressor();
  sc.recordExhaustedRun('conv-1', 'Q1', [], 'A1');
  sc.recordExhaustedRun('conv-2', 'Q2', [], 'A2');

  assert.ok(sc.hasCarryOver('conv-1'));
  assert.ok(sc.hasCarryOver('conv-2'));

  sc.clearCarryOver('conv-1');
  assert.ok(!sc.hasCarryOver('conv-1'));
  assert.ok(sc.hasCarryOver('conv-2'));
});

test('SessionCompressor: carry-over text is structured', () => {
  const sc = new SessionCompressor();
  const traces: TraceEventLike[] = [
    { type: 'action', content: 'Calling search — look for patterns', metadata: { toolName: 'search', goal: 'look for patterns' } },
    { type: 'observation', content: 'Found interesting_file.ts matching the pattern with 42 results' },
    { type: 'action', content: 'Calling read — read interesting_file.ts', metadata: { toolName: 'read', goal: 'read interesting_file.ts' } },
    { type: 'observation', content: 'export class Agent {\n  constructor() {...}\n  run() {...}\n}' },
    { type: 'observation', content: 'Branch completed after hitting total budget reached.' },
  ];

  sc.recordExhaustedRun('conv-1', 'Analyze the Agent class architecture', traces, 'The Agent class has a constructor and run method...');

  const carry = sc.getCarryOver('conv-1')!;
  assert.ok(carry.text.includes('search — look for patterns'), 'Should include tool with goal');
  assert.ok(carry.text.includes('read — read interesting_file.ts'), 'Should include read tool');
  assert.ok(carry.text.includes('Run exhausted step budget'), 'Should include status');
  assert.ok(carry.text.includes('Agent class'), 'Should include partial answer content');
});

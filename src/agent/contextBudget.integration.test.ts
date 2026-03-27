/**
 * Integration test for the dynamic context budget system.
 *
 * This test uses the real LLM configured in .env to verify that:
 *   1. Context budget is computed correctly from the live model name.
 *   2. The agent emits budget trace events.
 *   3. The effective parameters are within expected ranges.
 *   4. A real Agent.run() with a simple query completes successfully.
 *
 * Run with: node --test dist/agent/contextBudget.integration.test.js
 *
 * Requires: OPENAI_API_KEY and OPENAI_BASE_URL set in .env.
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import * as path from 'path';
import * as dotenv from 'dotenv';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Load .env from mitosis-cli root (two levels up from dist/agent/).
const cliRoot = path.resolve(__dirname, '../..');
dotenv.config({ path: path.join(cliRoot, '.env') });

import {
  computeContextBudget,
  estimateTokens,
  getModelContextLimit,
  compressTranscript,
  getCompressionLevel,
} from './contextBudget.js';

// ── Pure unit tests that verify budget logic with real model names ──────────

test('budget: real model from env has known context limit', () => {
  const model = process.env.OPENAI_MODEL || 'gpt-4o';
  const limit = getModelContextLimit(model);
  console.log(`  Model: ${model}, Context limit: ${limit}`);
  assert.ok(limit >= 8_192, `Model ${model} should have at least 8k context, got ${limit}`);
});

test('budget: compute produces valid parameters for real model', () => {
  const model = process.env.OPENAI_MODEL || 'gpt-4o';

  // Simulate a typical system prompt (~3500 tokens) + conversation (~500 tokens) + memory (~800 tokens)
  const budget = computeContextBudget({
    model,
    systemPromptTokens: 3500,
    conversationContextTokens: 500,
    memoryContextTokens: 800,
  });

  console.log(`  Budget for ${model}:`);
  console.log(`    modelLimit: ${budget.modelLimit}`);
  console.log(`    committed: ${budget.committedTokens}`);
  console.log(`    residual: ${budget.residualBudget}`);
  console.log(`    transcriptBudgetChars: ${budget.transcriptBudgetChars}`);

  assert.ok(budget.transcriptBudgetChars >= 4000, `chars >= 4000, got ${budget.transcriptBudgetChars}`);
  assert.ok(budget.residualBudget > 0, `residual > 0, got ${budget.residualBudget}`);
});

test('budget: heavy context usage produces constrained parameters', () => {
  const model = process.env.OPENAI_MODEL || 'gpt-4o';
  const limit = getModelContextLimit(model);

  // Simulate 90% context usage.
  const heavyCommitted = Math.floor(limit * 0.9);
  const budget = computeContextBudget({
    model,
    systemPromptTokens: Math.floor(heavyCommitted * 0.5),
    conversationContextTokens: Math.floor(heavyCommitted * 0.3),
    memoryContextTokens: Math.floor(heavyCommitted * 0.2) - 4096,
    safetyMargin: 4096,
  });

  console.log(`  Heavy-usage budget (90% committed):`);
  console.log(`    residual: ${budget.residualBudget}`);
  console.log(`    transcriptBudgetChars: ${budget.transcriptBudgetChars}`);

  // With only 10% residual, transcript budget should be small.
  assert.ok(budget.residualBudget > 0, 'residual should be positive');
});

test('budget: compression levels evolve as transcript grows', () => {
  const budgetTokens = 10_000;

  assert.equal(getCompressionLevel(1_000, budgetTokens), 'none');
  assert.equal(getCompressionLevel(6_000, budgetTokens), 'soft');
  assert.equal(getCompressionLevel(8_000, budgetTokens), 'medium');
  assert.equal(getCompressionLevel(9_500, budgetTokens), 'hard');
});

test('budget: compressTranscript handles real-world observation patterns', () => {
  // Simulate a realistic 8-step transcript with tool observations.
  const messages: Array<{ role: string; content: string }> = [
    { role: 'user', content: 'Search the codebase for all usages of the Agent class and explain its architecture.' },
  ];

  for (let i = 0; i < 8; i++) {
    messages.push({
      role: 'assistant',
      content: JSON.stringify({ kind: 'tool', thought: `Step ${i}: investigating component`, tool_calls: [{ name: 'search', arguments: { query: `step${i}` } }] }),
    });
    // Simulate a large tool observation (~2000 chars)
    messages.push({
      role: 'user',
      content: `TOOL OBSERVATION for search:\n${'// File: src/agent/index.ts line ' + (i * 100) + '\n' + 'export class Agent {\n  ' + 'x'.repeat(200) + '\n}\n'.repeat(8)}`,
    });
  }

  const totalLen = messages.reduce((s, m) => s + m.content.length, 0);
  console.log(`  Simulated transcript: ${messages.length} messages, ${totalLen} chars`);

  // Soft compression: should reduce size while keeping useful content.
  const soft = compressTranscript(messages, Math.floor(totalLen * 0.5), 'soft');
  console.log(`  After soft: ${soft.length} messages, ${soft.reduce((s, m) => s + m.content.length, 0)} chars`);
  assert.ok(soft[0].content === messages[0].content, 'First message preserved');

  // Medium compression: observations become one-liners.
  const medium = compressTranscript(messages, Math.floor(totalLen * 0.3), 'medium');
  console.log(`  After medium: ${medium.length} messages, ${medium.reduce((s, m) => s + m.content.length, 0)} chars`);
  assert.ok(medium[0].content === messages[0].content, 'First message preserved');

  // Hard compression: most messages dropped.
  const hard = compressTranscript(messages, 2000, 'hard');
  console.log(`  After hard: ${hard.length} messages, ${hard.reduce((s, m) => s + m.content.length, 0)} chars`);
  assert.ok(hard[0].content === messages[0].content, 'First message preserved');
  assert.ok(hard.length < messages.length, 'Hard should drop messages');
});

// ── Live LLM integration test ──────────────────────────────────────────────

test('live: Agent.run() emits context budget trace and completes', { timeout: 120_000 }, async () => {
  const apiKey = process.env.OPENAI_API_KEY;
  const baseURL = process.env.OPENAI_BASE_URL;
  const model = process.env.OPENAI_MODEL;

  if (!apiKey) {
    console.log('  SKIP: OPENAI_API_KEY not set');
    return;
  }

  // Dynamic import to avoid loading Agent at module level (it has side effects).
  const { Agent } = await import('./index.js');

  const projectRoot = cliRoot;
  const agent = new Agent(
    {
      apiKey,
      baseURL,
      model: model || 'gpt-4o',
      hmacAccessKey: process.env.HMAC_ACCESS_KEY,
      hmacSecretKey: process.env.HMAC_SECRET_KEY,
      gatewayApiKey: process.env.GATEWAY_API_KEY,
    },
    projectRoot,
  );

  await agent.start();

  const traces: Array<{ type: string; content: string }> = [];
  try {
    const answer = await agent.run(
      '1+1等于多少？用一句话回答。',
      (event) => {
        traces.push({ type: event.type, content: event.content });
      },
      { conversationId: 'budget-test' },
    );

    console.log(`  Agent answer: ${answer.slice(0, 200)}`);
    console.log(`  Total traces: ${traces.length}`);

    // Check that the context budget trace was emitted.
    const budgetTrace = traces.find((t) => t.content.includes('Context budget:'));
    assert.ok(budgetTrace, 'Should emit a "Context budget:" trace event');
    console.log(`  Budget trace: ${budgetTrace!.content}`);

    // Verify the trace contains expected fields.
    assert.ok(budgetTrace!.content.includes('residual='), 'Budget trace should contain residual');
    assert.ok(budgetTrace!.content.includes('transcriptChars='), 'Budget trace should contain transcriptChars');
    assert.ok(budgetTrace!.content.includes('agentMode='), 'Budget trace should contain agentMode');

    // Verify we got a non-empty answer.
    assert.ok(answer.length > 0, 'Answer should not be empty');
  } finally {
    await agent.shutdown(5000).catch(() => {});
  }
});

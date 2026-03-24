import test from 'node:test';
import assert from 'node:assert/strict';
import os from 'node:os';
import path from 'node:path';
import fs from 'node:fs';
import { Agent, type TraceEvent } from './index.js';

process.env.AUTO_QUEUE_MEMORY_SAVE = '0';
process.env.REACT_BEAM_SEARCH_ENABLED = '0';

function createTempProjectRoot(prefix: string): string {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), prefix));
  fs.mkdirSync(path.join(dir, '.mempedia', 'memory', 'index'), { recursive: true });
  return dir;
}

function installAgentTestDoubles(agent: Agent, answerForRequest: (request: string) => Promise<string> | string): void {
  const anyAgent = agent as any;
  anyAgent.retrieveRelevantContext = async () => ({
    contextText: '',
    recalledNodeIds: [],
    selectedNodeIds: [],
    rationale: 'test',
  });
  anyAgent.generateJsonPromptText = async ({ messages }: { messages: Array<{ content?: string }> }) => {
    const joined = Array.isArray(messages)
      ? messages.map((message) => String(message?.content || '')).join('\n\n')
      : '';
    const requestMatch = joined.match(/Original user request:\n([\s\S]*?)(?:\n\nStart with the root loop\.|\n\nActive branch:|$)/);
    const request = requestMatch?.[1]?.trim() || 'unknown';
    const answer = await answerForRequest(request);
    return JSON.stringify({
      kind: 'final',
      thought: 'Return final answer.',
      final_answer: answer,
      completion_summary: answer,
    });
  };
}

test('conversationId isolates follow-up grounding state', async () => {
  const projectRoot = createTempProjectRoot('mempedia-agent-conv-');
  const agent = new Agent({ apiKey: 'test-key' }, projectRoot);
  installAgentTestDoubles(agent, (request) => `ok:${request.slice(0, 8)}`);

  const tracesA1: TraceEvent[] = [];
  await agent.run('first topic', (event) => { tracesA1.push(event); }, { conversationId: 'thread-A' });
  assert.ok(tracesA1.some((event) => event.content.includes('Selected 0 recent conversation turns')));

  const tracesB: TraceEvent[] = [];
  await agent.run('继续', (event) => { tracesB.push(event); }, { conversationId: 'thread-B' });
  assert.ok(tracesB.some((event) => event.content.includes('Selected 0 recent conversation turns')));

  const tracesA2: TraceEvent[] = [];
  await agent.run('继续', (event) => { tracesA2.push(event); }, { conversationId: 'thread-A' });
  assert.ok(tracesA2.some((event) => event.content.includes('Selected 1 relevant recent conversation turn(s)')));

  agent.stop();
});

test('one Agent instance supports concurrent runs across different conversation ids', async () => {
  const projectRoot = createTempProjectRoot('mempedia-agent-concurrent-');
  const agent = new Agent({ apiKey: 'test-key' }, projectRoot);
  installAgentTestDoubles(agent, async (request) => {
    await new Promise((resolve) => setTimeout(resolve, 40));
    return request.includes('alpha') ? 'ok:alpha' : 'ok:beta';
  });

  const [alpha, beta] = await Promise.all([
    agent.run('alpha request', () => {}, { conversationId: 'thread-alpha', sessionId: 'session-alpha' }),
    agent.run('beta request', () => {}, { conversationId: 'thread-beta', sessionId: 'session-beta' }),
  ]);

  assert.equal(alpha, 'ok:alpha');
  assert.equal(beta, 'ok:beta');

  const tracesAlphaFollowUp: TraceEvent[] = [];
  await agent.run('继续', (event) => { tracesAlphaFollowUp.push(event); }, { conversationId: 'thread-alpha' });
  assert.ok(tracesAlphaFollowUp.some((event) => event.content.includes('Selected 1 relevant recent conversation turn(s)')));

  const tracesGamma: TraceEvent[] = [];
  await agent.run('继续', (event) => { tracesGamma.push(event); }, { conversationId: 'thread-gamma' });
  assert.ok(tracesGamma.some((event) => event.content.includes('Selected 0 recent conversation turns')));

  agent.stop();
});

test('beam planner forces structured outputs for custom model ids', () => {
  const projectRoot = createTempProjectRoot('mempedia-agent-structured-');
  const agent = new Agent({ apiKey: 'test-key', model: 'Doubao-Seed-2.0-pro' }, projectRoot);
  const anyAgent = agent as any;

  assert.equal(anyAgent.beamPlannerStructuredOpenai.supportsStructuredOutputs, true);
  assert.equal(anyAgent.beamPlannerCompatOpenai.supportsStructuredOutputs, false);

  agent.stop();
});
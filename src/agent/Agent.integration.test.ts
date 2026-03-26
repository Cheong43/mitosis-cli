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

test('unwraps raw planner JSON from final answer before returning to user', async () => {
  const projectRoot = createTempProjectRoot('mempedia-agent-final-unpack-');
  const agent = new Agent({ apiKey: 'test-key' }, projectRoot);
  installAgentTestDoubles(agent, () => '{"kind":"final","final_answer":"这是正常回答"}');

  const answer = await agent.run('测试请求', () => {}, { conversationId: 'thread-json-wrap' });
  assert.equal(answer, '这是正常回答');

  agent.stop();
});

test('planner fallback returns natural language error instead of echoing raw bash blocks', async () => {
  const projectRoot = createTempProjectRoot('mempedia-agent-fallback-natural-');
  const agent = new Agent({ apiKey: 'test-key' }, projectRoot);
  const anyAgent = agent as any;
  anyAgent.retrieveRelevantContext = async () => ({
    contextText: '',
    recalledNodeIds: [],
    selectedNodeIds: [],
    rationale: 'test',
  });
  anyAgent.generateJsonPromptText = async () => '```bash\nfind . -name "mempedia"\nls -la ./target/release/\n```';

  const answer = await agent.run('我已经重新build', () => {}, { conversationId: 'thread-fallback-natural' });
  assert.match(answer, /内部规划器输出成了命令或草稿/);
  assert.doesNotMatch(answer, /```bash/);
  assert.doesNotMatch(answer, /find \./);

  agent.stop();
});

test('persists recent conversation context across agent instances for the same conversation id', async () => {
  const projectRoot = createTempProjectRoot('mempedia-agent-persisted-context-');
  const conversationId = 'thread:persisted-context';

  const agent1 = new Agent({ apiKey: 'test-key' }, projectRoot);
  installAgentTestDoubles(agent1, () => '第一轮回答');
  await agent1.run('请记住这个话题：美光财报保存到 Mempedia', () => {}, { conversationId });
  agent1.stop();

  const statePath = path.join(projectRoot, '.mitosis', 'conversation_state', 'thread_persisted-context.json');
  assert.ok(fs.existsSync(statePath), 'expected persisted conversation state file to be created');

  const agent2 = new Agent({ apiKey: 'test-key' }, projectRoot);
  installAgentTestDoubles(agent2, (request) => `follow-up:${request}`);
  const traces: TraceEvent[] = [];
  await agent2.run('继续', (event) => { traces.push(event); }, { conversationId });

  assert.ok(traces.some((event) => event.content.includes('Recovered persisted thread context with 1 recent turn(s).')));
  assert.ok(traces.some((event) => event.content.includes('Selected 1 relevant recent conversation turn(s)')));
  assert.ok(traces.some((event) => event.content.includes('Injected persisted thread working state into the current run context.')));

  agent2.stop();
});

test('writes a turn summary journal entry for each persisted turn', async () => {
  const projectRoot = createTempProjectRoot('mempedia-agent-turn-summary-');
  const conversationId = 'thread:journal-check';
  const agent = new Agent({ apiKey: 'test-key' }, projectRoot);
  installAgentTestDoubles(agent, () => '这是用于摘要日志的回答。');

  await agent.run('记录这轮摘要', () => {}, { conversationId });

  const journalPath = path.join(projectRoot, '.mitosis', 'logs', 'thread_turn_summaries.jsonl');
  assert.ok(fs.existsSync(journalPath), 'expected turn summary journal to exist');
  const rows = fs.readFileSync(journalPath, 'utf-8').trim().split('\n').filter(Boolean).map((line) => JSON.parse(line));
  assert.equal(rows.length, 1);
  assert.equal(rows[0].conversation_id, conversationId);
  assert.equal(rows[0].user_intent, '记录这轮摘要');
  assert.match(rows[0].assistant_outcome, /这是用于摘要日志的回答/);
  assert.ok(Array.isArray(rows[0].tool_findings));

  agent.stop();
});

test('reconstructs thread context from turn summary journal when state file is missing', async () => {
  const projectRoot = createTempProjectRoot('mempedia-agent-journal-recovery-');
  const conversationId = 'thread:journal-recovery';

  const agent1 = new Agent({ apiKey: 'test-key' }, projectRoot);
  installAgentTestDoubles(agent1, () => '上一轮已经确认目标是继续保存美光财报。');
  await agent1.run('请继续跟进美光财报保存', () => {}, { conversationId });
  agent1.stop();

  const statePath = path.join(projectRoot, '.mitosis', 'conversation_state', 'thread_journal-recovery.json');
  if (fs.existsSync(statePath)) {
    fs.unlinkSync(statePath);
  }

  const agent2 = new Agent({ apiKey: 'test-key' }, projectRoot);
  installAgentTestDoubles(agent2, () => '已从 journal 恢复上下文。');
  const traces: TraceEvent[] = [];
  await agent2.run('继续', (event) => { traces.push(event); }, { conversationId });

  assert.ok(traces.some((event) => event.content.includes('Recovered persisted thread context with 1 recent turn(s).')));
  assert.ok(traces.some((event) => event.content.includes('Selected 1 relevant turn summary record(s) from journal replay.')));
  assert.ok(traces.some((event) => event.content.includes('Injected journal replay summaries into the current run context.')));

  agent2.stop();
});
